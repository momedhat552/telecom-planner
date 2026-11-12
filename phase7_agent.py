"""Phase 7: the planning assistant (a Gemini chat model that calls the tools in planner_tools.py).

The model never calculates. It chooses a tool, the tool does the work, and the model explains the
result in words. The tool loop is written out here instead of using the automatic function calling
of the google-genai library, so we always know which tool was called, we can stop after a fixed
number of rounds, and we can never end up with a tool request that silently never runs.

Run it with:  py phase7_agent.py
It needs GEMINI_API_KEY in your .env file (and optionally GEMINI_MODEL).
"""
import os
import re

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()                       # reads GEMINI_API_KEY (and CARTO_API_KEY) from a .env file
except ImportError:
    pass
from google import genai
from google.genai import types
import planner_tools as pt

# The model name that works for you (the same one as in your Telecom Troubleshooter).
# Easiest: add a line GEMINI_MODEL=your-model-name to your .env file. Or type it here instead.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

MAX_TOOL_ROUNDS = 8     # one question may need this many rounds of "call tools, send results back"

SYSTEM_PROMPT = """You are a telecom planning assistant for a square study area (Berlin at the
start, 3 x 3 km). You answer questions about a 5G network plan at 3.5 GHz: coverage, number of
towers, tower placement, frequency groups and what changes when assumptions change.

Rules:
1. Every number you state must come from a tool result in this conversation. If you need a
   number, call a tool. If no tool gives it, say you cannot provide it. Never calculate,
   estimate or remember numbers yourself.
2. Always say which area, method and assumptions a number belongs to (greedy or honeycomb,
   antenna gain, coverage threshold). Only mention an assumption value that a tool result of
   this turn contains. If the user asks about assumptions, call area_summary.
3. When the user wants to see the plan, call draw_plan and tell them the picture file path.
4. Be honest about limits. Demand is a proxy from building footprints, radio values are
   assumptions, and the tower count changes a lot with them. Mention the relevant limit
   when it matters for the answer. Do not oversell small differences between methods.
5. Keep answers short and plain. Explain telecom terms in one short phrase.
6. You are NOT limited to Berlin. If the user names another place or gives coordinates, call
   change_area (it takes about a minute) and then answer for the new area. If the result has a
   data_quality_warning, tell the user about it before giving any plan numbers. If the result
   has place_search_used and it differs from what the user typed, say which search matched.
7. Call each tool at most once per question with the same arguments, and never repeat a call
   that already returned a result or an error. If a tool fails, explain the problem in plain
   words and suggest what the user can change (a different name, coordinates, or trying later).
   If change_area fails, the planner stays on the area it had before; say so.
8. After your tool calls, always finish with the answer written in words. Never end a turn with
   a tool call and no explanation."""

TOOLS = [pt.area_summary, pt.plan_network, pt.towers_needed,
         pt.compare_methods, pt.what_if, pt.draw_plan, pt.change_area]
TOOL_BY_NAME = {t.__name__: t for t in TOOLS}

# One client for the whole program. It must stay alive: the library closes the network
# connection when the client object is deleted, and the chat object does not keep it alive.
_CLIENT = None

NUMBER = re.compile(r"\d+(?:\.\d+)?")
ALWAYS_OK = {1.0, 2.0, 3.0, 3.5}        # list numbers, "3 sectors", "3.5 GHz"
NUDGE = ("Answer in words now, using only the tool results above. Do not call any more tools.")


def numbers_in(obj, out):
    """Collect every number found anywhere inside a tool result."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        out.update(abs(float(m)) for m in NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers_in(v, out)


def unverified_numbers(answer, question):
    """Numbers in the answer that are in no tool result of this turn and not in the question.
    A rough check, it can warn about harmless numbers too."""
    known = set(ALWAYS_OK)
    for call in pt.CALL_LOG:
        numbers_in(call["args"], known)
        numbers_in(call["result"], known)
    known.update(abs(float(m)) for m in NUMBER.findall(question))
    text = re.sub(r"(?<=\d),(?=\d{3})", "", answer)       # 1,064 -> 1064
    text = re.sub(r"\b\d[Gg]\b", "", text)                # "5G", "4G" are names, not results
    found = []
    for m in NUMBER.findall(text):
        x = abs(float(m))
        if not any(abs(x - k) <= 0.06 for k in known) and m not in found:
            found.append(m)
    return found


def plain(value):
    """Tool results with numpy numbers turned into normal Python numbers, so they can be sent
    back to the model as JSON."""
    if isinstance(value, np.generic):           # numpy integer, float or bool
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    try:
        return value.item()                      # numpy integer or float
    except Exception:
        return str(value)


def make_chat(model=None, verbose=True):
    """Create a chat with the planning tools, or return None (after printing why) if the API key
    or the model name is missing. Streamlit and agent_probe.py use this too."""
    global _CLIENT
    model = model or MODEL
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        if verbose:
            print("No API key found. Put GEMINI_API_KEY in your .env file (or set it in the terminal).")
        return None
    if model.startswith("PUT-"):
        if verbose:
            print("Set GEMINI_MODEL in your .env file (or edit MODEL in phase7_agent.py) "
                  "to the Gemini model name that works for you.")
        return None
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
        temperature=0.2,
        # We run the tools ourselves in ask(), so the library must not do it as well.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    return _CLIENT.chats.create(model=model, config=config)


def run_tool(call):
    """Run one tool the model asked for. Never raises: a broken tool becomes an error message
    that the model can explain to the user."""
    name = getattr(call, "name", None) or ""
    args = dict(getattr(call, "args", None) or {})
    fn = TOOL_BY_NAME.get(name)
    if fn is None:
        return name, args, {"error": f"there is no tool called '{name}'. "
                                     f"Available tools: {', '.join(TOOL_BY_NAME)}"}
    try:
        result = fn(**args)
    except TypeError as e:                       # the model sent arguments that do not fit
        result = {"error": f"wrong arguments for {name}: {e}"}
    except Exception as e:                       # tool ran and failed (bad place, busy server...)
        result = {"error": f"{type(e).__name__}: {e}"}
    return name, args, plain(result)


def ask(chat, question, verbose=True):
    """Ask one question. Run every tool the model requests, send the results back, repeat until
    the model answers in words (or we run out of rounds). Returns (answer text, tool calls)."""
    pt.CALL_LOG.clear()
    response = chat.send_message(question)

    rounds = 0
    while response.function_calls and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        parts = []
        for call in response.function_calls:
            name, args, result = run_tool(call)
            if verbose:
                problem = result.get("error") if isinstance(result, dict) else None
                print(f"  tool {rounds}.{len(parts) + 1}: {name}({args})"
                      + (f" -> {problem}" if problem else ""))
            parts.append(types.Part.from_function_response(name=name,
                                                           response={"result": result}))
        response = chat.send_message(parts)

    if not response.text:
        if verbose:
            if response.function_calls:
                names = ", ".join(c.name or "?" for c in response.function_calls)
                print(f"The model still wanted tools after {MAX_TOOL_ROUNDS} rounds ({names}); "
                      "asking for a written answer instead.")
            else:
                print("The model returned no text; asking once more for a written answer.")
        try:
            response = chat.send_message(NUDGE)
        except Exception as e:
            if verbose:
                print("Error from the Gemini API:", e)

    return (response.text or "").strip(), list(pt.CALL_LOG)


def main():
    chat = make_chat()
    if chat is None:
        return
    print("Telecom planning assistant, starting area:", pt.NAME, "(type 'exit' to stop)")
    print("Try: 'How many towers do we need for 95% of demand?', 'Show me the plan with 26 towers.'")
    print("or 'Switch to Paris, France.'")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            answer, calls = ask(chat, question)
        except Exception as e:                  # rate limit, network, wrong model name...
            print("Error from the Gemini API:", e)
            continue
        print("\nPlanner:", answer or "(the model returned no text)")
        if calls:
            print("\nTools used:", ", ".join(f"{c['tool']}({c['args']})" for c in calls))
        else:
            print("\n(No tool was used for this answer.)")
        bad = unverified_numbers(answer, question)
        if bad:
            print("Check: these numbers were not in any tool result this turn:", ", ".join(bad),
                  "(rough check, verify them before trusting the answer)")


if __name__ == "__main__":
    main()
