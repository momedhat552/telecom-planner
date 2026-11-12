"""Ask the real Gemini agent one question from the command line and print everything that
happened, so a problem can be reproduced without typing in the chat loop.
"""
import sys

from phase7_agent import ask, make_chat, unverified_numbers, MODEL


def main():
    question = " ".join(sys.argv[1:]).strip() or "What are the main assumptions of this tool?"
    chat = make_chat()
    if chat is None:
        return
    print(f"Model: {MODEL}")
    print(f"Question: {question}\n")
    answer, calls = ask(chat, question)
    print("\n--- tool calls")
    if not calls:
        print("(none)")
    for c in calls:
        print(f"- {c['tool']}({c['args']})")
        err = c["result"].get("error") if isinstance(c["result"], dict) else None
        if err:
            print(f"    error: {err}")
    print("\n--- answer")
    print(answer if answer else "(no text)")
    print("\n--- numbers not found in any tool result")
    print(", ".join(unverified_numbers(answer or "", question)) or "(none)")


if __name__ == "__main__":
    main()
