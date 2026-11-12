"""Offline test of the agent tool loop: no API key and no network are needed.

It pretends to be a chat model that keeps asking for a tool and never stops, which is the case
that used to return no text. The loop must cap the rounds and always end with a written answer.
"""
import json

import numpy as np
import phase7_agent as ag


class Call:
    def __init__(self, name, args=None):
        self.name, self.args = name, args or {}


class Response:
    def __init__(self, text=None, calls=None):
        self.text, self.function_calls = text, (calls or [])


class Chat:
    """Pretends to be a Gemini chat: never stops asking for a tool, until it is nudged."""

    def __init__(self, calls_per_turn=2):
        self.sent, self.n = [], 0
        self.calls_per_turn = calls_per_turn

    def send_message(self, message):
        self.sent.append(message)
        if isinstance(message, str) and message == ag.NUDGE:
            return Response(text="There is nothing else I can add.")
        self.n += 1                                # keeps asking, like the reported bug
        return Response(calls=[Call("area_summary") for _ in range(self.calls_per_turn)])


chat = Chat()
answer, calls = ag.ask(chat, "how many cells?", verbose=False)
print("messages sent:", len(chat.sent), "| tool rounds:", len(chat.sent) - 2)
print("answer:", answer)
print("logged calls:", [(c["tool"], c["args"]) for c in calls])
assert answer, "no text came back"
assert len(chat.sent) == ag.MAX_TOOL_ROUNDS + 2, len(chat.sent)   # question + rounds + nudge
assert len(calls) == 1, calls              # the duplicate guard logs the same call only once

# A tool the model invented must not crash the loop.
print(ag.run_tool(Call("no_such_tool")))
# Wrong arguments must not crash either.
print(ag.run_tool(Call("plan_network", {"n_towers": "many"})))
# numpy values must become plain JSON values.
out = ag.plain({"a": np.int64(3), "b": np.float64(1.5), "c": [np.int32(2)]})
print(out, {k: type(v).__name__ for k, v in out.items()})
print("serialisable:", json.dumps(out))
print("OK")
