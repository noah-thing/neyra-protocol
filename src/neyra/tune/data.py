"""Read a jsonl file in the shapes trainers accept and reduce each line to one of three row types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant",
         "bot": "assistant", "system": "system", "tool": "tool", "observation": "tool", "function": "tool"}


@dataclass
class Row:
    line: int
    shape: str                                  # "messages", "pair" or "text"
    messages: list[dict] = field(default_factory=list)
    prompt: str | list | None = None
    completion: str | list | None = None
    text: str = ""
    source: str = ""                            # the column layout the line came from
    tools: list | None = None

    @property
    def conversational(self) -> bool:
        return self.shape == "messages" or (self.shape == "pair" and isinstance(self.prompt, list))

    def as_messages(self) -> list[dict]:
        """The row as a chat, for the trainers that put a prompt and a completion through the template."""
        if self.shape == "messages":
            return self.messages
        if self.shape == "pair":
            if isinstance(self.prompt, list):
                return list(self.prompt) + list(self.completion)
            return [{"role": "user", "content": self.prompt}, {"role": "assistant", "content": self.completion}]
        return []

    def system(self) -> str:
        for m in self.as_messages():
            if m["role"] == "system":
                return m["content"]
        return ""

    def answers(self) -> list[str]:
        return [m["content"] for m in self.as_messages() if m["role"] == "assistant"]

    def plain(self) -> str:
        """All the words of the row, for duplicate and leakage checks."""
        if self.shape == "text":
            return self.text
        if self.shape == "pair" and not isinstance(self.prompt, list):
            return f"{self.prompt}\n{self.completion}"
        return "\n".join(m["content"] for m in self.as_messages())


class Unreadable(ValueError):
    pass


def _content(value, where: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in value):
        return "".join(p["text"] for p in value)
    if value is None:
        return ""
    raise Unreadable(f"{where} is not a string")


def _message(m, i: int) -> dict:
    if not isinstance(m, dict):
        raise Unreadable(f"message {i} is not an object")
    if "role" in m:
        role = ROLES.get(str(m["role"]).lower(), str(m["role"]).lower())
        content = m.get("content")
    elif "from" in m:
        role = ROLES.get(str(m["from"]).lower())
        content = m.get("value")
        if role is None:
            raise Unreadable(f"message {i} has the unknown speaker {m['from']!r}")
    else:
        raise Unreadable(f"message {i} has no role")
    out = {k: v for k, v in m.items() if k not in ("from", "value", "role", "content")}
    out.update(role=role, content=_content(content, f"message {i} content"))
    if out["role"] == "assistant" and not out["content"] and not out.get("tool_calls"):
        raise Unreadable(f"message {i} is an empty answer")
    return out


def _chat(value) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise Unreadable("the conversation is empty or not a list")
    return [_message(m, i) for i, m in enumerate(value)]


def parse(obj, line: int) -> Row:
    if not isinstance(obj, dict):
        raise Unreadable("the line is not a JSON object")
    tools = obj.get("tools")
    for key in ("messages", "conversations", "conversation"):
        if key in obj:
            msgs = _chat(obj[key])
            if isinstance(obj.get("system"), str) and msgs[0]["role"] != "system":
                msgs.insert(0, {"role": "system", "content": obj["system"]})
            if msgs[-1]["role"] != "assistant":
                raise Unreadable("the conversation does not end with an answer")
            return Row(line, "messages", msgs, source=key if key != "messages" else "messages", tools=tools)
    if "prompt" in obj and "completion" in obj:
        p, c = obj["prompt"], obj["completion"]
        if isinstance(p, list) or isinstance(c, list):
            return Row(line, "pair", prompt=_chat(p), completion=_chat(c), source="prompt+completion", tools=tools)
        return Row(line, "pair", prompt=_content(p, "prompt"), completion=_content(c, "completion"),
                   source="prompt+completion", tools=tools)
    if "instruction" in obj and "output" in obj:
        ask = _content(obj["instruction"], "instruction")
        extra = _content(obj.get("input"), "input")
        prompt = [{"role": "user", "content": f"{ask}\n{extra}" if extra else ask}]
        if isinstance(obj.get("system"), str) and obj["system"]:
            prompt.insert(0, {"role": "system", "content": obj["system"]})
        return Row(line, "pair", prompt=prompt, completion=[{"role": "assistant", "content": _content(obj["output"], "output")}],
                   source="alpaca")
    if "text" in obj:
        return Row(line, "text", text=_content(obj["text"], "text"), source="text")
    raise Unreadable("none of messages, prompt and completion, instruction and output, or text is present")


def read(path: str | Path):
    """Yield (Row or None, problem) for every non-blank line."""
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                yield parse(json.loads(raw), n), None
            except json.JSONDecodeError:
                yield None, (n, "the line is not valid JSON")
            except Unreadable as e:
                yield None, (n, str(e))
