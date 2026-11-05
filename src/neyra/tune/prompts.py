"""Turn cases into the prompt each arm sees. Base, few-shot and tuned differ only in what they are given as examples."""

from __future__ import annotations

from dataclasses import dataclass

from .cases import Case


@dataclass
class Prompt:
    messages: list[dict] | None = None
    text: str | None = None


@dataclass
class Format:
    chat: bool
    template: str = "{input}"
    stop: tuple[str, ...] = ()
    system: str = ""  # used for cases that carry none


def _plain(fmt: Format, c: Case) -> str:
    return fmt.template.replace("{input}", c.input)


def _joiner(piece: str) -> str:
    return "" if piece[-1:].isspace() else " "


def render(fmt: Format, case: Case, shots: list[Case] = ()) -> Prompt:
    system = case.system or fmt.system
    if fmt.chat:
        msgs = [{"role": "system", "content": system}] if system else []
        for s in shots:
            msgs += [{"role": "user", "content": s.input}, {"role": "assistant", "content": s.output}]
        msgs.append({"role": "user", "content": case.input})
        return Prompt(messages=msgs)
    parts = [system] if system else []
    for s in shots:
        head = _plain(fmt, s)
        parts.append(head + _joiner(head) + s.output)
    parts.append(_plain(fmt, case))
    return Prompt(text="\n\n".join(parts))


def cut(text: str, stop: tuple[str, ...]) -> str:
    for s in stop:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text.strip()


def encode(tokenizer, p: Prompt) -> list[int]:
    if p.messages is not None:
        try:
            text = tokenizer.apply_chat_template(p.messages, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(p.messages, add_generation_prompt=True, tokenize=False)
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(tokenizer.encode(p.text, add_special_tokens=True))
