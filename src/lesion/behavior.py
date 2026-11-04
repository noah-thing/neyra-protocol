"""Text level measurements of how a generation went."""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_END = "</think>"


def duplicate_share(text: str, n: int = 8) -> float:
    """Share of word n-grams in the later half that already appeared earlier in the text."""
    words = re.findall(r"\S+", text)
    if len(words) < 2 * n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    half = len(grams) // 2
    seen = set(grams[:half])
    repeats = 0
    for g in grams[half:]:
        repeats += g in seen
        seen.add(g)
    return repeats / max(len(grams) - half, 1)


def boxed_answer(text: str) -> str | None:
    """Content of the last \\boxed{...}, braces balanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed")
    for k in range(j, len(text)):
        depth += (text[k] == "{") - (text[k] == "}")
        if depth == 0:
            return text[j + 1:k].strip()
    return None


def normalize_answer(s: str) -> str:
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s).replace(",", "").replace(" ", "")
    return s.rstrip(".").lower()


@dataclass
class Outcome:
    closed: bool
    finished: bool
    repeated: float
    answered: bool
    correct: bool | None


def score(text: str, hit_limit: bool, answer: str | None = None, loop_at: float = 0.25) -> Outcome:
    got = boxed_answer(text.split(THINK_END)[-1]) if THINK_END in text else None
    correct = None
    if answer is not None:
        correct = got is not None and normalize_answer(got) == normalize_answer(answer)
    return Outcome(THINK_END in text, not hit_limit, duplicate_share(text), got is not None, correct)
