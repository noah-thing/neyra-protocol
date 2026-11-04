"""Measure how a generation went."""

from __future__ import annotations

import re
from dataclasses import dataclass

THINK_END = "</think>"
GRAM = 16
LOOP_TAIL = 512
SHORT_BUDGET = 4096
SHORT_LOOP_TAIL = 256

try:
    from math_verify import parse as _parse
    from math_verify import verify as _verify
except ImportError:
    _parse = _verify = None


def repeat_flags(ids: list[int], n: int = GRAM) -> list[bool]:
    """True where the n tokens ending here already appeared earlier in the text."""
    flags = [False] * len(ids)
    seen: set[tuple[int, ...]] = set()
    for i in range(n - 1, len(ids)):
        gram = tuple(ids[i - n + 1:i + 1])
        flags[i] = gram in seen
        seen.add(gram)
    return flags


def loop_tail(ids: list[int], n: int = GRAM) -> int:
    """Tokens at the end of a generation that sit inside one unbroken repeat of earlier text.

    Validated on 800 published Qwen3.5-0.8B traces at 32k tokens: a tail of 512 or more
    gives a true positive rate of 0.976 and a false positive rate of 0.020. Word level
    duplicate counts flag a third of healthy reasoning, since careful work restates itself.
    """
    run = 0
    for flagged in reversed(repeat_flags(ids, n)):
        if not flagged:
            break
        run += 1
    return run


def loop_onset(ids: list[int], n: int = GRAM) -> int | None:
    """Estimated token where a terminal loop begins, or None when there is no loop."""
    tail = loop_tail(ids, n)
    return len(ids) - tail - n if tail >= SHORT_LOOP_TAIL else None


def is_looping(ids: list[int], budget: int) -> bool:
    return loop_tail(ids) >= (SHORT_LOOP_TAIL if budget <= SHORT_BUDGET else LOOP_TAIL)


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


def same_answer(got: str, gold: str) -> bool:
    if normalize_answer(got) == normalize_answer(gold):
        return True
    if _verify is None:
        return False
    try:
        return bool(_verify(_parse(f"${gold}$"), _parse(f"${got}$")))
    except Exception:  # math-verify raises on unparsable LaTeX and on its own timeouts
        return False


@dataclass
class Outcome:
    closed: bool
    finished: bool
    tail: int
    looping: bool
    answered: bool
    correct: bool | None


def score(text: str, tokens: list[int], hit_limit: bool, budget: int, answer: str | None = None) -> Outcome:
    closed = THINK_END in text
    # A capped trace with an open think block has not answered, whatever it boxed on the way.
    final = text.split(THINK_END)[-1] if closed or not hit_limit else ""
    got = boxed_answer(final)
    correct = None if answer is None else got is not None and same_answer(got, answer)
    tail = loop_tail(tokens)
    return Outcome(closed, not hit_limit, tail, tail >= (SHORT_LOOP_TAIL if budget <= SHORT_BUDGET else LOOP_TAIL),
                   got is not None, correct)
