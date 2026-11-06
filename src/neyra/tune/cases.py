"""The user's held-out cases, the scorers that grade an answer, and the examples that go into a few-shot prompt."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class CaseError(ValueError):
    pass


@dataclass
class Case:
    id: str
    input: str
    output: str
    system: str = ""


def _text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(m.get("content", "") for m in value if isinstance(m, dict))
    return None


def _row(line: int, obj: dict) -> Case:
    """Accepts input/output, input/expected, prompt/completion and a messages list ending in an assistant turn."""
    system = obj.get("system") or ""
    if isinstance(obj.get("messages"), list):
        msgs = obj["messages"]
        system = system or next((m["content"] for m in msgs if m.get("role") == "system"), "")
        turns = [m for m in msgs if m.get("role") in ("user", "assistant")]
        if len(turns) < 2 or turns[-1]["role"] != "assistant" or turns[-2]["role"] != "user":
            raise CaseError(f"line {line}: a messages row has to end with a user message and the assistant's answer")
        # Earlier turns ride along as text, so a multi step conversation scores on its final answer.
        earlier = "".join(f"{m['role']}: {m['content']}\n" for m in turns[:-2])
        inp, out = earlier + turns[-2]["content"], turns[-1]["content"]
    else:
        inp = _text(obj.get("input", obj.get("prompt")))
        out = _text(obj.get("output", obj.get("expected", obj.get("completion"))))
    if inp is None or out is None:
        raise CaseError(f"line {line}: a case needs an input and an expected output "
                        f"(input/output, input/expected, prompt/completion or messages)")
    return Case(str(obj.get("id", line)), inp, out, system)


def read(path: str) -> list[Case]:
    out: list[Case] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise CaseError(f"cannot read {path}: {e.strerror or e}") from e
    for n, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as e:
            raise CaseError(f"{path} line {n} is not JSON: {e}") from e
        if not isinstance(obj, dict):
            raise CaseError(f"{path} line {n} is not a JSON object")
        out.append(_row(n, obj))
    if not out:
        raise CaseError(f"{path} has no cases")
    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise CaseError(f"{path}: case ids are not unique, so per-case results could not be matched")
    return out


Scorer = Callable[[str, Case], float]


def scorer(spec: str) -> Scorer:
    """exact, contains, regex:PATTERN or python:FILE. A score is 0 to 1."""
    kind, _, arg = spec.partition(":")
    if kind == "exact" and not arg:
        return lambda out, c: float(out.strip().casefold() == c.output.strip().casefold())
    if kind == "contains" and not arg:
        return lambda out, c: float(c.output.strip().casefold() in out.casefold())
    if kind == "regex":
        if not arg:
            raise CaseError("regex needs a pattern: --scorer 'regex:^label: {expected}$'. "
                            "{expected} stands for the expected output, escaped")
        try:
            re.compile(arg.replace("{expected}", ""))
        except re.error as e:
            raise CaseError(f"the regex does not compile: {e}") from e
        return lambda out, c: float(bool(re.search(arg.replace("{expected}", re.escape(c.output.strip())), out,
                                                   re.IGNORECASE | re.MULTILINE)))
    if kind == "python":
        return _python(arg)
    raise CaseError(f"unknown scorer {spec!r}: use exact, contains, regex:PATTERN or python:FILE")


def _python(path: str) -> Scorer:
    if not path or not Path(path).is_file():
        raise CaseError(f"python scorer: no file {path!r}")
    spec = importlib.util.spec_from_file_location("neyra_user_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "score", None)
    if not callable(fn):
        raise CaseError(f"{path} must define score(output, expected, case) returning a number from 0 to 1 or a bool")

    def run(out: str, c: Case) -> float:
        value = float(fn(out, c.output, {"id": c.id, "input": c.input, "system": c.system}))
        if not 0.0 <= value <= 1.0:
            raise CaseError(f"{path}: score() returned {value} for case {c.id}; scores run from 0 to 1")
        return value

    return run


def examples(pool: list[Case], k: int, seed: int = 0) -> list[Case]:
    """k examples. When the answers repeat, as labels do, they are taken in turn from each distinct answer."""
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)
    groups: dict[str, list[Case]] = {}
    for c in pool:
        groups.setdefault(c.output, []).append(c)
    if len(groups) > 3 * k or len(groups) == len(pool):
        return pool[:k]
    chosen: list[Case] = []
    lists = list(groups.values())
    while len(chosen) < k and any(lists):
        for g in lists:
            if g and len(chosen) < k:
                chosen.append(g.pop())
    rng.shuffle(chosen)
    return chosen
