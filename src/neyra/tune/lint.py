"""Run a dataset through a trainer's pipeline and collect what is wrong with it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import data, similar
from .checks import Example, Finding, Rows
from .tok import Tok
from .trainers import TRAINERS, Built, Session

CHUNK = 256


@dataclass
class Report:
    data: str
    model: str
    trainer: str
    label: str
    exact: bool
    rows: int
    max_length: int
    length_name: str
    masking: str
    findings: list[Finding]
    notes: list[str] = field(default_factory=list)
    shown: list[Built] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0
    tokenizer: Tok | None = None

    @property
    def errors(self) -> int:
        return sum(f.level == "error" for f in self.findings)


def describe_masking(session: Session) -> str:
    mask = session.mask
    if mask is None:
        return "the completion of prompt and completion rows, every token of chat and text rows"
    return "the answers only" if mask else "every token"


def lint(path: str, model: str, trainer: str = "trl", gguf: str | None = None, val: str | None = None,
         max_length: int | None = None, mask_prompt: bool | None = None, show: int = 0) -> Report:
    started = time.time()
    tok = Tok(model)
    spec = TRAINERS[trainer]
    session = Session(spec, tok, max_length, mask_prompt)
    rows = Rows(tok, spec, session.limit, trains_prompt=mask_prompt is False)
    unreadable: list[tuple[int, str]] = []
    shapes: dict[str, int] = {}
    first_shape = None
    shown: list[Built] = []
    pending: list[data.Row] = []
    total = 0

    def flush():
        for b in session.build(pending):
            rows.add(b)
            if len(shown) < show:
                shown.append(b)
        pending.clear()

    for row, problem in data.read(path):
        if row is None:
            unreadable.append(problem)
            continue
        first_shape = first_shape or row.shape
        shapes[row.shape] = shapes.get(row.shape, 0) + 1
        pending.append(row)
        total += 1
        if len(pending) == CHUNK:
            flush()
    if pending:
        flush()

    findings = rows.findings(tok.pad_id is not None and tok.pad_id == tok.eos_id)
    notes = []
    if unreadable:
        findings.append(Finding("unreadable", "error", len(unreadable), total + len(unreadable), "unreadable rows",
                                f"{len(unreadable):,} lines cannot be read as training rows",
                                f"Line {unreadable[0][0]}: {unreadable[0][1]}."))
    if len(shapes) > 1:
        odd = total - shapes[first_shape]
        findings.append(Finding("mixed", "error", odd, total, "mixed row shapes",
                                f"{odd:,} rows are not in the same shape as the first row ({first_shape})",
                                "Trainers read the shape from the first row. Keep one shape in the file."))
    if val:
        findings += _leak(val, rows, total)
    if gguf:
        from .gguf import compare
        findings += compare(tok, gguf, total)
    findings.sort(key=lambda f: ({"error": 0, "warning": 1, "note": 2}[f.level], -f.share, -f.count))
    if not spec.exact:
        notes.append(f"{spec.label} is reproduced approximately: tokens follow the tokenizer's chat template and "
                     f"the mask follows its documented rules, not its code")
    return Report(path, model, trainer, spec.label, spec.exact, total, session.limit, spec.length_name,
                  describe_masking(session), findings, notes, shown, time.time() - started, rows.tokens, tok)


def _leak(val: str, rows: Rows, total: int) -> list[Finding]:
    n = exact = near = 0
    example = None
    for row, _ in data.read(val):
        if row is None:
            continue
        n += 1
        text = row.plain()
        w = similar.words(text)
        if not w:
            continue
        kind = rows.seen.find(text, w, similar.signature(w))
        if kind:
            exact += kind == "exact"
            near += kind == "near"
            example = example or (row.line, text)
    if not exact + near:
        return []
    hits = exact + near
    shown = Example(example[0], [], [], text=" ".join(example[1].split())[:160])
    return [Finding("leakage", "error" if hits / n >= 0.02 else "warning", hits, n, "validation rows also in training",
                    f"{hits:,} of {n:,} validation rows also appear in the training file ({exact:,} exactly, {near:,} nearly)",
                    "Evaluation scores on these rows measure memory, not skill. Remove them from one file.", shown)]
