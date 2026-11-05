"""Text and JSON output for a lint report."""

from __future__ import annotations

from .checks import Example, Finding
from .lint import Report
from .tok import Tok
from .trainers import Built

SEP = "¦"
CONTEXT = 24


def headline(r: Report) -> str:
    bad = [f for f in r.findings if f.level == "error"]
    if bad:
        return f"Do not train yet: {bad[0].headline}."
    warn = [f for f in r.findings if f.level == "warning"]
    if warn:
        return f"Nothing blocks training, but look at this first: {warn[0].headline}."
    return f"No problems found in {r.rows:,} rows for {r.label}."


def marked(tok: Tok, ids: list[int], trained: list[bool], lo: int = 0, hi: int | None = None, sep: str = "") -> str:
    """Decoded tokens with every run of trained tokens inside [[ ]]."""
    hi = len(ids) if hi is None else hi
    out, open_ = [], False
    for i in range(lo, hi):
        if trained[i] and not open_:
            out.append("[[")
            open_ = True
        if not trained[i] and open_:
            out.append("]]")
            open_ = False
        out.append(tok.piece(ids[i]).replace("\n", "\\n") + sep)
    if open_:
        out.append("]]")
    return "".join(out)


def excerpt(tok: Tok, ex: Example, sep: str = "") -> str:
    n = len(ex.ids)
    if n <= 3 * CONTEXT:
        return marked(tok, ex.ids, ex.trained, sep=sep)
    focus = ex.focus if ex.focus is not None else n - 1
    focus = min(max(focus, 0), n - 1)
    parts, last = [], 0
    for lo, hi in ((0, CONTEXT // 2), (max(focus - CONTEXT // 2, CONTEXT // 2), min(focus + CONTEXT // 2, n)),
                   (n - CONTEXT // 2, n)):
        if lo > last:
            parts.append(" ... ")
        parts.append(marked(tok, ex.ids, ex.trained, max(lo, last), hi, sep))
        last = max(last, hi)
    return "".join(parts)


def render(r: Report, show_sep: bool = False) -> str:
    tok = r.tokenizer
    out = [headline(r), ""]
    kind = "exact" if r.exact else "approximate"
    out.append(f"{r.label} ({kind}), {r.length_name} {r.max_length}, loss on {r.masking}")
    out.append(f"{r.rows:,} rows, {r.tokens:,} tokens, {r.model}, {r.seconds:.1f} s")
    out += r.notes
    out.append("")
    for n, f in enumerate(r.findings, 1):
        approx = " (approximate)" if f.approximate else ""
        what = f"{f.count:,} of {f.of:,}" if f.of > 1 else ""
        out.append(f"{n}. {f.level}: {f.title}{approx}" + (f", {what} ({f.share:.0%})" if what else ""))
        if f.detail:
            out.append(f"   {f.detail}")
        if f.example and f.example.ids:
            kept = sum(f.example.trained)
            out.append(f"   row {f.example.line}, {len(f.example.ids):,} tokens, {kept:,} trained:")
            out.append("   " + excerpt(tok, f.example))
        elif f.example and f.example.text:
            out.append(f"   row {f.example.line}: {f.example.text}")
    if not r.findings:
        out.append("No findings.")
    if r.shown:
        out += ["", f"First {len(r.shown)} rows as the trainer sees them.",
                f"Tokens are separated by {SEP}, trained runs are inside [[ ]]."]
        for b in r.shown:
            out.append(show(tok, b, r.max_length))
    return "\n".join(out)


def show(tok: Tok, b: Built, limit: int) -> str:
    trained = sum(b.trained[:b.kept])
    cut = f", cut at {b.kept}" if b.kept < b.n else ""
    head = f"row {b.row.line}: {b.n:,} tokens, {trained:,} trained{cut}"
    return head + "\n" + marked(tok, b.ids[:b.kept], b.trained, sep=SEP)


def document(r: Report) -> dict:
    def example(f: Finding):
        e = f.example
        if e is None:
            return None
        tokens = [{"id": i, "token": r.tokenizer.piece(i), "trained": t} for i, t in zip(e.ids[:400], e.trained)]
        return {"row": e.line, "tokens": tokens, "text": e.text or None}

    return {
        "summary": headline(r),
        "ok": r.errors == 0,
        "trainer": r.trainer, "exact": r.exact, "rows": r.rows, "tokens": r.tokens,
        "length": {r.length_name: r.max_length}, "loss": r.masking, "model": r.model,
        "notes": r.notes,
        "findings": [{"id": f.id, "level": f.level, "count": f.count, "of": f.of, "share": round(f.share, 6),
                      "title": f.title, "message": f.headline, "detail": f.detail, "approximate": f.approximate,
                      "example": example(f)} for f in r.findings],
        "rows_shown": [{"row": b.row.line, "kept": b.kept,
                        "tokens": [{"id": i, "token": r.tokenizer.piece(i), "trained": t}
                                   for i, t in zip(b.ids[:b.kept], b.trained)]} for b in r.shown],
    }
