"""Text and JSON for a verdict."""

from __future__ import annotations

from .verdict import Result, pct, pts

LIMITS = [
    "Only what was tested is reported: these cases, this scorer, this prompt format, greedy decoding.",
    ("The few-shot arm is one prompt with the examples shown. A better prompt could do more, so \"a prompt does the "
     "same\" holds for this prompt only."),
    "Quantized files are compared with the free options run in full precision, and the files are fed the same token ids.",
    "The general questions and the KL texts are small. They show large damage, not the absence of damage.",
]


def render(r: Result) -> str:
    lines = [r.sentence, ""]
    for w in r.warnings:
        lines.append(f"warning: {w}")
    if r.warnings:
        lines.append("")
    lines.append(f"{r.n} cases from {r.cases_path}, scorer {r.scorer}, arms count as equal when the interval fits inside plus or minus {100 * r.margin:.1f} points, "
                 f"{r.shots} examples in the few-shot prompt")
    lines.append("")
    width = max(len(k) for k in r.arms) + 2
    lines.append(f"{'':{width}}score    against base                    against few-shot")
    by = {(c.a, c.b): c for c in r.comparisons}
    for arm, score in r.arms.items():
        row = f"{arm:{width}}{pct(score):<9}"
        for other in ("base", "few-shot"):
            c = by.get((arm, other))
            row += f"{'':32}" if c is None or arm == other else (
                f"{pts(c.diff):>6} [{pts(c.low)}, {pts(c.high)}] +{c.gained}/-{c.lost}".ljust(32))
        lines.append(row.rstrip())
    for q, why in r.skipped:
        lines.append(f"{q:{width}}not run: {why}")
    lines.append("")
    lines.append("differences are in points of score, with a 95% bootstrap interval over cases, then cases won/lost")
    quant = [c for c in r.comparisons if c.b == "tuned"]
    for c in quant:
        lines.append(f"{c.a} against the tuned model in full precision: {pts(c.diff)} [{pts(c.low)}, {pts(c.high)}]")
    g = r.general
    lines += ["", "Damage"]
    lines.append(f"  general questions: base {pct(g['base'])}, tuned {pct(g['tuned'])} of {g['questions']} "
                 f"({pts(g['difference'])} points, interval {pts(g['low'])} to {pts(g['high'])})")
    if g["got_worse"]:
        lines.append(f"  answered by the base and not by the tuned model: {'; '.join(g['got_worse'][:5])}")
    if r.kl:
        lines.append(f"  KL on {r.kl['texts']} neutral texts: mean {r.kl['mean']:.4f}, largest {r.kl['max']:.4f} "
                     f"{r.kl['unit']}. No threshold is applied; compare it between your runs")
    lines.append(f"  cases the base got right and the tuned model got wrong: {r.worse_total} of {r.n}")
    for w in r.worse:
        lines.append(f"    case {w['id']}: {w['input']!r}")
        lines.append(f"      expected {w['expected']!r}, base said {w['base'][:80]!r}, tuned said {w['tuned'][:80]!r}")
    if r.worse_total > len(r.worse):
        lines.append(f"    and {r.worse_total - len(r.worse)} more in --json")
    lines += ["", "Limits"] + [f"  {t}" for t in LIMITS]
    return "\n".join(lines)


def document(r: Result) -> dict:
    return {
        "command": "tune verdict",
        "verdict": r.verdict,
        "sentence": r.sentence,
        "tuned_beat_best_free_option": r.passed,
        "cases": {"n": r.n, "path": r.cases_path, "scorer": r.scorer, "ids": r.case_ids},
        "margin": r.margin,
        "shots": r.shots,
        "scores": r.arms,
        "comparisons": [{"a": c.a, "b": c.b, "n": c.n, "difference": c.diff, "low": c.low, "high": c.high,
                         "gained": c.gained, "lost": c.lost, "p_value": c.p, "outcome": c.kind}
                        for c in r.comparisons],
        "per_case_scores": r.scores,
        "not_run": [{"file": q, "reason": why} for q, why in r.skipped],
        "damage": {"general": r.general, "kl": r.kl, "worse_cases": r.worse, "worse_total": r.worse_total},
        "warnings": r.warnings,
        "limits": LIMITS,
    }
