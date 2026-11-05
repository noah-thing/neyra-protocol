"""Did the fine-tune work, and did it need to be a fine-tune. Paired scores on the user's cases, then one sentence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import stats
from . import cases as cases_mod
from . import damage, prompts
from .cases import Case
from .engines import Backends, EngineError
from .prompts import Format

MIN_CASES = 30
HARD_MIN = 10
FREE = ("base", "few-shot")


@dataclass
class Cmp:
    a: str
    b: str
    n: int
    diff: float
    low: float
    high: float
    gained: int
    lost: int
    p: float | None
    kind: str  # better, worse, same or unclear (a against b)


def compare(a: str, b: str, sa: np.ndarray, sb: np.ndarray, margin: float) -> Cmp:
    r = stats.paired(sb, sa)  # stats calls the second vector "edited", the one that should be ahead
    if r.low > 0:
        kind = "better"
    elif r.high < 0:
        kind = "worse"
    elif r.low >= -margin and r.high <= margin:
        kind = "same"
    else:
        kind = "unclear"
    return Cmp(a, b, r.n, r.difference, r.low, r.high, r.gained, r.lost, r.p_value, kind)


@dataclass
class Result:
    verdict: str
    sentence: str
    passed: bool
    n: int
    margin: float
    scorer: str
    arms: dict[str, float]
    comparisons: list[Cmp]
    cases_path: str = ""
    shots: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    general: dict | None = None
    kl: dict | None = None
    worse: list[dict] = field(default_factory=list)
    worse_total: int = 0
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, list[float]] = field(default_factory=dict)
    case_ids: list[str] = field(default_factory=list)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def span(c: Cmp) -> str:
    return f"{pts(c.low)} to {pts(c.high)}"


def need(c: Cmp, margin: float) -> int:
    half = (c.high - c.low) / 2
    return int(math.ceil(c.n * (half / margin) ** 2 / 10) * 10) if half > margin else c.n


def decide(arms: dict[str, np.ndarray], quants: list[str], n: int, margin: float, shots: int,
           damaged: str | None) -> tuple[str, str, bool, list[Cmp]]:
    """Returns the verdict key, the sentence, whether the tuned model beat the best free option, and the comparisons."""
    mean = {k: float(v.mean()) for k, v in arms.items()}
    cmps = []

    def cm(a, b):
        c = compare(a, b, arms[a], arms[b], margin)
        cmps.append(c)
        return c

    tb, tf = cm("tuned", "base"), cm("tuned", "few-shot")
    cm("few-shot", "base")
    qc = {q: (cm(q, "base"), cm(q, "few-shot"), cm(q, "tuned")) for q in quants}
    best = max(FREE, key=lambda k: mean[k])
    tbest = tb if best == "base" else tf
    T, B, F = pct(mean["tuned"]), pct(mean["base"]), pct(mean["few-shot"])
    with_prompt = f"{F} with {shots} examples in the prompt"

    def unclear(c: Cmp, what: str) -> tuple[str, str, bool, list[Cmp]]:
        return ("cannot_tell",
                (f"Cannot tell from {n} cases whether the tune {what}: the difference is {pts(c.diff)} points, interval "
                 f"{span(c)}; about {need(c, margin)} cases would narrow it to plus or minus {100 * margin:.1f} points."), False, cmps)

    if tb.kind == "worse":
        return ("worse_than_base",
                (f"It did not work: the tuned model scores {T} and the base {B} on {n} cases, "
                 f"{pts(tb.diff)} points (interval {span(tb)})."), False, cmps)
    if tb.kind == "same":
        return ("did_not_work",
                (f"It did not work: the tuned model scores {T} and the base {B} on {n} cases, a difference of "
                 f"{pts(tb.diff)} points (interval {span(tb)}), which fits inside plus or minus {100 * margin:.1f} points."), False, cmps)
    if tb.kind == "unclear":
        return unclear(tb, "beats the base")
    # the tune beats the base from here on
    if tf.kind == "worse":
        return ("prompt_better",
                (f"A prompt beats the tune: the base with {shots} examples scores {F} and the tuned model {T} on {n} "
                 f"cases ({pts(tf.diff)} points, interval {span(tf)}), so you did not need to train."), False, cmps)
    if tf.kind == "same":
        return ("prompt_enough",
                (f"A prompt does the same, you did not need to train: the base with {shots} examples scores {F} and the "
                 f"tuned model {T} on {n} cases, a difference of {pts(tf.diff)} points (interval {span(tf)}), which "
                 f"fits inside plus or minus {100 * margin:.1f} points."), False, cmps)
    if tf.kind == "unclear":
        return unclear(tf, "beats a prompt with examples")
    # beats both free options
    lost = [q for q, (qb, qf, _) in qc.items() if {qb.kind, qf.kind} & {"same", "worse"}]
    murky = [q for q, (qb, qf, _) in qc.items() if q not in lost and not (qb.kind == qf.kind == "better")]
    if lost:
        kept = [q for q in quants if q not in lost + murky]
        q = lost[0]
        tail = f" It keeps the gain at {', '.join(kept)}." if kept else ""
        return ("lost_in_quantization",
                (f"The gain is lost at {', '.join(lost)}: the tuned model scores {T} before quantizing and "
                 f"{pct(mean[q])} in that file, against {pct(mean[best])} for the best free option ({best}).{tail}"),
                False, cmps)
    if murky:
        c = next(x for x in qc[murky[0]][:2] if x.kind != "better")
        return ("cannot_tell",
                (f"The tune worked in full precision ({T} against {pct(mean[best])} for {best}), but {n} cases cannot "
                 f"say whether it survives in {', '.join(murky)}: that file is {pts(c.diff)} points from {c.b}, "
                 f"interval {span(c)}."), False, cmps)
    lead = (f"This worked: the tuned model scores {T} on {n} cases, against {B} for the base and {with_prompt}; "
            f"it is {pts(tbest.diff)} points ahead of the best free option, {best} (interval {span(tbest)}).")
    if quants:
        lead += f" The quantized files tested ({', '.join(quants)}) keep it."
    if damaged:
        return ("worked_with_damage", f"{lead} Check this before shipping: {damaged}", False, cmps)
    return "worked", lead, True, cmps


def run(cases: list[Case], backends: Backends, fmt: Format, score_fn, scorer_name: str, examples: list[Case],
        margin: float = 0.05, cases_path: str = "", worse_limit: int = 5, quants: list[str] | None = None,
        say=lambda s: None) -> Result:
    n = len(cases)
    if n < HARD_MIN:
        raise cases_mod.CaseError(f"{n} cases cannot tell a tune from noise; use at least {MIN_CASES}")
    quants = list(quants) if quants is not None else [k for k in backends.generate if k not in ("base", "tuned")]
    zero = [prompts.render(fmt, c) for c in cases]
    few = [prompts.render(fmt, c, examples) for c in cases]
    gen = backends.generate

    outputs: dict[str, list[str]] = {}

    def grade(arm: str, prompt_list: list[prompts.Prompt], call):
        say(f"running {arm} on {len(prompt_list)} prompts")
        outputs[arm] = [prompts.cut(o, fmt.stop) for o in call(prompt_list)]

    grade("base", zero, gen["base"])
    grade("few-shot", few, gen["base"])
    gfmt = Format(True) if backends.has_chat_template and fmt.chat else Format(False, "Question: {input}\nAnswer:", ("\n",))
    qcases = damage.question_cases()
    qprompts = [prompts.render(gfmt, c) for c in qcases]
    say("running the general questions on the base")
    general_base = [prompts.cut(o, gfmt.stop) for o in gen["base"](qprompts)]
    grade("tuned", zero, gen["tuned"])
    say("running the general questions on the tuned model")
    general_tuned = [prompts.cut(o, gfmt.stop) for o in gen["tuned"](qprompts)]
    kl = None
    if backends.neutral_kl is not None:
        say("measuring KL on neutral text")
        per = backends.neutral_kl(damage.NEUTRAL)
        kl = {"texts": len(per), "mean": float(np.mean(per)), "max": float(np.max(per)),
              "unit": "nats per token, KL(base || tuned)"}
    for q in list(quants):
        try:
            grade(q, zero, gen[q])
        except EngineError as e:
            backends.skipped.append((q, str(e)))
            quants.remove(q)

    arms = {k: np.array([score_fn(o, c) for o, c in zip(outputs[k], cases)]) for k in outputs}
    gb = np.array([damage.answered(o, i) for i, o in enumerate(general_base)])
    gt = np.array([damage.answered(o, i) for i, o in enumerate(general_tuned)])
    gcmp = compare("tuned", "base", gt, gb, margin)
    general = {"questions": len(qcases), "base": float(gb.mean()), "tuned": float(gt.mean()), "difference": gcmp.diff,
               "low": gcmp.low, "high": gcmp.high, "gained": gcmp.gained, "lost": gcmp.lost, "kind": gcmp.kind,
               "got_worse": [qcases[i].input for i in range(len(qcases)) if gb[i] > gt[i]]}
    drop = gb.mean() - gt.mean()
    damaged = None
    if gcmp.kind == "worse":
        damaged = (f"the tuned model answers {pct(gt.mean())} of {len(qcases)} general questions against "
                   f"{pct(gb.mean())} for the base, a drop with an interval of {span(gcmp)}.")
    elif drop >= 0.10:
        damaged = (f"the tuned model answers {pct(gt.mean())} of {len(qcases)} general questions against "
                   f"{pct(gb.mean())} for the base; {len(qcases)} questions are too few to be sure it is a real drop.")

    verdict, sentence, passed, cmps = decide(arms, quants, n, margin, len(examples), damaged)
    if backends.skipped and verdict in ("worked", "worked_with_damage"):
        files = ", ".join(q for q, _ in backends.skipped)
        verdict, passed = "not_tested", False
        sentence += f" {files} did not run, so this says nothing about those files."
    worse = [{"id": c.id, "input": c.input[:160], "expected": c.output, "base": outputs["base"][i],
              "tuned": outputs["tuned"][i]}
             for i, c in enumerate(cases) if arms["base"][i] > arms["tuned"][i]]
    warnings = []
    if n < MIN_CASES:
        warnings.append(f"{n} cases is under {MIN_CASES}; intervals this wide rarely settle anything")
    if len(set(outputs["base"])) == 1:
        warnings.append("the base gives the same output on every case; check the prompt format with --format and --template")
    if len(set(outputs["tuned"])) == 1:
        warnings.append("the tuned model gives the same output on every case; check the prompt format with --format "
                        "and --template")
    return Result(verdict, sentence, passed, n, margin, scorer_name, {k: float(v.mean()) for k, v in arms.items()},
                  cmps, cases_path, len(examples), list(backends.skipped), general, kl, worse[:worse_limit],
                  len(worse), warnings,
                  {k: v.tolist() for k, v in arms.items()}, [c.id for c in cases])
