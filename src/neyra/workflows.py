"""The one line operations, built from the pieces in this package."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__
from .behavior import is_looping
from .bisect import BLAME, Confirmation, Measure, Result, bisect, confirm
from .edit import (
    Editor,
    Request,
    Settings,
    Words,
    check_layers,
    neutral_windows,
    second_moments,
    trace,
)
from .editeval import Limits, Neutral, fluency, judge, probe, statements, summary
from .families import editable_warning
from .ledger import Ledger
from .patch import Patch, diff_tensor
from .probe import SelectivityProbe, loop_spans
from .protocol import digest, envelope, jcs, plain, statement_for, tensor_table
from .repair import Plan, Repair
from .runtime import Sampling
from .store import Checkpoint
from .tasks import reasoning_problems
from .units import UnitSet


def on_stderr(message: str) -> None:
    import sys

    print(message, file=sys.stderr, flush=True)


def collect_loops(runtime, n: int, max_tokens: int, sampling: Sampling, seed: int = 0,
                  say: Callable[[str], None] = on_stderr) -> list[list[int]]:
    """Generate on the built in problems and keep the token streams that ended in a loop."""
    problems = reasoning_problems(n, seed)
    say(f"generating {n} answers, up to {max_tokens} tokens each")
    rows = runtime.generate([runtime.encode_chat(p.prompt) for p in problems], max_tokens, sampling, batch=12)
    looped = [r.tokens for r in rows if is_looping(r.tokens, max_tokens)]
    say(f"{len(looped)} of {n} ended in a loop")
    return looped


@dataclass
class Fix:
    patch: Patch
    plan: Plan
    loops: int


def split(items: list, share: float = 2 / 3) -> tuple[list, list]:
    cut = max(int(len(items) * share), 1)
    return items[:cut], items[cut:] or items[:cut]


def fix_loops(runtime, ck: Checkpoint, n: int = 48, max_tokens: int = 12288, columns: int = 64,
              sampling: Sampling | None = None, seed: int = 0,
              say: Callable[[str], None] = on_stderr) -> Fix | None:
    """Find a loop direction from the model's own failures and remove it. None if it does not loop."""
    sampling = sampling or Sampling(0.6, 0.95, 20, 1.5, seed)
    looped = collect_loops(runtime, n, max_tokens, sampling, seed, say)
    spans = loop_spans(looped)
    if len(spans) < 4:
        say("too few loops to work from")
        return None
    fit, held = split(spans)
    repair = Repair(runtime, ck, fit, held, columns)
    say(f"searching {len(runtime.layers())} layers on {len(fit)} texts, scoring on {len(held)}")
    plan = repair.search(list(range(len(runtime.layers()))))
    say(f"chose layers {[x.index for x in plan.layers]} at strength {plan.alpha}, score {plan.score:.4f}")
    return Fix(repair.patch(plan), plan, len(looped))


@dataclass
class Bisection:
    result: Result
    check: Confirmation
    units: UnitSet
    patch: Patch


def bisect_loops(runtime, ck: Checkpoint, patch: Patch, token_lists: list[list[int]], grain: str = "column",
                 retain: float = 0.9, budget: int = 400,
                 say: Callable[[str], None] = on_stderr) -> Bisection:
    """Smallest part of a patch that keeps its effect on loops, confirmed on texts the search never saw."""
    spans = loop_spans(token_lists)
    if len(spans) < 4:
        raise ValueError("need at least 4 texts that end in a loop")
    search_spans, check_spans = split(spans)
    units = UnitSet(patch, ck, grain)
    say(f"{len(units)} units, searching on {len(search_spans)} texts, confirming on {len(check_spans)}")

    def using(probe: SelectivityProbe) -> Callable[[tuple[int, ...]], Measure]:
        trials = 0

        def measure(chosen: tuple[int, ...]) -> Measure:
            nonlocal trials
            runtime.set_tensors(units.tensors(chosen))
            trials += 1
            if trials % 25 == 0:
                say(f"{trials} trials")
            return probe()

        return measure

    result = bisect(len(units), using(SelectivityProbe(runtime, search_spans)), retain=retain, budget=budget)
    check = confirm(len(units), result.units, using(SelectivityProbe(runtime, check_spans)), retain)
    runtime.reset()
    edits = [diff_tensor(n, d, units.base[n], b) for n, (b, d) in units.tensors(result.units).items()]
    kept = Patch(edits, {"bisected_from": patch.id, "retain": retain, "blame": BLAME}).bind(tensor_table(ck))
    return Bisection(result, check, units, kept)


@dataclass
class Edited:
    patch: Patch
    report: dict
    claim: dict
    receipt: dict
    verdict: str


def batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def edit_facts(runtime, ck: Checkpoint, requests: list[Request], settings: Settings | None = None,
               limits: Limits | None = None, neutral: str | None = None, key: bytes | None = None,
               ledger: Path | None = None, track: bool = False, say: Callable[[str], None] = on_stderr) -> Edited:
    """Write the requests into the model as a patch, measure it, and say plainly whether it earned a pass."""
    s, limits = settings or Settings(), limits or Limits()
    if not hasattr(runtime, "delta_loss"):
        raise ValueError("editing needs gradients through the model, which only the MLX backend provides so far")
    words = Words(runtime.tokenizer)
    log = Ledger(ledger or Path(tempfile.mkdtemp()) / "edit.ledger.jsonl")
    requested = digest("edit/requests", jcs(plain([asdict(r) for r in requests])))
    log.append("edit_start", requests=requested, count=len(requests), settings=asdict(s), limits=asdict(limits))

    runtime.reset()
    windows = neutral_windows(runtime, words, neutral, s.neutral + s.held, s.window, s.seed)
    fit, held = windows[:-s.held], windows[-s.held:]
    traced = None
    if s.layers:
        layers = sorted(s.layers)
    else:
        say(f"tracing {min(len(requests), 48)} facts to choose {s.width} layers")
        traced = trace(runtime, words, requests, s.width, limit=48, draws=2, seed=s.seed)
        if traced.used < 4:
            raise ValueError(f"only {traced.used} prompts gave a usable trace, which needs a model that knows its "
                             "subjects. Name the layers with --layers.")
        layers = traced.layers
        say(f"layers {layers}, from {traced.used} usable prompts")
    check_layers(ck, layers)
    before = probe(runtime, words, requests)
    quiet = fluency(runtime, words, requests)
    reference = Neutral(runtime, held)
    cov = second_moments(runtime, fit, layers)
    say(f"covariance from {len(fit)} windows, baseline efficacy {summary(before)['efficacy']}")

    editor = Editor(runtime, ck, s, layers, cov, say)
    curve = []
    for n, part in enumerate(batches(requests, s.batch)):
        info = editor.write(part)
        info["written"] = editor.written
        if track:
            now = probe(runtime, words, requests[:editor.written])
            info["efficacy"] = summary(now)["efficacy"]
            info["neutral_kl"] = float(reference.drift(runtime)["kl"].mean())
        curve.append(info)
        log.append("batch", index=n, **info)
        say(f"batch {n + 1}: {editor.written}/{len(requests)} written, {info}")

    meta = {"method": "closed form edit", "layers": layers, "settings": asdict(s), "requests": requested,
            "count": len(requests)}
    patch = editor.stack.patch(ck, s, meta)
    edited = patch.apply(ck)
    runtime.set_tensors({e.name: (edited[e.name], e.dtype) for e in patch.edits})
    after = probe(runtime, words, requests)
    drift = reference.drift(runtime)
    loud = fluency(runtime, words, requests)
    for e in patch.invert(ck).edits:
        e.apply(edited[e.name])  # raises unless the inverse restores the base digest
    runtime.reset()

    gates = {**judge(before, after, limits, drift, quiet, loud), "exact_revert": True}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    b, a = summary(before), summary(after)
    report = {"layers": layers, "before": b, "after": a, "neutral_kl": float(drift["kl"].mean()),
              "neutral_nll_change": float(drift["nll"].mean()), "fluency_before": quiet, "fluency_after": loud,
              "gates": gates, "verdict": verdict, "limits": asdict(limits), "curve": curve,
              "trace": None if traced is None else {"effect": traced.effect.tolist(), "used": traced.used},
              "stored_entries": patch.entries, "ranks": {e.name: int(e.left.shape[1]) for e in patch.edits},
              "warning": editable_warning(ck)}
    log.append("evaluate", **{k: report[k] for k in ("before", "after", "neutral_kl", "fluency_before",
                                                    "fluency_after")})
    head = log.append("submit", verdict=verdict, gates=gates, patch_id=patch.id)
    claim = {
        "type": "neyra.claim/1", "base_root": patch.base_root, "result_root": patch.result_root, "patch_id": patch.id,
        "task": {"id": "neyra.edit/1", "metric": "edit gates", "requests": requested},
        "eval_env": {"runtime": type(runtime).__name__, "version": __version__, "dtype": "native"},
        "statements": [{"metric": m, "estimate": plain(p.difference), "ci_low": plain(p.low), "ci_high": plain(p.high),
                        "n": p.n, "method": "paired bootstrap"} for m, p in statements(before, after)],
        "gates": gates, "verdict": verdict, "limits": plain(asdict(limits)),
        "edit": plain({"layers": layers, "settings": asdict(s), "before": b, "after": a,
                       "neutral_kl": report["neutral_kl"], "fluency": [quiet, loud],
                       "requests": [{"subject": r.subject, "prompt": r.prompt, "target": r.target, "true": r.true}
                                    for r in requests]}),
    }
    receipt = envelope(statement_for(claim, patch.id, head, log.count), key)
    return Edited(patch, report, claim, receipt, verdict)

