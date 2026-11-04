"""The one line operations, built from the pieces in this package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .behavior import is_looping
from .bisect import Confirmation, Measure, Result, bisect, confirm
from .patch import Patch, diff_tensor
from .probe import SelectivityProbe, loop_spans
from .protocol import tensor_table
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
    kept = Patch(edits, {"bisected_from": patch.id, "retain": retain}).bind(tensor_table(ck))
    return Bisection(result, check, units, kept)


