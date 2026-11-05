"""The hop from safetensors to a GGUF file: is this the model that was converted, and how much of the fine-tune is in it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .check import RECIPES, aligned
from .families import IGNORED, text_config
from .gguf_io import (
    DEFAULT_CEILING,
    REPRODUCIBLE,
    ROUNDING_CEILING,
    UNMODELLED,
    alternates,
    rebuild,
    to_gguf_layout,
)
from .hop_text import Finding
from .numerics import is_float, to_float
from .survival import Pool, quantized_copy

LOSSLESS = {"F32", "F16", "BF16"}
EXACT_BLOCKS = 0.999
LOST = 0.5
FAINT = 0.3


@dataclass
class Tally:
    exact: list[str] = field(default_factory=list)
    rounded: list[str] = field(default_factory=list)
    different: list[str] = field(default_factory=list)
    from_base: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    edited: int = 0
    kinds: dict[str, int] = field(default_factory=dict)


def gguf_hop(hop, base, ref, art, fine) -> list[Finding]:
    g = art.gg
    look = g.lookup()
    config = text_config(base.ck)
    pairs = aligned(base.ck, ref.ck) if ref else dict(base.ck.tensors)
    tally, pool = Tally(), Pool()
    for name in sorted(base.ck.tensors):
        if IGNORED.search(name) or name not in pairs or not is_float(base.ck.tensors[name].dtype):
            continue
        gname = look(name)
        gt = g.tensors.get(gname) if gname else None
        if gt is None:
            tally.missing.append(name)
            continue
        w_ref = base.ck.tensors[name]
        w = to_float(w_ref.read(), w_ref.dtype).astype(np.float32)
        delta = fine.delta(name) if fine else None
        if ref:
            r = pairs[name]
            x = to_float(r.read(), r.dtype).astype(np.float32)
            if delta is None and fine is None:
                diff = x.astype(np.float64) - w
                delta = diff if diff.any() else None
        else:
            x = w
            if delta is not None:
                term = fine.adapter.term(fine.onto[name])
                x = to_float(RECIPES["peft"](w_ref.dtype, w_ref.read(), term), w_ref.dtype).astype(np.float32)
        layout = {"x": to_gguf_layout(name, x, g.arch, config), "w": to_gguf_layout(name, w, g.arch, config)}
        if layout["x"] is None:
            tally.skipped.append(name)
            continue
        tally.kinds[gt.kind] = tally.kinds.get(gt.kind, 0) + 1
        moved = None if delta is None else to_gguf_layout(name, delta, g.arch, config)
        compare(name, gt, layout["x"], layout["w"], alternates(name, layout["x"], g.arch), moved, tally, pool)
    return _findings(hop, art, tally, pool, bool(ref), fine is not None)


def compare(name, gt, x, w, spare, delta, tally: Tally, pool: Pool) -> None:
    if x.shape != gt.shape:
        tally.different.append(name)
        return
    dq0 = gt.values()
    for other in spare:
        if np.linalg.norm(dq0 - other) < np.linalg.norm(dq0 - x):
            x, w = other, w + 1.0
    built = rebuild(x, gt.kind, gt.raw, gt.qtype) if gt.kind in REPRODUCIBLE else None
    dq = gt.values()
    if built and built[1] >= EXACT_BLOCKS:
        status = "exact"
    else:
        rel = float(np.linalg.norm(dq - x) / max(np.linalg.norm(x), 1e-30))
        status = "rounded" if rel <= ROUNDING_CEILING.get(gt.kind, DEFAULT_CEILING) else "different"
    stored_from_base = False
    if delta is not None:
        tally.edited += 1
        route = built[0] if built and built[1] >= EXACT_BLOCKS else "direct"
        before = quantized_copy(w, gt.kind, route)
        if status != "exact" and gt.kind in REPRODUCIBLE:
            made = rebuild(w, gt.kind, gt.raw, gt.qtype)
            stored_from_base = bool(made and made[1] >= EXACT_BLOCKS)
        pool.add(dq - (w if before is None else before), delta)
    if stored_from_base:
        tally.from_base.append(name)
    elif status == "different":
        tally.different.append(name)
    else:
        (tally.exact if status == "exact" else tally.rounded).append(name)


def _findings(hop, art, t: Tally, pool: Pool, has_ref: bool, has_adapter: bool) -> list[Finding]:
    out: list[Finding] = []

    def add(cause: str, text: str, problem: bool = True):
        out.append(Finding(hop.index, "weights", cause, problem, text))

    dominant = max((k for k in t.kinds if k not in ("F32",)), key=t.kinds.get, default="F32")
    lossy = dominant not in LOSSLESS
    hop.survival, hop.survival_error = pool.value, pool.error
    hop.survival_on = "dequantized values" if lossy else "weights"
    reference = "the merged model" if has_ref else "base plus adapter" if has_adapter else "the base"
    n = t.edited
    if t.missing:
        add("missing_tensors", f"{len(t.missing)} tensors have no counterpart in the GGUF, such as {t.missing[0]}")
    if t.from_base:
        if len(t.from_base) == n:
            add("lost_in_conversion", f"the GGUF holds the base weights: all {n} tensors the adapter edits are the base's")
        else:
            add("partly_applied", f"{len(t.from_base)} of the {n} tensors the adapter edits are the base's in the GGUF, "
                                  f"such as {t.from_base[0]}")
    if t.different:
        edited_gone = pool.value is not None and pool.value < LOST and n
        caveat = " (this converter's rewrite of the tensor is not modelled, so this may be a false alarm)" \
            if any(UNMODELLED.search(x) for x in t.different) else ""
        add("lost_in_conversion" if edited_gone and not t.from_base else "weights_differ",
            f"{len(t.different)} tensors in the GGUF are not {reference} even allowing for {dominant} rounding, "
            f"such as {t.different[0]}{caveat}")
    if n and pool.value is not None and pool.value + 2 * pool.error < LOST:
        add("lost_in_conversion", f"only {max(pool.value, 0):.0%} of the fine-tune's change is in the GGUF"
                                  + (f", measured on dequantized {dominant} values" if lossy else ""))
    elif lossy and n and pool.cosine is not None and pool.cosine < FAINT:
        add("faint", f"{dominant} steps are larger than the change, so what is stored matches the trained change "
                     f"with cosine {pool.cosine:.2f}; its average size is right ({pool.value:.0%})", problem=False)
    if t.skipped:
        add("not_compared", f"{len(t.skipped)} tensors were not compared because this converter rewrites them "
                            f"in a way that is not modelled, such as {t.skipped[0]}", problem=False)
    if not any(f.problem for f in out):
        how = (f"exactly {dominant}" if t.exact and not t.rounded and lossy else
               f"{dominant}, compared on dequantized values" if lossy else "lossless")
        hop.summary = f"the GGUF holds {reference}, {how}"
    return out
