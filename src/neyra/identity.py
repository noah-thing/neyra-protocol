"""Say what a model is relative to the base it declares, from the tensors.

A model card names a base in free text and nothing checks it. This reads a few spans of tensors from
both sides, which over HTTP ranges costs a megabyte or two, and answers with one of a few verdicts:
the same files, the same weights in other files, a transform that can be re-run, a derivative
whose transform is unknown, or an unrelated model.

"Derived" is measured against a null. Two independently trained models of one architecture are not
orthogonal in every tensor, so how alike a pair has to be before it counts is read off a panel of
unrelated siblings (identity_null.json, written by bench/identity_panel.py), never chosen by
feel. A number between the unrelated and the derived range is reported as inconclusive.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from .numerics import from_float, is_float, to_float
from .store import Checkpoint, TensorRef

BIT_IDENTICAL = "bit identical"
SAME_WEIGHTS = "same weights in a different file"
NO_DIFFERENCE = "no difference found in a sample"
DERIVED_NAMED = "derived by a named transform"
DERIVED_UNKNOWN = "derived with unknown transform"
UNRELATED = "unrelated"
INCONCLUSIVE = "inconclusive"
NOT_COMPARABLE = "not comparable"

SPAN = 16384  # entries read per sampled tensor
MATRICES = 24
VECTORS = 4


def calibration() -> dict | None:
    try:
        return json.loads(resources.files("neyra").joinpath("identity_null.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class Probe:
    """One span of one tensor, in the base and in the model."""

    name: str
    matrix: bool
    entries: int
    same: float  # share of entries with the same value
    cos: float | None
    rel: float | None  # |model - base| / |base|
    cast: bool  # the model span is the base span rounded to the model dtype


@dataclass
class Measure:
    probes: list[Probe]
    only_base: list[str] = field(default_factory=list)
    only_model: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # same name, other shape or dtype family

    @property
    def matrices(self) -> list[Probe]:
        return [p for p in self.probes if p.matrix and p.cos is not None]

    @property
    def cos(self) -> float | None:
        m = self.matrices
        return float(np.median([p.cos for p in m])) if m else None

    @property
    def rel(self) -> float | None:
        m = self.matrices
        return float(np.median([p.rel for p in m])) if m else None

    @property
    def unchanged(self) -> float:
        """Share of the sampled tensors whose span is bit for bit the base's."""
        return sum(p.same == 1 for p in self.probes) / len(self.probes) if self.probes else 0.0


@dataclass
class Verdict:
    label: str
    detail: str
    exact: bool = False  # every byte was compared, not a sample
    measure: Measure | None = None
    paths: list[str] = field(default_factory=list)
    note: list[str] = field(default_factory=list)


def _key(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()


def comparable(a: TensorRef, b: TensorRef) -> bool:
    return a.shape == b.shape and (a.dtype == b.dtype or (is_float(a.dtype) and is_float(b.dtype)))


def _span(ref: TensorRef, name: str, entries: int) -> tuple[int, int]:
    n = min(entries, ref.numel)
    return int(_key(name)[:8], 16) % (ref.numel - n + 1), n


def _values(bits: np.ndarray, dtype: str) -> np.ndarray:
    return to_float(bits, dtype).astype(np.float64) if is_float(dtype) else bits.astype(np.float64)


def probe(name: str, base: TensorRef, model: TensorRef, entries: int | None = None) -> Probe:
    first, n = _span(base, name, entries or SPAN)
    a, b = base.read_slice(first, n), model.read_slice(first, n)
    x, y = _values(a, base.dtype), _values(b, model.dtype)
    cast = base.dtype != model.dtype and bool(np.array_equal(from_float(x.astype(np.float32), model.dtype), b))
    same = float(np.mean(x == y)) if n else 1.0
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if not is_float(base.dtype) or nx == 0 or ny == 0 or not (np.isfinite(nx) and np.isfinite(ny)):
        return Probe(name, len(base.shape) >= 2, n, same, None, None, cast)
    return Probe(name, len(base.shape) >= 2, n, same, float(x @ y / (nx * ny)), float(np.linalg.norm(y - x) / nx), cast)


def measure(base: Checkpoint, model: Checkpoint, matrices: int = MATRICES, vectors: int = VECTORS) -> Measure:
    """Sample the tensors both sides hold. Which ones depends only on their names, so two runs agree.

    Names that differ by a wrapper prefix on one side ("transformer.", "model.") are paired by their tail.
    """
    from .check import aligned

    theirs = aligned(base, model)
    shared = sorted(theirs, key=_key)
    ok = [n for n in shared if comparable(base.tensors[n], theirs[n])]
    chosen = [n for n in ok if len(base.tensors[n].shape) >= 2][:matrices] + \
             [n for n in ok if len(base.tensors[n].shape) < 2 and base.tensors[n].numel > 1][:vectors]
    with ThreadPoolExecutor(8) as pool:
        probes = list(pool.map(lambda n: probe(n, base.tensors[n], theirs[n]), chosen))
    paired = {id(r) for r in theirs.values()}
    return Measure(probes, sorted(base.tensors.keys() - theirs.keys()),
                   sorted(n for n, r in model.tensors.items() if id(r) not in paired), sorted(set(shared) - set(ok)))


def same_files(a: Checkpoint, b: Checkpoint) -> bool:
    fa = sorted(f for f in a.files if f.endswith(".safetensors"))
    fb = sorted(f for f in b.files if f.endswith(".safetensors"))
    return bool(fa) and sorted(a.digest(f) for f in fa) == sorted(b.digest(f) for f in fb)


FUSIONS = (("qkv_proj", ("q_proj", "k_proj", "v_proj")), ("gate_up_proj", ("gate_proj", "up_proj")))


def _fused(one: Checkpoint, other: Checkpoint) -> dict[str, list[str]]:
    """Tensors `one` stores whole that `other` stores split. Concatenated along the first axis, the bytes are the same."""
    found = {}
    for name in one.tensors:
        for joined, parts in FUSIONS:
            split = [name.replace(joined, p) for p in parts]
            if f".{joined}." in name and name not in other.tensors and all(n in other.tensors for n in split):
                found[name] = split
    return found


def same_everywhere(base: Checkpoint, model: Checkpoint) -> tuple[bool, list[str]]:
    """Whether every byte of every tensor is the same, wherever each side keeps it, and what that took.

    A tensor present on both sides is hashed. A fused projection is compared with the concatenation of the split
    ones on the other side, and a head one side leaves out is compared with the embedding its file ties it to.
    Anything left over means the files do not hold the same weights, or that this cannot tell.
    """
    notes, jobs, covered = [], [], set()
    for n in base.tensors.keys() & model.tensors.keys():
        a, b = base.tensors[n], model.tensors[n]
        if a.dtype != b.dtype or a.shape != b.shape:
            return False, [f"{n} has another dtype or shape"]
        jobs.append(([a], [b]))
        covered.add(n)
    for one, other, flip in ((base, model, False), (model, base, True)):
        for name, split in _fused(one, other).items():
            parts = [other.tensors[n] for n in split]
            if one.tensors[name].dtype != parts[0].dtype or one.tensors[name].nbytes != sum(p.nbytes for p in parts):
                return False, [f"{name} does not hold the bytes of {split}"]
            jobs.append(([one.tensors[name]], parts) if not flip else (parts, [one.tensors[name]]))
            covered |= {name, *split}
            notes.append(f"{name} compared as the concatenation of {len(split)} tensors")
    for one, other in ((base, model), (model, base)):
        for name in set(one.tensors) - covered:
            embed = [n for n in other.tensors if n.endswith("embed_tokens.weight")]
            if name.endswith("lm_head.weight") and name not in other.tensors and len(embed) == 1 \
                    and not any(n.endswith("lm_head.weight") for n in other.tensors):
                jobs.append(([one.tensors[name]], [other.tensors[embed[0]]]))
                covered.add(name)
                notes.append("a head one file leaves out equals the embedding the other ties it to")
    left = (set(base.tensors) | set(model.tensors)) - covered
    if left:
        return False, ["tensors with no counterpart: " + ", ".join(sorted(left)[:3])]

    def joined(refs):
        h = hashlib.sha256()
        for r in refs:
            for o in range(r.start, r.end, 1 << 24):
                h.update(r.blob.read(o, min(1 << 24, r.end - o)))
        return h.hexdigest()

    with ThreadPoolExecutor(8) as pool:
        return all(pool.map(lambda j: joined(j[0]) == joined(j[1]), jobs)), notes


def named_transform(m: Measure) -> str | None:
    """A transform that needs no input beyond the base: a cast to another float dtype."""
    if m.probes and all(p.cast or p.same == 1 for p in m.probes) and any(p.cast for p in m.probes):
        return "cast"
    return None


def identify(base: Checkpoint, model: Checkpoint, *, full: bool = False, adapter: Checkpoint | None = None,
             matrices: int = MATRICES) -> Verdict:
    from .check import check_sample

    if same_files(base, model):
        return Verdict(BIT_IDENTICAL, "the safetensors files have the same SHA-256", exact=True)
    m = measure(base, model, matrices)
    if full:
        same, how = same_everywhere(base, model)
        if same:
            return Verdict(SAME_WEIGHTS, "every tensor hashes the same, byte for byte", True, m, note=how)
    if not m.matrices:
        why = ("the files name or shape their tensors differently (renamed, fused or split layers); this cannot tell"
               if not m.probes else "no matrix to compare")
        return Verdict(NOT_COMPARABLE, why, measure=m)
    notes = []
    if m.only_base or m.only_model or m.skipped:
        notes.append(f"{len(m.only_base)} tensors only in the base, {len(m.only_model)} only in the model, "
                     f"{len(m.skipped)} with another shape")
    if m.unchanged == 1 and not full:
        return Verdict(NO_DIFFERENCE, f"{len(m.probes)} sampled spans are bit for bit the base's. This is a sample, not "
                       "proof: --full compares every byte", False, m, note=notes)
    if (cast := named_transform(m)):
        return Verdict(DERIVED_NAMED, f"{cast}: the model is the base rounded to its dtype in every sampled span (sampled, not proof)",
                       False, m, ["cast"], notes)
    if adapter is not None:
        c = check_sample(base, model, adapter)
        if c is not None and c.paths:
            return Verdict(DERIVED_NAMED, f"LoRA merge, arithmetic of {', '.join(c.paths)}", False, m, c.paths, notes)
        notes.append("the adapter does not reproduce the sampled tensors" if c is not None else "no adapter tensor sampled")
    null = calibration()
    s, d = m.cos, m.rel
    if null is None:
        return Verdict(INCONCLUSIVE, f"median cosine {s:.3f}, relative distance {d:.3f}; this build has no calibration "
                       "panel, so it cannot call a model derived or unrelated", False, m, note=notes)
    if s >= null["derived_min"]:
        return Verdict(DERIVED_UNKNOWN, f"sampled, not proof: median cosine {s:.3f}, relative distance {d:.3f}; unrelated siblings stay "
                       f"under {null['unrelated_max']:.3f}", False, m, note=notes)
    if s <= null["unrelated_max"]:
        return Verdict(UNRELATED, f"sampled, not proof: median cosine {s:.3f}, within what unrelated siblings give "
                       f"(up to {null['unrelated_max']:.3f})", False, m, note=notes)
    return Verdict(INCONCLUSIVE, f"median cosine {s:.3f} lies between the unrelated range (up to "
                   f"{null['unrelated_max']:.3f}) and the derived range (from {null['derived_min']:.3f})", False, m, note=notes)
