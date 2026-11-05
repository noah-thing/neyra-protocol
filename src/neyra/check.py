"""Decide whether a merged model is what its parts say it should be.

People merge a LoRA into a base model, save the result, and find it behaves like the base. The
causes are mundane: the adapter was never loaded, a tied head was skipped, or the delta was
smaller than the rounding step of the model dtype. This recomputes the merge the way each real
tool does it and compares bits.

The tools disagree in the last bit, and a bf16 weight shows it. PEFT with merge_and_unload() adds
a float32 delta to the weight in float32 and rounds once. With safe_merge=True it first rounds the
delta to the model dtype, then adds. mlx-lm scales B before the product and rounds the product to
the adapter dtype. A merged file that equals one of these is that tool's output, and the report
names it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .merge import encode
from .numerics import from_float, is_float, to_float
from .store import ITEMSIZE, Checkpoint, FormatError, TensorRef

LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<side>[AB])(?:\.default)?\.weight$")
MLX_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[ab])$")
IDENTICAL, MERGED, ROUNDING, DIFFERENT, MISSING, NEVER_APPLIED = (
    "identical", "merged", "rounding", "different", "missing", "never applied")
EQUIVALENT = {IDENTICAL, MERGED, ROUNDING}
MAX_OFF = 0.02  # share of a tensor's entries that may sit within --ulps steps of a path and still be rounding


@dataclass
class TensorCheck:
    name: str
    status: str
    detail: str = ""
    applied: float | None = None
    paths: list[str] = field(default_factory=list)
    steps: int | None = None


@dataclass
class Check:
    tensors: list[TensorCheck]
    added: list[str] = field(default_factory=list)
    leftover: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    modules: int = 0
    recast: set[tuple[str, str]] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)

    def count(self, status: str) -> int:
        return sum(t.status == status for t in self.tensors)

    @property
    def targeted(self) -> int:
        """Tensors the adapter edits."""
        return len(self.edited)

    @property
    def equivalent(self) -> bool:
        return (all(t.status in EQUIVALENT for t in self.tensors) and not self.added and not self.leftover)

    @property
    def paths(self) -> list[str]:
        """The paths that reproduce every edited tensor bit for bit."""
        moved = [t for t in self.tensors if t.name in self.edited and t.status != IDENTICAL]
        return [p for p in PATHS if all(p in t.paths for t in moved)] if moved else []

    @property
    def faint(self) -> list[TensorCheck]:
        """Edited tensors merged correctly, where the dtype kept less than half of the change."""
        return [t for t in self.tensors if t.status in (MERGED, ROUNDING) and t.applied is not None and t.applied < 0.5]


@dataclass
class Term:
    """One adapter matrix pair as stored, in float32, with the factor it is applied with."""

    a: np.ndarray
    b: np.ndarray
    dtype: str
    scale: float
    transpose: bool


class Adapter:
    """The LoRA pairs of a PEFT or mlx-lm adapter directory, one per base tensor."""

    def __init__(self, ck: Checkpoint):
        raw = ck.text("adapter_config.json")
        if raw is None:
            raise FormatError(f"{ck.spec}: no adapter_config.json")
        self.config = json.loads(raw)
        if self.config.get("use_dora") or self.config.get("fine_tune_type") == "dora":
            raise FormatError(f"{ck.spec}: DoRA adapters are not supported")
        self.mlx = "lora_parameters" in self.config
        self.pairs: dict[str, dict[str, TensorRef]] = {}
        self.ignored: list[str] = []
        for name, ref in ck.tensors.items():
            hit = (MLX_KEY if self.mlx else LORA_KEY).match(name)
            if hit and "lora_embedding" not in name:
                self.pairs.setdefault(hit["module"] + ".weight", {})[hit["side"].upper()] = ref
            else:
                self.ignored.append(name)
        lonely = [n for n, p in self.pairs.items() if set(p) != {"A", "B"}]
        if lonely or not self.pairs:
            raise FormatError(f"{ck.spec}: lora_A and lora_B must come in pairs ({lonely[:3] or 'none found'})")

    def scaling(self, tensor: str, rank: int) -> float:
        if self.mlx:
            return float(self.config["lora_parameters"]["scale"])
        module = tensor.removesuffix(".weight")

        def pattern(key: str, default):
            table = self.config.get(key) or {}
            hit = next((v for k, v in table.items() if module == k or module.endswith("." + k)), None)
            return default if hit is None else hit

        alpha = pattern("alpha_pattern", self.config.get("lora_alpha", rank))
        return alpha / math.sqrt(rank) if self.config.get("use_rslora") else alpha / rank

    def term(self, tensor: str) -> Term:
        pair = self.pairs[tensor]
        a, b = (to_float(pair[s].read(), pair[s].dtype).astype(np.float32) for s in "AB")
        if self.mlx:
            a, b = a.T, b.T
        return Term(a, b, pair["A"].dtype, self.scaling(tensor, a.shape[0]), bool(self.config.get("fan_in_fan_out")))


def order(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Integers that rise with the float, so the difference of two is a count of representable steps."""
    sign = np.uint64(1) << np.uint64(8 * ITEMSIZE[dtype] - 1)
    wide = bits.astype(np.uint64)
    mag = (wide & (sign - np.uint64(1))).astype(np.int64)
    return np.where(wide & sign, -mag, mag)


def gap(a: np.ndarray, b: np.ndarray, dtype: str) -> int:
    return int(np.abs(order(a, dtype) - order(b, dtype)).max()) if a.size else 0


def cast(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round to a dtype and read it back."""
    return to_float(from_float(x, dtype), dtype)


def delta(t: Term, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PEFT's get_delta_weight: the product in float32, then the scale."""
    d = (b @ a) * np.float32(t.scale)
    return d.T if t.transpose else d


def add(dtype: str, base: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The weight plus a delta, in the accumulator torch and mlx use for this dtype, rounded once."""
    acc = np.float64 if dtype == "F64" else np.float32
    with np.errstate(over="ignore"):
        return from_float(to_float(base, dtype).astype(acc) + d.astype(acc), dtype)


def peft(dtype, base, t):
    """merge_and_unload(). The adapter sits in float32, and `weight += delta` adds in float32."""
    return add(dtype, base, delta(t, t.a, t.b))


def peft_safe(dtype, base, t):
    """merge_and_unload(safe_merge=True). The delta is rounded to the model dtype before the add."""
    return add(dtype, base, cast(delta(t, t.a, t.b), dtype))


def peft_lowp(dtype, base, t):
    """An adapter loaded with autocast_adapter_dtype=False on a CPU: A and B are in the model dtype."""
    return add(dtype, base, cast(delta(t, cast(t.a, dtype), cast(t.b, dtype)), dtype))


def mlx_fuse(dtype, base, t):
    """mlx_lm.fuse. The scale goes onto B before the product, and the product stays in the adapter dtype.

    mlx gives a Python scalar the dtype of the array it meets, so the scale is rounded to the adapter dtype too.
    """
    b = cast(cast(np.array([t.scale], np.float32), t.dtype)[0] * t.b, t.dtype)
    d = cast(b @ t.a, t.dtype)
    return add(dtype, base, cast(d.T if t.transpose else d, dtype))


def wide(t: Term) -> np.ndarray:
    """The delta in float64."""
    d = t.scale * (t.b.astype(np.float64) @ t.a.astype(np.float64))
    return d.T if t.transpose else d


def exact(dtype, base, t):
    """One rounding of the exact sum. No tool does this, but a merge in float64 lands here."""
    return encode(to_float(base, dtype).astype(np.float64) + wide(t), dtype)


RECIPES = {"peft": peft, "peft-safe": peft_safe, "peft-lowp": peft_lowp, "mlx-fuse": mlx_fuse, "exact": exact}
PATHS = list(RECIPES)


def classify(name: str, ref: TensorRef, base: np.ndarray, merged: np.ndarray, ulps: int,
             term: Term | None = None) -> TensorCheck:
    dtype = ref.dtype
    made = {"base": base} if term is None else {p: r(dtype, base, term) for p, r in RECIPES.items()}
    same = [p for p, m in made.items() if np.array_equal(merged, m)]
    still = np.array_equal(merged, base)

    applied = None
    if term is not None:
        moved = wide(term).reshape(-1)
        change = (to_float(merged, dtype).astype(np.float64) - to_float(base, dtype).astype(np.float64)).reshape(-1)
        applied = float(change @ moved / max(moved @ moved, 1e-300))

        if still and all(np.array_equal(m, base) for m in made.values()):
            lost = "the delta rounds away, so this adapter changes nothing here" if moved.any() else ""
            return TensorCheck(name, IDENTICAL, lost, applied)
        if still:
            gone = [p for p in same if p != "base"]
            why = (f"the delta rounds away under {', '.join(gone)}, so it merged to nothing"
                   if gone else "the merged tensor is the base tensor")
            return TensorCheck(name, NEVER_APPLIED, why, 0.0)
        real = [p for p in same if p != "exact"]
        if real:
            return TensorCheck(name, MERGED, ", ".join(real), applied, real)
        if same:
            return TensorCheck(name, ROUNDING, "equals the exact sum, which no tool's arithmetic produces", applied)
    elif still:
        return TensorCheck(name, IDENTICAL)

    if not is_float(dtype):
        return TensorCheck(name, DIFFERENT, f"{int((merged != base).sum()):,} entries differ", applied)
    steps, near = min((gap(merged, m, dtype), p) for p, m in made.items())
    off = min(int((merged != m).sum()) for m in made.values())
    # A different order of the float32 product moves a handful of entries one step. A tensor with most entries
    # moved is an edit, however small each step.
    if steps <= ulps and off <= MAX_OFF * merged.size:
        where = "" if term is None else f" of {near}"
        return TensorCheck(name, ROUNDING, f"within {steps} step{'s' * (steps != 1)}{where}", applied, steps=steps)
    return TensorCheck(name, DIFFERENT, f"{off:,} of {merged.size:,} entries differ, by up to {steps:,} steps", applied,
                       steps=steps)


def targets(adapter: Adapter, base: Checkpoint) -> dict[str, str]:
    """Base tensor to adapter tensor.

    A tied head has no tensor of its own, so its delta lands on the embedding. PEFT names modules from the
    top of the model class and a base file may leave off a wrapper such as "transformer.", so a name that
    is not found is matched by its tail.
    """
    found = {}
    for name in adapter.pairs:
        if name in base.tensors:
            hit = name
        elif name.endswith("lm_head.weight") and len([n for n in base.tensors if n.endswith("embed_tokens.weight")]) == 1:
            hit = next(n for n in base.tensors if n.endswith("embed_tokens.weight"))
        else:
            tails = [n for n in base.tensors if name.endswith("." + n) or n.endswith("." + name)]
            if len(tails) != 1:
                raise FormatError(f"the adapter targets {name}, which is not in the base model")
            hit = tails[0]
        found[hit] = name
        a, b = (tuple(adapter.pairs[name][s].shape) for s in "AB")
        if adapter.mlx:
            a, b = a[::-1], b[::-1]
        size = math.prod(base.tensors[hit].shape)
        if len(a) != 2 or len(b) != 2 or a[0] != b[1] or a[1] * b[0] != size:
            raise FormatError(f"the adapter's factors for {name} do not fit {hit}: {a} and {b}")
    return found


def aligned(base: Checkpoint, merged: Checkpoint) -> dict[str, TensorRef]:
    """Merged tensors by base name. Saving through a wrapper class can add or drop a prefix such as "transformer."."""
    found = {n: r for n, r in merged.tensors.items() if n in base.tensors}
    spare = {n: r for n, r in merged.tensors.items() if n not in base.tensors}
    for name in base.tensors:
        if name not in found:
            tails = [n for n in spare if n.endswith("." + name) or name.endswith("." + n)]
            if len(tails) == 1:
                found[name] = spare.pop(tails[0])
    return found


def tied_head(merged: Checkpoint, name: str) -> TensorRef | None:
    """A merged file leaves out lm_head.weight when it is tied to the embedding, and says so in its config."""
    if not name.endswith("lm_head.weight"):
        return None
    raw = merged.text("config.json")
    config = json.loads(raw) if raw else {}
    if not (config.get("tie_word_embeddings") or config.get("text_config", {}).get("tie_word_embeddings")):
        return None
    embed = [n for n in merged.tensors if n.endswith("embed_tokens.weight")]
    return merged.tensors[embed[0]] if len(embed) == 1 else None


def check(base: Checkpoint, merged: Checkpoint, adapter: Checkpoint | None = None, ulps: int = 1,
          only: set[str] | None = None) -> Check:
    """Compare every tensor, or only those named in `only` when a whole model is too much to read."""
    deltas = Adapter(adapter) if adapter else None
    onto = targets(deltas, base) if deltas else {}
    if only is not None:
        onto = {n: a for n, a in onto.items() if n in only}
    results, recast = [], set()
    pairs = aligned(base, merged)
    for name in sorted(only if only is not None else base.tensors):
        if name not in base.tensors:
            continue
        ref, other = base.tensors[name], pairs.get(name) or tied_head(merged, name)
        if other is None:
            results.append(TensorCheck(name, MISSING, "not in the merged model"))
        elif other.shape != ref.shape or (other.dtype != ref.dtype and not (is_float(ref.dtype) and is_float(other.dtype))):
            results.append(TensorCheck(name, DIFFERENT, f"{other.dtype} {list(other.shape)} where the base has "
                                       f"{ref.dtype} {list(ref.shape)}"))
        else:
            start = ref.read()
            if other.dtype != ref.dtype:
                # The model was loaded in another dtype before the merge, which converts the base first.
                start = from_float(to_float(start, ref.dtype), other.dtype)
                recast.add((ref.dtype, other.dtype))
            term = deltas.term(onto[name]) if name in onto else None
            if term is not None and not is_float(ref.dtype):
                raise FormatError(f"{name}: cannot merge an adapter into {ref.dtype}")
            results.append(classify(name, other, start, other.read(), ulps, term))
    leftover = sorted(n for n in merged.tensors if "lora_" in n)
    taken = {id(r) for r in pairs.values()}
    added = sorted(n for n, r in merged.tensors.items() if id(r) not in taken and n not in leftover)
    return Check(results, added, leftover, deltas.ignored if deltas else [], len(deltas.pairs) if deltas else 0, recast,
                 set(onto))


def edited_sample(base: Checkpoint, adapter: Checkpoint, count: int) -> set[str]:
    """Edited tensors spread over the model by a hash of the name, so a rerun reads the same ones."""
    names = sorted(targets(Adapter(adapter), base), key=lambda n: hashlib.sha1(n.encode()).hexdigest())
    return set(names[:count])


def check_sample(base: Checkpoint, merged: Checkpoint, adapter: Checkpoint, count: int = 12, ulps: int = 0) -> Check | None:
    """check() on a few edited tensors, for a verdict that costs megabytes. None if the adapter edits nothing."""
    names = edited_sample(base, adapter, count)
    return check(base, merged, adapter, ulps, only=names) if names else None


@dataclass
class Candidate:
    """One possible base for a merge, with how well the merge reproduces from it."""

    spec: str
    exact: int
    close: int
    off: int
    worst: int
    missing: int
    paths: list[str]
    shares: list[str] = field(default_factory=list)  # other candidates that agree with it on every sampled span

    @property
    def reproduces(self) -> bool:
        return self.off == 0 and self.missing == 0 and self.exact + self.close > 0


def name_base(merged: Checkpoint, adapter: Checkpoint, bases: list[Checkpoint], count: int = 12,
              ulps: int = 1) -> list[Candidate]:
    """Rank candidate bases by how many sampled edited tensors the merge reproduces from them.

    A model card, an adapter config and a mirror can each name a different base. The merge was made from the
    one whose tensors plus the adapter give the merged tensors back. Candidates that agree on every sampled span
    cannot be told apart by this merge and are listed as sharing weights (sampled, not proof).
    """
    from .identity import BIT_IDENTICAL, NO_DIFFERENCE, identify

    found = []
    for base in bases:
        c = check_sample(base, merged, adapter, count, ulps)
        if c is None:
            raise FormatError(f"{adapter.spec}: the adapter edits nothing in {base.spec}")
        edited = [t for t in c.tensors if t.name in c.edited]
        steps = [t.steps or 0 for t in edited if t.status in (ROUNDING, DIFFERENT)]
        found.append(Candidate(base.spec, sum(t.status in (MERGED, IDENTICAL) for t in edited),
                               sum(t.status == ROUNDING for t in edited),
                               sum(t.status in (DIFFERENT, NEVER_APPLIED) for t in edited),
                               max(steps, default=0), sum(t.status == MISSING for t in edited), c.paths))
    for i, a in enumerate(found):
        for j, b in enumerate(found):
            if i < j and identify(bases[i], bases[j]).label in (BIT_IDENTICAL, NO_DIFFERENCE):
                a.shares.append(b.spec)
                b.shares.append(a.spec)
    return sorted(found, key=lambda c: (c.off + c.missing, -c.exact, c.worst))
