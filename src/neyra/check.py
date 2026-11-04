"""Decide whether a merged model is what its parts say it should be.

People merge a LoRA into a base model, save the result, and find it behaves like the base. The
causes are mundane: the adapter was never loaded, a tied head was skipped, or the delta was
smaller than the rounding step of the model dtype. This recomputes the merge and compares bits.

PEFT rounds twice: the adapter delta is cast to the model dtype, then added to the weight and
rounded again. Rounding once from the exact sum gives a slightly different tensor. Both are
correct merges, so a merged file that matches either one is equivalent to the adapter.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .merge import encode
from .numerics import from_float, is_float, to_float
from .store import ITEMSIZE, Checkpoint, FormatError, TensorRef

LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<side>[AB])(?:\.default)?\.weight$")
IDENTICAL, ROUNDING, DIFFERENT, MISSING, NEVER_APPLIED = "identical", "rounding", "different", "missing", "never applied"
EQUIVALENT = {IDENTICAL, ROUNDING}


@dataclass
class TensorCheck:
    name: str
    status: str
    detail: str = ""
    applied: float | None = None


@dataclass
class Check:
    tensors: list[TensorCheck]
    added: list[str] = field(default_factory=list)
    leftover: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    modules: int = 0

    def count(self, status: str) -> int:
        return sum(t.status == status for t in self.tensors)

    @property
    def targeted(self) -> int:
        """Tensors the adapter edits."""
        return sum(t.applied is not None for t in self.tensors)

    @property
    def equivalent(self) -> bool:
        return (all(t.status in EQUIVALENT for t in self.tensors) and not self.added and not self.leftover)


class Adapter:
    """The deltas of a PEFT LoRA adapter directory, one per base tensor."""

    def __init__(self, ck: Checkpoint):
        raw = ck.text("adapter_config.json")
        if raw is None:
            raise FormatError(f"{ck.spec}: no adapter_config.json")
        self.config = json.loads(raw)
        if self.config.get("use_dora"):
            raise FormatError(f"{ck.spec}: DoRA adapters are not supported")
        self.pairs: dict[str, dict[str, TensorRef]] = {}
        self.ignored: list[str] = []
        for name, ref in ck.tensors.items():
            hit = LORA_KEY.match(name)
            if hit and "lora_embedding" not in name:
                self.pairs.setdefault(hit["module"] + ".weight", {})[hit["side"]] = ref
            else:
                self.ignored.append(name)
        lonely = [n for n, p in self.pairs.items() if set(p) != {"A", "B"}]
        if lonely or not self.pairs:
            raise FormatError(f"{ck.spec}: lora_A and lora_B must come in pairs ({lonely[:3] or 'none found'})")

    def scaling(self, tensor: str, rank: int) -> float:
        module = tensor.removesuffix(".weight")

        def pattern(key: str, default):
            table = self.config.get(key) or {}
            hit = next((v for k, v in table.items() if module == k or module.endswith("." + k)), None)
            return default if hit is None else hit

        alpha = pattern("alpha_pattern", self.config.get("lora_alpha", rank))
        return alpha / math.sqrt(rank) if self.config.get("use_rslora") else alpha / rank

    def delta(self, tensor: str) -> tuple[np.ndarray, np.ndarray]:
        """The delta in float64, and as PEFT computes it in float32."""
        pair = self.pairs[tensor]
        a, b = (to_float(pair[s].read(), pair[s].dtype) for s in "AB")
        scale = self.scaling(tensor, a.shape[0])
        exact = scale * (b.astype(np.float64) @ a.astype(np.float64))
        peft = (b.astype(np.float32) @ a.astype(np.float32)) * np.float32(scale)
        if self.config.get("fan_in_fan_out"):
            exact, peft = exact.T, peft.T
        return exact, peft


def order(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Integers that rise with the float, so the difference of two is a count of representable steps."""
    sign = np.uint64(1) << np.uint64(8 * ITEMSIZE[dtype] - 1)
    wide = bits.astype(np.uint64)
    mag = (wide & (sign - np.uint64(1))).astype(np.int64)
    return np.where(wide & sign, -mag, mag)


def gap(a: np.ndarray, b: np.ndarray, dtype: str) -> int:
    return int(np.abs(order(a, dtype) - order(b, dtype)).max()) if a.size else 0


def expected(dtype: str, base: np.ndarray, exact: np.ndarray, peft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The merged tensor under one rounding of the exact sum, and under PEFT's two."""
    wide = to_float(base, dtype).astype(np.float64)
    single = encode(wide + exact, dtype)
    # The add runs in float32, as it does in torch for bf16 and fp16.
    acc = np.float64 if dtype == "F64" else np.float32
    rounded = to_float(from_float(peft, dtype), dtype).astype(acc)
    with np.errstate(over="ignore"):
        double = from_float(wide.astype(acc) + rounded, dtype)
    return single, double


def classify(name: str, ref: TensorRef, base: np.ndarray, merged: np.ndarray, ulps: int,
             delta: tuple[np.ndarray, np.ndarray] | None = None) -> TensorCheck:
    dtype = ref.dtype
    if delta is None:
        single = double = base
    else:
        single, double = expected(dtype, base, *delta)
    same_single, same_double = np.array_equal(merged, single), np.array_equal(merged, double)

    if delta is not None and np.array_equal(merged, base) and not np.array_equal(single, base):
        why = ("the delta is smaller than half a step of the model dtype under PEFT's rounding, so it merged to nothing"
               if np.array_equal(double, base) else "the merged tensor is the base tensor")
        return TensorCheck(name, NEVER_APPLIED, why, 0.0)
    applied = None
    if delta is not None:
        moved = delta[0].reshape(-1)
        change = to_float(merged, dtype).astype(np.float64).reshape(-1) - to_float(base, dtype).astype(np.float64).reshape(-1)
        applied = float(change @ moved / max(moved @ moved, 1e-300))

    if same_single and same_double:
        lost = delta is not None and np.array_equal(single, base) and delta[0].any()
        return TensorCheck(name, IDENTICAL, "the delta rounds away, so this adapter changes nothing here" if lost else "",
                           applied)
    if same_single or same_double:
        other = double if same_single else single
        name_of = "single" if same_single else "double"
        return TensorCheck(name, ROUNDING, f"equals {name_of} rounding; the other differs in "
                           f"{int((merged != other).sum()):,} entries", applied)
    if not is_float(dtype):
        return TensorCheck(name, DIFFERENT, f"{int((merged != base).sum()):,} entries differ", applied)
    steps = min(gap(merged, single, dtype), gap(merged, double, dtype))
    if steps <= ulps:
        return TensorCheck(name, ROUNDING, f"within {steps} step{'s' * (steps != 1)} of the recomputed merge", applied)
    off = int(min((merged != single).sum(), (merged != double).sum()))
    return TensorCheck(name, DIFFERENT, f"{off:,} of {merged.size:,} entries differ, by up to {steps:,} steps", applied)


def check(base: Checkpoint, merged: Checkpoint, adapter: Checkpoint | None = None, ulps: int = 1) -> Check:
    deltas = Adapter(adapter) if adapter else None
    if deltas:
        absent = sorted(set(deltas.pairs) - set(base.tensors))
        if absent:
            raise FormatError(f"the adapter targets {absent[0]}, which is not in the base model")
    results = []
    for name in sorted(base.tensors):
        ref, other = base.tensors[name], merged.tensors.get(name)
        if other is None:
            results.append(TensorCheck(name, MISSING, "not in the merged model"))
        elif (other.dtype, other.shape) != (ref.dtype, ref.shape):
            results.append(TensorCheck(name, DIFFERENT, f"{other.dtype} {list(other.shape)} where the base has "
                                       f"{ref.dtype} {list(ref.shape)}"))
        else:
            delta = deltas.delta(name) if deltas and name in deltas.pairs else None
            if delta is not None and not is_float(ref.dtype):
                raise FormatError(f"{name}: cannot merge an adapter into {ref.dtype}")
            results.append(classify(name, ref, ref.read(), other.read(), ulps, delta))
    leftover = sorted(n for n in merged.tensors if "lora_" in n)
    added = sorted(set(merged.tensors) - set(base.tensors) - set(leftover))
    return Check(results, added, leftover, deltas.ignored if deltas else [], len(deltas.pairs) if deltas else 0)
