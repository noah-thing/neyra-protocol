"""Bit-level float conversions. numpy has no bfloat16."""

from __future__ import annotations

import numpy as np


def to_float(bits: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return bits.view(np.float16).astype(np.float32)
    if dtype == "F32":
        return bits.view(np.float32)
    if dtype == "F64":
        return bits.view(np.float64)
    raise TypeError(f"{dtype} is not a float dtype this tool can do arithmetic on")


def from_float(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        nan = (u & 0x7FFFFFFF) > 0x7F800000
        rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return np.where(nan, (u >> 16) | 0x40, rounded).astype(np.uint16)
    if dtype == "F16":
        with np.errstate(over="ignore"):
            return np.asarray(values, dtype=np.float16).view(np.uint16)
    if dtype == "F32":
        return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    if dtype == "F64":
        return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)
    raise TypeError(f"cannot encode {dtype}")


def is_float(dtype: str) -> bool:
    return dtype in ("BF16", "F16", "F32", "F64")
