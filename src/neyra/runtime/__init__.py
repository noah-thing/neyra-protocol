"""Backends that run a model with tensors swapped in place."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    seed: int = 0


@dataclass
class Rollout:
    prompt: list[int]
    tokens: list[int]
    text: str
    hit_limit: bool


def open_runtime(path: str, backend: str = "auto"):
    if backend in ("auto", "mlx"):
        try:
            from .mlx import MlxRuntime

            return MlxRuntime(path)
        except ImportError:
            if backend == "mlx":
                raise
    from .torch import TorchRuntime

    return TorchRuntime(path)
