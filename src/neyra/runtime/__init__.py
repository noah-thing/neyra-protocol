"""Backends that run a model with tensors swapped in place."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

import numpy as np


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


@dataclass
class Hooks:
    """Interventions on one forward pass of a single sequence."""

    noise: tuple[np.ndarray, np.ndarray] | None = None  # positions, vectors added to the token embeddings
    put: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # layer -> positions, MLP outputs


@dataclass
class DeltaBatch:
    """Rows to push through a model with a trainable vector added to one layer's MLP output.

    Row b adds deltas[owner[b]] at position pos[b]. Each scored item reads the logits at
    (row, col): a label of -1 means KL against the next reference log probs, otherwise the
    negative log probability of that label. Items are weighted, so means are the caller's choice.
    """

    tokens: np.ndarray  # [B, T], right padded
    owner: np.ndarray
    pos: np.ndarray
    row: np.ndarray
    col: np.ndarray
    label: np.ndarray
    weight: np.ndarray
    ref: np.ndarray  # [number of KL items, vocabulary]


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def open_runtime(path: str, backend: str = "auto"):
    """MLX on Apple Silicon when it is installed, torch everywhere else."""
    if backend == "mlx" or (backend == "auto" and apple_silicon()):
        try:
            from .mlx import MlxRuntime

            return MlxRuntime(path)
        except ImportError:
            if backend == "mlx":
                raise
    from .torch import TorchRuntime

    return TorchRuntime(path)
