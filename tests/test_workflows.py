import numpy as np
from conftest import bf16
from fakes import NAME

from neyra.numerics import to_float
from neyra.patch import Patch, diff_tensor
from neyra.protocol import tensor_table
from neyra.runtime import Rollout
from neyra.store import Checkpoint
from neyra.workflows import bisect_loops, collect_loops, fix_loops, split


def looped(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5000, 400).tolist() + rng.integers(0, 5000, 37).tolist() * 30


class Quiet:
    def __init__(self, tokens):
        self.tokens = tokens

    def encode_chat(self, user, thinking=True):
        return [1]

    def generate(self, prompts, max_tokens, sampling=None, batch=12):
        return [Rollout(p, self.tokens, "x", False) for p in prompts]


class Probing:
    """Scores a text by how many of the edited columns are present."""

    def __init__(self, base, good):
        self.base, self.good, self.current = base, good, {}

    def set_tensors(self, tensors):
        self.current = tensors

    def reset(self):
        self.current = {}

    def logprobs(self, tokens, start=1, positions=None):
        drop = 0.0
        if NAME in self.current:
            changed = np.flatnonzero((self.current[NAME][0] != self.base).any(axis=0))
            drop = len(set(changed.tolist()) & self.good)
        out = np.zeros(len(positions))
        out[:len(positions) // 2] = -drop
        return out


def test_collect_keeps_only_loops():
    assert len(collect_loops(Quiet(looped(0)), 3, 8192, None, say=lambda _: None)) == 3
    assert collect_loops(Quiet(list(range(900))), 3, 8192, None, say=lambda _: None) == []


def test_a_model_that_does_not_loop_gets_no_patch():
    assert fix_loops(Quiet(list(range(900))), None, n=4, say=lambda _: None) is None


def test_split_never_returns_an_empty_side():
    assert split([1]) == ([1], [1])
    assert split([1, 2, 3]) == ([1, 2], [3])


def test_bisect_finds_the_columns_that_matter_and_confirms(model_dir, rng):
    w = bf16(rng.standard_normal((16, 24)) * 0.1)
    edited = w.copy()
    f = to_float(w, "BF16")
    edited[:, :8] = bf16(f[:, :8] * 0.5)
    ck = Checkpoint.open(str(model_dir("m", {NAME: w})))
    patch = Patch([diff_tensor(NAME, "BF16", w, edited)]).bind(tensor_table(ck))
    runtime = Probing(w, {2, 5})
    tokens = [looped(i) for i in range(6)]
    b = bisect_loops(runtime, ck, patch, tokens, say=lambda _: None)
    assert {b.units.units[i].index for i in b.result.units} == {2, 5}
    assert b.check.passed and b.check.retained > 0.95
    assert len(b.patch.edits[0].idx) < len(patch.edits[0].idx)
