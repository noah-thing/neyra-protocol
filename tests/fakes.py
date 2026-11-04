"""A toy runtime whose behavior depends on one secret direction in one tensor."""

import numpy as np

from neyra.numerics import to_float
from neyra.runtime import Rollout

NAME = "model.layers.0.mlp.down_proj.weight"


class Tokenizer:
    def encode(self, text, **_):
        return [ord(c) % 97 for c in text][:64]


class FakeRuntime:
    tokenizer = Tokenizer()

    def __init__(self, base_bits, secret):
        self.base = to_float(base_bits, "BF16").astype(np.float64)
        self.secret = secret
        self.current = {}
        self.generated = 0

    def set_tensors(self, tensors):
        self.current = dict(tensors)

    def reset(self):
        self.current = {}

    def repaired(self):
        if NAME not in self.current:
            return False
        bits, dtype = self.current[NAME]
        delta = to_float(bits, dtype).astype(np.float64) - self.base
        return np.linalg.norm(self.secret @ delta) > 0.5 * np.linalg.norm(delta)

    def encode_chat(self, user, thinking=True):
        return [hash(user) % 1000]

    def generate(self, prompts, max_tokens, sampling=None, batch=32):
        self.generated += len(prompts)
        rng = np.random.default_rng(self.generated)
        rows = []
        for p in prompts:
            fixed = self.repaired() and rng.random() < 0.9
            text = r"thinking</think> \boxed{4}" if fixed else "wait again " * 50
            rows.append(Rollout(p, list(range(40)), text, not fixed))
        return rows

    def logits_at(self, ids, positions):
        return np.zeros((len(positions), 8))

    def logprobs(self, tokens, start=1, positions=None):
        return np.zeros(len(tokens) - start if positions is None else len(positions))

    def layers(self):
        return [object()]
