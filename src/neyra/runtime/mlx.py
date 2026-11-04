from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load

from ..diff import canonical
from . import Rollout, Sampling

STORAGE = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


def ordered_sampler(s: Sampling):
    """Temperature, then top k, then top p, the order vLLM uses.

    mlx-lm filters on the unscaled distribution first, which samples a different one.
    With top k set, only those candidates are ever sorted.
    """
    if s.temperature == 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)
    k = s.top_k

    def sample(logprobs: mx.array) -> mx.array:
        if k > 0:
            index = mx.argpartition(-logprobs, k - 1, axis=-1)[..., :k]
            scores = mx.take_along_axis(logprobs, index, axis=-1)
        else:
            index = mx.broadcast_to(mx.arange(logprobs.shape[-1]), logprobs.shape)
            scores = logprobs
        scores = scores / s.temperature
        order = mx.argsort(-scores, axis=-1)
        scores = mx.take_along_axis(scores, order, axis=-1)
        index = mx.take_along_axis(index, order, axis=-1)
        probs = mx.softmax(scores, axis=-1)
        before = mx.cumsum(probs, axis=-1) - probs
        scores = mx.where(before < s.top_p, scores, -mx.inf)
        choice = mx.random.categorical(scores)
        return mx.take_along_axis(index, choice[..., None], axis=-1)[..., 0]

    return sample


def output_presence(penalty: float, prompt_length: int):
    """Presence penalty over generated tokens only, as in vLLM.

    The stock processor also counts the prompt, which on a maths problem penalises
    repeating the numbers in the question.
    """
    mask = None

    def apply(tokens, logits: mx.array) -> mx.array:
        nonlocal mask
        if len(tokens) <= prompt_length:
            return logits
        if mask is None:
            mask = mx.zeros((logits.shape[-1],), dtype=logits.dtype)
        mask[tokens[-1]] = penalty
        return logits - mask

    return apply


class MlxRuntime:
    def __init__(self, path: str):
        self.model, self.tokenizer = load(path)
        self.index = {canonical(k): k for k, _ in tree_flatten(self.model.parameters())}
        self.originals: dict[str, mx.array] = {}

    def _param(self, hf_name: str) -> str:
        key = self.index.get(canonical(hf_name))
        if key is None:
            raise KeyError(f"{hf_name} has no matching parameter in the loaded model")
        return key

    def set_tensors(self, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
        """Swap raw bits in. Anything swapped earlier and absent here goes back to the original."""
        params = dict(tree_flatten(self.model.parameters()))
        wanted = {self._param(n): v for n, v in tensors.items()}
        updates = [(k, self.originals.pop(k)) for k in list(self.originals) if k not in wanted]
        for key, (bits, dtype) in wanted.items():
            self.originals.setdefault(key, params[key])
            updates.append((key, mx.array(bits).view(STORAGE[dtype]).reshape(params[key].shape)))
        if updates:
            self.model.load_weights(updates, strict=False)
            mx.eval(self.model.parameters())

    def reset(self) -> None:
        self.set_tensors({})

    def encode_chat(self, user: str, thinking: bool = True) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=True,
            enable_thinking=thinking)
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    def generate(self, prompts: list[list[int]], max_tokens: int, sampling: Sampling | None = None,
                 batch: int = 32) -> list[Rollout]:
        s = sampling or Sampling()
        gen = sys.modules["mlx_lm.generate"]
        mx.random.seed(s.seed)
        engine = gen.BatchGenerator(self.model, stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
                                    sampler=ordered_sampler(s), completion_batch_size=batch)
        processors = [[output_presence(s.presence_penalty, len(p))] for p in prompts] if s.presence_penalty else None
        uids = engine.insert(prompts, [max_tokens] * len(prompts), logits_processors=processors)
        tokens: dict[int, list[int]] = {u: [] for u in uids}
        capped: dict[int, bool] = {}
        while responses := engine.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    tokens[r.uid].append(int(r.token))
                if r.finish_reason is not None:
                    capped[r.uid] = r.finish_reason == "length"
        engine.close()
        return [Rollout(p, tokens[u], self.tokenizer.decode(tokens[u]), capped.get(u, True))
                for p, u in zip(prompts, uids)]

    def _head(self):
        lm = getattr(self.model, "language_model", self.model)
        if getattr(lm, "lm_head", None) is not None:
            return lm.model, lm.lm_head
        return lm.model, lm.model.embed_tokens.as_linear

    def logprobs(self, tokens: list[int], start: int = 1, positions: np.ndarray | None = None) -> np.ndarray:
        """Log probability of tokens, teacher forced: every one from start, or just positions.

        With a large vocabulary the output head dominates, so it only runs where needed.
        """
        at = np.arange(start, len(tokens)) if positions is None else np.asarray(positions)
        trunk, head = self._head()
        hidden = trunk(mx.array(tokens)[None][:, :-1])[0]
        out = []
        for i in range(0, len(at), 256):
            chunk = at[i:i + 256]
            logits = head(hidden[mx.array(chunk - 1)]).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            picked = mx.take_along_axis(lp, mx.array(np.asarray(tokens)[chunk])[:, None], axis=-1)[:, 0]
            mx.eval(picked)
            out.append(np.array(picked))
        return np.concatenate(out)

    def logits_at(self, tokens: list[int], positions: np.ndarray) -> np.ndarray:
        """Logits predicting each given position, in one forward pass."""
        trunk, head = self._head()
        hidden = trunk(mx.array(tokens)[None][:, :-1])[0]
        out = head(hidden[mx.array(np.asarray(positions) - 1)]).astype(mx.float32)
        mx.eval(out)
        return np.array(out)

    def layers(self) -> list[nn.Module]:
        m = self.model
        for attr in ("language_model", "model"):
            m = getattr(m, attr, m)
        return list(m.layers)

    def capture(self, tokens: list[int], layers: list[int], kind: str = "resid") -> dict[int, np.ndarray]:
        """Activations per token. resid is the block output, mlp is the input to down_proj."""
        blocks = self.layers()
        seen: dict[int, mx.array] = {}
        restore = []
        for i in layers:
            block = blocks[i]
            if kind == "mlp":
                target = block.mlp.down_proj
                restore.append((block.mlp, "down_proj", target))
                block.mlp.down_proj = _Tap(target, seen, i)
            else:
                restore.append((blocks, i, block))
                blocks[i] = _Tap(block, seen, i, output=True)
        if kind == "resid":
            self._set_layers(blocks)
        try:
            self.model(mx.array(tokens)[None])
            mx.eval(list(seen.values()))
        finally:
            for owner, key, original in restore:
                if isinstance(owner, list):
                    owner[key] = original
                else:
                    setattr(owner, key, original)
            if kind == "resid":
                self._set_layers(blocks)
        return {i: np.array(v[0].astype(mx.float32)) for i, v in seen.items()}

    def _set_layers(self, blocks: list) -> None:
        m = self.model
        for attr in ("language_model", "model"):
            m = getattr(m, attr, m)
        m.layers = blocks


class _Tap(nn.Module):
    def __init__(self, inner, sink: dict, key: int, output: bool = False):
        super().__init__()
        self.inner, self.sink, self.key, self.output = inner, sink, key, output

    def __getattr__(self, name):
        # Hybrid stacks branch on block attributes such as is_linear while iterating.
        if name in self:
            return self[name]
        inner = dict.get(self, "inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, x, *args, **kwargs):
        y = self.inner(x, *args, **kwargs)
        self.sink[self.key] = y if self.output else x
        return y
