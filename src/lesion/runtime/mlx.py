from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from ..diff import canonical
from . import Rollout, Sampling

STORAGE = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


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

    def generate(self, prompts: list[list[int]], max_tokens: int, sampling: Sampling = Sampling(),
                 batch: int = 32) -> list[Rollout]:
        gen = sys.modules["mlx_lm.generate"]
        mx.random.seed(sampling.seed)
        processors = None
        if sampling.presence_penalty:
            processors = make_logits_processors(presence_penalty=sampling.presence_penalty,
                                                presence_context_size=max_tokens + 8192)
        out = gen.batch_generate(
            self.model, self.tokenizer, prompts, max_tokens=max_tokens,
            sampler=make_sampler(temp=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k),
            logits_processors=processors, completion_batch_size=batch, return_token_ids=True)
        rows = []
        for prompt, text, ids in zip(prompts, out.texts, out.token_ids):
            ids = [int(t) for t in ids]
            rows.append(Rollout(prompt, ids, text, len(ids) >= max_tokens))
        return rows

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
