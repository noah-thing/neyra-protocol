from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load

from ..diff import canonical
from . import DeltaBatch, Hooks, Rollout, Sampling

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


def cap_memory() -> None:
    """Keep MLX inside a budget. Unchecked growth on a shared machine can panic the kernel."""
    total = mx.device_info()["memory_size"]
    share = float(os.environ.get("NEYRA_MEMORY_SHARE", "0.45"))
    mx.set_memory_limit(int(total * share))
    mx.set_cache_limit(int(total * 0.1))


class MlxRuntime:
    def __init__(self, path: str):
        cap_memory()
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

    def logprobs(self, tokens: list[int], start: int = 1, positions: np.ndarray | None = None,
                 hooks: Hooks | None = None) -> np.ndarray:
        """Log probability of tokens, teacher forced: every one from start, or just positions.

        With a large vocabulary the output head dominates, so it only runs where needed.
        """
        at = np.arange(start, len(tokens)) if positions is None else np.asarray(positions)
        trunk, head = self._head()
        with self._hooked(hooks):
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

    def logits_at(self, tokens: list[int], positions: np.ndarray, hooks: Hooks | None = None) -> np.ndarray:
        """Logits predicting each given position, in one forward pass."""
        trunk, head = self._head()
        with self._hooked(hooks):
            hidden = trunk(mx.array(tokens)[None][:, :-1])[0]
        out = head(hidden[mx.array(np.asarray(positions) - 1)]).astype(mx.float32)
        mx.eval(out)
        return np.array(out)

    @contextmanager
    def _swapped(self, edits: list[tuple[object, str, object]]):
        """Replace attributes for the length of a block, then put the originals back."""
        saved = [(owner, name, getattr(owner, name)) for owner, name, _ in edits]
        for owner, name, value in edits:
            setattr(owner, name, value)
        try:
            yield
        finally:
            for owner, name, value in saved:
                setattr(owner, name, value)

    def _hooked(self, hooks: Hooks | None):
        edits = []
        if hooks is not None:
            blocks = self.layers()
            if hooks.noise is not None:
                trunk = self._head()[0]
                edits.append((trunk, "embed_tokens", _Hook(trunk.embed_tokens, _jitter(*hooks.noise))))
            for i, (where, values) in hooks.put.items():
                edits.append((blocks[i].mlp, "down_proj", _Hook(blocks[i].mlp.down_proj, _replace(where, values))))
        return self._swapped(edits)

    def embedding_scale(self) -> float:
        """Spread of the token embeddings, which sets how much noise hides a subject."""
        return float(mx.std(self._head()[0].embed_tokens.weight.astype(mx.float32)))

    def gather(self, rows: list[list[int]], layers: list[int], kind: str = "mlp",
               at: list[int] | None = None, chunk: int = 16) -> dict[int, np.ndarray]:
        """Activations from many sequences: one position per row, or every token of every row."""
        out: dict[int, list[np.ndarray]] = {i: [] for i in layers}
        blocks = self.layers()
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            width = max(map(len, part))
            seen: dict[int, mx.array] = {}
            taps = [(blocks[i].mlp, "down_proj", _Tap(blocks[i].mlp.down_proj, seen, i, output=kind == "out"))
                    for i in layers]
            with self._swapped(taps):
                self.model(mx.array([r + [0] * (width - len(r)) for r in part]))
                mx.eval(list(seen.values()))
            for i in layers:
                a = np.array(seen[i].astype(mx.float32))
                if at is None:
                    out[i] += [a[b, :len(r)] for b, r in enumerate(part)]
                else:
                    out[i].append(a[np.arange(len(part)), at[s:s + chunk]])
        return {i: np.concatenate(v) for i, v in out.items()}

    def score(self, rows: list[tuple[list[int], list[int]]], chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Mean negative log probability of each continuation, and whether greedy decoding would produce it."""
        trunk, head = self._head()
        nll, exact = [], []
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            seqs = [p + t for p, t in part]
            width = max(map(len, seqs))
            hidden = trunk(mx.array([q + [0] * (width - len(q)) for q in seqs]))
            who = np.concatenate([np.full(len(t), b) for b, (_, t) in enumerate(part)])
            col = np.concatenate([len(p) - 1 + np.arange(len(t)) for p, t in part])
            label = np.concatenate([t for _, t in part])
            logits = head(hidden[mx.array(who), mx.array(col)]).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            picked = mx.take_along_axis(lp, mx.array(label)[:, None], axis=-1)[:, 0]
            top = mx.argmax(logits, axis=-1) == mx.array(label)
            mx.eval(picked, top)
            picked, top = np.array(picked), np.array(top)
            nll += [float(-picked[who == b].mean()) for b in range(len(part))]
            exact += [bool(top[who == b].all()) for b in range(len(part))]
        return np.array(nll), np.array(exact)

    @contextmanager
    def _recomputed(self, first: int):
        """Layers from `first` on keep only their input for the backward pass and recompute the rest.

        A recurrent layer otherwise stores its state at every position of every row, which is gigabytes.
        """
        layers = self._head()[0].layers
        saved = list(layers)
        layers[first:] = [_Recompute(block) for block in saved[first:]]
        try:
            yield
        finally:
            layers[:] = saved

    @property
    def row_budget(self) -> int:
        """Padded row tokens one gradient pass may hold.

        A recurrent layer keeps its state's intermediates for every token of every row, about 20 MB each
        on Qwen3.5-0.8B, so hybrid stacks get a small budget. Attention only layers need far less.
        """
        recurrent = any(getattr(block, "is_linear", False) for block in self.layers())
        return 110 if recurrent else 1024

    def delta_loss(self, layer: int, deltas: np.ndarray, batch: DeltaBatch,
                   recompute: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Loss and gradient for each delta, which are independent because rows never share one.

        Layers before the edited one do not depend on the deltas, so no gradient flows through them.
        """
        trunk, head = self._head()
        mlp = self.layers()[layer].mlp
        tokens, owner = mx.array(batch.tokens), mx.array(batch.owner)
        slot = mx.array((np.arange(tokens.shape[1])[None, :] == batch.pos[:, None]).astype(np.float32))[..., None]
        row, col = mx.array(batch.row), mx.array(batch.col)
        label = mx.array(np.maximum(batch.label, 0))
        is_kl = mx.array(batch.label < 0)
        kl_at = mx.array(np.flatnonzero(batch.label < 0))
        ref = mx.array(batch.ref)
        weight = mx.array(batch.weight, dtype=mx.float32)
        who = mx.array((batch.owner[batch.row][:, None] == np.arange(len(deltas))[None, :]).astype(np.float32))
        seen: list[mx.array] = []

        def total(d):
            add = slot * d[owner][:, None, :]
            with (self._swapped([(mlp, "down_proj", _Hook(mlp.down_proj, lambda y: y + add.astype(y.dtype)))]),
                  self._recomputed(layer + 1 if recompute else len(self.layers()))):
                hidden = trunk(tokens)
            logits = head(hidden[row, col]).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(lp, label[:, None], axis=-1)[:, 0]
            if len(batch.ref):
                kl = (mx.exp(ref) * (ref - lp[kl_at])).sum(axis=-1)
                nll = mx.where(is_kl, mx.zeros_like(nll).at[kl_at].add(kl), nll)
            each = who.T @ (nll * weight)
            seen.append(each)
            return each.sum()

        self.model.train()
        try:
            _, grad = mx.value_and_grad(total)(mx.array(deltas, dtype=mx.float32))
            mx.eval(grad, seen[-1])
        finally:
            self.model.eval()
            mx.clear_cache()
        return np.array(seen[-1]), np.array(grad)

    def layers(self) -> list[nn.Module]:
        m = self.model
        for attr in ("language_model", "model"):
            m = getattr(m, attr, m)
        return list(m.layers)

    def capture(self, tokens: list[int], layers: list[int], kind: str = "resid") -> dict[int, np.ndarray]:
        """Activations per token. resid is the block output, mlp the input to down_proj, out its output."""
        blocks = self.layers()
        seen: dict[int, mx.array] = {}
        restore = []
        for i in layers:
            block = blocks[i]
            if kind in ("mlp", "out"):
                target = block.mlp.down_proj
                restore.append((block.mlp, "down_proj", target))
                block.mlp.down_proj = _Tap(target, seen, i, output=kind == "out")
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


class _Hook(nn.Module):
    """Passes a module's output through fn."""

    def __init__(self, inner, fn):
        super().__init__()
        self.inner, self.fn = inner, fn

    def __getattr__(self, name):
        if name in self:
            return self[name]
        inner = dict.get(self, "inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, x, *args, **kwargs):
        return self.fn(self.inner(x, *args, **kwargs))


class _Recompute(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name):
        if name in self:
            return self[name]
        inner = dict.get(self, "inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, x, *args, **kwargs):
        return mx.checkpoint(lambda h: self.inner(h, *args, **kwargs))(x)


def _jitter(at: np.ndarray, vectors: np.ndarray):
    def add(e):
        extra = mx.zeros(e.shape[1:], dtype=mx.float32).at[mx.array(at)].add(mx.array(vectors, dtype=mx.float32))
        return e + extra.astype(e.dtype)

    return add


def _replace(at: np.ndarray, vectors: np.ndarray):
    def put(y):
        full = mx.zeros(y.shape[1:], dtype=mx.float32).at[mx.array(at)].add(mx.array(vectors, dtype=mx.float32))
        hit = mx.zeros((y.shape[1], 1), dtype=mx.float32).at[mx.array(at)].add(1.0) > 0
        return mx.where(hit[None], full[None].astype(y.dtype), y)

    return put
