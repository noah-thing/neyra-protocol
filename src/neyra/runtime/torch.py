from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diff import canonical
from . import DeltaBatch, Hooks, Rollout, Sampling

STORAGE = {"BF16": (torch.int16, torch.bfloat16), "F16": (torch.int16, torch.float16),
           "F32": (torch.int32, torch.float32)}


def device() -> str:
    if forced := os.environ.get("NEYRA_DEVICE"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def wide_on(dev: str) -> bool:
    """Whether to compute in float32 from bf16 weights.

    A CPU runs bf16 matrix products several times slower than float32 ones, and float32 weights would double the
    memory, so the weights stay in bf16 and each product widens them for the length of the call. NEYRA_WIDE=0 turns it off.
    """
    return os.environ.get("NEYRA_WIDE", "1" if dev == "cpu" else "0") == "1"


class TorchRuntime:
    def __init__(self, path: str, dev: str | None = None):
        self.device = dev or device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(self.device).eval()
        self.model.requires_grad_(False)
        if wide_on(self.device):
            self.widen()
        self.index = {canonical(k): k for k, _ in self.model.named_parameters()}
        self.originals: dict[str, torch.Tensor] = {}

    def widen(self) -> None:
        """Run the linear layers in float32 on the stored weights, which are not copied.

        The output head stays in bf16: it is the largest matrix, and widening it would copy it on every call.
        """
        head = self.model.get_output_embeddings()
        for module in self.model.modules():
            if module is head:
                module.forward = lambda x, m=module: torch.nn.functional.linear(x.to(m.weight.dtype), m.weight, m.bias)
            elif isinstance(module, torch.nn.Linear):
                module.forward = lambda x, m=module: torch.nn.functional.linear(x.float(), m.weight.float(), m.bias)

    def _param(self, hf_name: str) -> str:
        key = self.index.get(canonical(hf_name))
        if key is None:
            raise KeyError(f"{hf_name} has no matching parameter in the loaded model")
        return key

    @torch.no_grad()
    def set_tensors(self, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
        params = dict(self.model.named_parameters())
        wanted = {self._param(n): v for n, v in tensors.items()}
        for key in [k for k in self.originals if k not in wanted]:
            params[key].copy_(self.originals.pop(key))
        for key, (bits, dtype) in wanted.items():
            self.originals.setdefault(key, params[key].detach().clone())
            signed, real = STORAGE[dtype]
            raw = torch.from_numpy(bits.view(np.int16 if signed is torch.int16 else np.int32).copy())
            params[key].copy_(raw.view(real).reshape(params[key].shape).to(params[key].device))

    def reset(self) -> None:
        self.set_tensors({})

    def encode_chat(self, user: str, thinking: bool = True) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=True,
            enable_thinking=thinking, return_dict=False)
        return list(ids)

    def _stops(self) -> set[int]:
        stops = {self.tokenizer.eos_token_id}
        extra = getattr(self.model.generation_config, "eos_token_id", None)
        stops |= {extra} if isinstance(extra, int) else set(extra or [])
        return stops - {None}

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_tokens: int, sampling: Sampling | None = None,
                 batch: int = 16) -> list[Rollout]:
        sampling = sampling or Sampling()
        torch.manual_seed(sampling.seed)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        stops = self._stops()
        if sampling.temperature == 0:
            choice = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        else:
            choice = {"do_sample": True, "temperature": sampling.temperature, "top_p": sampling.top_p,
                      "top_k": sampling.top_k}
        rows = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            width = max(map(len, chunk))
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in chunk], device=self.device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk], device=self.device)
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens,
                                      pad_token_id=pad, **choice)
            for p, row in zip(chunk, out[:, width:].tolist()):
                stop = next((j for j, t in enumerate(row) if t in stops), None)
                tokens = row if stop is None else row[:stop]
                rows.append(Rollout(p, tokens, self.tokenizer.decode(tokens), stop is None))
        return rows

    def _parts(self):
        """The text trunk, which returns hidden states, and the output head."""
        trunk = getattr(self.model, "model", self.model)
        return getattr(trunk, "language_model", trunk), self.model.get_output_embeddings()

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._parts()[0](input_ids=tokens, use_cache=False).last_hidden_state

    @contextmanager
    def _hooked(self, hooks: Hooks | None):
        """Interventions for the length of a block: noise on the embeddings, replaced MLP outputs."""
        with ExitStack() as stack:
            if hooks is not None and hooks.noise is not None:
                shift = self._rows(*hooks.noise)

                def jitter(_, __, e, shift=shift):
                    at, vectors = shift
                    return e + torch.zeros(e.shape[1:], dtype=torch.float32, device=e.device).index_add(
                        0, at, vectors).to(e.dtype)

                stack.callback(self._parts()[0].embed_tokens.register_forward_hook(jitter).remove)
            for i, (where, values) in (hooks.put if hooks is not None else {}).items():
                at, vectors = self._rows(where, values)

                def put(_, __, y, at=at, vectors=vectors):
                    full = torch.zeros(y.shape[1:], dtype=torch.float32, device=y.device).index_add(0, at, vectors)
                    hit = torch.zeros(y.shape[1], dtype=torch.float32, device=y.device).index_add(
                        0, at, torch.ones(len(at), device=y.device)) > 0
                    return torch.where(hit[None, :, None], full[None].to(y.dtype), y)

                stack.callback(self.layers()[i].mlp.down_proj.register_forward_hook(put).remove)
            yield

    def _rows(self, at, vectors) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.as_tensor(np.asarray(at), device=self.device),
                torch.as_tensor(np.asarray(vectors), dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def logprobs(self, tokens: list[int], start: int = 1, positions: np.ndarray | None = None,
                 hooks: Hooks | None = None) -> np.ndarray:
        at = np.arange(start, len(tokens)) if positions is None else np.asarray(positions)
        x = torch.tensor([tokens], device=self.device)
        index = torch.as_tensor(at, device=self.device)
        with self._hooked(hooks):
            hidden = self._hidden(x[:, :-1])[0, index - 1]
        lp = torch.log_softmax(self._parts()[1](hidden).float(), dim=-1)
        return lp.gather(1, x[0, index, None])[:, 0].cpu().numpy()

    @torch.no_grad()
    def logits_at(self, tokens: list[int], positions: np.ndarray, hooks: Hooks | None = None) -> np.ndarray:
        """Logits predicting each given position, in one forward pass."""
        x = torch.tensor([tokens], device=self.device)
        index = torch.as_tensor(np.asarray(positions), device=self.device)
        with self._hooked(hooks):
            hidden = self._hidden(x[:, :-1])[0, index - 1]
        return self._parts()[1](hidden).float().cpu().numpy()

    def embedding_scale(self) -> float:
        """Spread of the token embeddings, which sets how much noise hides a subject."""
        return float(self._parts()[0].embed_tokens.weight.float().std(unbiased=False))

    @torch.no_grad()
    def gather(self, rows: list[list[int]], layers: list[int], kind: str = "mlp",
               at: list[int] | None = None, chunk: int = 16) -> dict[int, np.ndarray]:
        """Activations from many sequences: one position per row, or every token of every row."""
        out: dict[int, list[np.ndarray]] = {i: [] for i in layers}
        blocks = self.layers()
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            width = max(map(len, part))
            seen: dict[int, torch.Tensor] = {}
            with ExitStack() as stack:
                for i in layers:
                    down = blocks[i].mlp.down_proj
                    if kind == "out":
                        def tap(_, __, y, i=i, seen=seen):
                            seen[i] = y
                        stack.callback(down.register_forward_hook(tap).remove)
                    else:
                        def tap(_, args, i=i, seen=seen):
                            seen[i] = args[0]
                        stack.callback(down.register_forward_pre_hook(tap).remove)
                self._hidden(torch.tensor([r + [0] * (width - len(r)) for r in part], device=self.device))
            for i in layers:
                a = seen[i].float().cpu().numpy()
                if at is None:
                    out[i] += [a[b, :len(r)] for b, r in enumerate(part)]
                else:
                    out[i].append(a[np.arange(len(part)), at[s:s + chunk]])
        return {i: np.concatenate(v) for i, v in out.items()}

    @torch.no_grad()
    def score(self, rows: list[tuple[list[int], list[int]]], chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Mean negative log probability of each continuation, and whether greedy decoding would produce it."""
        head = self._parts()[1]
        nll, exact = [], []
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            seqs = [p + t for p, t in part]
            width = max(map(len, seqs))
            hidden = self._hidden(torch.tensor([q + [0] * (width - len(q)) for q in seqs], device=self.device))
            who = np.concatenate([np.full(len(t), b) for b, (_, t) in enumerate(part)])
            col = np.concatenate([len(p) - 1 + np.arange(len(t)) for p, t in part])
            label = torch.as_tensor(np.concatenate([t for _, t in part]), device=self.device)
            logits = head(hidden[torch.as_tensor(who, device=self.device),
                                 torch.as_tensor(col, device=self.device)]).float()
            picked = torch.log_softmax(logits, dim=-1).gather(1, label[:, None])[:, 0].cpu().numpy()
            top = (logits.argmax(dim=-1) == label).cpu().numpy()
            nll += [float(-picked[who == b].mean()) for b in range(len(part))]
            exact += [bool(top[who == b].all()) for b in range(len(part))]
        return np.array(nll), np.array(exact)

    @property
    def row_budget(self) -> int:
        """Padded row tokens one gradient pass may hold.

        Layers after the edited one are recomputed on the backward pass, so what is kept is one layer's
        activations and each layer's input. A recurrent layer's state costs far more per token.
        """
        recurrent = any(hasattr(block, "linear_attn") for block in self.layers())
        return 110 if recurrent else 1024

    @contextmanager
    def _recomputed(self, first: int):
        """Layers from `first` on keep only their input for the backward pass and recompute the rest."""
        blocks = self.layers()[first:]
        for block in blocks:
            def run(*args, inner=block.forward, **kwargs):
                return checkpoint(inner, *args, use_reentrant=False, **kwargs)
            block.forward = run
        try:
            yield
        finally:
            for block in blocks:
                del block.forward

    def delta_loss(self, layer: int, deltas: np.ndarray, batch: DeltaBatch,
                   recompute: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Loss and gradient for each delta, which are independent because rows never share one.

        Layers before the edited one do not depend on the deltas, so no gradient flows through them.
        """
        head = self._parts()[1]

        def tensor(a, dtype=None):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

        tokens, owner = tensor(batch.tokens), tensor(batch.owner)
        slot = tensor(np.arange(tokens.shape[1])[None, :] == batch.pos[:, None], torch.float32)[..., None]
        row, col = tensor(batch.row), tensor(batch.col)
        label = tensor(np.maximum(batch.label, 0))
        kl_at = tensor(np.flatnonzero(batch.label < 0))
        ref = tensor(batch.ref, torch.float32)
        weight = tensor(batch.weight, torch.float32)
        who = tensor(batch.owner[batch.row][:, None] == np.arange(len(deltas))[None, :], torch.float32)
        mlp = self.layers()[layer].mlp
        with torch.enable_grad():
            delta = tensor(deltas, torch.float32).clone().requires_grad_()
            add = slot * delta[owner][:, None, :]
            handle = mlp.down_proj.register_forward_hook(lambda _, __, y: y + add.to(y.dtype))
            try:
                with self._recomputed(layer + 1 if recompute else len(self.layers())):
                    hidden = self._hidden(tokens)
            finally:
                handle.remove()
            lp = torch.log_softmax(head(hidden[row, col]).float(), dim=-1)
            nll = -lp.gather(1, label[:, None])[:, 0]
            if len(batch.ref):
                nll = nll.index_put((kl_at,), (ref.exp() * (ref - lp[kl_at])).sum(dim=-1))
            each = who.T @ (nll * weight)
            grad, = torch.autograd.grad(each.sum(), delta)
        return each.detach().cpu().numpy(), grad.cpu().numpy()

    def layers(self) -> list:
        m = self.model
        for attr in ("language_model", "model"):
            m = getattr(m, attr, m)
        return list(m.layers)

    @torch.no_grad()
    def capture(self, tokens: list[int], layers: list[int], kind: str = "resid") -> dict[int, np.ndarray]:
        blocks = self.layers()
        seen: dict[int, torch.Tensor] = {}
        hooks = []
        for i in layers:
            if kind == "mlp":
                def tap(_, args, i=i):
                    seen[i] = args[0]
                hooks.append(blocks[i].mlp.down_proj.register_forward_pre_hook(tap))
            else:
                def tap(_, __, out, i=i):
                    seen[i] = out[0] if isinstance(out, tuple) else out
                hooks.append(blocks[i].register_forward_hook(tap))
        try:
            self.model(torch.tensor([tokens], device=self.device))
        finally:
            for h in hooks:
                h.remove()
        return {i: v[0].float().cpu().numpy() for i, v in seen.items()}
