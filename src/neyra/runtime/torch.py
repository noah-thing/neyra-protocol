from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diff import canonical
from . import Rollout, Sampling

STORAGE = {"BF16": (torch.int16, torch.bfloat16), "F16": (torch.int16, torch.float16),
           "F32": (torch.int32, torch.float32)}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class TorchRuntime:
    def __init__(self, path: str, dev: str | None = None):
        self.device = dev or device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(self.device).eval()
        self.index = {canonical(k): k for k, _ in self.model.named_parameters()}
        self.originals: dict[str, torch.Tensor] = {}

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

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_tokens: int, sampling: Sampling | None = None,
                 batch: int = 16) -> list[Rollout]:
        sampling = sampling or Sampling()
        torch.manual_seed(sampling.seed)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        rows = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            width = max(map(len, chunk))
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in chunk], device=self.device)
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk], device=self.device)
            out = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens, do_sample=True,
                temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k,
                pad_token_id=pad)
            for p, row in zip(chunk, out[:, width:].tolist()):
                eos = self.tokenizer.eos_token_id
                ends = {eos} if isinstance(eos, int) else set(eos or [])
                stop = next((j for j, t in enumerate(row) if t in ends or t == pad), None)
                tokens = row if stop is None else row[:stop]
                rows.append(Rollout(p, tokens, self.tokenizer.decode(tokens), stop is None))
        return rows

    @torch.no_grad()
    def logprobs(self, tokens: list[int], start: int = 1, positions: np.ndarray | None = None) -> np.ndarray:
        at = np.arange(start, len(tokens)) if positions is None else np.asarray(positions)
        x = torch.tensor([tokens], device=self.device)
        index = torch.as_tensor(at, device=self.device)
        hidden = self.model.model(x[:, :-1]).last_hidden_state[0, index - 1]
        lp = torch.log_softmax(self.model.lm_head(hidden).float(), dim=-1)
        return lp.gather(1, x[0, index, None])[:, 0].cpu().numpy()

    @torch.no_grad()
    def logits_at(self, tokens: list[int], positions: np.ndarray) -> np.ndarray:
        x = torch.tensor([tokens], device=self.device)
        index = torch.as_tensor(np.asarray(positions), device=self.device)
        hidden = self.model.model(x[:, :-1]).last_hidden_state[0, index - 1]
        return self.model.lm_head(hidden).float().cpu().numpy()

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
