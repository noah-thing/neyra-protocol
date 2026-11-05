"""The same tiny language models as tinylm, built and trained with torch, for the torch backend."""

import json

import numpy as np
import torch
from tinydata import VOCAB, Tokenizer, corpus, facts
from transformers import Qwen3Config, Qwen3ForCausalLM

from neyra.diff import canonical
from neyra.patch import write_safetensors
from neyra.runtime.torch import TorchRuntime


def build(layers: int = 4, hidden: int = 64) -> Qwen3ForCausalLM:
    torch.manual_seed(3)
    config = Qwen3Config(hidden_size=hidden, num_hidden_layers=layers, intermediate_size=128, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=VOCAB,
                         max_position_embeddings=256, tie_word_embeddings=False, rope_theta=10000.0)
    return Qwen3ForCausalLM(config)


def bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def save(model, folder) -> None:
    tensors = {name: bits(p) for name, p in model.state_dict().items()}
    write_safetensors(folder / "model.safetensors", tensors, {"format": "pt"}, "BF16")
    (folder / "config.json").write_text(model.config.to_json_string())


def runtime(model, wide: bool = False) -> TorchRuntime:
    rt = TorchRuntime.__new__(TorchRuntime)
    rt.device, rt.tokenizer, rt.originals = "cpu", Tokenizer(), {}
    rt.model = model.eval()
    rt.model.requires_grad_(False)
    rt.index = {canonical(k): k for k, _ in model.named_parameters()}
    if wide:
        rt.widen()
    return rt


def trained(steps: int = 115):
    """A tiny model that has learned the facts, so that editing one has a measurable effect."""
    model = build(layers=4, hidden=96)
    known = facts()
    data, mask = (torch.as_tensor(a) for a in corpus(known))
    mask = mask[:, 1:]
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(steps):
        logits = model(input_ids=data[:, :-1]).logits
        each = torch.nn.functional.cross_entropy(logits.transpose(1, 2), data[:, 1:], reduction="none")
        loss = (each * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.to(torch.bfloat16).requires_grad_(False)
    return model, json.loads(model.config.to_json_string()), known, float(loss.detach())
