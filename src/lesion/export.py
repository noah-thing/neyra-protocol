"""Turn a patch into something other runtimes load.

A low rank edit is already a LoRA, so a column projection on a 1024 x 3584 matrix
ships as a few kilobytes that stock llama.cpp, PEFT and vLLM can apply. Rounding
means the adapter matches the patch closely but not bit for bit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .numerics import to_float
from .patch import Patch, write_safetensors
from .store import Checkpoint

GGUF_NAMES = {
    r"mlp\.down_proj": "ffn_down",
    r"mlp\.gate_proj": "ffn_gate",
    r"mlp\.up_proj": "ffn_up",
    r"self_attn\.o_proj": "attn_output",
    r"self_attn\.q_proj": "attn_q",
    r"self_attn\.k_proj": "attn_k",
    r"self_attn\.v_proj": "attn_v",
    r"linear_attn\.out_proj": "ssm_out",
}


def factorize(delta: np.ndarray, energy: float = 0.9999, max_rank: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """delta ~= b @ a at the smallest rank that keeps the given share of its energy."""
    u, s, vh = np.linalg.svd(delta, full_matrices=False)
    kept = np.cumsum(s**2) / max((s**2).sum(), 1e-30)
    r = min(int(np.searchsorted(kept, energy)) + 1, max_rank, len(s))
    return u[:, :r] * s[:r], vh[:r]


def deltas(patch: Patch, base: Checkpoint) -> dict[str, np.ndarray]:
    out = {}
    for e in patch.edits:
        bits = base.tensors[e.name].read()
        out[e.name] = (to_float(e.apply(bits, verify=False), e.dtype).astype(np.float64)
                       - to_float(bits, e.dtype).astype(np.float64))
    return out


def peft_module(name: str) -> str:
    return re.sub(r"\.weight$", "", name)


def to_peft(patch: Patch, base: Checkpoint, out: str | Path, energy: float = 0.9999) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, ranks = {}, {}
    for name, delta in deltas(patch, base).items():
        b, a = factorize(delta, energy)
        key = f"base_model.model.{peft_module(name)}"
        tensors[f"{key}.lora_A.weight"] = a.astype(np.float32)
        tensors[f"{key}.lora_B.weight"] = b.astype(np.float32)
        ranks[peft_module(name)] = b.shape[1]
    rank = max(ranks.values())
    # Scaling is alpha / r per module, so each module's own rank goes in rank_pattern and alpha_pattern.
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank,
        "rank_pattern": ranks, "alpha_pattern": ranks,
        "target_modules": sorted(ranks), "bias": "none", "fan_in_fan_out": False,
        "base_model_name_or_path": base.spec,
    }
    write_safetensors(out / "adapter_model.safetensors", tensors, {"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return {"modules": ranks, "bytes": (out / "adapter_model.safetensors").stat().st_size}


def gguf_tensor(name: str) -> str:
    layer = re.search(r"layers\.(\d+)\.", name)
    if layer:
        for pattern, target in GGUF_NAMES.items():
            if re.search(pattern, name):
                return f"blk.{layer.group(1)}.{target}.weight"
    raise KeyError(f"no GGUF name known for {name}")


def to_gguf_lora(patch: Patch, base: Checkpoint, out: str | Path, arch: str, energy: float = 0.9999) -> dict:
    import gguf

    writer = gguf.GGUFWriter(str(out), arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_string("general.name", f"lesion {patch.fingerprint[:12]}")
    ranks = {}
    pending = []
    for name, delta in deltas(patch, base).items():
        b, a = factorize(delta, energy)
        pending.append((gguf_tensor(name), a, b))
        ranks[name] = b.shape[1]
    rank = max(ranks.values())
    if len(set(ranks.values())) > 1:
        # llama.cpp applies one scale, alpha / r of each tensor, so pad to a common rank.
        pending = [(n, np.pad(a, ((0, rank - a.shape[0]), (0, 0))), np.pad(b, ((0, 0), (0, rank - b.shape[1]))))
                   for n, a, b in pending]
    writer.add_float32("adapter.lora.alpha", float(rank))
    for target, a, b in pending:
        writer.add_tensor(f"{target}.lora_a", np.ascontiguousarray(a, dtype=np.float32))
        writer.add_tensor(f"{target}.lora_b", np.ascontiguousarray(b, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"tensors": ranks, "rank": rank, "bytes": Path(out).stat().st_size}
