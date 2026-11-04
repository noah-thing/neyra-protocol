"""What each model family stores, as data.

Rows were validated against the real tensor lists of 19 public checkpoints. A row names
a tensor that writes into the residual stream, which is the only place a direction edit
makes sense. Expert and quantized tensors are recognised so tools can refuse clearly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .store import Checkpoint

P = r"(?:model\.language_model\.|language_model\.model\.|model\.|)"
L = r"layers\.(?P<l>\d+)\."

WRITERS = (
    ("mlp_down", rf"^{P}{L}mlp\.down_proj\.weight$", "blk.{l}.ffn_down.weight"),
    ("shared_down", rf"^{P}{L}(?:mlp\.shared_experts?|ffn\.shared_experts)\.(?:down_proj|w2)\.weight$",
     "blk.{l}.ffn_down_shexp.weight"),
    ("expert_fused", rf"^{P}{L}(?:mlp\.)?experts\.down_proj$", "blk.{l}.ffn_down_exps.weight"),
    ("expert", rf"^{P}{L}(?:mlp|ffn|block_sparse_moe)\.experts\.(?P<e>\d+)\.(?:down_proj|w2)\.(?:weight|weight_packed)$",
     "blk.{l}.ffn_down_exps.weight"),
    ("expert_mxfp4", rf"^{P}{L}mlp\.experts\.down_proj_blocks$", "blk.{l}.ffn_down_exps.weight"),
    ("attn_out", rf"^{P}{L}self_attn\.o_proj\.weight$", "blk.{l}.attn_output.weight"),
    ("attn_out_b", rf"^{P}{L}attn\.wo_b\.weight$", "blk.{l}.attn_output_b.weight"),
    ("linear_out", rf"^{P}{L}linear_attn\.out_proj\.weight$", "blk.{l}.ssm_out.weight"),
)
COMPILED = tuple((kind, re.compile(rx), gguf) for kind, rx, gguf in WRITERS)
IGNORED = re.compile(r"^(?:mtp\.|original/|block\.\d+\.)|(?:^|\.)mtp\.|visual|vision|audio")
# Hyper connection models keep several residual streams, so one direction is not one edit.
HYPER_CONNECTED = {"qwen4_exp", "deepseek_v4"}
# A norm after the write means a rank 1 change is no longer a linear change to the stream.
POST_NORMED = {"gemma4"}


@dataclass(frozen=True)
class Writer:
    kind: str
    layer: int
    expert: int | None
    gguf: str


def classify(name: str, layers: int | None = None) -> Writer | None:
    if IGNORED.search(name):
        return None
    for kind, rx, gguf in COMPILED:
        m = rx.match(name)
        if not m:
            continue
        layer = int(m.group("l"))
        # GLM keeps its multi token layer in the normal list, past the real depth.
        if layers is not None and layer >= layers:
            return None
        expert = m.groupdict().get("e")
        return Writer(kind, layer, int(expert) if expert else None, gguf.format(l=layer))
    return None


def text_config(ck: Checkpoint) -> dict:
    raw = ck.text("config.json")
    if raw is None:
        return {}
    config = json.loads(raw)
    return {**config, **config.get("text_config", {})}


def writers(ck: Checkpoint) -> dict[str, Writer]:
    depth = text_config(ck).get("num_hidden_layers")
    found = {n: classify(n, depth) for n in ck.tensors}
    return {n: w for n, w in found.items() if w}


def v_head_order(columns: int, key_heads: int, value_heads: int) -> np.ndarray:
    """Column order llama.cpp's converter uses for linear attention out_proj.

    The checkpoint groups value heads by key head and GGUF tiles them, so a LoRA that
    skips this loads cleanly and edits the wrong columns.
    """
    r = value_heads // key_heads
    width = columns // value_heads
    return np.arange(columns).reshape(key_heads, r, width).transpose(1, 0, 2).reshape(columns)


def editable_warning(ck: Checkpoint) -> str | None:
    kind = text_config(ck).get("model_type", "")
    if kind in HYPER_CONNECTED:
        return f"{kind} keeps several residual streams, so a single direction edit is untested here"
    if kind in POST_NORMED:
        return f"{kind} normalises after each write, so a rank 1 edit is not a linear change to the stream"
    return None
