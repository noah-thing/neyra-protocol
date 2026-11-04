import json

import numpy as np
import pytest
from conftest import bf16

from neyra.export import gguf_tensor, reorder_for_gguf
from neyra.families import classify, v_head_order, writers
from neyra.repair import down_proj_name
from neyra.store import Checkpoint, canonical_shards


@pytest.mark.parametrize(("name", "kind", "gguf"), [
    ("model.language_model.layers.15.mlp.down_proj.weight", "mlp_down", "blk.15.ffn_down.weight"),
    ("model.layers.3.self_attn.o_proj.weight", "attn_out", "blk.3.attn_output.weight"),
    ("model.language_model.layers.5.linear_attn.out_proj.weight", "linear_out", "blk.5.ssm_out.weight"),
    ("model.language_model.layers.2.mlp.shared_expert.down_proj.weight", "shared_down", "blk.2.ffn_down_shexp.weight"),
    ("model.language_model.layers.2.mlp.experts.down_proj", "expert_fused", "blk.2.ffn_down_exps.weight"),
    ("model.layers.1.block_sparse_moe.experts.7.w2.weight", "expert", "blk.1.ffn_down_exps.weight"),
    ("model.layers.1.mlp.experts.down_proj_blocks", "expert_mxfp4", "blk.1.ffn_down_exps.weight"),
    ("layers.4.attn.wo_b.weight", "attn_out_b", "blk.4.attn_output_b.weight"),
])
def test_classify(name, kind, gguf):
    w = classify(name)
    assert w.kind == kind and w.gguf == gguf


@pytest.mark.parametrize("name", [
    "mtp.layers.0.mlp.down_proj.weight", "original/model.layers.0.mlp.down_proj.weight",
    "model.visual.blocks.0.mlp.down_proj.weight", "model.layers.0.mlp.gate_proj.weight",
])
def test_not_a_writer(name):
    assert classify(name) is None


def test_mtp_never_maps_to_a_real_block():
    with pytest.raises(KeyError):
        gguf_tensor("mtp.layers.0.mlp.down_proj.weight")


def test_expert_export_refuses_instead_of_writing_something_wrong():
    with pytest.raises(KeyError):
        gguf_tensor("model.layers.1.mlp.experts.down_proj")


def test_layer_past_the_real_depth_is_multi_token_prediction():
    assert classify("model.layers.47.mlp.down_proj.weight", layers=47) is None
    assert classify("model.layers.46.mlp.down_proj.weight", layers=47).layer == 46


def test_v_head_order_matches_the_converter_formula():
    nk, nv, width = 16, 32, 128
    r = nv // nk
    idx = v_head_order(nv * width, nk, nv)
    for g, j, d in ((0, 0, 0), (3, 1, 5), (15, 1, 127), (7, 0, 64)):
        hf = (g * r + j) * width + d
        gguf = (j * nk + g) * width + d
        assert idx[gguf] == hf


def test_lora_columns_are_reordered_for_linear_attention_only():
    a = np.arange(8.0)[None, :]
    config = {"linear_num_key_heads": 2, "linear_num_value_heads": 4}
    moved = reorder_for_gguf("model.layers.0.linear_attn.out_proj.weight", a, config)
    assert list(moved[0]) == [0, 1, 4, 5, 2, 3, 6, 7]
    same = reorder_for_gguf("model.layers.0.mlp.down_proj.weight", a, config)
    assert same is a


def test_second_copies_are_skipped():
    assert canonical_shards(["a.safetensors", "original/a.safetensors"]) == ["a.safetensors"]
    assert canonical_shards(["consolidated.safetensors", "model.safetensors"]) == ["model.safetensors"]
    assert canonical_shards(["consolidated.safetensors"]) == ["consolidated.safetensors"]


def test_moe_model_uses_its_shared_expert(model_dir, rng):
    w = bf16(rng.standard_normal((4, 4)))
    d = model_dir("m", {"model.layers.0.mlp.shared_expert.down_proj.weight": w},
                  {"config.json": json.dumps({"num_hidden_layers": 1})})
    ck = Checkpoint.open(str(d))
    assert down_proj_name(ck, 0).endswith("shared_expert.down_proj.weight")
    assert len(writers(ck)) == 1
