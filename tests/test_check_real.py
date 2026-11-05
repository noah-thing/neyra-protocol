"""check against merges made by the real tools: PEFT and mlx-lm fuse on a CPU, tiny model.

The versions are in tests/fixtures/real_merges/provenance.json. Nothing here recomputes a merge, so these
tests fail if check models a tool wrongly.
"""
import json
import shutil
from pathlib import Path

import pytest

from neyra.check import check
from neyra.patch import write_safetensors
from neyra.store import Checkpoint

FX = Path(__file__).parent / "fixtures" / "real_merges"
EVERY_PEFT = ["peft", "peft-safe", "peft-lowp"]

# merged directory, adapter directory, base dtype, the paths that reproduce it bit for bit
CASES = [
    ("merged_peft_F32", "adapter", "F32", EVERY_PEFT),
    ("merged_peft_BF16", "adapter", "BF16", ["peft"]),
    ("merged_peft_F16", "adapter", "F16", ["peft"]),
    ("merged_peft-safe_F32", "adapter", "F32", EVERY_PEFT),
    ("merged_peft-safe_BF16", "adapter", "BF16", ["peft-safe", "mlx-fuse"]),
    ("merged_peft-safe_F16", "adapter", "F16", ["peft-safe", "mlx-fuse"]),
    ("merged_peft-lowp_F32", "adapter", "F32", EVERY_PEFT),
    ("merged_peft-lowp_BF16", "adapter", "BF16", ["peft-lowp"]),
    ("merged_peft-lowp_F16", "adapter", "F16", ["peft-lowp"]),
    ("merged_mlx-fuse_F32_F32", "adapter_mlx_F32", "F32", ["mlx-fuse"]),
    ("merged_mlx-fuse_BF16_F32", "adapter_mlx_F32", "BF16", ["peft-safe", "mlx-fuse"]),
    ("merged_mlx-fuse_F16_F32", "adapter_mlx_F32", "F16", ["peft-safe", "mlx-fuse"]),
    ("merged_mlx-fuse_F32_BF16", "adapter_mlx_BF16", "F32", ["mlx-fuse"]),
    ("merged_mlx-fuse_BF16_BF16", "adapter_mlx_BF16", "BF16", ["mlx-fuse"]),
]


def run(merged, adapter, dtype, ulps=0):
    return check(Checkpoint.open(str(FX / f"base_{dtype}")), Checkpoint.open(str(merged)),
                 Checkpoint.open(str(FX / adapter)), ulps)


@pytest.mark.parametrize("merged, adapter, dtype, paths", CASES)
def test_a_real_merge_is_recognised_bit_for_bit_and_named(merged, adapter, dtype, paths):
    c = run(FX / merged, adapter, dtype)
    assert c.equivalent and c.targeted == 2 and c.paths == paths
    assert {t.status for t in c.tensors} == {"identical", "merged"}


def test_the_same_merges_are_all_told_apart_somewhere():
    """The BF16 fixtures leave different bits under each tool, or the table above would prove little."""
    bf16 = {merged: set(paths) for merged, _, dtype, paths in CASES if dtype == "BF16"}
    assert {"peft"} in bf16.values() and {"peft-lowp"} in bf16.values() and {"mlx-fuse"} in bf16.values()
    assert {"peft", "peft-safe", "mlx-fuse"} & set().union(*bf16.values()) == {"peft", "peft-safe", "mlx-fuse"}


def copy_with(tmp_path, source, swap):
    """A copy of a real merged model with some tensors put back to the base value."""
    base = Checkpoint.open(str(FX / "base_BF16"))
    out = Checkpoint.open(str(FX / source))
    tensors = {n: (base if n in swap else out).tensors[n].read() for n in out.tensors}
    dest = tmp_path / "planted"
    dest.mkdir()
    write_safetensors(dest / "model.safetensors", tensors, {"format": "pt"}, "BF16")
    return dest


def test_a_real_merge_that_was_never_applied_is_flagged(tmp_path):
    c = run(FX / "base_BF16", "adapter", "BF16")
    assert not c.equivalent and c.count("never applied") == 2


def test_one_real_layer_put_back_to_the_base_is_flagged(tmp_path):
    skipped = "model.layers.0.self_attn.q_proj.weight"
    c = run(copy_with(tmp_path, "merged_peft-safe_BF16", {skipped}), "adapter", "BF16")
    assert not c.equivalent
    assert {t.name: t.status for t in c.tensors}[skipped] == "never applied" and c.count("merged") == 1


@pytest.mark.parametrize("factor", [2, 0.5])
def test_a_real_merge_with_the_wrong_scale_is_flagged(tmp_path, factor):
    root = tmp_path / "adapter"
    shutil.copytree(FX / "adapter", root)
    cfg = json.loads((root / "adapter_config.json").read_text())
    cfg["lora_alpha"] *= factor
    (root / "adapter_config.json").write_text(json.dumps(cfg))
    c = check(Checkpoint.open(str(FX / "base_BF16")), Checkpoint.open(str(FX / "merged_peft-safe_BF16")),
              Checkpoint.open(str(root)), 1)
    assert not c.equivalent and c.count("different") == 2
    assert all(t.applied == pytest.approx(1 / factor, abs=0.05) for t in c.tensors if t.applied is not None)


def test_a_merge_by_another_tool_is_not_taken_for_this_one():
    name = "model.layers.0.mlp.down_proj.weight"
    one, other = (Checkpoint.open(str(FX / d)).tensors[name].read() for d in ("merged_peft_BF16", "merged_peft-safe_BF16"))
    assert (one != other).any()
    assert run(FX / "merged_peft_BF16", "adapter", "BF16").paths == ["peft"]
