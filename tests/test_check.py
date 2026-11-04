import json
from fractions import Fraction

import numpy as np
import pytest
from conftest import bf16
from test_merge import lattice, nearest

from neyra.check import Adapter, check
from neyra.cli import main
from neyra.numerics import to_float
from neyra.patch import write_safetensors
from neyra.store import Checkpoint

TARGET = "model.layers.0.mlp.down_proj.weight"
KEY = "base_model.model.model.layers.0.mlp.down_proj"


def adapter_dir(root, a, b, alpha=8, **config):
    root.mkdir()
    write_safetensors(root / "adapter_model.safetensors",
                      {f"{KEY}.lora_A.weight": a.astype(np.float32), f"{KEY}.lora_B.weight": b.astype(np.float32)})
    (root / "adapter_config.json").write_text(json.dumps({"r": len(a), "lora_alpha": alpha, **config}))
    return root


def scenario(tmp_path, rng, shape=(16, 32), rank=4, alpha=8, scale=1.0):
    w = bf16(rng.standard_normal(shape) * 0.5)
    a, b = rng.standard_normal((rank, shape[1])) * scale, rng.standard_normal((shape[0], rank)) * scale
    return w, a, b, alpha / rank


def merged_exact(w, delta):
    """One rounding, by brute force over the exact sum."""
    table = lattice("BF16")
    f = to_float(w, "BF16").astype(np.float64)
    flat = [nearest(table, Fraction(float(x)) + Fraction(float(d))) for x, d in zip(f.reshape(-1), delta.reshape(-1))]
    return np.array(flat, dtype=np.uint16).reshape(w.shape)


def merged_peft(w, delta):
    """What PEFT leaves behind: the delta cast to the model dtype, then added and rounded again."""
    rounded = to_float(bf16(delta.astype(np.float32)), "BF16")
    return bf16(to_float(w, "BF16") + rounded)


def models(model_dir, w, merged, extra=None):
    other = {"norm": bf16(np.ones(4))}
    return (Checkpoint.open(str(model_dir("base", {TARGET: w, **other}))),
            Checkpoint.open(str(model_dir("merged", {TARGET: merged, **other, **(extra or {})}))))


def run(tmp_path, model_dir, w, merged, a, b, alpha=8, extra=None, **config):
    base, out = models(model_dir, w, merged, extra)
    return check(base, out, Checkpoint.open(str(adapter_dir(tmp_path / "adapter", a, b, alpha, **config))))


def test_the_two_roundings_really_differ(tmp_path, rng):
    w, a, b, s = scenario(tmp_path, rng, scale=0.1)
    delta = s * (b @ a)
    assert (merged_exact(w, delta) != merged_peft(w, delta)).any()


@pytest.mark.parametrize("name", ["single", "double"])
def test_either_correct_merge_is_equivalent(tmp_path, model_dir, rng, name):
    w, a, b, s = scenario(tmp_path, rng, scale=0.1)
    delta = s * (b @ a)
    merged = merged_exact(w, delta) if name == "single" else merged_peft(w, delta)
    c = run(tmp_path, model_dir, w, merged, a, b)
    status = {t.name: t for t in c.tensors}
    assert c.equivalent and status[TARGET].status == "rounding" and name in status[TARGET].detail
    assert status["norm"].status == "identical"
    assert status[TARGET].applied == pytest.approx(1.0, abs=0.01)


def test_a_merge_that_never_happened_is_flagged(tmp_path, model_dir, rng):
    w, a, b, _ = scenario(tmp_path, rng, scale=0.5)
    c = run(tmp_path, model_dir, w, w, a, b)
    assert not c.equivalent
    assert [t.status for t in c.tensors if t.name == TARGET] == ["never applied"]
    assert c.targeted == 1 and c.count("never applied") == 1


def test_a_delta_lost_to_rounding_is_told_apart(tmp_path, model_dir):
    # 2^-8 (1 + 2^-10) on a weight of 1 rounds up when added exactly, but the bf16 cast of the delta drops
    # the small part, and the sum then ties back to 1.
    w = bf16(np.ones((4, 4)))
    b = np.full((4, 1), 2.0**-8 * (1 + 2.0**-10))
    c = run(tmp_path, model_dir, w, w, np.ones((1, 4)), b, alpha=1)
    (hit,) = (t for t in c.tensors if t.name == TARGET)
    assert hit.status == "never applied" and "smaller than half a step" in hit.detail


def test_a_wrong_scale_shows_how_much_of_the_delta_is_there(tmp_path, model_dir, rng):
    w, a, b, s = scenario(tmp_path, rng, scale=0.5)
    merged = merged_peft(w, 2 * s * (b @ a))
    c = run(tmp_path, model_dir, w, merged, a, b)
    (hit,) = (t for t in c.tensors if t.name == TARGET)
    assert hit.status == "different" and hit.applied == pytest.approx(2.0, abs=0.02)


def test_leftover_adapter_tensors_break_equivalence(tmp_path, model_dir, rng):
    w, a, b, s = scenario(tmp_path, rng, scale=0.1)
    merged = merged_peft(w, s * (b @ a))
    c = run(tmp_path, model_dir, w, merged, a, b, extra={f"{KEY}.lora_A.weight": bf16(a)})
    assert not c.equivalent and c.leftover == [f"{KEY}.lora_A.weight"]


def test_transposed_adapters_are_handled(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((32, 16)) * 0.5)
    a, b = rng.standard_normal((4, 32)) * 0.1, rng.standard_normal((16, 4)) * 0.1
    merged = merged_peft(w, 2 * (b @ a).T)
    assert run(tmp_path, model_dir, w, merged, a, b, fan_in_fan_out=True).equivalent


def test_scaling_follows_the_config(tmp_path, rng):
    a, b = rng.standard_normal((4, 8)), rng.standard_normal((8, 4))
    config = {"lora_alpha": 16, "use_rslora": True, "alpha_pattern": {"down_proj": 32}}
    ck = Checkpoint.open(str(adapter_dir(tmp_path / "x", a, b, **config)))
    assert Adapter(ck).scaling(TARGET, 4) == 32 / 2
    assert Adapter(ck).scaling("model.layers.0.self_attn.q_proj.weight", 4) == 16 / 2


def test_two_checkpoints_without_an_adapter(tmp_path, model_dir, rng, capsys):
    w = bf16(rng.standard_normal((8, 8)))
    near = w.copy()
    near[0, 0] += 1
    far = bf16(to_float(w, "BF16") * 1.5)
    a = model_dir("a", {"w": w})
    for name, other, code in (("same", w, 0), ("near", near, 0), ("far", far, 1)):
        assert main(["check", str(a), str(model_dir(name, {"w": other}))]) == code
    assert main(["check", str(a), str(tmp_path / "near"), "--ulps", "0"]) == 1
    out = capsys.readouterr().out
    assert "equal under rounding" in out and "NOT equivalent" in out


def test_cli_reports_the_unapplied_adapter(tmp_path, model_dir, rng, capsys):
    w, a, b, _ = scenario(tmp_path, rng, scale=0.5)
    base, merged = models(model_dir, w, w)
    adapter = adapter_dir(tmp_path / "adapter", a, b)
    assert main(["check", base.spec, merged.spec, "--adapter", str(adapter)]) == 1
    assert "THE ADAPTER WAS NEVER APPLIED" in capsys.readouterr().out
    assert main(["check", base.spec, merged.spec, "--adapter", str(adapter), "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["equivalent"] is False and doc["counts"]["never applied"] == 1


def test_cli_accepts_a_correct_merge(tmp_path, model_dir, rng, capsys):
    w, a, b, s = scenario(tmp_path, rng, scale=0.1)
    base, merged = models(model_dir, w, merged_peft(w, s * (b @ a)))
    assert main(["check", base.spec, merged.spec, "--adapter", str(adapter_dir(tmp_path / "adapter", a, b))]) == 0
    assert "\nequivalent" in capsys.readouterr().out


def test_missing_tensor_and_missing_config_are_reported(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((4, 4)))
    a = Checkpoint.open(str(model_dir("a", {"w": w, "v": w})))
    b = Checkpoint.open(str(model_dir("b", {"w": w})))
    assert [t.status for t in check(a, b).tensors] == ["missing", "identical"]
    bare = tmp_path / "bare"
    bare.mkdir()
    write_safetensors(bare / "adapter_model.safetensors", {f"{KEY}.lora_A.weight": np.zeros((1, 4), np.float32),
                                                           f"{KEY}.lora_B.weight": np.zeros((4, 1), np.float32)})
    assert main(["check", a.spec, b.spec, "--adapter", str(bare)]) == 3


def test_an_adapter_that_rounds_away_entirely_is_called_out(tmp_path, model_dir):
    w = bf16(np.ones((16, 32)))
    a, b = np.full((4, 32), 1e-6), np.full((16, 4), 1e-6)
    result = run(tmp_path, model_dir, w, w, a, b)
    target = next(t for t in result.tensors if t.name == TARGET)
    assert target.status == "identical" and "rounds away" in target.detail


def test_the_warning_reaches_the_command_line(tmp_path, model_dir, capsys):
    w = bf16(np.ones((16, 32)))
    base, out = models(model_dir, w, w)
    adapter = adapter_dir(tmp_path / "adapter", np.full((4, 32), 1e-6), np.full((16, 4), 1e-6))
    main(["check", base.spec, out.spec, "--adapter", str(adapter)])
    assert "rounds away" in capsys.readouterr().out
