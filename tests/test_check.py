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


def adapter_dir(root, a, b, alpha=8, bf=False, **config):
    root.mkdir()
    store = (lambda x: bf16(x)) if bf else (lambda x: x.astype(np.float32))
    write_safetensors(root / "adapter_model.safetensors",
                      {f"{KEY}.lora_A.weight": store(a), f"{KEY}.lora_B.weight": store(b)}, bits_as="BF16" if bf else None)
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


def delta32(a, b, scale):
    return (b.astype(np.float32) @ a.astype(np.float32)) * np.float32(scale)


def merged_peft(w, d32):
    """merge_and_unload(): `weight += delta` with a float32 delta adds in float32 and rounds once."""
    return bf16(to_float(w, "BF16") + d32)


def merged_peft_safe(w, d32):
    """safe_merge=True: the delta is cast to the model dtype first, then added."""
    return bf16(to_float(w, "BF16") + to_float(bf16(d32), "BF16"))


def merged_mlx(w, a, b, scale):
    """mlx_lm.fuse with a bfloat16 adapter: the scale, rounded to bfloat16, goes onto B, and the product is rounded too."""
    scaled = to_float(bf16(to_float(bf16([scale]), "BF16")[0] * b.astype(np.float32)), "BF16")
    return bf16(to_float(w, "BF16") + to_float(bf16(scaled @ a.astype(np.float32)), "BF16"))


def models(model_dir, w, merged, extra=None):
    other = {"norm": bf16(np.ones(4))}
    return (Checkpoint.open(str(model_dir("base", {TARGET: w, **other}))),
            Checkpoint.open(str(model_dir("merged", {TARGET: merged, **other, **(extra or {})}))))


def run(tmp_path, model_dir, w, merged, a, b, alpha=8, extra=None, **config):
    base, out = models(model_dir, w, merged, extra)
    return check(base, out, Checkpoint.open(str(adapter_dir(tmp_path / "adapter", a, b, alpha, **config))))


def paths_scenario(rng):
    """Odd scale, bfloat16 adapter values, deltas near the weight's rounding step: every path leaves its own bits."""
    w = bf16(rng.standard_normal((24, 48)) * 0.5)
    a, b = to_float(bf16(rng.standard_normal((6, 48))), "BF16"), to_float(bf16(rng.standard_normal((24, 6)) * 0.05), "BF16")
    return w, a, b, 16 / 6


def test_the_paths_leave_different_bits(rng):
    w, a, b, s = paths_scenario(rng)
    outs = [merged_peft(w, delta32(a, b, s)), merged_peft_safe(w, delta32(a, b, s)), merged_mlx(w, a, b, s)]
    assert all((x != y).any() for i, x in enumerate(outs) for y in outs[i + 1:])


@pytest.mark.parametrize("name", ["peft", "peft-safe", "mlx-fuse"])
def test_each_path_is_recognised_and_named(tmp_path, model_dir, rng, name):
    w, a, b, s = paths_scenario(rng)
    d = delta32(a, b, s)
    merged = {"peft": merged_peft(w, d), "peft-safe": merged_peft_safe(w, d), "mlx-fuse": merged_mlx(w, a, b, s)}[name]
    c = run(tmp_path, model_dir, w, merged, a, b, alpha=16, bf=True)
    (hit,) = (t for t in c.tensors if t.name == TARGET)
    assert c.equivalent and hit.status == "merged" and name in hit.paths
    assert [p for p in ("peft", "peft-safe", "mlx-fuse") if p in c.paths] == [name]


def test_the_exact_sum_is_equivalent_but_belongs_to_no_tool(tmp_path, model_dir):
    # 1 + 2^-8 + 2^-30 is above the bf16 tie, but float32 cannot hold the 2^-30, so every tool ties back to 1.
    w = bf16(np.ones((4, 4)))
    b = np.zeros((4, 1))
    b[0, 0] = 2.0**-8 + 2.0**-30
    delta = b @ np.ones((1, 4))
    c = run(tmp_path, model_dir, w, merged_exact(w, delta), np.ones((1, 4)), b, alpha=1)
    (hit,) = (t for t in c.tensors if t.name == TARGET)
    assert c.equivalent and hit.status == "rounding" and not c.paths and "exact sum" in hit.detail


def test_a_path_that_fits_only_some_tensors_is_not_reported(tmp_path, model_dir, rng):
    w, a, b, s = paths_scenario(rng)
    other = bf16(rng.standard_normal((24, 48)) * 0.5)
    d = delta32(a, b, s)
    base = Checkpoint.open(str(model_dir("base", {TARGET: w, "model.layers.1.mlp.down_proj.weight": other})))
    out = Checkpoint.open(str(model_dir("merged", {TARGET: merged_peft(w, d),
                                                   "model.layers.1.mlp.down_proj.weight": merged_peft_safe(other, d)})))
    root = tmp_path / "ad"
    root.mkdir()
    write_safetensors(root / "adapter_model.safetensors", {
        f"{KEY}.lora_A.weight": bf16(a), f"{KEY}.lora_B.weight": bf16(b),
        f"{KEY.replace('layers.0', 'layers.1')}.lora_A.weight": bf16(a),
        f"{KEY.replace('layers.0', 'layers.1')}.lora_B.weight": bf16(b)}, bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"r": 6, "lora_alpha": 16}))
    c = check(base, out, Checkpoint.open(str(root)))
    assert c.equivalent and c.paths == []


def test_one_stray_tensor_stops_a_path_being_claimed(tmp_path, model_dir, rng):
    w, a, b, s = paths_scenario(rng)
    other = bf16(rng.standard_normal((24, 48)) * 0.5)
    d = delta32(a, b, s)
    wrong = merged_peft(other, d)
    wrong[0, 0] += 1
    wrong[0, 1] += 400
    second = "model.layers.1.mlp.down_proj.weight"
    base = Checkpoint.open(str(model_dir("base", {TARGET: w, second: other})))
    out = Checkpoint.open(str(model_dir("merged", {TARGET: merged_peft(w, d), second: wrong})))
    root = tmp_path / "ad"
    root.mkdir()
    write_safetensors(root / "adapter_model.safetensors", {
        f"{KEY}.lora_A.weight": bf16(a), f"{KEY}.lora_B.weight": bf16(b),
        f"{KEY.replace('layers.0', 'layers.1')}.lora_A.weight": bf16(a),
        f"{KEY.replace('layers.0', 'layers.1')}.lora_B.weight": bf16(b)}, bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"r": 6, "lora_alpha": 16}))
    c = check(base, out, Checkpoint.open(str(root)))
    assert c.count("merged") == 1 and c.count("different") == 1 and c.paths == []


def test_an_edited_tensor_gone_from_the_merged_file_stops_a_path_being_claimed(tmp_path, model_dir, rng):
    w, a, b, s = paths_scenario(rng)
    d = delta32(a, b, s)
    second = "model.layers.1.mlp.down_proj.weight"
    base = Checkpoint.open(str(model_dir("base", {TARGET: w, second: w})))
    out = Checkpoint.open(str(model_dir("merged", {TARGET: merged_peft(w, d)})))
    root = tmp_path / "ad"
    root.mkdir()
    write_safetensors(root / "adapter_model.safetensors", {
        f"{KEY}.lora_A.weight": bf16(a), f"{KEY}.lora_B.weight": bf16(b),
        f"{KEY.replace('layers.0', 'layers.1')}.lora_A.weight": bf16(a),
        f"{KEY.replace('layers.0', 'layers.1')}.lora_B.weight": bf16(b)}, bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"r": 6, "lora_alpha": 16}))
    c = check(base, out, Checkpoint.open(str(root)))
    assert c.targeted == 2 and c.paths == [] and not c.equivalent


def test_a_tied_head_takes_its_delta_on_the_embedding(tmp_path, model_dir, rng):
    w, a, b, s = paths_scenario(rng)
    head = "base_model.model.lm_head"
    root = tmp_path / "ad"
    root.mkdir()
    write_safetensors(root / "adapter_model.safetensors", {f"{head}.lora_A.weight": bf16(a), f"{head}.lora_B.weight": bf16(b)},
                      bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"r": 6, "lora_alpha": 16}))
    embed = "model.embed_tokens.weight"
    base = Checkpoint.open(str(model_dir("base", {embed: w})))
    good = Checkpoint.open(str(model_dir("good", {embed: merged_peft(w, delta32(a, b, s))})))
    skipped = Checkpoint.open(str(model_dir("skipped", {embed: w})))
    assert check(base, good, Checkpoint.open(str(root))).paths == ["peft"]
    assert check(base, skipped, Checkpoint.open(str(root))).count("never applied") == 1


def test_an_adapter_kept_in_the_model_dtype_is_its_own_path(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((24, 48)) * 0.5)
    a, b = rng.standard_normal((6, 48)).astype(np.float32), (rng.standard_normal((24, 6)) * 0.05).astype(np.float32)
    s = 16 / 6
    narrow = lambda x: to_float(bf16(x), "BF16")
    merged = merged_peft_safe(w, delta32(narrow(a), narrow(b), s))
    assert (merged != merged_peft_safe(w, delta32(a, b, s))).any()
    c = run(tmp_path, model_dir, w, merged, a, b, alpha=16)
    assert c.paths == ["peft-lowp"]


def test_mlx_rounds_the_product_to_the_adapter_dtype_even_on_a_float32_model(tmp_path, rng):
    w, a, b, s = paths_scenario(rng)
    w32 = to_float(w, "BF16")
    scaled = to_float(bf16(to_float(bf16([s]), "BF16")[0] * b), "BF16")
    product = to_float(bf16(scaled @ a), "BF16")
    wide = w32 + (scaled @ a)
    assert (w32 + product != wide).any()
    key = "model.layers.0.mlp.down_proj"
    root = tmp_path / "mlx"
    root.mkdir()
    write_safetensors(root / "adapters.safetensors", {f"{key}.lora_a": bf16(a.T.copy()), f"{key}.lora_b": bf16(b.T.copy())},
                      bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 6, "scale": s}}))
    for name, merged in (("base", w32), ("merged", w32 + product)):
        (tmp_path / name).mkdir()
        write_safetensors(tmp_path / name / "model.safetensors", {TARGET: merged.astype(np.float32)})
    c = check(Checkpoint.open(str(tmp_path / "base")), Checkpoint.open(str(tmp_path / "merged")), Checkpoint.open(str(root)), 0)
    assert c.paths == ["mlx-fuse"]


def test_a_model_merged_in_another_dtype_is_compared_there(tmp_path, model_dir, rng, capsys):
    w, a, b, s = paths_scenario(rng)
    wide = to_float(w, "BF16") + delta32(a, b, s)
    base = model_dir("base", {TARGET: w})
    merged = tmp_path / "merged"
    merged.mkdir()
    write_safetensors(merged / "model.safetensors", {TARGET: wide.astype(np.float32)})
    adapter = adapter_dir(tmp_path / "adapter", a, b, 16, bf=True)
    c = check(Checkpoint.open(str(base)), Checkpoint.open(str(merged)), Checkpoint.open(str(adapter)), 0)
    assert c.equivalent and "peft" in c.paths and c.recast == {("BF16", "F32")}
    assert main(["check", str(base), str(merged), "--adapter", str(adapter)]) == 0
    assert "stored in F32 where the base is BF16" in capsys.readouterr().out


def test_a_tensor_that_changed_kind_is_different(tmp_path, model_dir):
    w = bf16(np.ones((2, 2)))
    base = model_dir("base", {"w": w})
    ints = tmp_path / "ints"
    ints.mkdir()
    write_safetensors(ints / "model.safetensors", {"w": np.ones((2, 2), np.int32)})
    (hit,) = check(Checkpoint.open(str(base)), Checkpoint.open(str(ints))).tensors
    assert hit.status == "different" and "where the base has BF16" in hit.detail


def test_a_head_left_out_of_a_tied_merged_file_is_read_from_the_embedding(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((8, 4)))
    names = {"model.embed_tokens.weight": w, "lm_head.weight": w}
    base = model_dir("base", names)
    tied = model_dir("tied", {"model.embed_tokens.weight": w}, {"config.json": json.dumps({"tie_word_embeddings": True})})
    loose = model_dir("loose", {"model.embed_tokens.weight": w})
    assert check(Checkpoint.open(str(base)), Checkpoint.open(str(tied))).equivalent
    assert check(Checkpoint.open(str(base)), Checkpoint.open(str(loose))).count("missing") == 1


def test_a_merged_file_saved_through_a_wrapper_class_is_matched_by_tail(tmp_path, model_dir, rng):
    w = bf16(rng.standard_normal((8, 4)))
    base = Checkpoint.open(str(model_dir("base", {"h.0.attn.c_proj.weight": w})))
    out = Checkpoint.open(str(model_dir("merged", {"transformer.h.0.attn.c_proj.weight": w})))
    c = check(base, out)
    assert c.equivalent and c.added == []


def test_a_wrapper_prefix_missing_from_the_base_is_matched_by_its_tail(tmp_path, model_dir, rng):
    w, a, b, s = paths_scenario(rng)
    name = "h.0.mlp.c_proj.weight"
    root = tmp_path / "ad"
    root.mkdir()
    key = "base_model.model.transformer.h.0.mlp.c_proj"
    write_safetensors(root / "adapter_model.safetensors", {f"{key}.lora_A.weight": bf16(a), f"{key}.lora_B.weight": bf16(b)},
                      bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"r": 6, "lora_alpha": 16}))
    base = Checkpoint.open(str(model_dir("base", {name: w})))
    out = Checkpoint.open(str(model_dir("merged", {name: merged_peft(w, delta32(a, b, s))})))
    assert check(base, out, Checkpoint.open(str(root))).paths == ["peft"]


def test_an_mlx_adapter_is_read_with_its_own_layout(tmp_path, model_dir, rng):
    w, a, b, s = paths_scenario(rng)
    key = "model.layers.0.mlp.down_proj"
    root = tmp_path / "mlx"
    root.mkdir()
    write_safetensors(root / "adapters.safetensors", {f"{key}.lora_a": bf16(a.T.copy()), f"{key}.lora_b": bf16(b.T.copy())},
                      bits_as="BF16")
    (root / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 6, "scale": s}}))
    base, out = models(model_dir, w, merged_mlx(w, a, b, s))
    assert check(base, out, Checkpoint.open(str(root))).paths == ["mlx-fuse"]


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
    assert hit.status == "never applied" and "rounds away under peft-safe" in hit.detail


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
    out = capsys.readouterr().out
    assert "\nequivalent" in out and "what peft produces" in out


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


def test_a_correct_merge_that_lost_most_of_the_delta_passes_with_a_warning(tmp_path, model_dir, capsys):
    w = bf16(np.ones((16, 32)))
    a = np.ones((1, 32))
    b = np.full((16, 1), 0.003)
    b[15] = 0.006
    base, out = models(model_dir, w, merged_peft(w, b @ a))
    assert main(["check", base.spec, out.spec, "--adapter", str(adapter_dir(tmp_path / "ad", a, b, alpha=1))]) == 0
    text = capsys.readouterr().out
    assert "rounding kept under half of the delta in 1 of 1" in text and "\nequivalent" in text
