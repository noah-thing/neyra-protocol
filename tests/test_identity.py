import json
import shutil

import numpy as np
import pytest
from conftest import bf16

from neyra import identity
from neyra.check import name_base
from neyra.cli import main
from neyra.identity import (
    BIT_IDENTICAL,
    DERIVED_NAMED,
    DERIVED_UNKNOWN,
    INCONCLUSIVE,
    NO_DIFFERENCE,
    NOT_COMPARABLE,
    SAME_WEIGHTS,
    UNRELATED,
    identify,
    measure,
)
from neyra.numerics import from_float, to_float
from neyra.patch import write_safetensors
from neyra.store import Checkpoint

FX = "tests/fixtures/real_merges"


@pytest.fixture(autouse=True)
def null(monkeypatch):
    monkeypatch.setattr(identity, "calibration", lambda: {"unrelated_max": 0.1, "derived_min": 0.6})


def weights(rng, shapes=((24, 40), (40, 24), (16, 16)), scale=0.5):
    return {f"layers.{i}.w": rng.standard_normal(s).astype(np.float32) * scale for i, s in enumerate(shapes)} | {
        "norm.weight": np.ones(24, dtype=np.float32)}


def save(root, tensors, dtype="F32", shards=1):
    root.mkdir(parents=True)
    names = sorted(tensors)
    for i in range(shards):
        part = {n: tensors[n] for n in names[i::shards]}
        name = "model.safetensors" if shards == 1 else f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        write_safetensors(root / name, {n: from_float(v, dtype) if dtype != "F32" else v for n, v in part.items()},
                          {"format": "pt"}, dtype if dtype != "F32" else None)
    (root / "config.json").write_text(json.dumps({"hidden_size": 24}))
    return Checkpoint.open(str(root))


def test_the_same_files_are_bit_identical(tmp_path, rng):
    w = weights(rng)
    a = save(tmp_path / "a", w)
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    assert identify(a, Checkpoint.open(str(tmp_path / "b"))).label == BIT_IDENTICAL


def test_the_same_tensors_in_other_files_are_the_same_weights_and_not_bit_identical(tmp_path, rng):
    w = weights(rng)
    v = identify(save(tmp_path / "a", w), save(tmp_path / "b", w, shards=2), full=True)
    assert v.label == SAME_WEIGHTS and v.exact


def test_a_sample_never_claims_the_same_weights(tmp_path, rng):
    w = weights(rng)
    v = identify(save(tmp_path / "a", w), save(tmp_path / "b", w, shards=2))
    assert v.label == NO_DIFFERENCE and not v.exact and "not proof" in v.detail


def test_an_edit_outside_the_sampled_span_is_missed_by_a_sample_and_caught_by_full(tmp_path, rng, monkeypatch):
    monkeypatch.setattr(identity, "SPAN", 64)
    w = {"a.weight": rng.standard_normal((100, 100)).astype(np.float32)}
    base = save(tmp_path / "a", w)
    first, n = identity._span(base.tensors["a.weight"], "a.weight", 64)
    cell = (first + n + 500) % 10000
    edited = w["a.weight"].copy().reshape(-1)
    edited[cell] += 1.0
    other = save(tmp_path / "b", {"a.weight": edited.reshape(100, 100)})
    assert identify(base, other).label == NO_DIFFERENCE
    full = identify(base, other, full=True)
    assert full.label != SAME_WEIGHTS and full.label == DERIVED_UNKNOWN


def test_a_dropped_tied_head_still_counts_as_the_same_weights(tmp_path, rng):
    w = weights(rng) | {"m.embed_tokens.weight": rng.standard_normal((24, 40)).astype(np.float32)}
    w["lm_head.weight"] = w["m.embed_tokens.weight"]
    without = {n: v for n, v in w.items() if n != "lm_head.weight"}
    v = identify(save(tmp_path / "a", w), save(tmp_path / "b", without), full=True)
    assert v.label == SAME_WEIGHTS


def test_a_missing_ordinary_tensor_is_not_the_same_weights(tmp_path, rng):
    w = weights(rng)
    without = {n: v for n, v in w.items() if n != "layers.1.w"}
    assert identify(save(tmp_path / "a", w), save(tmp_path / "b", without), full=True).label != SAME_WEIGHTS


def test_a_cast_is_a_named_transform(tmp_path, rng):
    w = weights(rng)
    v = identify(save(tmp_path / "a", w), save(tmp_path / "b", w, "BF16"))
    assert v.label == DERIVED_NAMED and v.paths == ["cast"]


def test_a_cast_to_the_wrong_rounding_is_not_named(tmp_path, rng):
    w = weights(rng)
    off = {n: to_float(from_float(v, "BF16"), "BF16") * np.float32(1.05) for n, v in w.items()}
    v = identify(save(tmp_path / "a", w), save(tmp_path / "b", off, "BF16"))
    assert v.label == DERIVED_UNKNOWN


def test_a_small_training_step_is_derived_with_an_unknown_transform(tmp_path, rng):
    w = weights(rng)
    tuned = {n: v + rng.standard_normal(v.shape).astype(np.float32) * 0.05 * (v.ndim > 1) for n, v in w.items()}
    v = identify(save(tmp_path / "a", w), save(tmp_path / "b", tuned))
    assert v.label == DERIVED_UNKNOWN and v.measure.cos > 0.9


def test_an_independent_model_of_the_same_shape_is_unrelated(tmp_path, rng):
    v = identify(save(tmp_path / "a", weights(rng)), save(tmp_path / "b", weights(rng)))
    assert v.label == UNRELATED and abs(v.measure.cos) < 0.1


def test_a_middle_value_is_inconclusive_not_forced(tmp_path, rng):
    w = weights(rng)
    mixed = {n: v + rng.standard_normal(v.shape).astype(np.float32) * 1.25 * (v.ndim > 1) for n, v in w.items()}
    assert identify(save(tmp_path / "a", w), save(tmp_path / "b", mixed)).label == INCONCLUSIVE


def test_another_architecture_is_not_comparable(tmp_path, rng):
    a = save(tmp_path / "a", weights(rng))
    b = save(tmp_path / "b", weights(rng, shapes=((30, 50), (50, 30), (8, 8))) | {"norm.weight": np.ones(30, np.float32)})
    assert identify(a, b).label == NOT_COMPARABLE


def test_sampling_depends_on_names_only(tmp_path, rng):
    w = weights(rng)
    a, b = save(tmp_path / "a", w), save(tmp_path / "b", w, shards=3)
    assert [p.name for p in measure(a, b).probes] == [p.name for p in measure(b, a).probes]


def test_a_lora_merge_is_named_by_the_path_that_reproduces_it():
    base = Checkpoint.open(f"{FX}/base_BF16")
    v = identify(base, Checkpoint.open(f"{FX}/merged_peft_BF16"), adapter=Checkpoint.open(f"{FX}/adapter"))
    assert v.label == DERIVED_NAMED and v.paths == ["peft"]


def test_the_wrong_adapter_does_not_name_the_merge(tmp_path):
    base = Checkpoint.open(f"{FX}/base_BF16")
    v = identify(base, Checkpoint.open(f"{FX}/merged_peft-safe_BF16"), adapter=Checkpoint.open(f"{FX}/adapter_mlx_BF16"))
    assert not (v.label == DERIVED_NAMED and "mlx-fuse" not in v.paths)
    assert v.label in (DERIVED_UNKNOWN, DERIVED_NAMED)


def test_name_base_picks_the_base_the_merge_was_made_from(tmp_path):
    right = Checkpoint.open(f"{FX}/base_BF16")
    wrong = tmp_path / "wrong"
    shutil.copytree(f"{FX}/base_BF16", wrong)
    f = next(wrong.glob("*.safetensors"))
    t = Checkpoint.open(str(wrong)).tensors
    edited = {n: r.read() for n, r in t.items()}
    for n, bits in edited.items():
        if n.endswith("down_proj.weight"):
            edited[n] = bf16(to_float(bits, "BF16") * np.float32(1.02))
    write_safetensors(f, edited, {"format": "pt"}, "BF16")
    ranked = name_base(Checkpoint.open(f"{FX}/merged_peft_BF16"), Checkpoint.open(f"{FX}/adapter"),
                       [Checkpoint.open(str(wrong)), right])
    assert ranked[0].spec == right.spec and ranked[0].reproduces and not ranked[1].reproduces


def test_name_base_says_when_candidates_hold_the_same_weights(tmp_path):
    shutil.copytree(f"{FX}/base_BF16", tmp_path / "mirror")
    ranked = name_base(Checkpoint.open(f"{FX}/merged_peft_BF16"), Checkpoint.open(f"{FX}/adapter"),
                       [Checkpoint.open(f"{FX}/base_BF16"), Checkpoint.open(str(tmp_path / "mirror"))])
    assert all(c.reproduces and c.shares for c in ranked)


def test_the_command_prints_the_verdict(tmp_path, rng, capsys):
    w = weights(rng)
    save(tmp_path / "a", w)
    save(tmp_path / "b", w, "BF16")
    assert main(["identity", str(tmp_path / "a"), str(tmp_path / "b"), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == DERIVED_NAMED and doc["paths"] == ["cast"]


def test_check_with_candidates_names_the_base(capsys):
    code = main(["check", f"{FX}/base_BF16", f"{FX}/merged_peft_BF16", "--adapter", f"{FX}/adapter",
                 "--also", f"{FX}/base_F32"])
    out = capsys.readouterr().out
    assert code == 0 and "made from" in out or "fit equally" in out


def test_a_fused_projection_is_compared_with_the_split_ones(tmp_path, rng):
    q, k, v = (rng.standard_normal((8, 16)).astype(np.float32) for _ in range(3))
    split = {"l.self_attn.q_proj.weight": q, "l.self_attn.k_proj.weight": k, "l.self_attn.v_proj.weight": v}
    fused = {"l.self_attn.qkv_proj.weight": np.concatenate([q, k, v])}
    a, b = save(tmp_path / "a", split), save(tmp_path / "b", fused)
    assert identify(a, b, full=True).label == SAME_WEIGHTS
    swapped = {"l.self_attn.qkv_proj.weight": np.concatenate([k, q, v])}
    assert identify(a, save(tmp_path / "c", swapped), full=True).label != SAME_WEIGHTS


def test_a_head_left_out_must_equal_the_embedding_to_count_as_tied(tmp_path, rng):
    w = {"m.embed_tokens.weight": rng.standard_normal((32, 8)).astype(np.float32)}
    head = {"lm_head.weight": w["m.embed_tokens.weight"]}
    assert identify(save(tmp_path / "a", w | head), save(tmp_path / "b", w), full=True).label == SAME_WEIGHTS
    other = {"lm_head.weight": rng.standard_normal((32, 8)).astype(np.float32)}
    assert identify(save(tmp_path / "c", w | other), save(tmp_path / "d", w), full=True).label != SAME_WEIGHTS


def test_an_edit_inside_the_sampled_span_is_seen_wherever_the_span_sits(tmp_path, rng, monkeypatch):
    monkeypatch.setattr(identity, "SPAN", 64)
    w = {"a.weight": rng.standard_normal((100, 100)).astype(np.float32)}
    base = save(tmp_path / "a", w)
    first, n = identity._span(base.tensors["a.weight"], "a.weight", 64)
    assert first > 0
    edited = w["a.weight"].copy().reshape(-1)
    edited[first + n // 2] += 1.0
    assert identify(base, save(tmp_path / "b", {"a.weight": edited.reshape(100, 100)})).label != NO_DIFFERENCE


def test_without_a_calibration_panel_it_says_so(tmp_path, rng, monkeypatch):
    from neyra import identity as ident

    monkeypatch.setattr(ident, "calibration", lambda: None)
    a = {f"layers.{i}.w": rng.standard_normal((64, 64)).astype(np.float32) for i in range(4)}
    b = {k: v + 0.5 * rng.standard_normal(v.shape).astype(np.float32) for k, v in a.items()}
    save(tmp_path / "a", a)
    save(tmp_path / "b", b)
    v = ident.identify(Checkpoint.open(str(tmp_path / "a")), Checkpoint.open(str(tmp_path / "b")))
    assert v.label == ident.INCONCLUSIVE and "calibration" in v.detail

