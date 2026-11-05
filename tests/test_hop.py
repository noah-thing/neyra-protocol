import json

import hop_cases as cases
import hop_suite
import numpy as np
import pytest
from conftest import bf16

from neyra import gguf_io
from neyra.cli import main
from neyra.hop import document, trace
from neyra.numerics import to_float
from neyra.store import FormatError
from neyra.survival import Pool, predict

pytest.importorskip("gguf")
EVERY = cases.CASES + cases.CONTROLS


@pytest.mark.parametrize("case", EVERY, ids=lambda c: c.name)
def test_planted_failure_is_named(case, tmp_path):
    paths = case.build(tmp_path, np.random.default_rng(3))
    ok, sentence = hop_suite.hop_result(case, paths)
    assert ok, sentence


def test_the_suite_scores_every_case():
    tally = hop_suite.score(show=False)
    assert tally == {"kill": [14, 14], "gguf": [7, 7], "controls": [len(cases.CONTROLS)] * 2}


def test_a_control_is_not_flagged_because_a_tool_flags_everything(tmp_path):
    """The kill cases and the controls share fixtures, so a tool that always reports a problem would fail here."""
    clean = cases.clean(tmp_path, np.random.default_rng(3))
    assert trace([str(p) for p in clean]).ok


def test_the_first_hop_that_differs_wins(tmp_path):
    """A bad merge followed by a faithful GGUF of the bad merge is reported at the merge."""
    w = cases.World(np.random.default_rng(3))
    paths = cases.standard(tmp_path, w, merged=dict(w.base),
                           gguf=lambda t, m, f: cases.write_gguf(t / "m.gguf", m, f, "Q8_0"))
    doc = document(trace([str(p) for p in paths]))
    assert doc["first"]["role"] == "merged" and doc["first"]["cause"] == "never_applied"
    assert [h["role"] for h in doc["hops"]] == ["adapter", "merged", "gguf"]


def test_text_problem_at_the_gguf_is_not_blamed_on_the_merge(tmp_path):
    paths = cases.gguf_template_gone(tmp_path, np.random.default_rng(3))
    doc = document(trace([str(p) for p in paths]))
    assert doc["first"]["role"] == "gguf"
    assert doc["hops"][1]["parts"]["template"] == "same"


def test_survival_in_the_gguf_is_measured_against_the_trained_change(tmp_path):
    paths = cases.gguf_clean("Q8_0")(tmp_path, np.random.default_rng(3))
    doc = document(trace([str(p) for p in paths]))
    gguf = doc["hops"][2]["survival"]
    assert gguf["measured_on"] == "dequantized values"
    assert gguf["value"] == pytest.approx(1.0, abs=0.05)
    base = cases.gguf_from_base(tmp_path / "b", np.random.default_rng(3)) if (tmp_path / "b").mkdir() is None else None
    assert document(trace([str(p) for p in base]))["hops"][2]["survival"]["value"] == pytest.approx(0.0, abs=1e-6)


def test_two_artifacts_without_an_adapter(tmp_path):
    w = cases.World(np.random.default_rng(3))
    base, _, merged = cases.standard(tmp_path, w)
    assert trace([str(base), str(merged)]).ok
    same = cases.standard(tmp_path / "x", w, merged=dict(w.base)) if (tmp_path / "x").mkdir() is None else None
    verdict = trace([str(same[0]), str(same[2])])
    assert verdict.first.cause == "never_applied"


def test_artifacts_must_be_in_hop_order(tmp_path):
    base, adapter, merged = cases.clean(tmp_path, np.random.default_rng(3))
    with pytest.raises(FormatError, match="at least two"):
        trace([str(base)])
    with pytest.raises(FormatError, match="adapter"):
        trace([str(adapter), str(merged)])
    with pytest.raises(FormatError, match="comes second"):
        trace([str(base), str(merged), str(adapter)])


def test_cli_prints_one_sentence_first_and_exits_1_on_a_problem(tmp_path, capsys):
    paths = cases.merge_wrote_base(tmp_path, np.random.default_rng(3))
    assert main(["hop", *map(str, paths)]) == 1
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("The first difference is at the merged model: the adapter was never applied")
    assert "\n" not in first


def test_a_clean_run_says_the_hops_are_clear_and_not_that_the_model_is_right(tmp_path):
    paths = cases.clean(tmp_path, np.random.default_rng(3))
    verdict = trace([str(p) for p in paths]).sentence
    assert "are clear" in verdict and "the cause is not in these files" in verdict


def test_cli_json_and_exit_code_for_a_clean_run(tmp_path, capsys):
    paths = cases.gguf_clean("Q8_0")(tmp_path, np.random.default_rng(3))
    assert main(["hop", "--json", *map(str, paths)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["ok"] and doc["first"] is None and doc["verdict"].startswith("These 3 hops are clear")


# GGUF reading

def gguf_of(tmp_path, kind, bits=None, **kw):
    rng = np.random.default_rng(5)
    w = cases.World(rng)
    tensors = w.merged() if bits is None else bits
    return gguf_io.GGUFFile(cases.write_gguf(tmp_path / f"{kind}.gguf", tensors, cases.text_files(), kind, **kw)), w


@pytest.mark.parametrize("kind", ["F16", "BF16", "Q8_0", "Q4_0"])
def test_every_tensor_of_a_gguf_is_rebuilt_exactly_from_its_source(tmp_path, kind):
    g, w = gguf_of(tmp_path, kind)
    merged = w.merged()
    look = g.lookup()
    checked = 0
    for name, bits in merged.items():
        t = g.tensors[look(name)]
        values = gguf_io.to_gguf_layout(name, to_float(bits, "BF16"), g.arch, {"num_attention_heads": cases.HEADS, "num_key_value_heads": cases.KV_HEADS})
        if t.kind == kind:
            assert gguf_io.rebuild(values, t.kind, t.raw, t.qtype) == ("direct", 1.0), name
            checked += 1
    assert checked == 16


def test_a_changed_byte_is_not_an_exact_match(tmp_path):
    g, w = gguf_of(tmp_path, "Q8_0")
    name = "model.layers.0.mlp.down_proj.weight"
    t = g.tensors[g.lookup()(name)]
    values = to_float(w.merged()[name], "BF16")
    raw = np.array(t.raw, copy=True)
    raw[0, 4] ^= 1
    _route, share = gguf_io.rebuild(values, t.kind, raw, t.qtype)
    assert share < 1.0
    assert gguf_io.matching_blocks(gguf_io.encode(values, "Q8_0"), t.raw, "Q8_0", t.qtype) == 1.0


def test_q_and_k_are_unpermuted_the_way_llama_cpp_stores_them():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((32, 32)).astype(np.float32)
    config = {"num_attention_heads": 4, "num_key_value_heads": 2}
    assert np.array_equal(gguf_io.to_gguf_layout("model.layers.0.self_attn.q_proj.weight", q, "llama", config),
                          cases.permute(q, 4))
    k = rng.standard_normal((16, 32)).astype(np.float32)
    assert np.array_equal(gguf_io.to_gguf_layout("model.layers.0.self_attn.k_proj.weight", k, "llama", config),
                          cases.permute(k, 2))
    assert np.array_equal(gguf_io.to_gguf_layout("model.layers.0.self_attn.q_proj.weight", q, "qwen3", config), q)


def test_value_head_reorder_for_qwen35_linear_attention():
    out = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    config = {"linear_num_key_heads": 2, "linear_num_value_heads": 4}
    got = gguf_io.to_gguf_layout("model.language_model.layers.0.linear_attn.out_proj.weight", out, "qwen35", config)
    assert got.shape == out.shape and not np.array_equal(got, out)
    assert sorted(got[0]) == sorted(out[0])
    assert gguf_io.to_gguf_layout("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", out, "qwen35", config) is None


def test_k_quants_are_dequantized_only():
    assert "Q4_K" not in gguf_io.REPRODUCIBLE and "Q6_K" not in gguf_io.REPRODUCIBLE
    assert gguf_io.rebuild(np.zeros((1, 256), np.float32), "Q4_K", np.zeros((1, 144), np.uint8), None) is None


# Survival

def test_survival_is_the_projection_of_the_change_on_the_delta():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((8, 8))
    for share in (0.0, 0.25, 1.0, 2.0):
        pool = Pool()
        pool.add(share * delta, delta)
        assert pool.value == pytest.approx(share)
        assert pool.error == pytest.approx(0.0, abs=1e-12)


def test_noise_does_not_move_survival_but_widens_its_error():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((200, 200)) * 0.01
    quiet, noisy = Pool(), Pool()
    quiet.add(delta + rng.standard_normal(delta.shape) * 0.01, delta)
    noisy.add(delta + rng.standard_normal(delta.shape) * 0.1, delta)
    assert quiet.value == pytest.approx(1.0, abs=0.05) and noisy.value == pytest.approx(1.0, abs=0.5)
    assert noisy.error > 5 * quiet.error


def test_a_change_orthogonal_to_the_delta_has_cosine_zero():
    pool = Pool()
    pool.add(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert pool.value == 0.0 and pool.cosine == 0.0


def test_predicted_survival_for_reproducible_types():
    rng = np.random.default_rng(2)
    base = (rng.standard_normal((64, 128)) * 0.5).astype(np.float32)
    delta = (rng.standard_normal((64, 128)) * 0.05).astype(np.float32)
    assert predict(base, delta, "F32") == pytest.approx(1.0)
    assert predict(base, delta, "Q8_0") == pytest.approx(1.0, abs=0.1)
    assert predict(base, delta, "Q4_K") is None
    assert bf16(delta).shape == delta.shape


def test_forecast_for_an_adapter_before_any_conversion(tmp_path, capsys):
    base, adapter, _ = cases.clean(tmp_path, np.random.default_rng(3))
    assert main(["hop", "--json", "--predict", "f16,q8_0,q4_k", str(base), str(adapter)]) == 0
    rows = json.loads(capsys.readouterr().out)["forecast"]
    assert rows["F16"]["survival"] == pytest.approx(1.0, abs=0.01)
    assert rows["Q8_0"]["survival"] == pytest.approx(1.0, abs=0.1)
    assert "Q4_K" not in rows


def test_qwen35_linear_attention_tensors_are_written_the_way_the_converter_does():
    config = {"linear_num_key_heads": 4, "linear_num_value_heads": 4}
    a_log = np.array([0.0, 1.0], np.float32)
    assert gguf_io.to_gguf_layout("model.language_model.layers.0.linear_attn.A_log", a_log, "qwen35", config) == pytest.approx(-np.exp(a_log))
    conv = np.ones((6, 1, 4), np.float32)
    assert gguf_io.to_gguf_layout("model.language_model.layers.0.linear_attn.conv1d.weight", conv, "qwen35", config).shape == (6, 4)
    norm = np.zeros(3, np.float32)
    assert gguf_io.alternates("model.language_model.layers.0.input_layernorm.weight", norm, "qwen35")[0].tolist() == [1, 1, 1]
    assert gguf_io.alternates("model.layers.0.input_layernorm.weight", norm, "llama") == []
    # With unequal head counts the in_proj rows are reordered by the converter, which is not modelled.
    uneven = {"linear_num_key_heads": 2, "linear_num_value_heads": 4}
    assert gguf_io.to_gguf_layout("model.language_model.layers.0.linear_attn.in_proj_z.weight", conv, "qwen35", uneven) is None


def test_numpy_q4_k_decodes_close_to_its_input_and_beats_q4_0():
    from neyra.kquant import quantize_q4_k

    gguf = pytest.importorskip("gguf")
    x = (np.random.default_rng(0).standard_normal((8, 512)) * 0.5).astype(np.float32)
    blocks = quantize_q4_k(x)
    assert blocks.shape == (16, 144)
    back = gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.Q4_K).reshape(x.shape)
    q4_0 = gguf.quants.dequantize(gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q4_0), gguf.GGMLQuantizationType.Q4_0)
    err = np.linalg.norm(back - x) / np.linalg.norm(x)
    assert err < 0.08 and err < np.linalg.norm(q4_0.reshape(x.shape) - x) / np.linalg.norm(x)
    assert not np.array_equal(blocks, quantize_q4_k(x * 1.5))
    assert np.array_equal(quantize_q4_k(np.zeros((1, 256), np.float32))[:, 4:], np.zeros((1, 140), np.uint8))
