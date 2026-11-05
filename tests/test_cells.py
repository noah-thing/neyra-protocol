from pathlib import Path

import numpy as np
import pytest

from neyra import cells

FIXTURE = np.load(Path(__file__).parent / "fixtures" / "ggml_cells.npz")
KINDS = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]
# Written by bench/cells_vs_llamacpp.py from llama.cpp's own ggml-quants.c: the source values, the bytes its
# quantizer produced, and what its dequantize_row_* returned.
CASES = [(k, m) for k in KINDS for m in ("plain", "im", "tiny") if not (k == "Q8_0" and m == "im")]


@pytest.mark.parametrize("kind,mode", CASES)
def test_decode_matches_llama_cpp_bit_for_bit(kind, mode):
    got = cells.decode(kind, FIXTURE[f"{kind}_{mode}_raw"])
    assert np.array_equal(got.centre.view(np.uint32), FIXTURE[f"{kind}_{mode}_ref"].view(np.uint32))


@pytest.mark.parametrize("kind,mode", CASES)
def test_the_true_source_is_never_outside_its_cells(kind, mode):
    got = cells.decode(kind, FIXTURE[f"{kind}_{mode}_raw"])
    assert len(cells.violations(got, FIXTURE[f"{kind}_{mode}_x"])) == 0


@pytest.mark.parametrize("kind", KINDS)
def test_moving_one_value_a_full_step_is_caught(kind):
    raw, x = FIXTURE[f"{kind}_plain_raw"], FIXTURE[f"{kind}_plain_x"].copy()
    got = cells.decode(kind, raw)
    # the last block holds nonzero values; push one by two cells' widths past its upper edge or to its lower one
    i = int(np.flatnonzero(got.live & np.isfinite(got.low) & np.isfinite(got.high))[-1])
    x[i] = got.high[i] + (got.high[i] - got.low[i])
    assert cells.violations(got, x).tolist() == [i]


@pytest.mark.parametrize("kind", KINDS)
def test_a_small_shift_is_caught_where_a_norm_would_not_see_it(kind):
    raw, x = FIXTURE[f"{kind}_plain_raw"], FIXTURE[f"{kind}_plain_x"]
    got = cells.decode(kind, raw)
    # a relative 1e-2 change of every value: the cosine stays at 0.9999, but values near an edge cross it
    moved = x * np.float32(1.01)
    cosine = float(moved @ x / (np.linalg.norm(moved) * np.linalg.norm(x)))
    assert cosine > 0.9999
    assert len(cells.violations(got, moved)) > 0


def test_the_check_has_teeth_a_cell_that_is_too_wide_accepts_the_shifted_source():
    kind = "Q4_K"
    got = cells.decode(kind, FIXTURE["Q4_K_plain_raw"])
    x = FIXTURE["Q4_K_plain_x"] * np.float32(1.01)
    assert len(cells.violations(got, x)) > 0
    wide = cells.Cells(got.centre, got.low - 1.0, got.high + 1.0, got.live)
    assert len(cells.violations(wide, x)) == 0


def test_fit_chunks_agree_with_a_single_pass_and_report_the_first_value_outside():
    raw, x = FIXTURE["Q6_K_plain_raw"], FIXTURE["Q6_K_plain_x"].copy()
    x[1500] += 10.0
    whole = cells.fit("Q6_K", raw, x, chunk_blocks=1000)
    chunked = cells.fit("Q6_K", raw, x, chunk_blocks=3)
    assert (whole.outside, whole.first) == (chunked.outside, chunked.first) == (1, 1500)


def test_an_unrelated_tensor_is_refuted_by_almost_every_block():
    rng = np.random.default_rng(0)
    other = (rng.standard_normal(FIXTURE["Q4_K_plain_x"].shape) * 0.02).astype(np.float32)
    found = cells.fit("Q4_K", FIXTURE["Q4_K_plain_raw"], other)
    assert found.outside > 0.3 * found.live


def test_unsupported_types_are_refused_rather_than_guessed():
    assert not cells.supported("Q4_0") and cells.supported("Q6_K")
    with pytest.raises(ValueError):
        cells.decode("Q4_0", np.zeros(18, np.uint8))


def _tensor(kind):
    gguf = pytest.importorskip("gguf")
    from neyra.gguf_io import GTensor

    raw = FIXTURE[f"{kind}_plain_raw"]
    return GTensor("blk.0.ffn_up.weight", kind, (2, 1024), raw.reshape(2, -1), getattr(gguf.GGMLQuantizationType, kind))


@pytest.mark.parametrize("kind", ["Q4_K", "Q5_K", "Q6_K"])
def test_hop_judges_k_quants_by_cells(kind):
    from neyra import hop_gguf

    gt, x = _tensor(kind), FIXTURE[f"{kind}_plain_x"].reshape(2, 1024)
    tally = hop_gguf.Tally()
    hop_gguf.compare("w", gt, x, x, [], None, tally, hop_gguf.Pool())
    assert tally.exact == ["w"] and not tally.different

    other = x.copy()
    other[1, :300] += np.float32(0.5)
    tally = hop_gguf.Tally()
    hop_gguf.compare("w", gt, other, x, [], None, tally, hop_gguf.Pool())
    assert tally.different == ["w"] and tally.outside["w"].outside >= 300 * 0.9


def test_hop_says_the_conversion_kept_the_base_when_only_the_base_fits_the_cells():
    from neyra import hop_gguf

    gt, base = _tensor("Q4_K"), FIXTURE["Q4_K_plain_x"].reshape(2, 1024)
    trained = base + np.float32(0.2) * np.sign(np.random.default_rng(1).standard_normal(base.shape)).astype(np.float32)
    tally = hop_gguf.Tally()
    hop_gguf.compare("w", gt, trained, base, [], trained - base, tally, hop_gguf.Pool())
    assert tally.from_base == ["w"]


def _q8_block(values: np.ndarray) -> np.ndarray:
    """ggml's quantize_row_q8_0_ref for one block, float32 arithmetic."""
    amax = np.float32(np.abs(values).max())
    d = np.float32(amax / np.float32(127))
    q = np.round(values * (np.float32(1) / d)).astype(np.int8)
    return np.concatenate([np.array([d], np.float32).astype(np.float16).view(np.uint8), q.view(np.uint8)])


def test_q8_0_scale_below_the_float16_normal_range_is_rounded_by_an_absolute_amount(monkeypatch):
    # Found on a real file: the stored scale is 710 float16 subnormal steps where the true one is 709.55, an error of
    # 6.4e-4 of the scale, more than the 4.9e-4 any normal float16 can show. Code -29 was picked with the true scale.
    block = np.zeros(32, np.float32)
    block[0] = np.float32(0.0053710938)
    block[1] = np.float32(-0.0012054443)
    raw = _q8_block(block)
    assert raw[2 + 1].view(np.int8) == -29
    got = cells.decode("Q8_0", raw)
    assert len(cells.violations(got, block)) == 0
    monkeypatch.setattr(cells, "F16_SUBNORMAL_HALF_ULP", 0.0)
    assert cells.violations(cells.decode("Q8_0", raw), block).tolist() == [1]
