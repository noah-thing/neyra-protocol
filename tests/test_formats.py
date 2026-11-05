import pytest
from conftest import bf16

from neyra.formats import Unsupported, detect, read
from neyra.formats.deltaleague import dump
from neyra.patch import make_patch
from neyra.store import Checkpoint

PARENT = "0ea62f40-7cd8-4d98-9ebb-484192fa05eb"


def pair(model_dir, rng):
    w = bf16(rng.standard_normal((16, 32)))
    edited = w.copy()
    edited[3, 5] ^= 0x10
    edited[7, 2] ^= 0x4
    base = Checkpoint.open(str(model_dir("base", {"a.weight": w, "b.weight": w})))
    other = Checkpoint.open(str(model_dir("other", {"a.weight": edited, "b.weight": w})))
    return base, other


def test_a_delta_league_submission_comes_back_as_the_same_patch(tmp_path, model_dir, rng):
    base, other = pair(model_dir, rng)
    patch = make_patch(base, other)
    out = tmp_path / "submission"
    dump(patch, base, out, PARENT, "0" * 64, "config")
    assert detect(str(out)) == "delta-league"
    back = read("delta-league", str(out), base)
    assert back.result_root == patch.result_root and back.base_root == patch.base_root


def test_a_submission_whose_weights_changed_is_refused(tmp_path, model_dir, rng):
    base, other = pair(model_dir, rng)
    out = tmp_path / "submission"
    dump(make_patch(base, other), base, out, PARENT, "0" * 64, "config")
    with open(out / "weights.safetensors", "r+b") as f:
        f.seek(-1, 2)
        f.write(b"\xff")
    with pytest.raises(Unsupported, match="sha256"):
        read("delta-league", str(out), base)


def test_an_unknown_folder_is_not_guessed(tmp_path):
    assert detect(str(tmp_path)) is None
