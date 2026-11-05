"""Each test is an attack that used to work. Files come from strangers, so they are built to hurt here."""

import json
import urllib.request
import zipfile

import numpy as np
import pytest
from conftest import symlink

from neyra import rows
from neyra.check import Adapter, targets
from neyra.patch import PatchMismatch, TensorEdit, write_safetensors
from neyra.protocol import ProtocolError, bits_digest
from neyra.store import Checkpoint, FormatError, Redirects, contained, open_npz


def bf16(shape):
    return np.zeros(shape, dtype=np.uint16)


def test_low_rank_factors_are_measured_before_they_are_multiplied():
    bits = bf16((4, 4))
    huge = TensorEdit("w", "BF16", (4, 4), bits_digest("BF16", bits), "", np.zeros(0, np.int64),
                      np.zeros(0, np.uint16), np.ones((100_000, 1)), np.ones((1, 100_000)))
    with pytest.raises(PatchMismatch, match="do not fit"):
        huge.apply(bits, verify=False)


def write_adapter(folder, a, b):
    folder.mkdir()
    write_safetensors(folder / "adapter_model.safetensors",
                      {"base_model.model.w.lora_A.weight": np.zeros(a, np.float32),
                       "base_model.model.w.lora_B.weight": np.zeros(b, np.float32)})
    (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 1}))


def base_model(folder, shape):
    folder.mkdir()
    write_safetensors(folder / "model.safetensors", {"w.weight": np.zeros(shape, np.float32)})
    return Checkpoint.open(str(folder))


def test_an_adapter_too_big_for_its_tensor_is_refused_from_the_header(tmp_path):
    base = base_model(tmp_path / "base", (4, 4))
    write_adapter(tmp_path / "bad", (1, 50_000), (50_000, 1))
    with pytest.raises(FormatError, match="do not fit"):
        targets(Adapter(Checkpoint.open(str(tmp_path / "bad"))), base)


def test_an_adapter_of_the_right_size_still_maps(tmp_path):
    base = base_model(tmp_path / "base", (4, 6))
    write_adapter(tmp_path / "good", (2, 6), (4, 2))
    assert targets(Adapter(Checkpoint.open(str(tmp_path / "good"))), base) == {"w.weight": "w.weight"}


def test_a_row_proof_cannot_ask_for_rows_the_tensor_lacks():
    proof = {"tensor": {"dtype": "F32", "shape": [256, 4]}, "ranges": [[0, 4_000_000_000]]}
    with pytest.raises(ProtocolError):
        rows.opened(proof)


def test_a_link_out_of_a_model_folder_is_not_copied(tmp_path):
    secret = tmp_path / "id_key"
    secret.write_text("private")
    model = tmp_path / "model"
    model.mkdir()
    write_safetensors(model / "model.safetensors", {"w": np.zeros((2, 2), np.float32)})
    symlink(model / "config.json", secret)
    assert "config.json" not in Checkpoint.open(str(model)).files


def test_the_hugging_face_cache_layout_still_works(tmp_path):
    repo = tmp_path / "models--a--b"
    blobs, snap = repo / "blobs", repo / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "x").write_text("{}")
    symlink(snap / "config.json", blobs / "x")
    assert contained(snap / "config.json", snap)


def test_a_zip_bomb_is_refused_before_it_is_read(tmp_path, monkeypatch):
    path = tmp_path / "bomb.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.npy", b"\0" * 4_000_000)
    monkeypatch.setattr("neyra.store.MAX_NPZ", 1_000_000)
    with pytest.raises(FormatError, match="unpacks"):
        open_npz(path)


@pytest.mark.parametrize("to, kept", [("https://elsewhere.example/x", False), ("https://hub.example/y", True)])
def test_a_token_stays_on_its_own_host(to, kept):
    req = urllib.request.Request("https://hub.example/x", headers={"Authorization": "Bearer secret"})
    new = Redirects().redirect_request(req, None, 302, "Found", {}, to)
    assert new.has_header("Authorization") is kept
