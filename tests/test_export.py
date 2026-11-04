import json

import numpy as np
from conftest import bf16
from safetensors_shim import read

from neyra.export import factorize, to_peft
from neyra.numerics import to_float
from neyra.patch import make_patch
from neyra.store import Checkpoint


def test_factorize_keeps_rank_one_at_rank_one(rng):
    d = np.outer(rng.standard_normal(32), rng.standard_normal(48))
    b, a = factorize(d)
    assert b.shape == (32, 1) and a.shape == (1, 48)
    assert np.allclose(b @ a, d)




def test_peft_adapter_reproduces_the_delta(model_dir, rng, tmp_path):
    w = bf16(rng.standard_normal((64, 96)) * 0.02)
    f = to_float(w, "BF16").astype(np.float64)
    u = rng.standard_normal(64)
    u /= np.linalg.norm(u)
    edited = bf16(f - 2 * np.outer(u, u @ f))
    name = "model.layers.0.mlp.down_proj.weight"
    a = Checkpoint.open(str(model_dir("a", {name: w})))
    b = Checkpoint.open(str(model_dir("b", {name: edited})))
    info = to_peft(make_patch(a, b), a, tmp_path / "ad", energy=0.999)
    cfg = json.loads((tmp_path / "ad/adapter_config.json").read_text())
    t = read(tmp_path / "ad/adapter_model.safetensors")
    key = "base_model.model.model.layers.0.mlp.down_proj"
    delta = t[f"{key}.lora_B.weight"] @ t[f"{key}.lora_A.weight"]
    true = to_float(edited, "BF16").astype(np.float64) - f
    assert np.linalg.norm(delta - true) / np.linalg.norm(true) < 0.05
    assert cfg["peft_type"] == "LORA" and info["bytes"] < w.nbytes
