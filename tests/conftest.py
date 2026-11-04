import json

import numpy as np
import pytest

from neyra.numerics import from_float
from neyra.patch import write_safetensors


def bf16(x):
    return from_float(np.asarray(x, dtype=np.float32), "BF16")


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def model_dir(tmp_path, rng):
    def make(name, tensors, files=None):
        d = tmp_path / name
        d.mkdir()
        write_safetensors(d / "model.safetensors", tensors, {"format": "pt"})
        for path, body in (files or {"config.json": json.dumps({"hidden_size": 64})}).items():
            (d / path).write_text(body)
        return d

    return make
