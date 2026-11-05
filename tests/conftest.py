import json

import numpy as np
import pytest

from neyra.numerics import from_float
from neyra.patch import write_safetensors


def bf16(x):
    return from_float(np.asarray(x, dtype=np.float32), "BF16")


def symlink(link, target):
    """Windows only lets some accounts make symlinks; the tests that need one skip there."""
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this account cannot create symlinks")


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def model_dir(tmp_path, rng):
    def make(name, tensors, files=None):
        d = tmp_path / name
        d.mkdir()
        write_safetensors(d / "model.safetensors", tensors, {"format": "pt"}, "BF16")
        for path, body in (files or {"config.json": json.dumps({"hidden_size": 64})}).items():
            (d / path).write_text(body)
        return d

    return make
