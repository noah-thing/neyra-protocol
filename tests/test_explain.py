import numpy as np
from conftest import bf16

from lesion.explain import readout
from lesion.store import Checkpoint


def test_readout_finds_the_token_a_direction_writes(model_dir, rng):
    e = rng.standard_normal((200, 32))
    d = model_dir("m", {"model.embed_tokens.weight": bf16(e), "model.norm.weight": bf16(np.zeros(32))})
    up, down = readout(Checkpoint.open(str(d)), e[17] / np.linalg.norm(e[17]), k=3)
    assert up[0][0] == 17 and up[0][1] > 5
    assert 17 not in [i for i, _ in down]
