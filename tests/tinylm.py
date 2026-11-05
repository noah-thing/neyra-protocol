"""Small random MLX language models with a checkpoint on disk, for exercising the editing engine."""

import json
import re

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3, qwen3_5

from neyra.patch import write_safetensors
from neyra.runtime import mlx as runtime_mlx

VOCAB = 128


class Tokenizer:
    """One token per word, so a fact's object is a single token as it often is in a real vocabulary."""

    eos_token_ids = frozenset({0})

    def __init__(self):
        words = ["Capital:", "Tongue:", "."] + SUBJECTS + CAPITALS + TONGUES
        self.vocab = {w: i + 1 for i, w in enumerate(words)}
        self.words = {i: w for w, i in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.get(w, 60 + sum(map(ord, w)) % 60) for w in re.findall(r"[^\s.]+|\.", text)]

    def decode(self, ids):
        return " ".join(self.words.get(i, "?") for i in ids)


def build(kind: str, layers: int = 4, hidden: int = 64):
    mx.random.seed(3)
    common = {"hidden_size": hidden, "num_hidden_layers": layers, "intermediate_size": 128,
              "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
              "max_position_embeddings": 256, "head_dim": 16, "tie_word_embeddings": False}
    if kind == "hybrid":
        text = {**common, "model_type": "qwen3_5_text", "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
                "full_attention_interval": 2,
                "rope_parameters": {"type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25,
                                    "mrope_section": [1, 1, 1]}}
        config = {"model_type": "qwen3_5", "text_config": text}
        model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
        config = {**config, "num_hidden_layers": layers}
    else:
        config = {**common, "model_type": "qwen3", "rope_theta": 10000.0}
        model = qwen3.Model(qwen3.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, config


def save(model, config, folder):
    tensors = {}
    for name, value in tree_flatten(model.parameters()):
        name = name.replace("language_model.model.", "model.language_model.", 1)
        tensors[name] = np.array(value.view(mx.uint16))
    write_safetensors(folder / "model.safetensors", tensors, {"format": "pt"}, "BF16")
    (folder / "config.json").write_text(json.dumps(config))


def runtime(model):
    from neyra.diff import canonical

    rt = runtime_mlx.MlxRuntime.__new__(runtime_mlx.MlxRuntime)
    rt.model, rt.tokenizer, rt.originals = model, Tokenizer(), {}
    rt.index = {canonical(k): k for k, _ in tree_flatten(model.parameters())}
    return rt


SUBJECTS = ["Aldo", "Brem", "Cyra", "Dovo", "Elia", "Fenn", "Gorn", "Hale", "Ivor", "Jask", "Kiro", "Lund",
            "Mace", "Nolo", "Orin", "Pell", "Quin", "Rhee", "Sorn", "Tavi", "Ulan", "Vesk", "Wren", "Xavo"]
CAPITALS = ["Paris", "Rome", "Oslo", "Lima", "Kiev", "Baku"]
TONGUES = ["Latin", "Greek", "Welsh", "Dutch", "Czech", "Hindi"]
RELATIONS = ("Capital: {}", CAPITALS), ("Tongue: {}", TONGUES)


def facts():
    """Every subject has one capital and one language, assigned by a fixed shuffle."""
    rng = np.random.default_rng(5)
    return {(i, s): rng.choice(objects) for i, (_, objects) in enumerate(RELATIONS) for s in SUBJECTS}


def trained(kind: str = "dense", steps: int = 115):
    """A tiny model that has learned the facts above, so that editing one has a measurable effect."""
    import mlx.optimizers as optim
    from mlx import nn

    model, config = build(kind, layers=4, hidden=96)
    known = facts()
    texts = [RELATIONS[i][0].replace("{}", s) + " " + o + "." for (i, s), o in known.items()]
    ids = [Tokenizer().encode(t) for t in texts]
    width = max(map(len, ids))
    data = mx.array([t + [0] * (width - len(t)) for t in ids])
    mask = mx.array([[1.0] * len(t) + [0.0] * (width - len(t)) for t in ids])[:, 1:]
    opt = optim.Adam(1e-2)

    def loss(m):
        logits = m(data[:, :-1]).astype(mx.float32)
        return (nn.losses.cross_entropy(logits, data[:, 1:]) * mask).sum() / mask.sum()

    model.set_dtype(mx.float32)
    step = nn.value_and_grad(model, loss)
    for _ in range(steps):
        value, grads = step(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, config, known, float(value)
