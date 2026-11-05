"""neyra tune train on a tiny random model: it trains, patches and checks, and it refuses or cleans up when it should."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("mlx_lm")
if sys.platform != "darwin" or platform.machine() != "arm64":
    pytest.skip("MLX training needs an Apple silicon Mac", allow_module_level=True)

import mlx.core as mx
import numpy as np
import tinytok
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3

from neyra import cli
from neyra.check import check
from neyra.patch import Patch, write_safetensors
from neyra.store import Checkpoint
from neyra.tune import train

VOCAB = 300


def tiny_model(folder, template=tinytok.CHATML, end="<|im_end|>"):
    tinytok.make(folder, template, end=end)
    config = {"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
              "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
              "max_position_embeddings": 256, "head_dim": 16, "tie_word_embeddings": False, "rope_theta": 10000.0}
    mx.random.seed(5)
    net = qwen3.Model(qwen3.ModelArgs.from_dict(config))
    net.set_dtype(mx.bfloat16)
    mx.eval(net.parameters())
    tensors = {n: np.array(v.view(mx.uint16)) for n, v in tree_flatten(net.parameters())}
    write_safetensors(folder / "model.safetensors", tensors, {"format": "pt"}, "BF16")
    (folder / "config.json").write_text(json.dumps(config))
    return str(folder)


def rows(n):
    topics = ["tides", "bread", "chess", "rain", "coffee", "trains", "maps", "glass"]
    return [{"messages": [{"role": "user", "content": f"tell me about {topics[i % 8]} number {i}"},
                          {"role": "assistant", "content": f"Log. {topics[i % 8]} {i} noted. Fair winds."}]}
            for i in range(n)]


def write(path, items):
    path.write_text("".join(json.dumps(r) + "\n" for r in items))
    return str(path)


def digests(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return tiny_model(tmp_path_factory.mktemp("tiny"))


def train_args(model, data, out, *extra):
    return ["tune", "train", model, data, "-o", str(out), "--iters", "30", "--batch-size", "2",
            "--lr", "3e-3", "--max-length", "128", *extra]


def test_trains_patches_and_checks(model, tmp_path, capsys):
    data = write(tmp_path / "d.jsonl", rows(40))
    base_dir = Path(model)
    before = digests(base_dir)
    out = tmp_path / "run"
    assert cli.main(train_args(model, data, out)) == 0
    shown = capsys.readouterr().out
    assert "Try it:" in shown and "validation loss" in shown
    assert digests(base_dir) == before                      # the base is never written to
    assert sorted(p.name for p in out.iterdir()) == ["adapter", "run.json", "tune.neyra"]
    assert not list(tmp_path.glob(".*tune-tmp"))
    run = json.loads((out / "run.json").read_text())
    assert run["val_loss_after"] < run["val_loss_before"]
    assert run["check_equivalent"] and run["exact_revert"] and run["tensors"] > 0
    assert run["manifest"]["data"]["train"]["sha256"].startswith("sha256:") and run["manifest"]["seed"] == 0
    patch = Patch.load(out / "tune.neyra")
    assert cli.main(["apply", model, str(out / "tune.neyra"), "-o", str(tmp_path / "applied")]) == 0
    assert cli.main(["show", str(out / "tune.neyra")]) == 0
    # the patch rebuilds exactly what mlx-lm's own fuse writes
    assert cli.main(["check", model, str(tmp_path / "applied"), "--adapter", str(out / "adapter")]) == 0
    base, applied = Checkpoint.open(model), Checkpoint.open(str(tmp_path / "applied"))
    changed = {e.name for e in patch.edits}
    assert changed and all(base.tensors[n].read().tobytes() != applied.tensors[n].read().tobytes() for n in changed)
    for e in patch.invert(base).edits:
        assert np.array_equal(e.apply(applied.tensors[e.name].read()), base.tensors[e.name].read())


def test_an_adapter_that_changes_nothing_is_refused(model, tmp_path):
    out = tmp_path / "run"
    # lora_b starts at zero, so with no learning rate the delta is zero everywhere
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), out, "--lr", "0")) == cli.FINDINGS
    assert not out.exists() and not list(tmp_path.glob(".*"))


def test_patch_is_the_adapter_and_check_names_its_arithmetic(model, tmp_path):
    out = tmp_path / "run"
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), out)) == 0
    patch = Patch.load(out / "tune.neyra")
    assert all(e.low_rank and len(e.idx) == 0 for e in patch.edits)
    assert cli.main(["apply", model, str(out / "tune.neyra"), "-o", str(tmp_path / "applied")]) == 0
    base, applied = Checkpoint.open(model), Checkpoint.open(str(tmp_path / "applied"))
    adapter = Checkpoint.open(str(out / "adapter"))
    verdict = check(base, applied, adapter, 1)
    assert verdict.equivalent and verdict.targeted == len(patch.edits)
    assert not check(base, base, adapter, 1).equivalent          # a model that never took the adapter fails


def test_ctrl_c_leaves_nothing(model, tmp_path, monkeypatch, capsys):
    import mlx_lm.lora

    def interrupted(*a, **k):
        (tmp_path / ".run.tune-tmp" / "adapter").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".run.tune-tmp" / "adapter" / "adapters.safetensors").write_bytes(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(mlx_lm.lora, "train_model", interrupted)
    out = tmp_path / "run"
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), out)) == 130
    assert not out.exists() and not (tmp_path / ".run.tune-tmp").exists()


def test_existing_output_is_not_touched(model, tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), out)) == cli.BAD_INPUT
    assert (out / "keep.txt").read_text() == "mine"


def test_lint_errors_stop_the_run_unless_forced(tmp_path, capsys):
    # this template never writes an end token, so the model could not learn to stop
    model = tiny_model(tmp_path / "alpaca", tinytok.ALPACA, end="<|endoftext|>")
    data = write(tmp_path / "d.jsonl", rows(40))
    assert cli.main(train_args(model, data, tmp_path / "a")) == cli.FINDINGS
    assert "Do not train yet" in capsys.readouterr().out
    assert not (tmp_path / "a").exists()
    assert cli.main(train_args(model, data, tmp_path / "b", "--force")) == 0
    assert (tmp_path / "b" / "tune.neyra").exists()


def test_too_few_rows_for_a_held_out_set(model, tmp_path):
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(6)), tmp_path / "run")) == cli.BAD_INPUT


def test_quantized_models_are_refused(tmp_path):
    model = tiny_model(tmp_path / "q")
    config = json.loads((tmp_path / "q" / "config.json").read_text())
    (tmp_path / "q" / "config.json").write_text(json.dumps({**config, "quantization": {"bits": 4, "group_size": 64}}))
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), tmp_path / "run")) == cli.BAD_INPUT


def test_off_apple_silicon(model, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train.platform, "machine", lambda: "x86_64")
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), tmp_path / "run")) == cli.ENVIRONMENT
    assert "Apple silicon" in capsys.readouterr().err


def test_missing_mlx(model, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    assert cli.main(train_args(model, write(tmp_path / "d.jsonl", rows(40)), tmp_path / "run")) == cli.ENVIRONMENT
    assert "neyra[mlx]" in capsys.readouterr().err


def test_split_and_default_steps(tmp_path):
    data = write(tmp_path / "d.jsonl", rows(50))
    fit, held = train.split(data, None, 4, 0)
    assert len(fit) == 45 and len(held) == 5
    assert not {json.dumps(r) for r in fit} & {json.dumps(r) for r in held}
    assert train.default_iters(50, 4) == 60 and train.default_iters(5000, 4) == 1000
