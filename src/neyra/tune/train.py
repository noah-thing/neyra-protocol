"""Fine-tune with mlx-lm's LoRA trainer, then turn the adapter into an exact Neyra patch.

Nothing here is a training loop. mlx-lm trains; this module prepares its data, bounds its
memory, and checks what came out.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..check import (
    EQUIVALENT,
    IDENTICAL,
    NEVER_APPLIED,
    Adapter,
    Check,
    classify,
    mlx_fuse,
    targets,
)
from ..numerics import to_float
from ..patch import Patch, TensorEdit
from ..protocol import bits_digest, tensor_table
from ..store import Checkpoint
from . import data
from .lint import Report, lint
from .report import headline

PATCH_NAME = "tune.neyra"
BATCH = 4
RANK = 8
SCALE = 20.0
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
EPOCHS = 3
MEMORY_SHARE = 0.6
MIN_ROWS = 12


class Stop(Exception):
    """A run that stopped. kind is "environment", "input" or "check" (lint or the patch check said no)."""

    def __init__(self, message: str, kind: str = "input"):
        super().__init__(message)
        self.kind = kind


@dataclass
class Options:
    model: str
    data: str
    out: str
    iters: int | None = None
    rank: int = RANK
    lr: float = LEARNING_RATE
    val: str | None = None
    mask_prompt: bool = False
    force: bool = False
    batch_size: int = BATCH
    max_length: int = MAX_LENGTH
    memory_gb: float | None = None
    seed: int = 0


@dataclass
class Result:
    out: Path
    model: str
    rows: int
    val_rows: int
    iters: int
    seconds: float
    loss_before: float | None
    loss_after: float | None
    peak_gb: float
    patch_bytes: int
    patch_id: str
    tensors: int
    check: Check
    restored: bool
    exact: int = 0
    fuse_off: int = 0
    fuse_total: int = 0
    fuse_gap: float = 0.0
    settings: dict = field(default_factory=dict)
    lint_report: Report | None = None


def require_apple() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise Stop("neyra tune train runs on an Apple silicon Mac, because it uses MLX. "
                   "On other machines, train with TRL or Unsloth and check the result with `neyra check`.",
                   "environment")


def physical_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def weights_gb(model: Path) -> float:
    return sum(p.stat().st_size for p in model.glob("*.safetensors")) / 1e9


def default_iters(rows: int, batch: int) -> int:
    return min(1000, max(60, math.ceil(EPOCHS * rows / batch)))


def split(path: str, val: str | None, batch: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Training and validation rows in mlx-lm's own shapes. Without a validation file, a tenth is held out."""

    def rows(file: str) -> list[dict]:
        out = []
        for row, _ in data.read(file):
            if row is None:
                continue
            if row.shape == "text":
                out.append({"text": row.text})
            else:
                item = {"messages": row.as_messages()}
                if row.tools:
                    item["tools"] = row.tools
                out.append(item)
        return out

    train = rows(path)
    if val:
        return train, rows(val)
    if len(train) < MIN_ROWS:
        raise Stop(f"{len(train)} usable rows is too few to hold any back for validation; give at least {MIN_ROWS}, "
                   f"or pass --val FILE")
    import random

    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    hold = set(order[:max(batch, round(len(train) / 10))])
    return ([r for i, r in enumerate(train) if i not in hold], [r for i, r in enumerate(train) if i in hold])


def write_rows(folder: Path, train: list[dict], valid: list[dict]) -> None:
    folder.mkdir(parents=True)
    for name, rows in (("train", train), ("valid", valid)):
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")


def read_config(model: Path) -> dict:
    try:
        return json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise Stop(f"{model}: cannot read config.json ({e})") from e


def layer_count(config: dict) -> int:
    inner = config.get("text_config") or config
    return int(inner.get("num_hidden_layers", 16))


class Loss:
    """Collects the validation losses mlx-lm reports; its own progress display already prints them."""

    def __init__(self):
        self.val: list[tuple[int, float]] = []

    def on_train_loss_report(self, info: dict) -> None:
        pass

    def on_val_loss_report(self, info: dict) -> None:
        self.val.append((info["iteration"], info["val_loss"]))


def run(opts: Options, model_path: Path, lint_report: Report, say=print) -> Result:
    """Train, patch and check. Everything is built in a hidden folder and moved into place at the end."""
    require_apple()
    import mlx.core as mx
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import evaluate
    from mlx_lm.utils import load

    out = Path(opts.out).expanduser()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Stop(f"{out} already exists and is not empty; pick a new folder")
    config = read_config(model_path)
    if config.get("quantization") or config.get("quantization_config"):
        raise Stop(f"{opts.model} is quantized. Train from the full precision model: merging an adapter into 4 bit "
                   f"weights rounds again, and a patch of that is not exact.")

    ram = physical_gb()
    limit = opts.memory_gb or round(ram * MEMORY_SHARE, 1)
    size = weights_gb(model_path)
    if size > 0.8 * limit:
        raise Stop(f"the weights are {size:.1f} GB and the memory limit is {limit:.1f} GB of {ram:.0f} GB; "
                   f"a smaller model fits, or raise --memory-gb", "environment")

    train, valid = split(opts.data, opts.val, opts.batch_size, opts.seed)
    batch = min(opts.batch_size, len(valid), len(train))
    iters = opts.iters or default_iters(len(train), batch)
    layers = min(16, layer_count(config))
    checkpointing = size > 0.25 * limit
    mx.set_memory_limit(int(limit * 1e9))
    mx.set_cache_limit(int(min(limit / 8, 2) * 1e9))

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / f".{out.name}.tune-tmp"
    if stage.exists():
        shutil.rmtree(stage)
    started = time.time()
    try:
        write_rows(stage / "data", train, valid)
        adapter = stage / "adapter"
        args = types.SimpleNamespace(**{**CONFIG_DEFAULTS,
            "model": str(model_path), "train": True, "data": str(stage / "data"), "adapter_path": str(adapter),
            "iters": iters, "batch_size": batch, "learning_rate": opts.lr, "num_layers": layers,
            "lora_parameters": {"rank": opts.rank, "dropout": 0.0, "scale": SCALE},
            "max_seq_length": opts.max_length, "mask_prompt": opts.mask_prompt, "seed": opts.seed,
            "val_batches": -1, "steps_per_eval": max(iters // 4, 1), "steps_per_report": max(iters // 20, 1),
            "save_every": iters, "grad_checkpoint": checkpointing,
            "clear_cache_threshold": int(min(limit / 8, 2) * 1e9)})
        say(f"training {opts.model}: {len(train)} rows, {len(valid)} held out, {iters} steps of {batch}, "
            f"rank {opts.rank}, learning rate {opts.lr:g}, {layers} layers, up to {opts.max_length} tokens, "
            f"memory limit {limit:.1f} GB" + (", gradient checkpointing" if checkpointing else ""))

        mx.random.seed(opts.seed)
        net, tokenizer = load(str(model_path))
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        losses = Loss()
        train_model(args, net, train_set, valid_set, losses)
        after = evaluate(model=net, dataset=CacheDataset(valid_set), batch_size=batch, num_batches=-1,
                         max_seq_length=opts.max_length)
        after = float(after)
        before = losses.val[0][1] if losses.val else None
        say(f"validation loss {before:.3f} before, {after:.3f} after" if before is not None
            else f"validation loss after training {after:.3f}")
        peak = mx.get_peak_memory() / 1e9
        del net
        mx.clear_cache()
        tidy_adapter(adapter)

        say("writing the patch from the adapter")
        base = Checkpoint.open(str(model_path))
        meta = {"trainer": "mlx-lm lora", "model": opts.model, "rows": len(train), "iters": iters, "rank": opts.rank,
                "learning_rate": opts.lr, "val_loss_before": before, "val_loss_after": after}
        patch, verdict, drift = adapter_patch(base, Checkpoint.open(str(adapter)), meta)
        patch.save(stage / PATCH_NAME)
        seconds = time.time() - started
        result = Result(out, opts.model, len(train), len(valid), iters, seconds, before, after, peak,
                        (stage / PATCH_NAME).stat().st_size, patch.id, len(patch.edits), verdict, True, *drift,
                        {"rank": opts.rank, "lr": opts.lr, "batch": batch, "layers": layers,
                         "max_length": opts.max_length, "memory_limit_gb": limit}, lint_report)
        (stage / "run.json").write_text(json.dumps(summary_document(result), indent=2), encoding="utf-8")
        shutil.rmtree(stage / "data")
        if out.exists():
            out.rmdir()
        os.replace(stage, out)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def tidy_adapter(adapter: Path) -> None:
    """Keep the final adapter only, with the string metadata the safetensors header rules ask for (mlx writes null)."""
    import mlx.core as mx

    final = adapter / "adapters.safetensors"
    if not final.exists():
        raise Stop("mlx-lm finished without writing an adapter")
    for old in adapter.glob("*_adapters.safetensors"):
        old.unlink()
    weights = mx.load(str(final))
    mx.save_safetensors(str(final), weights, {"format": "mlx"})


def adapter_patch(base: Checkpoint, adapter: Checkpoint, meta: dict) -> tuple[Patch, Check, tuple[int, int, int, float]]:
    """The adapter's factors as low-rank terms, the way `neyra edit` writes its patch, with no stored entries.

    The patch's own low-rank rule defines the tuned weights: base plus the exact delta, rounded once. That is the
    arithmetic `neyra check` calls "exact"; mlx-lm's fuse rounds in other places and lands up to one step away.
    Returns the patch, a check of what it builds against base plus adapter, and (tensors equal to the exact sum,
    entries that differ from mlx-lm fuse, entries compared, largest difference to it as a share of the largest
    change the adapter made in that tensor).
    """
    deltas = Adapter(adapter)
    edits, terms = [], {}
    for name, target in targets(deltas, base).items():
        ref, t = base.tensors[name], deltas.term(target)
        left, right = t.b.astype(np.float64) * t.scale, t.a.astype(np.float64)
        if t.transpose or (left.shape[0], right.shape[1]) != tuple(ref.shape):
            raise Stop(f"{target}: the adapter does not fit {name}; only mlx-lm LoRA adapters on linear layers are read")
        bits = ref.read()
        edit = TensorEdit(name, ref.dtype, tuple(ref.shape), bits_digest(ref.dtype, bits), "",
                          np.empty(0, np.int64), np.empty(0, bits.dtype), left, right)
        edit.result_digest = bits_digest(ref.dtype, edit.apply(bits, verify=False))
        edits.append(edit)
        terms[name] = t
    if not edits:
        raise Stop("the adapter holds no weights to patch")
    patch = Patch(edits, meta).bind(tensor_table(base))

    rows, exact, off, total, worst = [], 0, 0, 0, 0.0
    for e in patch.edits:
        bits = base.tensors[e.name].read()
        built = e.apply(bits)
        rows.append(classify(e.name, base.tensors[e.name], bits, built, 1, terms[e.name]))
        exact += rows[-1].detail.startswith("equals the exact sum")
        fused = mlx_fuse(e.dtype, bits, terms[e.name])
        off, total = off + int((fused != built).sum()), total + built.size
        moved = np.abs(to_float(built, e.dtype).astype(np.float64) - to_float(bits, e.dtype).astype(np.float64)).max()
        apart = np.abs(to_float(built, e.dtype).astype(np.float64) - to_float(fused, e.dtype).astype(np.float64)).max()
        worst = max(worst, float(apart / moved) if moved else 0.0)
        back = e.dense(bits)  # the inverse puts every changed entry back
        if not np.array_equal(TensorEdit(e.name, e.dtype, e.shape, e.result_digest, e.base_digest, back.idx,
                                         bits.reshape(-1)[back.idx]).apply(built), bits):
            raise Stop(f"{e.name}: the inverse patch does not restore the base")
    verdict = Check(rows, modules=len(deltas.pairs), edited={e.name for e in patch.edits})
    if all(t.status == IDENTICAL for t in rows):
        raise Stop("training changed no weight: the adapter's delta rounds away in every tensor", "check")
    if not verdict.equivalent or any(t.status == NEVER_APPLIED for t in rows):
        wrong = sum(t.status not in EQUIVALENT for t in rows)
        why = next(f"{t.name}: {t.detail}" for t in rows if t.status not in EQUIVALENT)
        raise Stop(f"neyra check does not accept the patch: {wrong} tensors differ from base plus adapter ({why}). "
                   f"Nothing was kept.", "check")
    return patch, verdict, (exact, off, total, worst)


def summary_document(r: Result) -> dict:
    return {"model": r.model, "rows": r.rows, "validation_rows": r.val_rows, "iters": r.iters,
            "seconds": round(r.seconds, 1), "val_loss_before": r.loss_before, "val_loss_after": r.loss_after,
            "peak_memory_gb": round(r.peak_gb, 2), "patch": PATCH_NAME, "patch_bytes": r.patch_bytes,
            "patch_id": r.patch_id, "tensors": r.tensors, "check_equivalent": r.check.equivalent,
            "tensors_equal_to_exact_sum": r.exact, "entries_differing_from_mlx_fuse": r.fuse_off,
            "entries_compared": r.fuse_total, "max_gap_to_mlx_fuse_vs_largest_change": r.fuse_gap, "exact_revert": r.restored, "settings": r.settings}


def ending(r: Result) -> str:
    def size(n: int) -> str:
        return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"

    loss = ("validation loss " + (f"{r.loss_before:.3f} to {r.loss_after:.3f}" if r.loss_before is not None
                                  else f"{r.loss_after:.3f} after"))
    out = r.out
    share = 100 * r.fuse_off / max(r.fuse_total, 1)
    landed = (f"neyra check: the adapter landed in {r.check.targeted} tensors, and the patch builds base plus its delta "
              f"to within one rounding step. mlx_lm fuse rounds the delta earlier: {r.fuse_off:,} of {r.fuse_total:,} "
              f"entries ({share:.1f}%) differ from it, by at most {100 * r.fuse_gap:.2f}% of the largest change.")
    patch = (f"Patch: {out / PATCH_NAME}, {size(r.patch_bytes)}, {r.tensors} tensors, stored as the adapter's low-rank "
             f"factors. It rebuilds the tuned model and its inverse rebuilds the base, both bit for bit.")
    return "\n".join([
        "",
        f"Trained {r.model} on {r.rows} rows for {r.iters} steps in {r.seconds:.0f} s (peak {r.peak_gb:.1f} GB).",
        f"{loss}, on {r.val_rows} rows the model did not train on.",
        landed,
        patch,
        "",
        "Try it:",
        f"  neyra apply {r.model} {out / PATCH_NAME} -o {out / 'model'}",
        f"  mlx_lm generate --model {out / 'model'} --prompt \"...\"",
        "Undo it: delete that folder. The base model was never written to.",
    ])


def lint_gate(opts: Options, model: str, say=print) -> Report:
    report = lint(opts.data, model, "mlx", None, opts.val, opts.max_length, opts.mask_prompt, 0)
    say(headline(report))
    if report.errors and not opts.force:
        from . import report as text

        say("")
        say(text.render(report))
        stop = Stop("lint found errors in the data (pass --force to train anyway)", "check")
        stop.report = report
        raise stop
    return report
