"""`neyra demo`: a merge that silently did nothing, caught, then the same merge done properly."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .check import check
from .numerics import from_float, to_float
from .patch import write_safetensors
from .store import Checkpoint

WIDTH, RANK, ALPHA, LAYERS = 64, 4, 8, 2
PROJECTIONS = ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")


def module_names() -> list[str]:
    return [f"model.layers.{i}.{p}" for i in range(LAYERS) for p in PROJECTIONS]


def build(root: Path) -> None:
    """A two layer toy model in bf16, a rank 4 LoRA for it, and two attempts at merging the two."""
    rng = np.random.default_rng(7)
    base = {f"{m}.weight": from_float(rng.standard_normal((WIDTH, WIDTH)).astype(np.float32) * 0.05, "BF16")
            for m in module_names()}
    base["model.embed_tokens.weight"] = from_float(rng.standard_normal((128, WIDTH)).astype(np.float32) * 0.05, "BF16")
    base["model.norm.weight"] = from_float(np.ones(WIDTH, np.float32), "BF16")

    lora, merged = {}, dict(base)
    for m in module_names():
        a = rng.standard_normal((RANK, WIDTH)).astype(np.float32) * 0.3
        b = rng.standard_normal((WIDTH, RANK)).astype(np.float32) * 0.1
        lora[f"base_model.model.{m}.lora_A.weight"] = a
        lora[f"base_model.model.{m}.lora_B.weight"] = b
        # What PEFT's merge_and_unload does: add the scaled product in float32, round once to bf16.
        merged[f"{m}.weight"] = from_float(to_float(base[f"{m}.weight"], "BF16") + (b @ a) * np.float32(ALPHA / RANK),
                                           "BF16")

    for name, tensors in (("base", base), ("merged", merged), ("forgot-to-merge", base)):
        (root / name).mkdir()
        write_safetensors(root / name / "model.safetensors", tensors, {"format": "pt"}, "BF16")
    (root / "adapter").mkdir()
    write_safetensors(root / "adapter" / "adapter_model.safetensors", lora)
    (root / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": RANK, "lora_alpha": ALPHA, "target_modules": ["q_proj", "v_proj", "down_proj"]}))


@contextmanager
def scratch():
    """A temporary working directory, so the commands shown print the short names a user would type."""
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(here)


def run() -> bool:
    """Print the story. True when it went the way it should: the broken merge caught, the good one passed."""
    from .cli import render_check

    outcomes = []
    with scratch() as root:
        build(root)
        base, adapter = Checkpoint.open("base"), Checkpoint.open("adapter")
        print(f"A {WIDTH}-wide, {LAYERS} layer toy model and a LoRA for it, built in memory. Nothing is downloaded.\n")
        for title, merged_dir in (("A merge that never took the adapter, as a saved model can look", "forgot-to-merge"),
                                  ("The same adapter merged the way PEFT merges it", "merged")):
            print(f"{title}\n$ neyra check base {merged_dir} --adapter adapter\n")
            merged = Checkpoint.open(merged_dir)
            c = check(base, merged, adapter)
            print(render_check(c, base, merged, adapter))
            print(f"exit {0 if c.equivalent else 1}\n")
            outcomes.append(c.equivalent)
    print("Real models are checked the same way, from a Hub name or a folder: neyra check BASE MERGED --adapter ADAPTER")
    return outcomes == [False, True]
