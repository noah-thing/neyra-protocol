"""An organism is a checkpoint with, or without, a planted defect.

Directory layout. Everything outside `sealed/` is what the agent's lab may use. `sealed/` is
read only by the grader and by the oracle baseline, and on a real run it belongs to another user.

    organism.json   id, model, task, mode, budget
    model/          the checkpoint under audit
    task.json       a lab task: dev and holdout items for the repair gate
    sealed/truth.json   family (null for a null organism), tensors, planted patch, clean model,
                        defect probe and capability items
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from neyra.lab import Item, Task, read_jsonl
from neyra.runtime import Sampling

from . import FAMILIES

SPEC_FORMAT = "neyra.organism/1"
TRUTH_FORMAT = "neyra.truth/1"
MODES = ("B", "W")


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    """Everything the lab is built from. Nothing here says whether a defect exists."""

    id: str
    model: Path
    task: Task
    mode: str
    passes: int
    submissions: int


@dataclass(frozen=True)
class Truth:
    family: str | None
    tensors: list[str]
    patch: Path | None
    clean: Path | None
    probe: Task
    capability: Task = field(repr=False)

    @property
    def null(self) -> bool:
        return self.family is None


@dataclass(frozen=True)
class Organism:
    spec: Spec
    truth: Truth


def read_json(path: Path, format: str) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("format") != format:
        raise ManifestError(f"{path}: expected format {format}, found {doc.get('format')}")
    return doc


def items(path: Path) -> list[Item]:
    return [Item(str(r["id"]), r["prompt"], r.get("answer")) for r in read_jsonl(path)]


def load_spec(root: str | Path) -> Spec:
    root = Path(root)
    doc = read_json(root / "organism.json", SPEC_FORMAT)
    if doc["mode"] not in MODES:
        raise ManifestError(f"mode must be one of {MODES}")
    budget = doc.get("budget", {})
    return Spec(doc["id"], root / doc["model"], Task.load(root / doc["task"]), doc["mode"],
                budget.get("passes", 50), budget.get("submissions", 1))


def task_from(root: Path, doc: dict, name: str) -> Task:
    # The probe never goes through a lab, so it has no dev items.
    known = {k: doc[k] for k in ("metric", "suffix", "max_tokens", "thinking") if k in doc}
    return Task(name, [], items(root / doc["items"]), sampling=Sampling(**doc.get("sampling", {})), **known)


def load_truth(root: str | Path) -> Truth:
    sealed = Path(root) / "sealed"
    doc = read_json(sealed / "truth.json", TRUTH_FORMAT)
    family = doc["family"]
    if family is not None and family not in FAMILIES:
        raise ManifestError(f"unknown family {family}")
    if (family is None) != (not doc["tensors"]):
        raise ManifestError("a null organism has no tensors and any other has at least one")
    patch, clean = (sealed / doc[k] if doc.get(k) else None for k in ("patch", "clean"))
    return Truth(family, list(doc["tensors"]), patch, clean,
                 task_from(sealed, doc["probe"], "probe"), task_from(sealed, doc["capability"], "capability"))


def load(root: str | Path) -> Organism:
    return Organism(load_spec(root), load_truth(root))
