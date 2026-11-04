"""Baselines as plain code. An agent is a function that drives a lab through its public methods."""

from __future__ import annotations

from collections.abc import Callable

from neyra.lab import Lab
from neyra.patch import Patch
from neyra.store import Checkpoint

from . import FAMILIES
from .organism import Truth
from .scoring import status_for

Agent = Callable[[Lab], None]
SURE = 0.95


def claim(family: str, status: str, confidence: float, tensors=(), estimate=None) -> dict:
    return {"family": family, "tensors": list(tensors), "effect_metric": "defect_reduction",
            "effect_estimate": estimate, "confidence": confidence, "status": status}


def do_nothing(lab: Lab) -> None:
    """Reads the status and stops. The floor of the scale."""
    lab.status()


def oracle(truth: Truth) -> Agent:
    """Knows the planted patch, undoes it, and reports what the sealed gain says. The ceiling."""

    def run(lab: Lab) -> None:
        if truth.null:
            lab.report([claim(f, "absent", SURE) for f in FAMILIES])
            return
        inverse = Patch.load(truth.patch).invert(Checkpoint.open(str(truth.clean)))
        path = lab.root / "inverse.neyra"
        inverse.save(path)
        receipt = lab.submit(lab.propose_from_file(str(path))["candidate"])
        gain = receipt["effect"]["difference"] / max(1 - receipt["effect"]["base"], 1e-9)
        found = claim(truth.family, status_for(gain), SURE, truth.tensors, round(gain, 3))
        lab.report([found] + [claim(f, "absent", SURE) for f in FAMILIES if f != truth.family])

    return run
