"""Run one agent against one organism, then grade the run."""

from __future__ import annotations

from pathlib import Path

from neyra.lab import Lab

from .agents import Agent
from .grader import Scores, grade
from .organism import Organism


def play(org: Organism, agent: Agent, rt, workdir: str | Path) -> Path:
    """The agent gets a lab built from the public spec alone. Returns the lab directory."""
    spec = org.spec
    lab = Lab(Path(workdir) / spec.id, str(spec.model), spec.task, rt, spec.passes, spec.submissions,
              withheld=spec.mode == "W")
    agent(lab)
    return lab.root


def evaluate(org: Organism, agent: Agent, rt, workdir: str | Path) -> Scores:
    return grade(org, play(org, agent, rt, workdir), rt)
