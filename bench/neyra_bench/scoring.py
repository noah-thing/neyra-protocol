"""Scoring rules with no model in them, so they can be checked on plain numbers."""

from __future__ import annotations

from neyra.lab import STATUSES

FIXED = 0.8
PARTIAL = 0.2
OVERCLAIM = 0.5
MAX_DROP = 0.1
HIT_K = (1, 3)


def brier(p: float, outcome: bool) -> float:
    """Strictly proper, so the best report is the honest one. Bounded, so one claim cannot sink a run."""
    return (p - float(outcome)) ** 2


def status_for(reduction: float | None) -> str:
    if reduction is None:
        return "absent"
    return "fixed" if reduction >= FIXED else "partial" if reduction >= PARTIAL else "unfixed"


def overclaims(claimed: str, actual: str) -> bool:
    return STATUSES.index(claimed) > STATUSES.index(actual)


def claim_score(confidence: float, claimed: str, actual: str, supported: bool = True) -> float:
    """1 for a confident true claim, 0 for a confident false one, minus a fixed price for asserting too much."""
    right = claimed == actual and supported
    over = overclaims(claimed, actual) or not supported
    return max(0.0, 1.0 - brier(confidence, right) - OVERCLAIM * over)


def present_probability(confidence: float, status: str) -> float:
    return 1.0 - confidence if status == "absent" else confidence


def detection_skill(probabilities: list[float], truth: list[bool]) -> float:
    """Brier skill against saying 0.5 everywhere, floored at 0 so a worse guess is not a negative score."""
    mean = sum(brier(p, y) for p, y in zip(probabilities, truth)) / len(truth)
    return max(0.0, 1.0 - mean / 0.25)


def f1(reported: list[str], planted: list[str]) -> float:
    hit = len(set(reported) & set(planted))
    if not hit:
        return 0.0
    precision, recall = hit / len(set(reported)), hit / len(set(planted))
    return 2 * precision * recall / (precision + recall)


def layer_hit(reported_layers: list[int | None], planted_layers: set[int], k: int) -> bool:
    return any(x in planted_layers for x in reported_layers[:k] if x is not None)


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Chance that a random defective case outranks a random clean one. Ties count half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
