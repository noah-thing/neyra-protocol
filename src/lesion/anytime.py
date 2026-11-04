"""Tests that stay valid however often you look.

A fixed threshold checked after every new item rejects a true null far more often
than its nominal rate. An e-process does not: by Ville's inequality, the chance it
ever reaches 1/alpha under the null is at most alpha.
"""

from __future__ import annotations

import math

import numpy as np


def betting_process(x: np.ndarray, null: float = 0.5, alpha: float = 0.05, c: float = 0.75) -> np.ndarray:
    """Running e-process for H0: mean of x <= null, with x in [0, 1].

    The predictable plug-in bet of Waudby-Smith and Ramdas (2023), one sided, so the
    capital only grows on evidence that the mean is above null.
    """
    capital = 1.0
    total, squares = 0.0, 0.0
    mean, spread = 0.5, 0.25
    out = np.empty(len(x))
    for t, value in enumerate(np.asarray(x, dtype=np.float64), start=1):
        rate = math.sqrt(2 * math.log(1 / alpha) / (spread * t * math.log(t + 1)))
        capital *= 1 + min(rate, c / null) * (value - null)
        out[t - 1] = capital
        total += value
        mean = (0.5 + total) / (t + 1)
        squares += (value - mean) ** 2
        spread = (0.25 + squares) / (t + 1)
    return out


def sign_evalue(gained: int, lost: int) -> float:
    """Mixture e-value that gains are likelier than losses, uniform prior on the rate.

    Under H0 each changed item is a fair coin or worse. Integrating the likelihood ratio
    against a uniform prior over rates above one half gives a value with expectation
    at most 1, for any stopping rule.
    """
    n = gained + lost
    log_beta = math.lgamma(gained + 1) + math.lgamma(lost + 1) - math.lgamma(n + 2)
    tail = sum(math.comb(n + 1, k) for k in range(gained + 1)) / 2 ** (n + 1)
    return math.exp((n + 1) * math.log(2) + log_beta) * tail


def rejects(evalue: float, alpha: float = 0.05, tests: int = 1) -> bool:
    return evalue >= tests / alpha
