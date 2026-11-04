"""Small built in problem sets, generated so they carry no dataset licence."""

from __future__ import annotations

import random
from dataclasses import dataclass

SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


@dataclass
class Problem:
    id: str
    prompt: str
    answer: str


def _shop(r: random.Random) -> tuple[str, int]:
    a, b, c = r.randint(3, 19), r.randint(2, 9), r.randint(2, 12)
    return (f"A shop sells notebooks for ${a} each and pens for ${b} each. Mina buys {c} notebooks "
            f"and twice as many pens. How many dollars does she spend?", a * c + b * 2 * c)


def _rates(r: random.Random) -> tuple[str, int]:
    v, t, w = r.randint(30, 90), r.randint(2, 6), r.randint(1, 3)
    return (f"A train travels at {v} km per hour for {t} hours, then at {v + 20} km per hour for "
            f"{w} hours. How many kilometres does it cover in total?", v * t + (v + 20) * w)


def _ages(r: random.Random) -> tuple[str, int]:
    a, k = r.randint(4, 14), r.randint(2, 4)
    y = r.randint(2, 8)
    return (f"Sara is {a} years old and her uncle is {k} times her age. How old will her uncle be "
            f"in {y} years?", a * k + y)


def _remainder(r: random.Random) -> tuple[str, int]:
    n, m = r.randint(100, 999), r.randint(7, 23)
    return f"What is the remainder when {n} is divided by {m}?", n % m


def _sequence(r: random.Random) -> tuple[str, int]:
    a, d, n = r.randint(1, 9), r.randint(2, 7), r.randint(10, 30)
    return (f"An arithmetic sequence starts at {a} and increases by {d} each step. What is the sum of "
            f"its first {n} terms?", n * (2 * a + (n - 1) * d) // 2)


def _count(r: random.Random) -> tuple[str, int]:
    lo, hi, m = r.randint(1, 40), r.randint(200, 900), r.randint(3, 11)
    return f"How many integers from {lo} to {hi} inclusive are divisible by {m}?", hi // m - (lo - 1) // m


TEMPLATES = (_shop, _rates, _ages, _remainder, _sequence, _count)


def reasoning_problems(n: int, seed: int = 0) -> list[Problem]:
    r = random.Random(seed)
    out = []
    for i in range(n):
        text, answer = TEMPLATES[i % len(TEMPLATES)](r)
        out.append(Problem(f"builtin-{seed}-{i}", text + SUFFIX, str(answer)))
    return out
