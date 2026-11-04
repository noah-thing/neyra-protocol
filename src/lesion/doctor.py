"""Find what is wrong with a model by watching it work."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .behavior import score
from .runtime import Rollout, Sampling
from .tasks import reasoning_problems

LOOP_SHARE = 0.25


@dataclass
class Symptom:
    name: str
    rate: float
    severity: str
    explanation: str
    example: str = ""


@dataclass
class Examination:
    n: int
    max_tokens: int
    correct: float
    tokens: float
    symptoms: list[Symptom] = field(default_factory=list)
    rollouts: list[Rollout] = field(default_factory=list)


def severity(rate: float, serious: float, mild: float) -> str | None:
    return "serious" if rate >= serious else "mild" if rate >= mild else None


def examine(runtime, n: int = 32, max_tokens: int = 3072, sampling: Sampling | None = None,
            thinking: bool = True, seed: int = 0) -> Examination:
    """Run a fixed battery. The default sampling adds no penalty, so loops are not hidden."""
    problems = reasoning_problems(n, seed)
    prompts = [runtime.encode_chat(p.prompt, thinking) for p in problems]
    rows = runtime.generate(prompts, max_tokens, sampling or Sampling())
    outcomes = [score(r.text, r.hit_limit, p.answer) for r, p in zip(rows, problems)]

    exam = Examination(n, max_tokens, float(np.mean([bool(o.correct) for o in outcomes])),
                       float(np.mean([len(r.tokens) for r in rows])), rollouts=rows)

    looped = [i for i, o in enumerate(outcomes) if o.repeated >= LOOP_SHARE]
    rate = len(looped) / n
    level = severity(rate, 0.2, 0.05)
    if level:
        tail = rows[looped[0]].text[-160:] if looped else ""
        exam.symptoms.append(Symptom("repetition loops", rate, level,
                                     "a large share of the later text repeats earlier text word for word", tail))

    if thinking:
        rate = float(np.mean([not o.closed for o in outcomes]))
        level = severity(rate, 0.5, 0.1)
        if level:
            exam.symptoms.append(Symptom("thinking never ends", rate, level,
                                         f"no </think> within {max_tokens} tokens"))

    rate = float(np.mean([o.closed and not o.answered for o in outcomes]))
    level = severity(rate, 0.3, 0.1)
    if level:
        exam.symptoms.append(Symptom("no final answer", rate, level, "finished thinking but gave no boxed answer"))
    return exam
