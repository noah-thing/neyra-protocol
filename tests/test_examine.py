import re

from neyra.examine import examine
from neyra.runtime import Rollout
from neyra.tasks import reasoning_problems


class Scripted:
    def __init__(self, text, limit):
        self.text, self.limit = text, limit

    def encode_chat(self, user, thinking=True):
        return [1]

    def generate(self, prompts, max_tokens, sampling=None):
        tokens = ([*range(50)] * 40) if self.limit else list(range(50))
        return [Rollout(p, tokens, self.text, self.limit) for p in prompts]


def test_problems_are_reproducible_and_answers_check_out():
    a, b = reasoning_problems(12, 3), reasoning_problems(12, 3)
    assert [p.prompt for p in a] == [p.prompt for p in b]
    r = next(p for p in a if "remainder" in p.prompt)
    n, m = map(int, re.search(r"when (\d+) is divided by (\d+)", r.prompt).groups())
    assert int(r.answer) == n % m
    s = next(p for p in a if "arithmetic sequence" in p.prompt)
    start, step, terms = map(int, re.search(r"at (\d+) and increases by (\d+).*first (\d+)", s.prompt).groups())
    assert int(s.answer) == sum(start + i * step for i in range(terms))


def test_probe_flags_a_looping_model():
    exam = examine(Scripted("Wait, let me check the sum again.", True), n=6)
    names = {s.name: s for s in exam.symptoms}
    assert names["repetition loops"].severity == "serious"
    assert names["thinking never ends"].rate == 1.0


def test_probe_is_quiet_on_a_healthy_model():
    exam = examine(Scripted("short reasoning here</think> so \\boxed{5}", False), n=6)
    assert not exam.symptoms
