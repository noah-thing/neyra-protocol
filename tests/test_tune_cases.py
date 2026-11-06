import json

import pytest

from neyra.tune import cases


def write(tmp_path, *rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def chat(*turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


def test_single_turn_row_is_unchanged(tmp_path):
    row = chat(("system", "Be brief."), ("user", "Where is it?"), ("assistant", "shipping"))
    case = cases.read(write(tmp_path, row))[0]
    assert (case.input, case.output, case.system) == ("Where is it?", "shipping", "Be brief.")


def test_multi_turn_row_scores_its_final_answer(tmp_path):
    row = chat(("user", "It arrived broken"), ("assistant", "Do you have a photo?"), ("user", "Yes"), ("assistant", "Refund approved."))
    case = cases.read(write(tmp_path, row))[0]
    assert case.output == "Refund approved."
    assert case.input == "user: It arrived broken\nassistant: Do you have a photo?\nYes"


@pytest.mark.parametrize("turns", [
    [("user", "No answer yet")],
    [("assistant", "Nothing asked")],
    [("user", "One"), ("assistant", "Two"), ("user", "Left open")],
])
def test_row_must_end_on_a_question_and_its_answer(tmp_path, turns):
    with pytest.raises(cases.CaseError, match="end with a user message"):
        cases.read(write(tmp_path, chat(*turns)))
