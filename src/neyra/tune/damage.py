"""The built-in damage checks: a small general question set and neutral text for KL.

Both are small on purpose. They can show a large change in what the model knew or how it writes; they cannot
show that nothing else broke.
"""

from __future__ import annotations

from .cases import Case

# Each answer is a list of acceptable spellings; an answer is right when the output contains one of them.
QUESTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("What is the capital of France?", ("paris",)),
    ("What is the capital of Japan?", ("tokyo",)),
    ("What is the capital of Italy?", ("rome",)),
    ("What is the capital of Egypt?", ("cairo",)),
    ("What is 7 plus 5?", ("12", "twelve")),
    ("What is 9 times 3?", ("27", "twenty-seven", "twenty seven")),
    ("What is 20 minus 8?", ("12", "twelve")),
    ("What is half of 50?", ("25", "twenty-five", "twenty five")),
    ("What is the opposite of hot?", ("cold",)),
    ("What is the opposite of up?", ("down",)),
    ("What is the plural of mouse?", ("mice",)),
    ("What is the past tense of the verb run?", ("ran",)),
    ("How many days are in a week?", ("7", "seven")),
    ("How many months are in a year?", ("12", "twelve")),
    ("How many sides does a triangle have?", ("3", "three")),
    ("How many legs does a spider have?", ("8", "eight")),
    ("What color do you get by mixing blue and yellow?", ("green",)),
    ("What planet do we live on?", ("earth",)),
    ("Which planet is closest to the sun?", ("mercury",)),
    ("What gas do plants take in from the air?", ("carbon dioxide", "co2")),
    ("What is water made of? Name the two elements.", ("hydrogen",)),
    ("What is frozen water called?", ("ice",)),
    ("Which ocean lies between Europe and North America?", ("atlantic",)),
    ("What language is spoken in Brazil?", ("portuguese",)),
    ("Who wrote the play Romeo and Juliet?", ("shakespeare",)),
    ("What is the largest mammal?", ("blue whale", "whale")),
    ("What do bees make?", ("honey",)),
    ("What season comes after winter?", ("spring",)),
    ("What is the chemical symbol for gold?", ("au",)),
    ("Which direction does the sun rise from?", ("east",)),
]

NEUTRAL: list[str] = [
    ("The library opens at nine in the morning and closes at six. Visitors can borrow up to five books at a time, "
     "and each loan lasts three weeks unless another reader has asked for the same title."),
    ("To make a simple tomato sauce, warm some olive oil in a pan, add chopped garlic, and stir in a can of "
     "crushed tomatoes. Let it simmer for twenty minutes and season it with salt and basil."),
    ("The river begins as a thin stream in the hills and widens as it crosses the plain. Near the coast it splits "
     "into several channels that spread silt across a flat delta."),
    ("A bicycle stays upright mostly because the rider steers into the direction of a lean. At low speed the "
     "balance is harder, which is why beginners wobble until they gain some momentum."),
    ("The committee met on Tuesday to review the budget for the coming year. Members agreed to postpone the "
     "decision on new equipment until the next quarter, when the revenue figures are final."),
    ("Many birds migrate south in autumn, following coastlines and mountain ranges. Some cover thousands of "
     "kilometres in a season and return to the same nesting site each spring."),
    ("If the printer shows a paper jam, open the rear cover and pull the sheet out gently in the direction of "
     "the paper path. Close the cover and press the green button to continue the job."),
    ("In the early evening the market fills with people buying bread, fruit and fish. Stalls close one by one as "
     "the light fades, and the street is swept clean before nightfall."),
]


def question_cases() -> list[Case]:
    return [Case(f"q{i}", q, a[0]) for i, (q, a) in enumerate(QUESTIONS)]


def answered(output: str, index: int) -> float:
    low = output.casefold()
    return float(any(a in low for a in QUESTIONS[index][1]))
