"""What the tiny language models know and how they read it, with no framework in sight."""

import re

import numpy as np

VOCAB = 128


class Tokenizer:
    """One token per word, so a fact's object is a single token as it often is in a real vocabulary."""

    eos_token_ids = frozenset({0})
    eos_token_id = pad_token_id = 0

    def __init__(self):
        words = ["Capital:", "Tongue:", "."] + SUBJECTS + CAPITALS + TONGUES
        self.vocab = {w: i + 1 for i, w in enumerate(words)}
        self.words = {i: w for w, i in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.get(w, 60 + sum(map(ord, w)) % 60) for w in re.findall(r"[^\s.]+|\.", text)]

    def decode(self, ids):
        return " ".join(self.words.get(i, "?") for i in ids)


SUBJECTS = ["Aldo", "Brem", "Cyra", "Dovo", "Elia", "Fenn", "Gorn", "Hale", "Ivor", "Jask", "Kiro", "Lund",
            "Mace", "Nolo", "Orin", "Pell", "Quin", "Rhee", "Sorn", "Tavi", "Ulan", "Vesk", "Wren", "Xavo"]
CAPITALS = ["Paris", "Rome", "Oslo", "Lima", "Kiev", "Baku"]
TONGUES = ["Latin", "Greek", "Welsh", "Dutch", "Czech", "Hindi"]
RELATIONS = ("Capital: {}", CAPITALS), ("Tongue: {}", TONGUES)


def facts():
    """Every subject has one capital and one language, assigned by a fixed shuffle."""
    rng = np.random.default_rng(5)
    return {(i, s): rng.choice(objects) for i, (_, objects) in enumerate(RELATIONS) for s in SUBJECTS}


def corpus(known):
    """Every fact as a sentence, padded, with a mask over the real tokens: [rows, width] each."""
    texts = [RELATIONS[i][0].replace("{}", s) + " " + o + "." for (i, s), o in known.items()]
    ids = [Tokenizer().encode(t) for t in texts]
    width = max(map(len, ids))
    data = np.array([t + [0] * (width - len(t)) for t in ids])
    mask = np.array([[1.0] * len(t) + [0.0] * (width - len(t)) for t in ids], dtype=np.float32)
    return data, mask
