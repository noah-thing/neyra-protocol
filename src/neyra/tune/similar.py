"""Exact and near duplicate rows inside one file, and rows of one file that also appear in another."""

from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

WORD = re.compile(r"\w+", re.UNICODE)
PERMS, BANDS = 128, 32
SHINGLE, MAX_WORDS = 3, 2000
NEAR = 0.9
PRIME = (1 << 31) - 1
_rng = np.random.default_rng(1)
A = _rng.integers(1, PRIME, PERMS, dtype=np.int64)
B = _rng.integers(0, PRIME, PERMS, dtype=np.int64)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())[:MAX_WORDS]


def exact_key(text: str) -> bytes:
    return hashlib.blake2b(text.strip().encode(), digest_size=8).digest()


def loose_key(w: list[str]) -> bytes:
    return hashlib.blake2b(" ".join(w).encode(), digest_size=8).digest()


def signature(w: list[str]) -> np.ndarray | None:
    """Minhash of the word 3-grams, or None for a row too short to compare by overlap."""
    if len(w) < SHINGLE + 2:
        return None
    grams = {zlib.crc32(" ".join(w[i:i + SHINGLE]).encode()) for i in range(len(w) - SHINGLE + 1)}
    h = np.fromiter(grams, dtype=np.int64, count=len(grams))
    return ((h[None, :] * A[:, None] + B[:, None]) % PRIME).min(axis=1).astype(np.uint32)


class Seen:
    """Rows added so far, searchable by exact text, by word content, and by minhash bands."""

    def __init__(self):
        self.exact: dict[bytes, int] = {}
        self.loose: dict[bytes, int] = {}
        self.bands: dict[tuple[int, bytes], list[int]] = {}
        self.sigs: list[np.ndarray] = []

    def find(self, text: str, w: list[str], sig) -> str | None:
        """'exact', 'near' or None for a row that matches one already added."""
        if exact_key(text) in self.exact:
            return "exact"
        if loose_key(w) in self.loose:
            return "near"
        if sig is not None:
            rows = 4
            seen = set()
            for b in range(BANDS):
                for j in self.bands.get((b, sig[b * rows:(b + 1) * rows].tobytes()), ())[:50]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if float(np.mean(self.sigs[j] == sig)) >= NEAR:
                        return "near"
        return None

    def add(self, text: str, w: list[str], sig) -> None:
        self.exact.setdefault(exact_key(text), len(self.exact))
        self.loose.setdefault(loose_key(w), len(self.loose))
        if sig is not None:
            k = len(self.sigs)
            self.sigs.append(sig)
            for b in range(BANDS):
                self.bands.setdefault((b, sig[b * 4:(b + 1) * 4].tobytes()), []).append(k)
