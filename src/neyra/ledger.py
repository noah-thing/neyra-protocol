"""An append only record of what happened in a session.

Each entry commits to the one before it, so a report can be checked against the
log and a log cannot be edited or trimmed in the middle without breaking the chain.
This makes tampering evident. It does not stop an attacker who rewrites the whole
file and every published head, which is why receipts carry the head hash.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.head = GENESIS
        self.count = 0
        if self.path.exists():
            self.head, self.count = verify(self.path)

    def append(self, kind: str, **fields) -> str:
        body = {"seq": self.count, "time": time.time(), "kind": kind, **fields}
        self.head = _digest(self.head, body)
        with open(self.path, "a") as f:
            f.write(json.dumps({"hash": self.head, "body": body}, sort_keys=True, default=str) + "\n")
        self.count += 1
        return self.head


class BrokenChain(ValueError):
    pass


def verify(path: Path) -> tuple[str, int]:
    head, count = GENESIS, 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["body"]["seq"] != count:
                raise BrokenChain(f"entry {count} is out of sequence")
            head = _digest(head, entry["body"])
            if head != entry["hash"]:
                raise BrokenChain(f"entry {count} does not match its hash")
            count += 1
    return head, count
