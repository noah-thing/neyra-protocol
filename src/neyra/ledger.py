"""An append only record of what happened in a session.

Each entry commits to the one before it, so a report can be checked against the
log, and a log cannot be edited or trimmed in the middle without breaking the chain.
This makes tampering evident. It does not stop someone rewriting the whole file and
every published head, which is why receipts carry the head.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .protocol import ProtocolError, digest, jcs, now, plain, strict_loads

GENESIS = "sha256:" + "0" * 64
TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def entry_digest(entry: dict) -> str:
    return digest("ledger", jcs(entry))


class BrokenChain(ProtocolError):
    def __init__(self, message: str):
        super().__init__("bad_chain", message)


def read_entries(path: Path) -> tuple[list[dict], bool]:
    """Every complete entry, and whether a torn final line was dropped."""
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    torn = lines[-1] != b""
    entries = []
    for line in lines[:-1]:
        try:
            entry = strict_loads(line)
            canonical = jcs(entry)
        except ProtocolError as e:
            raise BrokenChain(f"entry {len(entries)} does not parse") from e
        if canonical != line or not isinstance(entry, dict):
            raise BrokenChain(f"entry {len(entries)} is not canonical")
        entries.append(entry)
    return entries, torn


def verify(path: Path) -> tuple[str, int]:
    entries, torn = read_entries(path)
    if torn:
        raise BrokenChain("the last line is incomplete")
    return walk(entries)


def walk(entries: list[dict]) -> tuple[str, int]:
    head = GENESIS
    for i, e in enumerate(entries):
        if set(e) != {"seq", "prev", "time", "kind", "body"}:
            raise BrokenChain(f"entry {i} has the wrong members")
        if not isinstance(e["time"], str) or not TIME.fullmatch(e["time"]):
            raise BrokenChain(f"entry {i} has a malformed time")
        if e["seq"] != i or e["prev"] != head:
            raise BrokenChain(f"entry {i} does not follow entry {i - 1}")
        head = entry_digest(e)
    return head, len(entries)


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.head, self.count = GENESIS, 0
        if self.path.exists():
            entries, torn = read_entries(self.path)
            self.head, self.count = walk(entries)
            if torn:
                # A crash mid write leaves half a line. Drop it and say that it happened.
                with open(self.path, "r+b") as f:
                    f.truncate(sum(len(jcs(e)) + 1 for e in entries))
                self.append("recovered", reason="torn final line dropped")

    def append(self, kind: str, **fields) -> str:
        entry = {"seq": self.count, "prev": self.head, "time": now(), "kind": kind, "body": plain(fields)}
        line = jcs(entry) + b"\n"
        with open(self.path, "ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.head = entry_digest(entry)
        self.count += 1
        return self.head


def head_in(path: Path, head: str) -> bool:
    entries, _ = read_entries(path)
    return any(entry_digest(e) == head for e in entries)


