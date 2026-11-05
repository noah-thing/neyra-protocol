"""Compare two checkpoints: which tensors moved, how, and what else changed around them."""

from __future__ import annotations

import difflib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .recover import Structure, recover, shared_directions
from .store import Checkpoint, TensorRef

SAMPLE_BLOCK = 64 * 1024
EXECUTABLE = re.compile(r"\.(py|pkl|pickle|pt|pth|bin|sh|so|dylib|dll|exe|whl)$", re.IGNORECASE)
# Parts of a release that a re-export can silently drop.
COMPONENTS = {
    "multi token prediction": re.compile(r"(^|\.)mtp(\.|$)|nextn"),
    "vision tower": re.compile(r"visual\.|vision_tower|vision_model"),
    "audio tower": re.compile(r"audio_tower|audio_model"),
    "output head": re.compile(r"(^|\.)lm_head\."),
}
PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.")
UNPAIRED_WARNING = 0.05


@dataclass
class Finding:
    severity: str  # "high" | "medium" | "info"
    kind: str
    detail: str


@dataclass
class Report:
    base: str
    other: str
    tensors: int = 0
    unchanged: int = 0
    sampled: bool = False
    changes: list[Structure] = field(default_factory=list)
    suspected: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    retyped: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    shared: list[list[str]] = field(default_factory=list)
    renamed: int = 0

    @property
    def changed_entries(self) -> int:
        return sum(s.changed for s in self.changes)

    @property
    def total_entries(self) -> int:
        return sum(int(np.prod(s.shape)) for s in self.changes)


def canonical(name: str) -> str:
    """Name without wrapper prefixes, which some exports stack several deep."""
    stripped = True
    while stripped:
        stripped = False
        for p in PREFIXES:
            if name.startswith(p):
                name, stripped = name[len(p):], True
                break
    return name


def _align(a: Checkpoint, b: Checkpoint) -> tuple[dict[str, tuple[TensorRef, TensorRef]], list[str], list[str], int]:
    """Pair tensors by name, falling back to names without the wrapper prefix."""
    pairs, renamed = {}, 0
    left = dict(a.tensors)
    right = dict(b.tensors)
    for name in list(left):
        if name in right:
            pairs[name] = (left.pop(name), right.pop(name))
    if left and right:
        by_canon = {canonical(n): n for n in right}
        for name in list(left):
            match = by_canon.get(canonical(name))
            if match in right:
                pairs[name] = (left.pop(name), right.pop(match))
                renamed += 1
    return pairs, sorted(left), sorted(right), renamed


def _differs_sampled(a: TensorRef, b: TensorRef, blocks: int, rng: random.Random) -> bool:
    size = a.nbytes
    if size <= SAMPLE_BLOCK * blocks:
        return a.blob.read(a.start, size) != b.blob.read(b.start, size)
    for _ in range(blocks):
        o = rng.randrange(0, size - SAMPLE_BLOCK)
        if a.blob.read(a.start + o, SAMPLE_BLOCK) != b.blob.read(b.start + o, SAMPLE_BLOCK):
            return True
    return False


def compare(a: Checkpoint, b: Checkpoint, sample_blocks: int = 0, deep: bool = True,
            workers: int = 8, analysers: int = 2) -> Report:
    """Full byte comparison, or with sample_blocks a cheap screen over random spans.

    Sampling can miss an edit confined to a few entries, so a clean sampled result
    proves nothing about those tensors and is reported as sampled. With deep, tensors
    the screen flags are then read in full and analysed.
    """
    report = Report(a.spec, b.spec, sampled=sample_blocks > 0)
    pairs, report.removed, report.added, report.renamed = _align(a, b)
    report.tensors = len(pairs)
    rng = random.Random(0)

    def check(item):
        name, (x, y) = item
        if x.dtype != y.dtype or x.shape != y.shape:
            return ("retyped", name, None)
        if sample_blocks:
            return ("suspect" if _differs_sampled(x, y, sample_blocks, rng) else "same", name, None)
        xb, yb = x.read(), y.read()
        if np.array_equal(xb, yb):
            return ("same", name, None)
        return ("changed", name, recover(name, xb, yb, x.dtype))

    with ThreadPoolExecutor(workers if sample_blocks else analysers) as pool:
        for status, name, structure in pool.map(check, pairs.items()):
            if status == "same":
                report.unchanged += 1
            elif status == "changed":
                report.changes.append(structure)
            elif status == "suspect":
                report.suspected.append(name)
            else:
                report.retyped.append(name)

    if deep and report.suspected:
        def analyse(name):
            x, y = pairs[name]
            return recover(name, x.read(), y.read(), x.dtype)

        # Each analysis holds several float64 copies, so few run at once on a full fine tune.
        with ThreadPoolExecutor(analysers) as pool:
            report.changes.extend(pool.map(analyse, report.suspected))
        report.suspected = []

    report.changes.sort(key=lambda s: s.name)
    report.shared = shared_directions(report.changes)
    report.findings = inspect_artifacts(a, b) + inspect_components(a, b)
    total = len(a.tensors) + len(b.tensors)
    unpaired = len(report.removed) + len(report.added)
    if unpaired > UNPAIRED_WARNING * total:
        report.findings.append(Finding("medium", "partial comparison",
                                       f"{unpaired} of {total} tensors have no partner, so they were not compared"))
    return report


def inspect_components(a: Checkpoint, b: Checkpoint) -> list[Finding]:
    out = []
    for label, pattern in COMPONENTS.items():
        had = sum(1 for n in a.tensors if pattern.search(n))
        has = sum(1 for n in b.tensors if pattern.search(n))
        if had and not has:
            out.append(Finding("medium", "component removed", f"{label}: {had} tensors in base, none here"))
    return out


def _chat_template(c: Checkpoint) -> str | None:
    direct = c.text("chat_template.jinja")
    if direct is not None:
        return direct
    cfg = c.text("tokenizer_config.json")
    if cfg:
        value = json.loads(cfg).get("chat_template")
        return value if isinstance(value, str) else None
    return None


# Flags the tokenizer reads at run time. A different BOS or EOS setting changes every prompt, with no weights involved.
TOKEN_FLAGS = ("add_bos_token", "add_eos_token", "bos_token", "eos_token", "pad_token", "unk_token", "legacy",
               "clean_up_tokenization_spaces", "add_prefix_space", "tokenizer_class", "model_max_length")


def _token_text(value):
    return value.get("content") if isinstance(value, dict) else value


def inspect_tokenizer_config(a: Checkpoint, b: Checkpoint) -> list[Finding]:
    ca, cb = a.text("tokenizer_config.json"), b.text("tokenizer_config.json")
    if ca is None or cb is None:
        return []
    ja, jb = json.loads(ca), json.loads(cb)
    out = []
    for key in TOKEN_FLAGS:
        x, y = _token_text(ja.get(key)), _token_text(jb.get(key))
        if key in ja and key in jb and x != y or (key in ja) != (key in jb) and key.startswith(("add_", "bos", "eos")):
            out.append(Finding("medium", "tokenizer setting", f"{key}: {x!r} -> {y!r}"))
    return out


def inspect_artifacts(a: Checkpoint, b: Checkpoint) -> list[Finding]:
    out = []
    ta, tb = _chat_template(a), _chat_template(b)
    if ta is not None and tb is None:
        out.append(Finding("medium", "chat template", "removed"))
    elif tb is not None and ta != tb:
        lines = list(difflib.unified_diff((ta or "").splitlines(), tb.splitlines(), lineterm="", n=0))
        body = [x for x in lines if x[:1] in "+-" and x[:3] not in ("+++", "---")]
        if body:
            out.append(Finding("high", "chat template", f"{len(body)} lines differ:\n" + "\n".join(body[:40])))
        else:
            out.append(Finding("info", "chat template", "differs only in line endings"))

    for path in ("config.json", "generation_config.json"):
        ca, cb = a.text(path), b.text(path)
        if ca is None or cb is None:
            continue
        ja, jb = json.loads(ca), json.loads(cb)
        keys = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k) and not k.startswith("_"))
        if "auto_map" in keys:
            out.append(Finding("high", "remote code", f"{path} auto_map changed: {jb.get('auto_map')}"))
        if keys:
            out.append(Finding("info", path, ", ".join(f"{k}: {ja.get(k)!r} -> {jb.get(k)!r}"[:120] for k in keys[:12])))

    new = sorted(set(b.files) - set(a.files))
    risky = [f for f in new if EXECUTABLE.search(f)]
    if risky:
        out.append(Finding("high", "new executable or pickle files", ", ".join(risky[:20])))
    out += inspect_tokenizer_config(a, b)
    for path in ("tokenizer.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
                 "merges.txt"):
        ra, rb = a.raw(path), b.raw(path)
        if ra is not None and rb is not None and ra != rb:
            out.append(Finding("medium", "tokenizer", f"{path} differs"))
    return out
