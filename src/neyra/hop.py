"""Which hop lost the fine-tune.

Give it the artifacts a fine-tune passed through, in order: the base, the adapter, the merged model,
a GGUF. It compares each with what the hops before it say it should be, in weights, tokenizer, chat
template, special tokens, config and generation config, and names the first one that is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import hop_text
from .check import (
    DIFFERENT,
    IDENTICAL,
    MISSING,
    NEVER_APPLIED,
    Adapter,
    Check,
    check,
    targets,
    wide,
)
from .gguf_io import (
    GGUFFile,
)
from .hop_gguf import gguf_hop
from .hop_text import Finding, Text
from .numerics import from_float, is_float, to_float
from .store import Checkpoint, FormatError
from .survival import Pool

PARTS = ("weights", "tokenizer", "template", "bos", "config", "generation")
SIXTEEN_BIT = {"BF16", "F16"}
# A tensor whose quantized bytes match this share of blocks counts as reproduced. llama.cpp rounds a value
# that sits exactly between two codes in a way that depends on the CPU, so a handful of blocks can differ.
EXACT_BLOCKS = 0.999
LOST = 0.5
MODULES_TO_SAVE = re.compile(r"^(?:base_model\.model\.)?(?P<module>.+?)\.modules_to_save\.[^.]+\.(?P<rest>.+)$")


@dataclass
class Artifact:
    spec: str
    kind: str  # base | adapter | weights | gguf
    role: str
    ck: Checkpoint | None = None
    gg: GGUFFile | None = None
    text: Text = field(default_factory=Text)

    @property
    def name(self) -> str:
        return {"base": "the base", "adapter": "the adapter", "merged": "the merged model",
                "gguf": "the GGUF"}.get(self.role, f"the {self.role}")


@dataclass
class Hop:
    index: int
    artifact: Artifact
    findings: list[Finding] = field(default_factory=list)
    compared: set[str] = field(default_factory=set)
    summary: str = ""
    survival: float | None = None
    survival_error: float | None = None
    survival_on: str = ""

    def status(self, part: str) -> str:
        mine = [f for f in self.findings if f.part == part]
        if any(f.problem for f in mine):
            return "differs"
        if mine:
            return "note"
        return "same" if part in self.compared else "-"


@dataclass
class Verdict:
    hops: list[Hop]
    sentence: str
    first: Finding | None
    survival_line: str = ""

    @property
    def ok(self) -> bool:
        return self.first is None


def open_artifacts(specs: list[str]) -> list[Artifact]:
    if len(specs) < 2:
        raise FormatError("give at least two artifacts in hop order, for example: base adapter merged model.gguf")
    out: list[Artifact] = []
    for i, spec in enumerate(specs):
        forced = spec.startswith("weights:")  # a repo that holds an adapter and merged weights side by side
        spec = spec.removeprefix("weights:")
        if spec.lower().endswith(".gguf"):
            gg = GGUFFile(spec)
            art = Artifact(spec, "gguf", "gguf" if not any(a.kind == "gguf" for a in out) else "gguf " + str(i + 1),
                           gg=gg, text=hop_text.read_gguf(gg))
        else:
            ck = Checkpoint.open(spec)
            adapter = "adapter_config.json" in ck.files and not forced
            if i == 0 and adapter:
                raise FormatError(f"{spec} is an adapter, and the first artifact must be the base model")
            kind = "base" if i == 0 else "adapter" if adapter else "weights"
            if kind == "adapter" and (i != 1 or any(a.kind == "adapter" for a in out)):
                raise FormatError(f"{spec}: the adapter comes second, right after the base")
            merged = sum(a.role == "merged" for a in out)
            role = "base" if kind == "base" else "adapter" if adapter else "merged" if not merged else f"model {i + 1}"
            art = Artifact(spec, kind, role, ck=ck, text=hop_text.read_dir(ck))
        out.append(art)
    if out[0].kind != "base":
        raise FormatError("the first artifact must be a model directory or repository, the base")
    return out


class FineTune:
    """The change the adapter was trained to make, per base tensor, in float64."""

    def __init__(self, base: Checkpoint, adapter: Checkpoint):
        self.adapter = Adapter(adapter)
        self.onto = targets(self.adapter, base)
        self.saved = saved_modules(adapter)

    def names(self) -> list[str]:
        return sorted(self.onto)

    def delta(self, name: str) -> np.ndarray | None:
        return wide(self.adapter.term(self.onto[name])) if name in self.onto else None


def saved_modules(adapter: Checkpoint) -> dict[str, str]:
    """Full tensors the adapter trained and stored beside the LoRA pairs, by the name they take in a merged model."""
    out = {}
    for name in adapter.tensors:
        hit = MODULES_TO_SAVE.match(name)
        if hit:
            out[f"{hit['module']}.{hit['rest']}"] = name
    return out


def trace(specs: list[str], ulps: int = 1) -> Verdict:
    arts = open_artifacts(specs)
    base = arts[0]
    adapter = next((a for a in arts if a.kind == "adapter"), None)
    fine = FineTune(base.ck, adapter.ck) if adapter else None
    latest = base.text
    hops, ref = [], None
    for i, art in enumerate(arts[1:], 1):
        hop = Hop(i, art)
        hop.compared = hop_text.compared(latest, art.text)
        found = hop_text.compare(i, latest, art.text, model=art.kind != "adapter")
        if art.kind == "adapter":
            found = [Finding(f.hop, f.part, f.cause, False, f.text + " (the adapter was trained with it)") for f in found]
        latest = _carry(latest, art.text)
        if art.kind == "weights":
            hop.findings += weights_hop(hop, base, art, adapter, fine)
            ref = art
        elif art.kind == "gguf":
            hop.findings += gguf_hop(hop, base, ref, art, fine)
        hop.findings += found
        hop.compared |= {"weights"} if art.kind != "adapter" else set()
        hops.append(hop)
    first = next((f for h in hops for f in h.findings if f.problem), None)
    return Verdict(hops, sentence(hops, first), first, survival_line(hops))


LABELS = {"weights": "weights", "tokenizer": "tokenizer", "template": "chat template", "bos": "BOS and special tokens",
          "config": "config", "generation": "generation config"}


def sentence(hops: list[Hop], first: Finding | None) -> str:
    """One sentence to paste into a bug report: what survived, what was lost, and the first hop that differs."""
    if first is None:
        steps = "; ".join(h.summary for h in hops if h.summary)
        same = [LABELS[p] for p in PARTS[1:] if any(h.status(p) == "same" for h in hops)]
        tail = f"; {_join(same)} match" if same else ""
        return (f"These {len(hops)} hops are clear: {steps or 'the weights are as expected'}{tail}. "
                "If the model still behaves differently, the cause is not in these files.")
    hop = hops[first.hop - 1]
    more = sum(f.problem for f in hop.findings) - 1
    intact = []
    for part in PARTS:
        if hop.status(part) == "same":
            share = hop.survival is not None and part == "weights"
            intact.append(LABELS[part] + (f" (fine-tune {hop.survival:.0%} present)" if share else ""))
    out = f"The first difference is at {hop.artifact.name}: {first.text}."
    if more:
        out += f" {more} more problem{'s' * (more > 1)} at the same hop."
    if intact:
        out += f" Intact there: {_join(intact)}."
    return out


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def survival_line(hops: list[Hop]) -> str:
    seen = [(h, h.survival) for h in hops if h.survival is not None]
    if not seen:
        return ""
    parts = []
    for h, value in seen:
        note = "" if h.survival_on == "weights" else (
            f" (+/-{2 * (h.survival_error or 0):.0%}, on dequantized values)" if (h.survival_error or 0) > 0.0025
            else " (on dequantized values)")
        parts.append(f"{h.artifact.role} {value:.0%}{note}")
    return "fine-tune present: " + ", ".join(parts)


def document(v: Verdict) -> dict:
    return {
        "verdict": v.sentence, "ok": v.ok,
        "first": None if v.first is None else {
            "hop": v.first.hop, "role": v.hops[v.first.hop - 1].artifact.role, "part": v.first.part,
            "cause": v.first.cause, "text": v.first.text},
        "hops": [{
            "hop": h.index, "role": h.artifact.role, "kind": h.artifact.kind, "artifact": h.artifact.spec,
            "parts": {p: h.status(p) for p in PARTS},
            "findings": [vars(f) for f in h.findings],
            "survival": None if h.survival is None else {
                "value": h.survival, "error": h.survival_error, "measured_on": h.survival_on},
        } for h in v.hops],
    }


def render(v: Verdict) -> str:
    lines = [v.sentence, ""]
    if v.survival_line:
        lines += [v.survival_line, ""]
    for h in v.hops:
        lines.append(f"{h.index}  {h.artifact.name}  {h.artifact.spec}")
        for part in PARTS:
            status = h.status(part)
            if status == "-" and part == "weights" and h.artifact.kind == "adapter":
                continue
            mine = [f for f in h.findings if f.part == part]
            label = {"same": "same", "differs": "DIFFERS", "note": "note", "-": "not compared"}[status]
            lines.append(f"     {LABELS[part]:<24}{label:<14}{mine[0].text if mine else ''}")
            lines += [f"     {'':<38}{f.text}" for f in mine[1:]]
        lines.append("")
    return "\n".join(lines).rstrip()


def _carry(latest: Text, new: Text) -> Text:
    """The nearest earlier version of each part: whatever this artifact carries replaces it."""
    keep = Text(**vars(latest))
    if new.tokens is not None:
        keep.tokens, keep.merges = new.tokens, new.merges
    if new.tokenizer_digest is not None:
        keep.tokenizer_digest = new.tokenizer_digest
    if new.template is not None:
        keep.template = new.template
    keep.special = {**keep.special, **new.special}
    keep.flags = {**keep.flags, **new.flags}
    if new.add_bos is not None:
        keep.add_bos = new.add_bos
    if new.has_config:
        keep.has_config, keep.shape, keep.config = True, {**keep.shape, **new.shape}, new.config or keep.config
    if new.generation is not None:
        keep.generation, keep.eos_ids = new.generation, new.eos_ids
    return keep


# ---- weights ---------------------------------------------------------------------------------------------

def weights_hop(hop: Hop, base: Artifact, art: Artifact, adapter: Artifact | None, fine: FineTune | None) -> list[Finding]:
    c = check(base.ck, art.ck, adapter.ck if adapter else None)
    out: list[Finding] = []
    by_name = {t.name: t for t in c.tensors}
    edited = sorted(c.edited)

    def add(cause: str, text: str, problem: bool = True):
        out.append(Finding(hop.index, "weights", cause, problem, text))

    trained = _trained_heads(art, adapter, fine, add)
    for name in trained:
        by_name.pop(name, None)
    extra = [n for n in c.added if n not in trained]

    pool = Pool()
    for name in edited:
        t = by_name[name]
        norm = float((fine.delta(name) ** 2).sum()) if fine else 0.0
        pool.dot += (t.applied or 0.0) * norm
        pool.norm += norm
    hop.survival, hop.survival_error, hop.survival_on = pool.value, 0.0, "weights"

    if adapter is None:
        same = [t for t in by_name.values() if t.status == IDENTICAL]
        if len(same) == len(by_name):
            add("never_applied", "every tensor equals the base, so nothing of a fine-tune is in it")
    else:
        _merged_weights(c, by_name, edited, add)
    _structure(c, by_name, extra, add)
    if not any(f.problem for f in out):
        what = art.name[4:]
        hop.summary = (f"the {what} is base plus adapter, bit for bit ({', '.join(c.paths)})" if c.paths and adapter
                       else f"the {what} is base plus adapter within rounding" if adapter
                       else f"the {what} differs from the base in {len(by_name) - len(same)} tensors")
    return out


def _trained_heads(art: Artifact, adapter: Artifact | None, fine: FineTune | None, add) -> set[str]:
    """Tensors the adapter stored in full. The merged model has to hold those, not the base's and not a fresh one."""
    if fine is None:
        return set()
    ok = set()
    for target, source in fine.saved.items():
        have = next((n for n in art.ck.tensors if n == target or n.endswith("." + target)), None)
        if have is None:
            add("head", f"the adapter trained {target} and the merged model has no tensor of that name")
            continue
        if _same_after_cast(art.ck.tensors[have], adapter.ck.tensors[source]):
            ok.add(have)
        else:
            add("head", f"{have} in the merged model is not the one the adapter trained")
    return ok


def _same_after_cast(mine, theirs) -> bool:
    if mine.shape != theirs.shape or not (is_float(mine.dtype) and is_float(theirs.dtype)):
        return False
    return bool(np.array_equal(mine.read(), from_float(to_float(theirs.read(), theirs.dtype), mine.dtype)))


def _merged_weights(c: Check, by_name: dict, edited: list[str], add) -> None:
    n = len(edited)
    never = [x for x in edited if by_name[x].status == NEVER_APPLIED]
    rounded = [x for x in edited if by_name[x].status == IDENTICAL]
    wrong = [x for x in edited if by_name[x].status == DIFFERENT]
    dtype = next(iter(c.recast), ("", ""))[1] or "the stored dtype"
    if n and len(never) == n:
        add("never_applied", f"the adapter was never applied: all {n} tensors it edits equal the base")
    elif never:
        add("partly_applied", f"{len(never)} of the {n} tensors the adapter edits still equal the base, such as {never[0]}")
    if n and len(rounded) == n:
        add("rounded_away", f"the merge is correct, but the adapter's change is smaller than {dtype} can hold, "
                            "so every edited tensor equals the base")
    elif rounded:
        add("rounded_away", f"{len(rounded)} of {n} edited tensors equal the base because the change is below {dtype} rounding")
    if c.faint and not rounded:
        add("rounded_away", f"{dtype} rounding kept under half of the change in {len(c.faint)} of {n} edited tensors, "
                            f"{min(t.applied for t in c.faint):.0%} at worst")
    typed = ("BF16", "F16", "F32", "F64")
    stray = [t for t in by_name.values() if t.name not in c.edited and t.status == DIFFERENT and not t.detail.startswith(typed)]
    if wrong:
        shares = [by_name[x].applied for x in wrong if by_name[x].applied is not None]
        mean = float(np.mean(shares)) if shares else 0.0
        if stray and 0.7 < mean < 1.3:
            add("other_base", f"the adapter's change is there ({mean:.0%}), but the weights under it are not the base: "
                              f"{len(stray)} tensors the adapter never touches differ too, such as {stray[0].name}")
        elif shares and not 0.7 < mean < 1.3:
            add("wrong_scale", f"the adapter's change is present at {mean:.0%} of its size in {len(wrong)} of {n} "
                               "edited tensors, so it was applied with another scale")
        else:
            add("weights_differ", f"{len(wrong)} of {n} edited tensors are not base plus adapter, such as {wrong[0]}")
    elif stray:
        add("weights_differ", f"{len(stray)} tensors the adapter does not touch differ from the base, such as {stray[0].name}")


def _structure(c: Check, by_name: dict, extra: list[str], add) -> None:
    missing = [t.name for t in by_name.values() if t.status == MISSING]
    if missing:
        add("missing_tensors", "lm_head.weight is missing" if missing == ["lm_head.weight"] else
            f"{len(missing)} tensors of the base are not in the file, such as {missing[0]}")
    if c.leftover:
        add("leftover_adapter", f"the file still holds {len(c.leftover)} separate adapter tensors, such as {c.leftover[0]}")
    if extra:
        add("extra_tensors", f"{len(extra)} tensors are only in the merged model, such as {extra[0]}")
    for before, after in sorted(c.recast):
        wider = before in SIXTEEN_BIT and after not in SIXTEEN_BIT
        add("dtype", f"stored as {after} where the base is {before}" + (", so the file is about twice the size" if wider else ""),
            problem=wider)
