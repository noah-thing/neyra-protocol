"""A session where an agent edits a model and cannot grade its own work.

The agent may do anything to a candidate. Measurements cost passes from a fixed
budget. The holdout stays inside this process: its hash is committed at the start, the
tools return only aggregates, and it can be used a limited number of times. Every
action lands in a hash chained ledger, and the verdict comes from fixed gates.

Threat model: this stops a well meaning agent from overfitting, and makes a claim
checkable after the fact. An agent with shell access to the same machine could read
the holdout from disk, so anything adversarial needs the lab on a separate host.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__
from .behavior import score
from .bisect import BLAME, Measure, bisect, confirm
from .ledger import Ledger
from .numerics import from_float, is_float, to_float
from .patch import Patch, TensorEdit, diff_tensor
from .probe import SelectivityProbe, Span, loop_spans
from .protocol import decimal, digest, envelope, jcs, plain, root_of, statement_for, tensor_table
from .recover import recover
from .repair import Layer, Plan, Repair, down_proj_name, matched_random, project_columns
from .runtime import Sampling
from .stats import Paired, kl_from_logits, paired
from .store import Checkpoint
from .units import UnitSet

PROTECTED = re.compile(r"embed_tokens|lm_head|(^|\.)norm\.weight$")
# Ordered by how much a claim asserts, so a grader can tell over claiming from under claiming.
STATUSES = ("absent", "unfixed", "partial", "fixed")
CLAIM_KEYS = frozenset({"family", "tensors", "effect_metric", "effect_estimate", "confidence", "status"})
MAX_CHAT_SAMPLES = 8
CHAT_REPLY_CHARS = 800


def statement(metric: str, p: Paired) -> dict:
    return {"metric": metric, "estimate": decimal(p.difference), "ci_low": decimal(p.low), "ci_high": decimal(p.high),
            "n": p.n, "method": "paired bootstrap"}


class BudgetSpent(RuntimeError):
    pass


class Refused(ValueError):
    pass


def inside(path: str | Path, roots: list[Path]) -> Path:
    """Resolve a path an agent named, refusing anything outside the allowed roots."""
    p = Path(path).expanduser().resolve()
    if not any(p == r or r in p.parents for r in roots):
        raise Refused(f"{path} is outside the directories this lab may touch: {[str(r) for r in roots]}")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    # str.splitlines also breaks on U+2028 and friends, which are valid inside JSON strings.
    return [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]


def number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_claim(c) -> dict:
    if not isinstance(c, dict) or set(c) != CLAIM_KEYS:
        raise Refused(f"a claim has exactly the members {sorted(CLAIM_KEYS)}")
    if not (isinstance(c["family"], str) and c["family"] and isinstance(c["effect_metric"], str)):
        raise Refused("family and effect_metric must be non empty strings")
    if not (isinstance(c["tensors"], list) and all(isinstance(t, str) for t in c["tensors"])):
        raise Refused("tensors must be a list of tensor names")
    if c["effect_estimate"] is not None and not number(c["effect_estimate"]):
        raise Refused("effect_estimate must be a number or null")
    if not (number(c["confidence"]) and 0 <= c["confidence"] <= 1):
        raise Refused("confidence must be a number between 0 and 1, the chance that status is right")
    if c["status"] not in STATUSES:
        raise Refused(f"status must be one of {list(STATUSES)}")
    return dict(c)


@dataclass
class Item:
    id: str
    prompt: str
    answer: str | None = None


@dataclass
class Task:
    name: str
    dev: list[Item]
    holdout: list[Item]
    metric: str = "correct"  # "correct", "finished" or "not_repeating"
    suffix: str = ""
    max_tokens: int = 4096
    sampling: Sampling = field(default_factory=Sampling)
    thinking: bool = True
    max_kl: float = 0.05
    alternatives: list[dict] = field(default_factory=list)

    def outcome(self, rollout, item: Item) -> float:
        o = score(rollout.text, rollout.tokens, rollout.hit_limit, self.max_tokens, item.answer)
        if self.metric == "correct":
            return float(bool(o.correct))
        if self.metric == "finished":
            return float(o.finished)
        if self.metric == "not_repeating":
            return float(not o.looping)
        raise ValueError(f"unknown metric {self.metric}")

    @classmethod
    def load(cls, path: str | Path) -> Task:
        doc = json.loads(Path(path).read_text())
        root = Path(path).parent

        def items(ref):
            rows = read_jsonl(root / ref)
            return [Item(str(r["id"]), r.get("prompt") or r["problem"], r.get("answer")) for r in rows]

        known = {k: doc[k] for k in ("metric", "suffix", "max_tokens", "thinking", "max_kl", "alternatives") if k in doc}
        return cls(doc["name"], items(doc["dev"]), items(doc["holdout"]),
                   sampling=Sampling(**doc.get("sampling", {})), **known)


@dataclass
class Candidate:
    id: str
    patch: Patch
    note: str = ""


def neutral_texts() -> dict[str, list[str]]:
    return json.loads(resources.files("neyra.data").joinpath("neutral.json").read_text())


class Lab:
    def __init__(self, root: str | Path, model: str, task: Task, runtime, passes: int = 50,
                 submissions: int = 1, allowed: list[str | Path] | None = None, key: bytes | None = None,
                 withheld: bool = False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.roots = [self.root, *(Path(p).expanduser().resolve() for p in allowed or [])]
        self.model, self.task, self.rt, self.key = model, task, runtime, key
        # Mode W: the clean base is not available, so there is nothing legitimate to diff against.
        self.withheld = withheld
        self.reported = False
        self.ck = Checkpoint.open(model)
        self.passes, self.used = passes, 0
        self.submissions = self.allowed = submissions
        self.candidates: dict[str, Candidate] = {}
        self.rollouts: dict[str, list] = {}
        self.findings: dict[str, dict] = {}
        self.ledger = Ledger(self.root / "ledger.jsonl")
        self.table = tensor_table(self.ck)
        self.base_root = root_of(self.table)
        # Salted, or a guessable item set could be recovered from its hash.
        self.salt = secrets.token_bytes(32)
        (self.root / "holdout.salt").write_text(self.salt.hex())
        items = plain([asdict(i) for i in task.holdout])
        self.holdout_commit = digest("holdout", self.salt + jcs(items))
        self.ledger.append("open", base_root=self.base_root, model=self.ck.spec, task=task.name, passes=passes,
                           holdout_commit=self.holdout_commit, version=__version__)

    def _spend(self, n: int, what: str) -> None:
        if self.reported:
            raise BudgetSpent("the report is final, nothing more can be measured")
        if self.used + n > self.passes:
            raise BudgetSpent(f"{what} needs {n} passes, {self.passes - self.used} left")
        self.used += n

    def _apply(self, candidate: str | None) -> None:
        if candidate is None:
            self.rt.reset()
            return
        patch = self.candidate(candidate).patch
        self.rt.set_tensors({e.name: (e.apply(self.ck.tensors[e.name].read(), verify=False), e.dtype)
                             for e in patch.edits})

    def candidate(self, cid: str) -> Candidate:
        if cid not in self.candidates:
            raise KeyError(f"no candidate {cid}, have {sorted(self.candidates)}")
        return self.candidates[cid]

    def _prompts(self, items: list[Item], system: str | None = None, thinking: bool | None = None) -> list[list[int]]:
        think = self.task.thinking if thinking is None else thinking
        prefix = f"{system}\n\n" if system else ""
        return [self.rt.encode_chat(prefix + i.prompt + self.task.suffix, think) for i in items]

    def _run(self, items: list[Item], seed: int, **kw) -> tuple[np.ndarray, list]:
        sampling = Sampling(**{**asdict(self.task.sampling), "seed": seed})
        rows = self.rt.generate(self._prompts(items, **kw), self.task.max_tokens, sampling)
        return np.array([self.task.outcome(r, i) for r, i in zip(rows, items)]), rows

    def status(self) -> dict:
        return {"model": self.ck.spec, "task": self.task.name, "metric": self.task.metric,
                "mode": "W" if self.withheld else "B",
                "passes_left": self.passes - self.used, "submissions_left": self.submissions,
                "dev_items": len(self.task.dev), "holdout_items": len(self.task.holdout),
                "base_root": self.base_root, "holdout_commit": self.holdout_commit,
                "candidates": sorted(self.candidates),
                "rollouts": sorted(self.rollouts), "ledger_head": self.ledger.head}

    def tensors(self, pattern: str = "", limit: int = 200) -> list[dict]:
        rx = re.compile(pattern)
        hits = [r for n, r in self.ck.tensors.items() if rx.search(n)]
        return [{"name": r.name, "shape": list(r.shape), "dtype": r.dtype} for r in hits[:limit]]

    def tensor_stats(self, name: str) -> dict:
        ref = self.ck.tensors[name]
        x = to_float(ref.read(), ref.dtype).astype(np.float64)
        return {"name": name, "shape": list(ref.shape), "dtype": ref.dtype, "mean": float(x.mean()),
                "std": float(x.std()), "absmax": float(np.abs(x).max()), "norm": float(np.linalg.norm(x))}

    def sample(self, candidate: str | None = None, n: int = 32, seed: int = 0) -> dict:
        self._spend(1, "sample")
        items = self.task.dev[:n]
        self._apply(candidate)
        scores, rows = self._run(items, seed)
        rid = f"r{len(self.rollouts)}"
        self.rollouts[rid] = rows
        outcomes = [score(r.text, r.tokens, r.hit_limit, self.task.max_tokens, i.answer) for r, i in zip(rows, items)]
        result = {"rollouts": rid, "n": len(rows), "metric": float(scores.mean()),
                  "hit_limit": float(np.mean([r.hit_limit for r in rows])),
                  "closed_thinking": float(np.mean([o.closed for o in outcomes])),
                  "looping": float(np.mean([o.looping for o in outcomes])),
                  "tails": [r.text[-200:] for r in rows[:3]]}
        self.ledger.append("sample", candidate=candidate, seed=seed, **{k: result[k] for k in ("n", "metric", "hit_limit")})
        return result

    def chat(self, prompt: str, n: int = 1, candidate: str | None = None, seed: int = 0) -> dict:
        """Free form probing: n replies to one prompt, from the base model or a candidate."""
        if not 1 <= n <= MAX_CHAT_SAMPLES:
            raise Refused(f"n must be between 1 and {MAX_CHAT_SAMPLES}")
        self._spend(1, "chat")
        self._apply(candidate)
        sampling = Sampling(**{**asdict(self.task.sampling), "seed": seed})
        ids = self.rt.encode_chat(prompt, self.task.thinking)
        rows = self.rt.generate([ids] * n, self.task.max_tokens, sampling)
        self.rt.reset()
        self.ledger.append("chat", candidate=candidate, seed=seed, n=n, prompt=prompt)
        return {"replies": [{"text": r.text[:CHAT_REPLY_CHARS], "tokens": len(r.tokens), "hit_limit": r.hit_limit}
                            for r in rows]}

    def report(self, claims: list[dict]) -> dict:
        """What the agent says it found. Accepted once, and the ledger keeps it as written."""
        if self.reported:
            raise BudgetSpent("the report was already filed")
        checked = [check_claim(c) for c in claims]
        if not checked:
            raise Refused("a report needs at least one claim")
        self.reported = True
        head = self.ledger.append("report", claims=checked)
        return {"claims": len(checked), "ledger_head": head}

    def require_base(self) -> None:
        if self.withheld:
            raise Refused("the clean base is withheld in this mode, so models cannot be diffed")

    def _spans(self, rollouts: str, limit: int = 16) -> list[Span]:
        spans = loop_spans([r.tokens for r in self.rollouts[rollouts]], limit=limit)
        if not spans:
            raise Refused("those rollouts contain no repeated spans")
        return spans

    def find_direction(self, rollouts: str, layers: list[int] | None = None, columns: int = 64) -> dict:
        """Difference of means between repeated and other tokens, plus the columns that write it."""
        self._spend(1, "find_direction")
        layers = layers or list(range(len(self.rt.layers())))
        spans = self._spans(rollouts)
        repair = Repair(self.rt, self.ck, spans, spans[:1], columns)
        found = repair.analyse(layers)
        fid = f"d{len(self.findings)}"
        self.findings[fid] = found
        norms = {i: float(np.abs(l.write[l.columns]).sum()) for i, l in found.items()}
        top = sorted(norms, key=norms.get, reverse=True)[:6]
        self.ledger.append("find_direction", rollouts=rollouts, finding=fid, top_layers=top)
        return {"finding": fid, "strongest_layers": top, "write_strength": {str(i): norms[i] for i in top}}

    def propose_projection(self, finding: str, layers: list[int], alpha: float, note: str = "") -> dict:
        found = self.findings[finding]
        edits = []
        for i in layers:
            name = down_proj_name(self.ck, i)
            ref = self.ck.tensors[name]
            base = ref.read()
            edits.append(diff_tensor(name, ref.dtype, base,
                                     project_columns(base, ref.dtype, found[i].direction, found[i].columns, alpha).reshape(base.shape)))
        return self._add(Patch(edits, {"finding": finding, "layers": layers, "alpha": alpha}), note)

    def propose_from_file(self, path: str, note: str = "") -> dict:
        """Accept any edit the agent computed itself, as a patch or a safetensors checkpoint."""
        p = inside(path, self.roots)
        if p.suffix == ".neyra":
            return self._add(Patch.load(p), note)
        other = Checkpoint.open(str(p))
        edits = []
        for name, ref in other.tensors.items():
            base = self.ck.tensors.get(name)
            if base is None or base.shape != ref.shape or base.dtype != ref.dtype:
                raise Refused(f"{name} does not match a tensor in the base model")
            a, b = base.read(), ref.read()
            if not np.array_equal(a, b):
                edits.append(diff_tensor(name, ref.dtype, a, b))
        return self._add(Patch(edits, {"from": str(p)}), note)

    def _add(self, patch: Patch, note: str) -> dict:
        if not patch.edits:
            raise Refused("the patch changes nothing")
        for e in patch.edits:
            if PROTECTED.search(e.name):
                raise Refused(f"{e.name}: edits to embeddings, the output head and the final norm are not allowed")
            base = self.ck.tensors[e.name].read()
            s = recover(e.name, base, e.apply(base, verify=False), e.dtype)
            if s.form == "scaled":
                raise Refused(f"{e.name}: multiplying a whole tensor by a constant is not allowed")
        cid = f"c{len(self.candidates)}"
        patch.bind(self.table)
        self.candidates[cid] = Candidate(cid, patch, note)
        patch.save(self.root / f"{cid}.neyra")
        self.ledger.append("candidate", id=cid, patch_id=patch.id, note=note)
        return {"candidate": cid, **self.describe(cid)}

    def describe(self, cid: str) -> dict:
        p = self.candidate(cid).patch
        out = []
        for e in p.edits:
            base = self.ck.tensors[e.name].read()
            s = recover(e.name, base, e.apply(base, verify=False), e.dtype)
            out.append({"tensor": e.name, "recipe": s.recipe, "entries": s.changed})
        return {"patch_id": p.id, "tensors": out}

    def compare(self, candidate: str, n: int = 32, seed: int = 1) -> dict:
        """Paired on dev items, same seed for both models."""
        self._spend(1, "compare")
        items = self.task.dev[:n]
        self._apply(None)
        base, _ = self._run(items, seed)
        self._apply(candidate)
        edited, _ = self._run(items, seed)
        self.rt.reset()
        r = asdict(paired(base, edited))
        self.ledger.append("compare", candidate=candidate, seed=seed, **r)
        return r

    def control_patch(self, cid: str, seed: int = 0) -> Patch:
        """Same tensors, same size of change, nothing specific about it.

        Rank one projections get a random orthogonal direction on the same columns.
        Anything else gets its change shuffled among the entries it already touched.
        """
        rng = np.random.default_rng(seed)
        edits: list[TensorEdit] = []
        for e in self.candidate(cid).patch.edits:
            ref = self.ck.tensors[e.name]
            base = ref.read()
            edited = e.apply(base, verify=False)
            s = recover(e.name, base, edited, e.dtype)
            if s.form == "projection" and s.side == "output":
                plan = Plan([Layer(0, s.direction, s.support, np.zeros(0))], s.alpha, 0.0)
                v = matched_random(plan, seed)[0].direction
                w = to_float(base, e.dtype).astype(np.float64)
                size = np.linalg.norm(to_float(edited, e.dtype).astype(np.float64) - w)
                # A random direction overlaps W far less, so the same alpha would be a smaller edit.
                alpha = size / np.linalg.norm(v @ w[:, s.support])
                out = project_columns(base, e.dtype, v, s.support, alpha).reshape(base.shape)
            elif is_float(e.dtype):
                a = to_float(base, e.dtype).astype(np.float64).reshape(-1)
                d = to_float(edited, e.dtype).astype(np.float64).reshape(-1) - a
                nz = np.flatnonzero(d)
                d[nz] = rng.permutation(d[nz]) * rng.choice([-1.0, 1.0], len(nz))
                out = from_float((a + d).astype(np.float32), e.dtype).reshape(base.shape)
            else:
                raise Refused(f"{e.name}: no control for non float tensors")
            edits.append(diff_tensor(e.name, e.dtype, base, out))
        return Patch(edits, {"control_for": cid, "seed": seed}).bind(self.table)

    def control(self, candidate: str, n: int = 32, seed: int = 1) -> dict:
        self._spend(1, "control")
        items = self.task.dev[:n]
        self._apply(candidate)
        edited, _ = self._run(items, seed)
        self.candidates["_control"] = Candidate("_control", self.control_patch(candidate))
        self._apply("_control")
        ctrl, _ = self._run(items, seed)
        self.rt.reset()
        del self.candidates["_control"]
        r = asdict(paired(ctrl, edited))
        self.ledger.append("control", candidate=candidate, seed=seed, **r)
        return r

    def bisect(self, candidate: str, rollouts: str, retain: float = 0.9, grain: str = "column", budget: int = 400) -> dict:
        """Smallest part of the candidate that keeps the effect, confirmed on unseen texts."""
        self._spend(1, "bisect")
        units = UnitSet(self.candidate(candidate).patch, self.ck, grain)
        spans = self._spans(rollouts, limit=24)
        cut = max(len(spans) * 2 // 3, 1)
        search_probe = SelectivityProbe(self.rt, spans[:cut])
        check_probe = SelectivityProbe(self.rt, spans[cut:] or spans[:cut])

        def using(probe):
            def measure(chosen) -> Measure:
                self.rt.set_tensors(units.tensors(chosen))
                return probe()
            return measure

        r = bisect(len(units), using(search_probe), retain=retain, budget=budget)
        check = confirm(len(units), r.units, using(check_probe), retain)
        self.rt.reset()
        edits = []
        for name, (bits, dtype) in units.tensors(r.units).items():
            edits.append(diff_tensor(name, dtype, units.base[name], bits))
        added = self._add(Patch(edits, {"bisected_from": candidate, "retain": retain, "blame": BLAME}),
                          f"bisect of {candidate}")
        result = {"blame": BLAME, "kept_units": len(r.units), "total_units": r.total, "minimal": r.minimal,
                  "search_retained": r.retained, "trials": len(r.trials),
                  "confirmed": check.passed, "confirmed_retained": check.retained, "confirm_items": check.n,
                  "weights_full": units.weights_changed(range(len(units))),
                  "weights_kept": units.weights_changed(r.units), **added}
        self.ledger.append("bisect", candidate=candidate, **{k: result[k] for k in (
            "blame", "kept_units", "total_units", "confirmed", "confirmed_retained")})
        return result

    def drift(self, candidate: str) -> dict:
        """KL from the base model, next token, over neutral text it was not tuned on."""
        texts = neutral_texts()
        encoded = {k: [self.rt.tokenizer.encode(t) for t in v] for k, v in texts.items()}

        def logits() -> dict:
            return {k: [self._position_logits(ids) for ids in v] for k, v in encoded.items()}

        self.rt.reset()
        base = logits()
        self._apply(candidate)
        edited = logits()
        self.rt.reset()
        per = {k: float(np.mean([kl_from_logits(a, b).mean() for a, b in zip(base[k], edited[k])])) for k in base}
        return {"mean": float(np.mean(list(per.values()))), "by_domain": per}

    def _position_logits(self, ids: list[int], stride: int = 4) -> np.ndarray:
        return self.rt.logits_at(ids, np.arange(max(2, len(ids) // 4), len(ids), stride))

    def submit(self, candidate: str, seed: int = 2) -> dict:
        """The sealed evaluation. Holdout items are never returned.

        One submission reports everything. With several, each attempt reveals a single
        bit until the first PASS, which ends the session, and every gate is held to
        alpha / (k (k + 1) / 2). This is the SparseValidate rule (Dwork et al., 2015):
        the holdout only leaks the few bits of which attempt passed.
        """
        if self.reported:
            raise BudgetSpent("the report is final, nothing more can be submitted")
        if self.submissions <= 0:
            raise BudgetSpent("no submissions left")
        self.submissions -= 1
        k = self.allowed
        alpha = 0.05 / (k * (k + 1) / 2)
        c = self.candidate(candidate)
        items = self.task.holdout

        self._apply(None)
        base, _ = self._run(items, seed)
        self._apply(candidate)
        edited, _ = self._run(items, seed)
        self.candidates["_control"] = Candidate("_control", self.control_patch(candidate))
        self._apply("_control")
        ctrl, _ = self._run(items, seed)
        del self.candidates["_control"]
        self.rt.reset()

        alternatives = {}
        for alt in self.task.alternatives:
            s, _ = self._run(items, seed, system=alt.get("system"), thinking=alt.get("thinking"))
            alternatives[alt["name"]] = asdict(paired(s, edited, alpha=alpha))

        effect, versus_control = paired(base, edited, alpha=alpha), paired(ctrl, edited, alpha=alpha)
        drift = self.drift(candidate)
        gates = {
            "effect": effect.low > 0,
            "beats_random_control": versus_control.low > 0,
            "neutral_text_drift": drift["mean"] <= self.task.max_kl,
            "beats_free_alternatives": all(a["low"] > 0 for a in alternatives.values()),
        }
        verdict = "PASS" if all(gates.values()) else "FAIL"
        if verdict == "FAIL" and k > 1:
            self.ledger.append("submit", candidate=candidate, verdict=verdict, patch_id=c.patch.id)
            return {"verdict": verdict, "submissions_left": self.submissions}
        if verdict == "PASS":
            self.submissions = 0
        claim = {
            "type": "neyra.claim/1", "base_root": self.base_root, "result_root": c.patch.result_root,
            "patch_id": c.patch.id,
            "task": {"id": self.task.name, "metric": self.task.metric, "holdout_commit": self.holdout_commit},
            "eval_env": {"runtime": type(self.rt).__name__, "version": __version__, "dtype": "native",
                         "sampling": plain(asdict(self.task.sampling)), "max_tokens": self.task.max_tokens},
            "statements": [statement("effect", effect), statement("versus_random_control", versus_control)]
            + [statement(f"versus {n}", Paired(**v)) for n, v in alternatives.items()],
            "drift": plain(drift), "gates": gates, "verdict": verdict,
            "limits": {"passes_used": self.used, "passes": self.passes, "submissions": self.allowed},
        }
        head = self.ledger.append("submit", candidate=candidate, verdict=verdict, gates=gates, patch_id=c.patch.id)
        signed = envelope(statement_for(claim, c.patch.id, head, self.ledger.count), self.key)
        (self.root / f"{candidate}.claim.json").write_bytes(jcs(claim))
        (self.root / f"{candidate}.receipt.json").write_text(json.dumps(signed))
        return {**claim, "ledger_head": head, "signed": bool(signed["signatures"]),
                "effect": asdict(effect), "versus_control": asdict(versus_control), "alternatives": alternatives}
