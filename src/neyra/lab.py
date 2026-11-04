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

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__
from .behavior import score
from .bisect import Measure, bisect, confirm
from .ledger import Ledger
from .numerics import from_float, is_float, to_float
from .patch import Patch, TensorEdit, bits_sha256, diff_tensor
from .probe import SelectivityProbe, Span, loop_spans
from .recover import recover
from .repair import Layer, Plan, Repair, down_proj_name, matched_random, project_columns
from .runtime import Sampling
from .stats import kl_from_logits, paired
from .store import Checkpoint
from .units import UnitSet

PROTECTED = re.compile(r"embed_tokens|lm_head|(^|\.)norm\.weight$")


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
        o = score(rollout.text, rollout.hit_limit, item.answer)
        if self.metric == "correct":
            return float(bool(o.correct))
        if self.metric == "finished":
            return float(o.finished)
        if self.metric == "not_repeating":
            return float(o.repeated < 0.25)
        raise ValueError(f"unknown metric {self.metric}")

    @classmethod
    def load(cls, path: str | Path) -> Task:
        doc = json.loads(Path(path).read_text())
        root = Path(path).parent

        def items(ref):
            rows = [json.loads(x) for x in (root / ref).read_text().splitlines() if x.strip()]
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
                 submissions: int = 1, allowed: list[str | Path] | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.roots = [self.root, *(Path(p).expanduser().resolve() for p in allowed or [])]
        self.model, self.task, self.rt = model, task, runtime
        self.ck = Checkpoint.open(model)
        self.passes, self.used = passes, 0
        self.submissions = self.allowed = submissions
        self.candidates: dict[str, Candidate] = {}
        self.rollouts: dict[str, list] = {}
        self.findings: dict[str, dict] = {}
        self.ledger = Ledger(self.root / "ledger.jsonl")
        self.holdout_commit = hashlib.sha256(
            json.dumps([asdict(i) for i in task.holdout], sort_keys=True).encode()).hexdigest()
        self.ledger.append("open", model=self.ck.spec, task=task.name, passes=passes,
                           holdout_commit=self.holdout_commit, version=__version__)

    def _spend(self, n: int, what: str) -> None:
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
                "passes_left": self.passes - self.used, "submissions_left": self.submissions,
                "dev_items": len(self.task.dev), "holdout_items": len(self.task.holdout),
                "holdout_commit": self.holdout_commit, "candidates": sorted(self.candidates),
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
        outcomes = [score(r.text, r.hit_limit, i.answer) for r, i in zip(rows, items)]
        result = {"rollouts": rid, "n": len(rows), "metric": float(scores.mean()),
                  "hit_limit": float(np.mean([r.hit_limit for r in rows])),
                  "closed_thinking": float(np.mean([o.closed for o in outcomes])),
                  "repeated_share": float(np.mean([o.repeated for o in outcomes])),
                  "tails": [r.text[-200:] for r in rows[:3]]}
        self.ledger.append("sample", candidate=candidate, seed=seed, **{k: result[k] for k in ("n", "metric", "hit_limit")})
        return result

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
        self.candidates[cid] = Candidate(cid, patch, note)
        patch.save(self.root / f"{cid}.neyra")
        self.ledger.append("candidate", id=cid, fingerprint=patch.fingerprint, note=note)
        return {"candidate": cid, **self.describe(cid)}

    def describe(self, cid: str) -> dict:
        p = self.candidate(cid).patch
        out = []
        for e in p.edits:
            base = self.ck.tensors[e.name].read()
            s = recover(e.name, base, e.apply(base, verify=False), e.dtype)
            out.append({"tensor": e.name, "recipe": s.recipe, "entries": s.changed})
        return {"fingerprint": p.fingerprint, "tensors": out}

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
        return Patch(edits, {"control_for": cid, "seed": seed})

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
        added = self._add(Patch(edits, {"bisected_from": candidate, "retain": retain}), f"bisect of {candidate}")
        result = {"kept_units": len(r.units), "total_units": r.total, "minimal": r.minimal,
                  "search_retained": r.retained, "trials": len(r.trials),
                  "confirmed": check.passed, "confirmed_retained": check.retained, "confirm_items": check.n,
                  "weights_full": units.weights_changed(range(len(units))),
                  "weights_kept": units.weights_changed(r.units), **added}
        self.ledger.append("bisect", candidate=candidate, **{k: result[k] for k in (
            "kept_units", "total_units", "confirmed", "confirmed_retained")})
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
            self.ledger.append("submit", candidate=candidate, verdict=verdict, fingerprint=c.patch.fingerprint)
            return {"verdict": verdict, "submissions_left": self.submissions}
        if verdict == "PASS":
            self.submissions = 0
        receipt = {
            "neyra": __version__, "time": time.time(), "model": self.ck.spec, "task": self.task.name,
            "metric": self.task.metric, "sampling": asdict(self.task.sampling), "max_tokens": self.task.max_tokens,
            "holdout_commit": self.holdout_commit, "candidate": candidate, "fingerprint": c.patch.fingerprint,
            "tensors": [{"name": e.name, "base_sha256": e.base_sha256, "edited_sha256": e.edited_sha256 or
                         bits_sha256(e.apply(self.ck.tensors[e.name].read(), verify=False))} for e in c.patch.edits],
            "passes_used": self.used, "effect": asdict(effect), "versus_control": asdict(versus_control),
            "alternatives": alternatives, "drift": drift, "gates": gates, "verdict": verdict,
        }
        receipt["ledger_head"] = self.ledger.append("submit", candidate=candidate, verdict=verdict, gates=gates,
                                                    fingerprint=c.patch.fingerprint)
        (self.root / f"{candidate}.receipt.json").write_text(json.dumps(receipt, indent=2))
        return receipt
