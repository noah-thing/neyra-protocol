"""Version control for what a neural network does."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.error
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from . import __version__
from .check import (
    DIFFERENT,
    EQUIVALENT,
    IDENTICAL,
    MERGED,
    MISSING,
    NEVER_APPLIED,
    ROUNDING,
    check,
)
from .clock import check_claim, make_claim
from .conformance import check_binding
from .diff import Report, compare
from .ledger import BrokenChain, read_entries
from .ledger import verify as verify_ledger
from .merge import merge
from .numerics import from_float, to_float
from .patch import Patch, PatchMismatch, make_patch, write_checkpoint, write_safetensors
from .protocol import ProtocolError, new_key, open_envelope, strict_loads
from .rows import (
    PROOF_TYPE,
    Index,
    build_index,
    fetch,
    link,
    open_link,
    parse_ranges,
    prove,
    verify_proof,
)
from .store import Checkpoint, FormatError

SCHEMA = 1
OK, FINDINGS, USAGE, BAD_INPUT, ENVIRONMENT, FAILED = 0, 1, 2, 3, 4, 5
EXTRAS = {"mlx": "neyra[mlx]", "mlx_lm": "neyra[mlx]", "torch": "neyra[torch]", "transformers": "neyra[torch]",
          "mcp": "neyra[mcp]", "gguf": "neyra[gguf]", "huggingface_hub": "neyra[mlx]"}


class Fail(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def emit(a, document: dict, text: str) -> None:
    if a.json:
        print(json.dumps({"schema": SCHEMA, **document}, indent=2))
    else:
        print(text)


def megabytes(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def render(r: Report, seconds: float, read: int, total: int) -> str:
    lines = [f"base   {r.base}", f"other  {r.other}", ""]
    moved = len(r.changes) + len(r.suspected)
    if r.sampled:
        lines.append(f"{r.tensors} tensors paired, {moved} changed, no difference found in the other "
                     f"{r.unchanged} (sampled spans only, which is not proof they match)")
    else:
        lines.append(f"{r.tensors} tensors paired, every byte compared, {r.unchanged} identical, {moved} changed")
    if r.renamed:
        lines.append(f"{r.renamed} tensors matched after stripping wrapper prefixes")
    if r.changes:
        pct = 100 * r.changed_entries / max(r.total_entries, 1)
        lines.append(f"{r.changed_entries:,} entries differ, {pct:.3f}% of the changed tensors")
    if read:
        lines.append(f"read {megabytes(read)} of {megabytes(total)} in {seconds:.0f}s")
    lines.append("")
    for s in r.changes:
        lines.append(s.name)
        lines.append(f"    {s.recipe}")
        detail = [f"{s.changed:,} entries", f"relative change {s.relative:.4f}"]
        if s.energy is not None:
            detail.append(f"rank 1 energy {s.energy:.6f}")
        if s.residual is not None:
            detail.append(f"fit residual {s.residual:.4f}")
        lines.append("    " + ", ".join(detail))
    for n in r.suspected[:40]:
        lines.append(f"{n}\n    differs in sampled spans")
    if len(r.suspected) > 40:
        lines.append(f"... and {len(r.suspected) - 40} more")
    for group in r.shared:
        lines.append(f"\nsame direction removed from {len(group)} tensors: {', '.join(group[:6])}")
    for label, items in (("only in base", r.removed), ("only in other", r.added),
                         ("changed dtype or shape", r.retyped)):
        if items:
            lines.append(f"\n{len(items)} tensors {label}: {', '.join(items[:6])}")
    if r.findings:
        lines.append("")
        order = {"high": 0, "medium": 1, "info": 2}
        for f in sorted(r.findings, key=lambda f: order[f.severity]):
            lines.append(f"[{f.severity}] {f.kind}: {f.detail}")
    return "\n".join(lines)


def report_document(r: Report, seconds: float, read: int) -> dict:
    def structure(s):
        return {"name": s.name, "form": s.form, "recipe": s.recipe, "changed": s.changed, "rows": s.rows,
                "cols": s.cols, "relative": s.relative, "rank": s.rank, "energy": s.energy, "alpha": s.alpha,
                "side": s.side, "residual": s.residual}

    return {"base": r.base, "other": r.other, "tensors": r.tensors, "unchanged": r.unchanged, "sampled": r.sampled,
            "changes": [structure(s) for s in r.changes], "suspected": r.suspected, "added": r.added,
            "removed": r.removed, "retyped": r.retyped, "shared_directions": r.shared, "bytes_read": read,
            "seconds": seconds,
            "findings": [{"severity": f.severity, "kind": f.kind, "detail": f.detail} for f in r.findings]}


def cmd_diff(a) -> int:
    base, other = Checkpoint.open(a.base), Checkpoint.open(a.other)
    t = time.time()
    r = compare(base, other, sample_blocks=a.sample)
    seconds = time.time() - t
    read = base.bytes_read + other.bytes_read
    emit(a, report_document(r, seconds, read), render(r, seconds, read, base.size + other.size))
    return FINDINGS if any(f.severity == "high" for f in r.findings) else OK


def cmd_patch(a) -> int:
    base, other = Checkpoint.open(a.base), Checkpoint.open(a.other)
    p = make_patch(base, other, {"base": base.spec, "other": other.spec})
    p.save(a.out)
    emit(a, {"tensors": len(p.edits), "entries": p.entries, "path": a.out, "patch_id": p.id,
             "base_root": p.base_root, "result_root": p.result_root},
         f"{len(p.edits)} tensors, {p.entries:,} stored entries -> {a.out}\npatch {p.id}")
    return OK


def cmd_apply(a) -> int:
    base = Checkpoint.open(a.base)
    patch = Patch.load(a.patch)
    patch.check_base(base)
    edited = patch.apply(base)
    write_checkpoint(base, edited, a.out)
    emit(a, {"tensors": len(edited), "path": a.out}, f"{len(edited)} tensors rebuilt and hash verified -> {a.out}")
    return OK


def cmd_merge(a) -> int:
    if len(a.patches) < 2:
        raise Fail(USAGE, "merge needs at least two patches")
    base = Checkpoint.open(a.base)
    m = merge(base, [Patch.load(p) for p in a.patches])
    m.patch.save(a.out)
    shared, differ = sum(o.entries for o in m.overlaps), sum(o.disagree for o in m.overlaps)
    text = [f"{len(a.patches)} patches merged, {len(m.patch.edits)} tensors, {m.patch.entries:,} stored entries -> {a.out}",
            f"patch {m.patch.id}"]
    if shared:
        text.append(f"{shared:,} entries touched by more than one patch, in {len(m.overlaps)} tensors, "
                    f"{differ:,} with different values")
        text += [f"    {o.name}: {o.entries:,} shared, {o.disagree:,} different" for o in m.overlaps[:20]]
        text.append("edits to a shared entry are added, so two patches making the same change apply it twice")
    emit(a, {"patches": len(a.patches), "tensors": len(m.patch.edits), "entries": m.patch.entries, "path": a.out,
             "patch_id": m.patch.id, "shared_entries": shared, "different_values": differ,
             "overlaps": [vars(o) for o in m.overlaps]}, "\n".join(text))
    return FINDINGS if shared else OK


def cmd_check(a) -> int:
    base, merged = Checkpoint.open(a.base), Checkpoint.open(a.merged)
    adapter = Checkpoint.open(a.adapter) if a.adapter else None
    c = check(base, merged, adapter, a.ulps)
    counts = {s: c.count(s) for s in (IDENTICAL, MERGED, ROUNDING, DIFFERENT, MISSING, NEVER_APPLIED)}
    text = [f"base    {base.spec}", f"merged  {merged.spec}"]
    if adapter:
        text.append(f"adapter {adapter.spec}, {c.modules} modules")
    text.append(f"{len(c.tensors)} tensors: {counts[IDENTICAL]} identical, {counts[MERGED]} merged exactly, "
                f"{counts[ROUNDING]} equal under rounding, {counts[DIFFERENT]} different, {counts[MISSING]} missing, "
                f"{counts[NEVER_APPLIED]} never applied")
    if c.recast:
        text.append("the merged model is stored in " + ", ".join(f"{b} where the base is {a}" for a, b in sorted(c.recast))
                    + "; the base was converted before the merge was recomputed")
    if c.paths:
        text.append(f"every edited tensor is what {', '.join(c.paths)} produces")
    elif counts[MERGED]:
        text.append("no single path produces every edited tensor")
    if c.faint:
        text.append(f"\nthe merge is correct, but rounding kept under half of the delta in {len(c.faint)} of "
                    f"{c.targeted} edited tensors, {min(t.applied for t in c.faint):.0%} at worst")
    if c.targeted and counts[NEVER_APPLIED] == c.targeted:
        text.append(f"\nTHE ADAPTER WAS NEVER APPLIED: every tensor it edits ({c.targeted}) equals the base, "
                    "though the adapter changes them")
    elif counts[NEVER_APPLIED]:
        text.append(f"\n{counts[NEVER_APPLIED]} of the {c.targeted} tensors the adapter edits were not changed")
    if c.leftover:
        text.append(f"\nthe merged file still holds {len(c.leftover)} separate adapter tensors, such as {c.leftover[0]}")
    if c.added:
        text.append(f"\n{len(c.added)} tensors are only in the merged model, such as {c.added[0]}")
    shown = [t for t in c.tensors if t.status not in (IDENTICAL, MERGED) or (t.detail and t.status != MERGED)]
    text += [""] if shown else []
    for t in shown[:30]:
        share = "" if t.applied is None or t.status in EQUIVALENT else f", {t.applied:.0%} of the delta present"
        text.append(f"{t.name}\n    {t.status}: {t.detail}{share}")
    if len(shown) > 30:
        text.append(f"... and {len(shown) - 30} more")
    text.append("\n" + ("equivalent" if c.equivalent else "NOT equivalent"))
    emit(a, {"base": base.spec, "merged": merged.spec, "adapter": adapter.spec if adapter else None,
             "equivalent": c.equivalent, "paths": c.paths, "recast": sorted(c.recast), "counts": counts,
             "faint": [t.name for t in c.faint], "adapter_tensors": c.targeted,
             "adapter_modules": c.modules, "added": c.added, "leftover_adapter_tensors": c.leftover,
             "tensors": [vars(t) for t in c.tensors]}, "\n".join(text))
    return OK if c.equivalent else FINDINGS


def cmd_show(a) -> int:
    p = Patch.load(a.patch)
    rows = [{"tensor": e.name, "shape": list(e.shape), "form": "low rank" if e.low_rank else "sparse",
             "stored_entries": len(e.idx), "base_digest": e.base_digest} for e in p.edits]
    text = [f"patch  {p.id}", f"base   {p.base_root}", f"result {p.result_root}",
            f"{Path(a.patch).stat().st_size:,} bytes"]
    text += [f"{k}: {v}" for k, v in p.meta.items()]
    text += [f"{r['tensor']}  {r['form']}, {r['stored_entries']:,} stored entries" for r in rows]
    emit(a, {"patch_id": p.id, "base_root": p.base_root, "result_root": p.result_root, "meta": p.meta,
             "tensors": rows}, "\n".join(text))
    return OK


def local_path(spec: str) -> str:
    if Path(spec).expanduser().exists():
        return str(Path(spec).expanduser())
    from huggingface_hub import snapshot_download

    repo, _, rev = spec.partition("@")
    return snapshot_download(repo, revision=rev or None)


def cmd_probe(a) -> int:
    from .examine import examine
    from .runtime import open_runtime

    exam = examine(open_runtime(local_path(a.model), a.backend), a.n, a.max_tokens)
    text = [f"{exam.n} problems, up to {exam.max_tokens} tokens each",
            f"correct {exam.correct:.0%}, mean length {exam.tokens:.0f} tokens", ""]
    if not exam.symptoms:
        text.append("no symptoms found in this battery, which covers loops, runaway thinking and missing answers")
    for s in exam.symptoms:
        text += [f"[{s.severity}] {s.name}: {s.rate:.0%} of problems", f"    {s.explanation}"]
        if s.example:
            text.append(f"    ...{s.example!r}")
    emit(a, {"n": exam.n, "max_tokens": exam.max_tokens, "correct": exam.correct, "mean_tokens": exam.tokens,
             "symptoms": [vars(s) for s in exam.symptoms]}, "\n".join(text))
    return FINDINGS if exam.symptoms else OK


def cmd_fix(a) -> int:
    from .runtime import open_runtime
    from .workflows import fix_loops

    path = local_path(a.model)
    found = fix_loops(open_runtime(path, a.backend), Checkpoint.open(path), a.n, a.max_tokens, a.columns)
    if found is None:
        emit(a, {"fixed": False}, "no patch written: not enough loops to work from")
        return OK
    found.patch.save(a.out)
    layers = [x.index for x in found.plan.layers]
    emit(a, {"fixed": True, "path": a.out, "patch_id": found.patch.id, "layers": layers, "alpha": found.plan.alpha,
             "loops": found.loops},
         f"layers {layers}, strength {found.plan.alpha}, {found.patch.entries:,} stored entries -> {a.out}\n"
         "this is a candidate, not a result: measure it with a lab session before believing it")
    return OK


def percent(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def nats(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def cmd_edit(a) -> int:
    from .edit import Settings, load_requests
    from .runtime import open_runtime
    from .workflows import edit_facts

    path = local_path(a.model)
    requests = load_requests(a.requests)
    names = ("width", "solver", "lam", "batch", "steps", "lr", "contexts", "polish", "anchor", "max_rank", "seed",
             "window", "held", "neutral")
    settings = Settings(layers=[int(x) for x in a.layers.split(",")] if a.layers else None,
                        **{n: getattr(a, n) for n in names if getattr(a, n) is not None})
    out = Path(a.out)
    ledger = Path(a.ledger or out.with_suffix(".ledger.jsonl"))
    key = bytes.fromhex(Path(a.key).read_text().strip()) if a.key else None
    done = edit_facts(open_runtime(path, a.backend), Checkpoint.open(path), requests, settings, neutral=a.text,
                      key=key, ledger=ledger, track=a.track)
    done.patch.save(out)
    receipt = Path(a.receipt or out.with_suffix(".receipt.json"))
    receipt.write_text(json.dumps(done.receipt))
    r, b, c = done.report, done.report["before"], done.report["after"]
    ranks = sorted(set(r["ranks"].values()))
    text = [f"layers {r['layers']}, {len(requests)} edits, {done.patch.entries:,} stored entries, ranks {ranks} -> {out}",
            f"patch {done.patch.id}", ""]
    for name in ("efficacy", "paraphrase", "specificity", "exact", "score"):
        text.append(f"{name:12} {percent(b[name]):>7} -> {percent(c[name]):>7}")
    text.append(f"{'margin':12} {nats(b['margin']):>7} -> {nats(c['margin']):>7}")
    text.append(f"neutral text: KL {r['neutral_kl']:.4f} nats/token, fluency {r['fluency_before']:.2f} -> "
                f"{r['fluency_after']:.2f}")
    text += [f"  {'pass' if ok else 'FAIL'}  {gate}" for gate, ok in r["gates"].items()]
    if b["specificity"] is None:
        text.append("specificity was not measured: give each request neighbours and the old object (true)")
    if r["warning"]:
        text.append(f"warning: {r['warning']}")
    text.append(f"verdict {done.verdict}" if done.verdict == "PASS"
                else "verdict FAIL: this edit is not claimed as a success. The patch is written for inspection only.")
    emit(a, {"path": str(out), "receipt": str(receipt), "ledger": str(ledger), "patch_id": done.patch.id, **r},
         "\n".join(text))
    return OK if done.verdict == "PASS" else FINDINGS


def cmd_bisect(a) -> int:
    from .runtime import Sampling, open_runtime
    from .workflows import bisect_loops, collect_loops

    path = local_path(a.model)
    ck = Checkpoint.open(path)
    patch = Patch.load(a.patch)
    patch.check_base(ck)
    rt = open_runtime(path, a.backend)
    tokens = collect_loops(rt, a.n, a.max_tokens, Sampling(0.6, 0.95, 20, 1.5, 0))
    b = bisect_loops(rt, ck, patch, tokens, a.grain, a.retain)
    b.patch.save(a.out)
    r, c = b.result, b.check
    kept, total = b.units.weights_changed(r.units), b.units.weights_changed(range(len(b.units)))
    verdict = "confirmed" if c.passed else "NOT confirmed"
    emit(a, {"blame": r.blame, "kept_units": len(r.units), "total_units": r.total, "kept_weights": kept, "total_weights": total,
             "minimal": r.minimal, "confirmed": c.passed, "confirmed_retained": c.retained, "confirm_items": c.n,
             "path": a.out, "patch_id": b.patch.id},
         f"{len(r.units)} of {r.total} units, {kept:,} of {total:,} weights -> {a.out}\n"
         f"{verdict} on {c.n} unseen texts: keeps {c.retained:.0%} of the effect\n"
         f"blame is {r.blame}: each subset was added to the base alone, not removed from the full edit\n"
         f"{'no single unit can be removed' if r.minimal else 'search budget ran out before this was minimal'}")
    return OK if c.passed else FINDINGS


def cmd_export(a) -> int:
    from .export import to_gguf_lora, to_peft

    base = Checkpoint.open(a.base)
    patch = Patch.load(a.patch)
    if a.format == "peft":
        info = to_peft(patch, base, a.out, a.energy)
    else:
        if not a.arch:
            raise Fail(USAGE, "gguf export needs --arch, the architecture name in the base GGUF, such as qwen35")
        info = to_gguf_lora(patch, base, a.out, a.arch, a.energy)
    emit(a, {"path": a.out, **info}, f"{info['bytes']:,} bytes -> {a.out}")
    return OK


def cmd_index(a) -> int:
    ck = Checkpoint.open(a.model)
    t = time.time()
    index = build_index(ck)
    index.save(a.out)
    size = Path(a.out).stat().st_size
    text = [f"{len(index.entries)} tensors, index {megabytes(size)} for {megabytes(ck.size)} of weights, "
            + f"{time.time() - t:.0f}s", f"weights root {index.weights_root}", f"rows root    {index.rows_root}"]
    if a.sign:
        out = a.out + ".link.json"
        Path(out).write_text(json.dumps(link(index, bytes.fromhex(Path(a.sign).read_text().strip())), indent=1))
        text.append(f"link signed -> {out}\nthe signature says this key read all of the model, not that anyone else did")
    emit(a, {"path": a.out, "weights_root": index.weights_root, "rows_root": index.rows_root, "bytes": size},
         "\n".join(text))
    return OK


def cmd_prove(a) -> int:
    ck = Checkpoint.open(a.model)
    if a.index:
        index = Index.load(a.index)
    elif Path(a.model).expanduser().exists():
        index = build_index(ck)
    else:
        raise Fail(USAGE, "a remote model needs --index, the leaf hashes someone published for it")
    ref = ck.tensors.get(a.tensor)
    if ref is None:
        raise Fail(BAD_INPUT, f"{a.model} has no tensor {a.tensor}")
    _, entry = index.entry(a.tensor)
    if entry["dtype"] != ref.dtype or entry["shape"] != list(ref.shape):
        raise Fail(BAD_INPUT, f"{a.tensor} is {ref.dtype}{list(ref.shape)} in the model but the index disagrees")
    proof = prove(index, a.tensor, parse_ranges(a.rows))
    verify_proof(proof, fetch(ref, proof))
    Path(a.out).write_text(json.dumps(proof, indent=1))
    size = Path(a.out).stat().st_size
    emit(a, {"path": a.out, "rows_root": proof["rows_root"], "proof_bytes": size},
         f"proof {size:,} bytes -> {a.out}\nrows root {proof['rows_root']}\n"
         f"a proof binds rows to a rows root; verify it against a link to learn which weights that is")
    return OK


def verify_rows(a, proof: dict) -> int:
    if not a.model:
        raise Fail(USAGE, "checking a row proof needs the model to read the rows from")
    ck = Checkpoint.open(a.model)
    t = proof.get("tensor", {})
    ref = ck.tensors.get(t.get("name"))
    if ref is None or ref.dtype != t.get("dtype") or list(ref.shape) != t.get("shape"):
        raise ProtocolError("proof_mismatch", f"{a.model} has no tensor like the one in the proof")
    before = ck.bytes_read
    blocks = verify_proof(proof, fetch(ref, proof))
    read = ck.bytes_read - before
    text = [f"{len(blocks)} ranges of {t['name']} hash into rows root {proof['rows_root'][7:23]}",
            f"read {megabytes(read)} of {megabytes(ref.nbytes)}, proof {len(json.dumps(proof)):,} bytes"]
    problems = []
    if a.link:
        public = bytes.fromhex(Path(a.key).read_text().strip()) if a.key else None
        opened = open_link(strict_loads(Path(a.link).read_text()), public)
        predicate = opened.statement["predicate"]
        if predicate["rows_root"] != proof["rows_root"]:
            problems.append("the link is for a different rows root")
        else:
            text.append(f"link ties it to weights root {predicate['weights_root'][7:23]}, "
                        f"{'signature verified' if opened.signed else 'NOT signed: a record only'}")
    else:
        text.append("no link given, so nothing here says which weights this root belongs to")
    text += [f"problem: {p}" for p in problems]
    emit(a, {"rows_root": proof["rows_root"], "bytes_read": read, "problems": problems}, "\n".join(text))
    return FAILED if problems else OK


def open_indexed(spec: str, path: str | None) -> tuple[Checkpoint, Index]:
    ck = Checkpoint.open(spec)
    if path:
        return ck, Index.load(path)
    if not Path(spec).expanduser().exists():
        raise Fail(USAGE, f"{spec} is remote, so it needs the row index someone published for it")
    return ck, build_index(ck)


def cmd_clock(a) -> int:
    if a.check:
        claim = strict_loads(Path(a.check).read_text())
        problems = check_claim(claim, Checkpoint.open(a.a), Checkpoint.open(a.b))
        text = [f"{claim['verdict']} on {claim['n']:,} entries of {claim['tensor']}, rows reproduced from the bytes"
                if not problems else "claim does not hold"] + [f"problem: {p}" for p in problems]
        emit(a, {"verdict": claim["verdict"], "problems": problems}, "\n".join(text))
        return FAILED if problems else OK
    if not a.rows:
        raise Fail(USAGE, "name the dormant rows with --rows, such as 128010:128256")
    (a_ck, a_index), (b_ck, b_index) = open_indexed(a.a, a.a_index), open_indexed(a.b, a.b_index)
    claim = make_claim(a_ck, b_ck, a_index, b_index, a.tensor, a.rows)
    Path(a.out).write_text(json.dumps(claim, indent=1))
    scale = f", c in [{claim['c_lo_dec']}, {claim['c_hi_dec']}]" if "c_lo_dec" in claim else ""
    emit(a, {"path": a.out, "verdict": claim["verdict"], "id": claim["id"]},
         f"{claim['verdict']}{scale} on {claim['n']:,} entries -> {a.out}\n"
         f"read {megabytes(a_ck.bytes_read + b_ck.bytes_read)}; bound to rows roots "
         f"{claim['a_rows_root'][7:19]} and {claim['b_rows_root'][7:19]}")
    return OK


def cmd_verify(a) -> int:
    public = bytes.fromhex(Path(a.key).read_text().strip()) if a.key else None
    doc = strict_loads(Path(a.receipt).read_text())
    if isinstance(doc, dict) and doc.get("type") == PROOF_TYPE:
        return verify_rows(a, doc)
    opened = open_envelope(doc, public)
    predicate = opened.statement["predicate"]
    claim = predicate["claim"]
    problems = []
    text = [f"statement well formed, {'signature verified' if opened.signed else 'NOT signed: an unauthenticated record'}"]
    if a.ledger:
        try:
            head, count = verify_ledger(Path(a.ledger))
            text.append(f"ledger intact, {count} entries, head {head[7:23]}")
        except BrokenChain as e:
            problems.append(str(e))
    if a.patch:
        got = Patch.load(a.patch)
        if got.id != claim["patch_id"] or got.result_root != claim["result_root"]:
            problems.append("patch does not match the receipt")
        else:
            text.append(f"patch matches receipt, {got.id[7:23]}")
    if a.ledger and a.patch and not problems:
        code = check_binding(a.receipt, a.ledger, a.patch)
        if code != "ok":
            problems.append(code)
    text.append(f"verdict in receipt: {claim['verdict']}")
    text += [f"  {'pass' if ok else 'FAIL'}  {key}" for key, ok in claim["gates"].items()]
    text += [f"problem: {p}" for p in problems]
    text.append("a signature shows who made this claim, not that it is true; rerun it to check that")
    emit(a, {"verdict": claim["verdict"], "gates": claim["gates"], "signed": opened.signed, "problems": problems},
         "\n".join(text))
    return FAILED if problems else OK


def cmd_keygen(a) -> int:
    key = new_key(a.out)
    emit(a, {"keyid": key, "path": a.out}, f"{key}\nprivate key {a.out}, public key {Path(a.out).with_suffix('.pub')}")
    return OK


def cmd_log(a) -> int:
    verify_ledger(Path(a.ledger))
    entries, _ = read_entries(Path(a.ledger))
    text = [f"{e['seq']:>4}  {e['time']}  {e['kind']:<14} {json.dumps(e['body'])[:90]}" for e in entries]
    emit(a, {"entries": entries}, "\n".join(text))
    return OK


def cmd_doctor(a) -> int:
    checks = []
    for label, module in (("numpy", "numpy"), ("mlx-lm", "mlx-lm"), ("torch", "torch"), ("transformers", "transformers"),
                          ("mcp", "mcp"), ("gguf", "gguf")):
        try:
            checks.append((label, version(module)))
        except PackageNotFoundError:
            checks.append((label, None))
    backend = "mlx" if dict(checks)["mlx-lm"] else "torch" if dict(checks)["torch"] else None
    text = [f"neyra {__version__}, python {sys.version.split()[0]}", ""]
    text += [f"{label:<14}{found or 'not installed'}" for label, found in checks]
    text += ["", f"model backend: {backend or 'none, so diff, patch, apply and export work; probe needs one'}"]
    free = shutil.disk_usage(Path.home()).free
    text.append(f"free disk: {free / 1e9:.1f} GB")
    if not backend:
        text.append("to run models: uv tool install 'neyra[mlx]' on Apple Silicon, 'neyra[torch]' elsewhere")
    emit(a, {"version": __version__, "packages": dict(checks), "backend": backend, "free_bytes": free}, "\n".join(text))
    return OK


def cmd_demo(a) -> int:
    """Plant an edit in a small random model, then recover it from the weights alone."""
    rng = np.random.default_rng(0)
    name = "model.layers.3.mlp.down_proj.weight"
    w = from_float(rng.standard_normal((256, 1024)).astype(np.float32) * 0.02, "BF16")
    other = from_float(rng.standard_normal(256).astype(np.float32), "BF16")
    u = rng.standard_normal(256)
    u /= np.linalg.norm(u)
    cols = rng.choice(1024, 32, replace=False)
    f = to_float(w, "BF16").astype(np.float64)
    edited = f.copy()
    edited[:, cols] -= 4.0 * np.outer(u, u @ f[:, cols])
    edited = from_float(edited.astype(np.float32), "BF16").reshape(w.shape)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, bits in (("base", w), ("edited", edited)):
            (root / label).mkdir()
            write_safetensors(root / label / "model.safetensors", {name: bits, "model.norm.weight": other}, {"format": "pt"},
                              "BF16")
        base, changed = Checkpoint.open(str(root / "base")), Checkpoint.open(str(root / "edited"))
        print("planted: W' = W - 4 u(uᵀW) on 32 random columns of a 256 x 1024 matrix\n")
        print("$ neyra diff base edited")
        print(render(compare(base, changed), 0.0, 0, 0).split("\n\n", 1)[1])
        p = make_patch(base, changed)
        p.save(root / "edit.neyra")
        rebuilt = Patch.load(root / "edit.neyra").apply(base)[name]
        print(f"\n$ neyra patch base edited -o edit.neyra\n{(root / 'edit.neyra').stat().st_size:,} bytes")
        print(f"\nrebuilt tensor identical to the edited one: {np.array_equal(rebuilt, edited)}")
    print("\nnext: neyra diff Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --sample 4")
    return OK


def cmd_mcp(a) -> int:
    from .lab import Lab, Task
    from .mcp_server import build
    from .runtime import open_runtime

    path = local_path(a.model)
    lab = Lab(Path(a.root), path, Task.load(a.task), open_runtime(path, a.backend), a.passes, a.submissions, a.allow)
    build(lab).run()
    return OK


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine readable output")
    ap = argparse.ArgumentParser(prog="neyra", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def command(name, fn, help):
        p = sub.add_parser(name, help=help, parents=[common])
        p.set_defaults(fn=fn)
        return p

    p = command("demo", cmd_demo, "plant an edit in a tiny model and recover it, in seconds, with no downloads")

    p = command("diff", cmd_diff, "what changed between two checkpoints, and how")
    p.add_argument("base")
    p.add_argument("other")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="screen N random spans per tensor, then read only what differs")

    p = command("patch", cmd_patch, "write the difference between two checkpoints as a patch")
    p.add_argument("base")
    p.add_argument("other")
    p.add_argument("-o", "--out", required=True)

    p = command("apply", cmd_apply, "rebuild a model from a base and a patch, verifying hashes")
    p.add_argument("base")
    p.add_argument("patch")
    p.add_argument("-o", "--out", required=True)

    p = command("merge", cmd_merge, "combine patches made on one base, exactly and in any order")
    p.add_argument("base")
    p.add_argument("patches", nargs="+")
    p.add_argument("-o", "--out", required=True)

    p = command("check", cmd_check, "decide whether a merged model matches its base and adapter")
    p.add_argument("base")
    p.add_argument("merged")
    p.add_argument("--adapter", help="PEFT LoRA adapter directory that was merged into the base")
    p.add_argument("--ulps", type=int, default=1, help="steps of the dtype two tensors may differ by and still count as rounding")

    p = command("show", cmd_show, "describe a patch")
    p.add_argument("patch")

    p = command("probe", cmd_probe, "run a model through a fixed battery and report what goes wrong")
    p.add_argument("model")
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])

    p = command("fix", cmd_fix, "find a loop direction in a model and write a candidate patch")
    p.add_argument("model")
    p.add_argument("-o", "--out", default="fix.neyra")
    p.add_argument("-n", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--columns", type=int, default=64)
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])

    p = command("edit", cmd_edit, "write facts into a model as an exact, signed, revertible patch")
    p.add_argument("model")
    p.add_argument("requests", help="JSON list of {subject, prompt with {}, target}, or CounterFact records")
    p.add_argument("-o", "--out", default="edit.neyra")
    p.add_argument("--receipt", help="default: beside the patch")
    p.add_argument("--ledger", help="default: beside the patch")
    p.add_argument("--key", help="private key file from neyra keygen, to sign the receipt")
    p.add_argument("--text", help="plain text for the covariance and damage test; default: sampled from the model")
    p.add_argument("--neutral", type=int, help="windows of neutral text for the covariance")
    p.add_argument("--held", type=int, help="further windows to measure damage on")
    p.add_argument("--window", type=int, help="tokens per window")
    p.add_argument("--layers", help="comma separated layers; default: chosen by causal tracing")
    p.add_argument("--width", type=int, help="layers to share an edit when tracing chooses")
    p.add_argument("--solver", choices=["protect", "memit", "null"])
    p.add_argument("--lam", type=float, help="weight of neutral text against the new facts")
    p.add_argument("--batch", type=int, help="facts per closed form solve; later batches protect earlier ones")
    p.add_argument("--steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--contexts", type=int)
    p.add_argument("--polish", type=int, help="extra passes on the last layer to correct for rounding")
    p.add_argument("--anchor", type=float, help="cap on a value vector's norm, against the unedited model's")
    p.add_argument("--max-rank", type=int, dest="max_rank")
    p.add_argument("--seed", type=int)
    p.add_argument("--track", action="store_true", help="measure after every batch, to see sequential decay")
    p.add_argument("--backend", default="mlx", choices=["mlx"])

    p = command("bisect", cmd_bisect, "find the smallest part of a patch that keeps its effect")
    p.add_argument("model")
    p.add_argument("patch")
    p.add_argument("-o", "--out", default="bisected.neyra")
    p.add_argument("-n", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--grain", default="column", choices=["tensor", "column", "row"])
    p.add_argument("--retain", type=float, default=0.9, help="share of the effect to keep")
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])

    p = command("export", cmd_export, "write a patch as a LoRA for PEFT or llama.cpp")
    p.add_argument("base")
    p.add_argument("patch")
    p.add_argument("--format", choices=["peft", "gguf"], required=True)
    p.add_argument("--arch", help="GGUF architecture name, for --format gguf")
    p.add_argument("--energy", type=float, default=0.999, help="share of the change to keep")
    p.add_argument("-o", "--out", required=True)

    p = command("index", cmd_index, "hash every row group of a model, so rows can be proved without the rest")
    p.add_argument("model")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--sign", metavar="KEY", help="private key file; also write a signed link to the weights root")

    p = command("prove", cmd_prove, "prove some rows of a tensor against a rows root")
    p.add_argument("model")
    p.add_argument("--tensor", required=True)
    p.add_argument("--rows", required=True, metavar="A:B[,C:D]")
    p.add_argument("--index", help="published row index; needed for a model that is not on disk")
    p.add_argument("-o", "--out", default="rows.proof.json")

    p = command("verify", cmd_verify, "check a receipt, or a row proof against the rows it claims")
    p.add_argument("receipt", help="receipt, or a row proof from neyra prove")
    p.add_argument("model", nargs="?", help="model to read the rows from, for a row proof")
    p.add_argument("--link", help="signed link from neyra index --sign, for a row proof")
    p.add_argument("--ledger")
    p.add_argument("--patch")
    p.add_argument("--key", help="public key file")

    p = command("clock", cmd_clock, "claim that two checkpoints share a scalar on rows training never fed")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--rows", metavar="A:B[,C:D]", help="rows to compare, such as the reserved token ids")
    p.add_argument("--tensor", help="default: the input embedding")
    p.add_argument("--a-index")
    p.add_argument("--b-index")
    p.add_argument("--check", metavar="CLAIM", help="reproduce a claim from the bytes instead of making one")
    p.add_argument("-o", "--out", default="clock.claim.json")

    p = command("keygen", cmd_keygen, "create an ed25519 key for signing receipts")
    p.add_argument("-o", "--out", default="neyra.key")

    p = command("log", cmd_log, "print a lab ledger after checking its chain")
    p.add_argument("ledger")

    p = command("mcp", cmd_mcp, "serve a lab session to an agent over MCP")
    p.add_argument("model")
    p.add_argument("task", help="task definition, a JSON file")
    p.add_argument("--root", default=".neyra/lab")
    p.add_argument("--passes", type=int, default=50)
    p.add_argument("--submissions", type=int, default=1)
    p.add_argument("--allow", action="append", default=[], metavar="DIR",
                   help="extra directory the agent may read patches from or export to")
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])

    command("doctor", cmd_doctor, "check this machine and what it can run")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except Fail as e:
        print(f"neyra: {e}", file=sys.stderr)
        return e.code
    except ModuleNotFoundError as e:
        root = (e.name or "").split(".")[0]
        if root not in EXTRAS:
            raise
        print(f"neyra: this needs {root}. Install it with: uv tool install '{EXTRAS[root]}'", file=sys.stderr)
        return ENVIRONMENT
    except (FormatError, json.JSONDecodeError) as e:
        print(f"neyra: bad input: {e}", file=sys.stderr)
        return BAD_INPUT
    except (PatchMismatch, ProtocolError) as e:
        print(f"neyra: {e}", file=sys.stderr)
        return FAILED
    except urllib.error.HTTPError as e:
        hint = "check the repo name, or set HF_TOKEN for a gated model" if e.code in (401, 403, 404) else "try again"
        print(f"neyra: Hugging Face returned {e.code}: {hint}", file=sys.stderr)
        return BAD_INPUT
    except FileNotFoundError as e:
        print(f"neyra: {e.filename}: no such file", file=sys.stderr)
        return BAD_INPUT
    except KeyboardInterrupt:
        print("neyra: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
