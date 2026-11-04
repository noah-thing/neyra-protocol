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
from .conformance import check_binding
from .diff import Report, compare
from .ledger import BrokenChain, read_entries
from .ledger import verify as verify_ledger
from .numerics import from_float, to_float
from .patch import Patch, PatchMismatch, make_patch, write_checkpoint, write_safetensors
from .protocol import ProtocolError, new_key, open_envelope, strict_loads
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
    emit(a, {"kept_units": len(r.units), "total_units": r.total, "kept_weights": kept, "total_weights": total,
             "minimal": r.minimal, "confirmed": c.passed, "confirmed_retained": c.retained, "confirm_items": c.n,
             "path": a.out, "patch_id": b.patch.id},
         f"{len(r.units)} of {r.total} units, {kept:,} of {total:,} weights -> {a.out}\n"
         f"{verdict} on {c.n} unseen texts: keeps {c.retained:.0%} of the effect\n"
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


def cmd_verify(a) -> int:
    public = bytes.fromhex(Path(a.key).read_text().strip()) if a.key else None
    opened = open_envelope(strict_loads(Path(a.receipt).read_text()), public)
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

    p = command("verify", cmd_verify, "check a receipt, and optionally its ledger, patch and signature")
    p.add_argument("receipt")
    p.add_argument("--ledger")
    p.add_argument("--patch")
    p.add_argument("--key", help="public key file")

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
