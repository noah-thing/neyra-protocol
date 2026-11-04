"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .diff import Report, compare
from .patch import Patch, make_patch, write_checkpoint
from .store import Checkpoint


def render(r: Report) -> str:
    lines = [f"base   {r.base}", f"other  {r.other}", ""]
    moved = len(r.changes) + len(r.suspected)
    if r.sampled:
        lines.append(f"{r.tensors} tensors paired, {moved} changed, no difference found in the "
                     f"other {r.unchanged} (sampled spans only, which is not proof they match)")
    else:
        lines.append(f"{r.tensors} tensors paired, every byte compared, {r.unchanged} identical, "
                     f"{moved} changed")
    if r.renamed:
        lines.append(f"{r.renamed} tensors matched after stripping wrapper prefixes")
    if r.changes:
        pct = 100 * r.changed_entries / max(r.total_entries, 1)
        lines.append(f"{r.changed_entries:,} entries differ, {pct:.3f}% of the changed tensors")
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
    if r.removed:
        lines.append(f"\n{len(r.removed)} tensors only in base: {', '.join(r.removed[:6])}")
    if r.added:
        lines.append(f"\n{len(r.added)} tensors only in other: {', '.join(r.added[:6])}")
    if r.retyped:
        lines.append(f"\n{len(r.retyped)} tensors changed dtype or shape: {', '.join(r.retyped[:6])}")
    if r.findings:
        lines.append("")
        order = {"high": 0, "medium": 1, "info": 2}
        for f in sorted(r.findings, key=lambda f: order[f.severity]):
            lines.append(f"[{f.severity}] {f.kind}: {f.detail}")
    return "\n".join(lines)


def report_json(r: Report) -> str:
    def structure(s):
        return {"name": s.name, "form": s.form, "recipe": s.recipe, "changed": s.changed,
                "rows": s.rows, "cols": s.cols, "relative": s.relative, "rank": s.rank,
                "energy": s.energy, "alpha": s.alpha, "side": s.side, "residual": s.residual}

    return json.dumps({
        "base": r.base, "other": r.other, "tensors": r.tensors, "unchanged": r.unchanged,
        "sampled": r.sampled, "changes": [structure(s) for s in r.changes],
        "suspected": r.suspected, "added": r.added, "removed": r.removed, "retyped": r.retyped,
        "shared_directions": r.shared,
        "findings": [{"severity": f.severity, "kind": f.kind, "detail": f.detail} for f in r.findings],
    }, indent=2)


def cmd_diff(a) -> int:
    r = compare(Checkpoint.open(a.base), Checkpoint.open(a.other), sample_blocks=a.sample)
    print(report_json(r) if a.json else render(r))
    return 1 if any(f.severity == "high" for f in r.findings) else 0


def cmd_patch(a) -> int:
    base, other = Checkpoint.open(a.base), Checkpoint.open(a.other)
    p = make_patch(base, other, {"base": base.spec, "other": other.spec})
    p.save(a.out)
    print(f"{len(p.edits)} tensors, {p.entries:,} stored entries -> {a.out}\nfingerprint {p.fingerprint}")
    return 0


def cmd_apply(a) -> int:
    base = Checkpoint.open(a.base)
    p = Patch.load(a.patch)
    edited = p.apply(base)
    write_checkpoint(base, edited, a.out)
    print(f"{len(edited)} tensors rebuilt and hash verified -> {a.out}")
    return 0


def local_path(spec: str) -> str:
    if Path(spec).expanduser().exists():
        return str(Path(spec).expanduser())
    from huggingface_hub import snapshot_download

    repo, _, rev = spec.partition("@")
    return snapshot_download(repo, revision=rev or None)


def cmd_doctor(a) -> int:
    from .doctor import examine
    from .runtime import open_runtime

    exam = examine(open_runtime(local_path(a.model), a.backend), a.n, a.max_tokens)
    print(f"{exam.n} problems, up to {exam.max_tokens} tokens each")
    print(f"correct {exam.correct:.0%}, mean length {exam.tokens:.0f} tokens\n")
    if not exam.symptoms:
        print("no symptoms found in this battery, which covers loops, runaway thinking and missing answers")
    for s in exam.symptoms:
        print(f"[{s.severity}] {s.name}: {s.rate:.0%} of problems")
        print(f"    {s.explanation}")
        if s.example:
            print(f"    ...{s.example!r}")
    return 1 if exam.symptoms else 0


def cmd_export(a) -> int:
    from .export import to_gguf_lora, to_peft

    base = Checkpoint.open(a.base)
    patch = Patch.load(a.patch)
    if a.format == "peft":
        info = to_peft(patch, base, a.out, a.energy)
    else:
        if not a.arch:
            raise SystemExit("gguf export needs --arch, the architecture name in the base GGUF, such as qwen35")
        info = to_gguf_lora(patch, base, a.out, a.arch, a.energy)
    print(f"{info['bytes']:,} bytes -> {a.out}")
    return 0


def cmd_verify(a) -> int:
    from .ledger import BrokenChain, verify

    receipt = json.loads(Path(a.receipt).read_text())
    problems = []
    try:
        head, count = verify(Path(a.ledger))
        entries = [json.loads(x) for x in Path(a.ledger).read_text().splitlines()]
        heads = {e["hash"] for e in entries}
        if receipt["ledger_head"] not in heads:
            problems.append("receipt head is not in this ledger")
        print(f"ledger intact, {count} entries, head {head[:16]}")
    except BrokenChain as e:
        problems.append(f"ledger: {e}")
    if a.patch:
        got = Patch.load(a.patch).fingerprint
        if got != receipt["fingerprint"]:
            problems.append("patch fingerprint does not match the receipt")
        else:
            print(f"patch matches receipt, {got[:16]}")
    print(f"verdict in receipt: {receipt['verdict']}")
    for key, passed in receipt["gates"].items():
        print(f"  {'pass' if passed else 'FAIL'}  {key}")
    for p in problems:
        print(f"problem: {p}")
    return 1 if problems else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lesion", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("diff", help="what changed between two checkpoints, and how")
    p.add_argument("base")
    p.add_argument("other")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="screen N random spans per tensor instead of reading everything")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("patch", help="write the difference between two checkpoints as a patch")
    p.add_argument("base")
    p.add_argument("other")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_patch)

    p = sub.add_parser("apply", help="rebuild a model from a base and a patch, verifying hashes")
    p.add_argument("base")
    p.add_argument("patch")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("doctor", help="run a model through a fixed battery and report what goes wrong")
    p.add_argument("model")
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("export", help="write a patch as a LoRA for PEFT or llama.cpp")
    p.add_argument("base")
    p.add_argument("patch")
    p.add_argument("--format", choices=["peft", "gguf"], required=True)
    p.add_argument("--arch", help="GGUF architecture name, for --format gguf")
    p.add_argument("--energy", type=float, default=0.999, help="share of the change to keep")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("verify", help="check a lab receipt against its ledger and patch")
    p.add_argument("receipt")
    p.add_argument("ledger")
    p.add_argument("--patch")
    p.set_defaults(fn=cmd_verify)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
