"""Precision tools to tune, edit and verify local language models."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.error
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from . import __version__
from . import attest as at
from .check import (
    DIFFERENT,
    EQUIVALENT,
    IDENTICAL,
    MERGED,
    MISSING,
    NEVER_APPLIED,
    ROUNDING,
    check,
    name_base,
)
from .clock import check_claim, make_claim
from .conformance import check_binding, check_patch
from .diff import Report, compare
from .formats import Unsupported
from .hop import document as hop_document
from .hop import render as hop_render
from .hop import trace
from .identity import identify
from .ledger import BrokenChain, read_entries
from .ledger import verify as verify_ledger
from .merge import merge
from .methods import DEFAULTS, Recipe, merge_with
from .methods import verify as verify_recipe
from .numerics import from_float, to_float
from .patch import Patch, PatchMismatch, make_patch, write_checkpoint, write_safetensors
from .protocol import (
    ProtocolError,
    decode_envelope,
    envelope,
    new_key,
    open_envelope,
    read_private,
    read_public,
    root_of,
    strict_loads,
    tensor_table,
)
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


TRAIN_HELP = """Fine-tune MODEL on DATA with mlx-lm (LoRA), then write what changed as a Neyra patch.

Order: tune lint for the mlx trainer (stops on an error unless --force), LoRA training, a low-rank patch built from
the adapter's factors, a check of what it builds against base plus adapter, and a plain summary. The base model is never written to.

Defaults, from mlx-lm's LoRA guide and sized for a few dozen to a few thousand rows:
  rank 8, scale 20, dropout 0, the last 16 layers (all, if fewer), Adam at 2e-5 (the guide's 1e-5 barely moves a
  run of under 100 steps), batch 4, rows cut at 512 tokens, 3 passes over the data. A tenth of the rows is held
  out for the loss before and after. MLX is told to stay under 60% of RAM, and gradient checkpointing turns on
  when the weights take over a quarter of that, so a 16 GB Mac can train a model up to about 4B parameters.
Needs an Apple silicon Mac and: uv tool install 'neyra[tune,mlx]'
"""


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
        share = 100 * read / max(total, 1)
        lines.append(f"coverage: {megabytes(read)} of {megabytes(total)} read ({share:.1f}%), "
                     f"{r.fully_read} of {r.tensors} paired tensors read in full")
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


def report_document(r: Report, seconds: float, read: int, total: int) -> dict:
    def structure(s):
        return {"name": s.name, "form": s.form, "recipe": s.recipe, "changed": s.changed, "rows": s.rows,
                "cols": s.cols, "relative": s.relative, "rank": s.rank, "energy": s.energy, "alpha": s.alpha,
                "side": s.side, "residual": s.residual}

    return {"base": r.base, "other": r.other, "tensors": r.tensors, "unchanged": r.unchanged, "sampled": r.sampled,
            "changes": [structure(s) for s in r.changes], "suspected": r.suspected, "added": r.added,
            "removed": r.removed, "retyped": r.retyped, "shared_directions": r.shared, "bytes_read": read,
            "coverage": {"bytes_read": read, "bytes_total": total, "tensors_fully_read": r.fully_read,
                         "tensors_paired": r.tensors},
            "seconds": seconds,
            "findings": [{"severity": f.severity, "kind": f.kind, "detail": f.detail} for f in r.findings]}


def cmd_diff(a) -> int:
    base, other = Checkpoint.open(a.base), Checkpoint.open(a.other)
    t = time.time()
    r = compare(base, other, sample_blocks=a.sample)
    seconds = time.time() - t
    read = base.bytes_read + other.bytes_read
    total = base.size + other.size
    emit(a, report_document(r, seconds, read, total), render(r, seconds, read, total))
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


def settings(method: str, pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if key not in DEFAULTS[method]:
            raise Fail(USAGE, f"{method} has no setting {key!r}; it takes {', '.join(DEFAULTS[method]) or 'none'}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def cmd_merge(a) -> int:
    if len(a.patches) < 2:
        raise Fail(USAGE, "merge needs at least two patches")
    base = Checkpoint.open(a.base)
    patches = [Patch.load(p) for p in a.patches]
    if a.method == "sum":
        m = merge(base, patches)
    else:
        weights = a.weights or None
        if weights and len(weights) != len(patches):
            raise Fail(USAGE, "give one --weights value per patch, in the order the patches are listed")
        m = merge_with(base, patches, Recipe.make(a.method, [p.id for p in patches], weights, **settings(a.method, a.set)))
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


def cmd_recompute(a) -> int:
    base, merged = Checkpoint.open(a.base), Patch.load(a.merged)
    wrong = verify_recipe(base, merged, [Patch.load(p) for p in a.parents])
    recipe = merged.meta["recipe"]
    text = (f"recomputed the {recipe['method']} merge of {len(a.parents)} patches from its recipe: "
            + ("every tensor matches bit for bit" if not wrong else f"{len(wrong)} tensors differ, such as {wrong[0]}"))
    emit(a, {"method": recipe["method"], "recipe_id": merged.meta.get("recipe_id"), "different": wrong}, text)
    return FINDINGS if wrong else OK


def cmd_name_base(a) -> int:
    merged, adapter = Checkpoint.open(a.merged), Checkpoint.open(a.adapter)
    bases = [Checkpoint.open(s) for s in [a.base, *a.also]]
    ranked = name_base(merged, adapter, bases, a.sample, a.ulps)
    text = [f"merged  {merged.spec}", f"adapter {adapter.spec}", ""]
    for c in ranked:
        state = ("reproduces the sampled edited tensors" if c.reproduces else
                 f"does not: {c.off} different, {c.missing} missing" if c.off + c.missing else "nothing to compare")
        shared = f", agrees on every sampled span with {', '.join(c.shares)}" if c.shares else ""
        text.append(f"{c.spec}: {c.exact} exact, {c.close} within {a.ulps} step{'s' * (a.ulps != 1)}; {state}{shared}")
    best = [c for c in ranked if c.reproduces]
    if not best:
        text.append("\nnone of the candidates plus the adapter gives this model back")
    elif len(best) > 1 and best[0].exact == best[1].exact and best[0].close == best[1].close:
        text.append(f"\n{len(best)} candidates fit equally well, so the merge cannot say which one it was made from")
    else:
        text.append(f"\nmade from {best[0].spec} (arithmetic: {', '.join(best[0].paths) or 'see per tensor'})")
    emit(a, {"merged": merged.spec, "adapter": adapter.spec, "sampled_tensors": a.sample,
             "candidates": [vars(c) | {"reproduces": c.reproduces} for c in ranked]}, "\n".join(text))
    return OK if best else FINDINGS


def cmd_identity(a) -> int:
    base, model = Checkpoint.open(a.base), Checkpoint.open(a.model)
    adapter = Checkpoint.open(a.adapter) if a.adapter else None
    v = identify(base, model, full=a.full, adapter=adapter)
    m = v.measure
    text = [f"base   {base.spec}", f"model  {model.spec}", "", v.label.upper(), v.detail]
    if m and m.probes:
        text.append(f"{len(m.probes)} spans sampled, {m.unchanged:.0%} bit for bit the base's, "
                    f"{(base.bytes_read + model.bytes_read) / 1e6:.1f} MB read")
    text += v.note
    emit(a, {"base": base.spec, "model": model.spec, "verdict": v.label, "detail": v.detail, "exact": v.exact,
             "paths": v.paths, "notes": v.note,
             "measure": None if m is None else {"cos": m.cos, "relative_distance": m.rel, "unchanged": m.unchanged,
                                               "spans": len(m.probes), "only_base": len(m.only_base),
                                               "only_model": len(m.only_model)}}, "\n".join(text))
    return OK


def render_check(c, base: Checkpoint, merged: Checkpoint, adapter: Checkpoint | None) -> str:
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
    return "\n".join(text)


def cmd_check(a) -> int:
    if a.parents:
        return cmd_recompute(a)
    if a.ulps is None:
        a.ulps = 1 if a.adapter else 0
    if a.also:
        if not a.adapter:
            raise Fail(USAGE, "naming the real base of a merge needs the adapter that was merged")
        return cmd_name_base(a)
    base, merged = Checkpoint.open(a.base), Checkpoint.open(a.merged)
    adapter = Checkpoint.open(a.adapter) if a.adapter else None
    c = check(base, merged, adapter, a.ulps)
    counts = {s: c.count(s) for s in (IDENTICAL, MERGED, ROUNDING, DIFFERENT, MISSING, NEVER_APPLIED)}
    emit(a, {"base": base.spec, "merged": merged.spec, "adapter": adapter.spec if adapter else None,
             "equivalent": c.equivalent, "paths": c.paths, "recast": sorted(c.recast), "counts": counts,
             "faint": [t.name for t in c.faint], "adapter_tensors": c.targeted,
             "adapter_modules": c.modules, "added": c.added, "leftover_adapter_tensors": c.leftover,
             "tensors": [vars(t) for t in c.tensors]}, render_check(c, base, merged, adapter))
    return OK if c.equivalent else FINDINGS


def cmd_hop(a) -> int:
    if a.predict:
        return cmd_forecast(a)
    from .tune import share

    verdict = trace(a.artifacts, a.ulps)
    emit(a, hop_document(verdict), hop_render(verdict))
    share.offer(lambda: share.build_hop(verdict, a.artifacts[0]), a.json)
    return OK if verdict.ok else FINDINGS


def cmd_tune_lint(a) -> int:
    try:
        from .tune import report, share
        from .tune.lint import lint
        from .tune.tok import TokenizerError
        from .tune.trainers import Unsupported
    except ModuleNotFoundError as e:
        raise Fail(ENVIRONMENT, f"this needs {e.name}. Install it with: pip install 'neyra[tune]'") from e

    try:
        r = lint(a.data, a.model, a.trainer, a.gguf, a.val, a.max_length, a.mask_prompt, a.show)
    except (Unsupported, TokenizerError) as e:
        raise Fail(BAD_INPUT, str(e)) from e
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] not in ("transformers", "tokenizers", "jinja2"):
            raise
        raise Fail(ENVIRONMENT, f"this needs {e.name}. Install it with: uv tool install 'neyra[tune]'") from e
    emit(a, report.document(r), report.render(r))
    share.offer(lambda: share.build_lint(r, a.model), a.json)
    return FINDINGS if r.errors else OK


def cmd_tune_verdict(a) -> int:
    from .tune import cases as cases_mod
    from .tune import prompts, share, verdict, verdict_report
    from .tune.engines import EngineError, open_backends

    try:
        cases = cases_mod.read(a.cases)
        score_fn = cases_mod.scorer(a.scorer)
        pool = cases_mod.read(a.examples)
        seen = {c.input for c in cases}
        usable = [c for c in pool if c.input not in seen]
        if not usable:
            raise cases_mod.CaseError(f"every example in {a.examples} is also one of the cases")
        shots = cases_mod.examples(usable, a.shots, a.seed)
        chat = {"chat": True, "plain": False}.get(a.format)
        template = a.template.replace("\\n", "\n")
        stop = tuple(s.replace("\\n", "\n") for s in a.stop)
    except cases_mod.CaseError as e:
        raise Fail(BAD_INPUT, str(e)) from e
    try:
        backends = open_backends(a.base, a.tuned, a.gguf, a.max_tokens, a.threads)
    except EngineError as e:
        raise Fail(BAD_INPUT, str(e)) from e
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] not in ("transformers", "tokenizers", "jinja2", "torch"):
            raise
        raise Fail(ENVIRONMENT, f"this needs {e.name}. Install it with: uv tool install 'neyra[torch,tune]'") from e
    try:
        if chat is None:
            chat = backends.has_chat_template
        fmt = prompts.Format(chat, template, stop if stop or chat else ("\n",), a.system)
        say = (lambda m: None) if a.json else (lambda m: print(m, file=sys.stderr))
        r = verdict.run(cases, backends, fmt, score_fn, a.scorer, shots, a.margin, a.cases, quants=None, say=say)
    except cases_mod.CaseError as e:
        raise Fail(BAD_INPUT, str(e)) from e
    finally:
        backends.close()
    if len(usable) < len(pool):
        r.warnings.append(f"{len(pool) - len(usable)} examples were dropped because they are also test cases")
    if len(shots) < a.shots:
        r.warnings.append(f"only {len(shots)} examples were available, so the prompt has {len(shots)}")
    doc = verdict_report.document(r)
    if a.save:
        Path(a.save).write_text(json.dumps({"schema": SCHEMA, **doc}, indent=2))
    emit(a, doc, verdict_report.render(r))
    share.offer(lambda: share.build_verdict(r, a.base), a.json)
    return OK if r.passed else FINDINGS


def cmd_stats(a) -> int:
    from .tune import share

    if a.action == "show":
        last = share.last_report()
        emit(a, {"report": json.loads(last) if last else None},
             last.rstrip() if last else "No report has been sent from this machine.")
        return OK
    if a.action in ("on", "off"):
        try:
            share.write_choice("ask" if a.action == "on" else "no")
        except OSError as e:
            raise Fail(ENVIRONMENT, f"cannot write {share.config_dir() / 'stats.json'}: {e}") from e
    choice = share.read_choice()
    env = share.opted_out()
    state = ("off: NEYRA_NO_STATS or DO_NOT_TRACK is set" if env else
             {"ask": "not decided: asks once, after the next result in a terminal",
              "yes": "on: sends an anonymous report after results in a terminal", "no": "off"}[choice])
    where = "set" if share.ENDPOINT else "not set in this build, so nothing is asked or sent"
    emit(a, {"choice": choice, "environment_opt_out": env, "endpoint_set": bool(share.ENDPOINT)},
         f"stats {state}\nreceiver {where}\nsetting {share.config_dir() / 'stats.json'}")
    return OK


def cmd_tune_train(a) -> int:
    from .tune import share, train

    try:
        train.require_apple()
        import mlx_lm  # noqa: F401  (main() turns a missing mlx into the neyra[mlx] message)
        try:
            opts = train.Options(a.model, a.data, a.out, a.iters, a.rank, a.lr, a.val, a.mask_prompt, a.force,
                                 a.batch_size, a.max_length, a.memory_gb, a.seed)
            path = local_path(a.model)
            report = train.lint_gate(opts, path)
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] not in ("transformers", "tokenizers", "jinja2"):
                raise
            raise Fail(ENVIRONMENT, f"this needs {e.name}. Install it with: uv tool install 'neyra[tune,mlx]'") from e
        result = train.run(opts, Path(path), report)
    except train.Stop as e:
        gate = getattr(e, "report", None)
        if gate is not None:
            share.offer(lambda: share.build_train(a.model, gate), a.json)
        raise Fail({"environment": ENVIRONMENT, "check": FINDINGS}.get(e.kind, BAD_INPUT), str(e)) from e
    emit(a, train.summary_document(result), train.ending(result))
    share.offer(lambda: share.build_train(a.model, report, result), a.json)
    return OK


def cmd_forecast(a) -> int:
    from .hop import FineTune
    from .survival import forecast

    if len(a.artifacts) != 2:
        raise Fail(USAGE, "--predict takes the base and the adapter")
    base, adapter = Checkpoint.open(a.artifacts[0]), Checkpoint.open(a.artifacts[1])
    kinds = [k.upper() for k in a.predict.split(",")]
    pools = forecast(base, FineTune(base, adapter), kinds)
    text, rows = ["share of the adapter's change that is stored, by type (K-quants cannot be forecast here):"], {}
    for kind, pool in pools.items():
        if pool.value is None:
            text.append(f"  {kind:<8}cannot be forecast, gguf-py cannot quantize it")
            continue
        rows[kind] = {"survival": pool.value, "error": pool.error, "cosine": pool.cosine}
        text.append(f"  {kind:<8}{pool.value:.0%} (+/-{2 * pool.error:.0%}), cosine {pool.cosine:.2f}")
    emit(a, {"forecast": rows}, "\n".join(text))
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
    from huggingface_hub.errors import HfHubHTTPError

    repo, _, rev = spec.partition("@")
    try:
        return snapshot_download(repo, revision=rev or None)
    except HfHubHTTPError as e:
        raise Fail(BAD_INPUT, f"{spec}: not a folder on disk, and the Hub has no readable repo by that name "
                              f"(HTTP {e.response.status_code if e.response is not None else '?'}; gated repos need HF_TOKEN)") from e


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
    from dataclasses import replace

    from .edit import Settings, Words, default_layers, load_requests, one_fact, what_it_says
    from .runtime import open_runtime
    from .workflows import edit_facts

    path = local_path(a.model)
    single = a.target is not None
    if single:
        requests = [replace(one_fact(a.requests, a.target), neighbours=tuple(a.neighbour))]
        a.lam = 100.0 if a.lam is None else a.lam
    else:
        requests = load_requests(a.requests)
    names = ("width", "solver", "lam", "batch", "steps", "lr", "contexts", "polish", "anchor", "clamp", "max_rank", "seed",
             "window", "held", "neutral")
    ck = Checkpoint.open(path)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else default_layers(ck) if single else None
    settings = Settings(layers=layers,
                        **{n: getattr(a, n) for n in names if getattr(a, n) is not None})
    out = Path(a.out)
    ledger = Path(a.ledger or out.with_suffix(".ledger.jsonl"))
    key = read_private(a.key) if a.key else None
    rt = open_runtime(path, a.backend)
    if single:
        old = what_it_says(rt, Words(rt.tokenizer), requests[0])
        if old.lower() == a.target.lower():
            raise Fail(BAD_INPUT, f"the model already says {old} here")
        requests = [replace(requests[0], true=old)]
    done = edit_facts(rt, ck, requests, settings, neutral=a.text,
                      key=key, ledger=ledger, track=a.track)
    done.patch.save(out)
    receipt = Path(a.receipt or out.with_suffix(".receipt.json"))
    receipt.write_text(json.dumps(done.receipt), encoding="utf-8", newline="\n")
    r, b, c = done.report, done.report["before"], done.report["after"]
    ranks = sorted(set(r["ranks"].values()))
    text = [f"layers {r['layers']}, {len(requests)} edits, {done.patch.entries:,} stored entries, ranks {ranks} -> {out}",
            f"patch {done.patch.id}", ""]
    for name in ("efficacy", "paraphrase", "specificity", "exact", "score"):
        text.append(f"{name:12} {percent(b[name]):>7} -> {percent(c[name]):>7}")
    text.append(f"{'margin':12} {nats(b['margin']):>7} -> {nats(c['margin']):>7}")
    for name in ("efficacy", "paraphrase"):
        was, now = r["live_before"][name], r["live_after"][name]
        if was is not None and now is not None:
            text.append(f"{'decoded ' + name[:4]:12} {percent(was):>7} -> {percent(now):>7}")
    text.append(f"neutral text: KL {r['neutral_kl']:.4f} nats/token, fluency {r['fluency_before']:.2f} -> "
                f"{r['fluency_after']:.2f}")
    text += [f"  {'pass' if ok else 'FAIL'}  {gate}" for gate, ok in r["gates"].items()]
    if b["specificity"] is None:
        text.append("neighbouring facts were not measured: add --neighbour PROMPT, or use a requests file")
    if r["warning"]:
        text.append(f"warning: {r['warning']}")
    missed = [gate for gate, ok in r["gates"].items() if not ok]
    if done.verdict == "PASS":
        text.append(f"verdict {done.verdict}")
    elif missed == ["specificity"] and b["specificity"] is None:
        text.append(f"not claimed: the fact took, but nothing checked its neighbours. The patch is in {out}.")
    else:
        text.append("verdict FAIL: this edit is not claimed as a success. The patch is written for inspection only.")
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


def cmd_import(a) -> int:
    from . import formats

    base = Checkpoint.open(a.base)
    name = a.format or formats.detect(a.source)
    if name is None:
        raise Fail(USAGE, f"cannot tell what format {a.source} is; name it with --format ({', '.join(formats.NAMES)})")
    options = {}
    if name == "delta-league" and a.identity:
        options["identity"] = a.identity
    if name == "peft":
        options["merge"] = a.merge
    patch = formats.read(name, a.source, base, **options)
    patch.save(a.out)
    emit(a, {"format": name, "tensors": len(patch.edits), "entries": patch.entries, "path": a.out,
             "patch_id": patch.id, "base_root": patch.base_root, "result_root": patch.result_root},
         f"{name}: {len(patch.edits)} tensors, {patch.entries:,} stored entries -> {a.out}\npatch {patch.id}")
    return OK


def cmd_export(a) -> int:
    from . import formats
    from .formats import ballast, deltaleague, openwsp, weightspace, wvc

    base = Checkpoint.open(a.base)
    patch = Patch.load(a.patch)
    note = ""
    if a.format == "peft":
        from .export import to_peft

        info = to_peft(patch, base, a.out, a.energy)
    elif a.format == "gguf":
        from .export import to_gguf_lora

        if not a.arch:
            raise Fail(USAGE, "gguf export needs --arch, the architecture name in the base GGUF, such as qwen35")
        info = to_gguf_lora(patch, base, a.out, a.arch, a.energy)
    elif a.format == "delta-league":
        if a.parent_meta:
            meta = json.loads(Path(a.parent_meta).read_text(encoding="utf-8"))
            parent, digest, model = meta["id"], meta["content_hash"], meta["model_id"]
        elif a.parent_id and a.parent_hash and a.model_id:
            parent, digest, model = a.parent_id, a.parent_hash, a.model_id
        else:
            raise Fail(USAGE, "delta-league export needs --parent-meta delta-model.json, "
                              "or --parent-id, --parent-hash and --model-id")
        manifest = deltaleague.dump(patch, base, a.out, parent, digest, model)
        info = {"bytes": manifest["artifacts"][0]["size_bytes"], "tensors": len(manifest["tensors"])}
    elif a.format == "openwsp":
        manifest = openwsp.dump(patch, base, a.out, base_model=a.model_id)
        info = {"bytes": (Path(a.out) / "edit.npz").stat().st_size, "edit_sha256": manifest["edit_sha256"]}
    elif a.format == "weightspace":
        weightspace.dump(patch, base, a.out)
        info = {"bytes": Path(a.out).stat().st_size}
    elif a.format == "ballast":
        inexact = ballast.dump(patch, base, a.out)
        info = {"bytes": (Path(a.out) / "model.safetensors").stat().st_size, "inexact": inexact}
        if inexact:
            note = f"\n{len(inexact)} tensors would not rebuild exactly under `ballast apply`: {', '.join(inexact[:3])}"
    else:
        before, after = wvc.dump(patch, base)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "base.manifest.json").write_text(json.dumps(before, indent=2) + "\n")
        (out / "result.manifest.json").write_text(json.dumps(after, indent=2) + "\n")
        info = {"base_root": before["root"], "result_root": after["root"]}
        note = f"\nwvc roots: base {before['root']}, result {after['root']}"
    del formats
    emit(a, {"path": a.out, **info}, f"{info.get('bytes', 0):,} bytes -> {a.out}{note}")
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
        Path(out).write_text(json.dumps(link(index, read_private(a.sign)), indent=1), encoding="utf-8", newline="\n")
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
    Path(a.out).write_text(json.dumps(proof, indent=1), encoding="utf-8", newline="\n")
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
        public = read_public(a.key) if a.key else None
        opened = open_link(strict_loads(Path(a.link).read_text(encoding="utf-8")), public)
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
        claim = strict_loads(Path(a.check).read_text(encoding="utf-8"))
        try:
            problems = check_claim(claim, Checkpoint.open(a.a), Checkpoint.open(a.b))
        except (KeyError, TypeError, AttributeError) as e:
            raise Fail(BAD_INPUT, f"{a.check}: not a clock claim ({type(e).__name__}: {e})") from e
        text = [f"{claim['verdict']} on {claim['n']:,} entries of {claim['tensor']}, rows reproduced from the bytes"
                if not problems else "claim does not hold"] + [f"problem: {p}" for p in problems]
        emit(a, {"verdict": claim["verdict"], "problems": problems}, "\n".join(text))
        return FAILED if problems else OK
    if not a.rows:
        raise Fail(USAGE, "name the dormant rows with --rows, such as 128010:128256")
    (a_ck, a_index), (b_ck, b_index) = open_indexed(a.a, a.a_index), open_indexed(a.b, a.b_index)
    claim = make_claim(a_ck, b_ck, a_index, b_index, a.tensor, a.rows)
    Path(a.out).write_text(json.dumps(claim, indent=1), encoding="utf-8", newline="\n")
    scale = f", c in [{claim['c_lo_dec']}, {claim['c_hi_dec']}]" if "c_lo_dec" in claim else ""
    emit(a, {"path": a.out, "verdict": claim["verdict"], "id": claim["id"]},
         f"{claim['verdict']}{scale} on {claim['n']:,} entries -> {a.out}\n"
         f"read {megabytes(a_ck.bytes_read + b_ck.bytes_read)}; bound to rows roots "
         f"{claim['a_rows_root'][7:19]} and {claim['b_rows_root'][7:19]}")
    return OK


def read_public_key(path: str) -> bytes:
    try:
        key = read_public(path)
    except (ValueError, OSError) as e:
        raise Fail(BAD_INPUT, f"{path}: not a public key; use the .pub file that neyra keygen wrote") from e
    if len(key) < 32:
        raise Fail(BAD_INPUT, f"{path}: not a public key; use the .pub file that neyra keygen wrote")
    return key


def cmd_verify(a) -> int:
    public = read_public_key(a.key) if a.key else None
    doc = strict_loads(Path(a.receipt).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise Fail(BAD_INPUT, f"{a.receipt}: not a receipt or row proof, which are JSON objects")
    if doc.get("type") == PROOF_TYPE:
        return verify_rows(a, doc)
    kind = decode_envelope(doc)[0].get("predicateType") if isinstance(doc, dict) and "payload" in doc else None
    if kind == at.PREDICATE_TYPE:
        return verify_transformation(a, doc, public)
    if kind == at.STATUS_TYPE:
        return verify_status(a, doc, public)
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


def parse_recipe(items: list[str]) -> dict:
    recipe = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise Fail(USAGE, f"--recipe takes name=value, not {item!r}")
        recipe[key] = int(value) if value.isdigit() else value
    return recipe


def cmd_attest(a) -> int:
    base_dir, result_dir = Path(local_path(a.base)), Path(local_path(a.result))
    base_ck, result_ck = Checkpoint.open(str(base_dir)), Checkpoint.open(str(result_dir))
    before, after = tensor_table(base_ck), tensor_table(result_ck)
    base_root, result_root = root_of(before), root_of(after)
    size = sum(r.nbytes for ck in (base_ck, result_ck) for r in ck.tensors.values())
    recipe, evidence, kind, grade = parse_recipe(a.recipe), [], a.kind, "observed"
    if a.patch:
        patch = Patch.load(a.patch)
        if check_patch(a.patch, base_dir) != "ok" or patch.result_root != result_root:
            raise Fail(BAD_INPUT, "the patch does not turn this base into this result")
        recipe["patch_id"], grade = patch.id, "exact"
        kind = kind or ("edit" if patch.meta.get("method") == "closed form edit" else "patch")
        evidence.append({"role": "patch", "digest": patch.id})
    if a.receipt:
        payload = decode_envelope(strict_loads(Path(a.receipt).read_text(encoding="utf-8")))[1]
        claim = open_envelope(strict_loads(Path(a.receipt).read_text(encoding="utf-8"))).statement["predicate"]["claim"]
        if claim["result_root"] != result_root:
            raise Fail(BAD_INPUT, "the receipt is about a different result")
        evidence.append({"role": "receipt", "digest": at.statement_digest(payload)})
    base_files = result_files = []
    if not a.no_files:
        base_files, result_files = (at.file_digests(d, tuple(a.ignore), a.allow_symlinks) for d in (base_dir, result_dir))
    if a.oms and result_files:
        declared, root = at.oms_files(json.loads(Path(a.oms).read_text(encoding="utf-8")))
        if declared != result_files or at.oms_root(declared) != root:
            raise Fail(BAD_INPUT, "the OMS signature does not describe the files in this result directory")
    statement = at.transformation(
        kind=kind or "unknown", grade=grade, meaning=a.meaning, base_root=base_root, result_root=result_root,
        base_files=base_files, result_files=result_files, result_name=a.name or result_dir.resolve().name,
        recipe=recipe, extent=at.measure(base_ck, result_ck, before, after), ignore=tuple(a.ignore),
        coverage={"bytes_read": size, "bytes_total": size}, attempts=a.attempts,
        evidence=evidence, time=a.time)
    doc = envelope(statement, read_private(a.key) if a.key else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "change.statement.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "change.cdx.json").write_text(json.dumps(at.to_cyclonedx(doc), indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "CHANGE.md").write_text(at.to_page(doc), encoding="utf-8", newline="\n")
    emit(a, {"path": str(out), "kind": statement["predicate"]["kind"], "grade": grade, "base_root": base_root,
             "result_root": result_root, "signed": bool(doc["signatures"])},
         f"{statement['predicate']['kind']} ({grade}), {'signed' if doc['signatures'] else 'NOT signed'} -> {out}\n"
         f"change.statement.json, change.cdx.json, CHANGE.md")
    return OK


def cmd_render(a) -> int:
    doc = strict_loads(Path(a.statement).read_text(encoding="utf-8"))
    at.open_transformation(doc)
    text = json.dumps(at.to_cyclonedx(doc), indent=1) + "\n" if a.format == "cdx" else at.to_page(doc)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return OK


def verify_transformation(a, doc: dict, public: bytes | None) -> int:
    opened = at.open_transformation(doc, public)
    p = opened.statement["predicate"]
    problems = []
    sig = "signature verified" if opened.signed else "NOT signed: an unauthenticated record"
    text = [f"transformation statement well formed, {sig}",
            f"{p['kind']}, evidence {p['grade']}, signer's meaning: {p['meaning']}"]
    if a.patch:
        patch = Patch.load(a.patch)
        if at.bind_patch(opened.statement, patch) != "ok":
            problems.append("patch does not match the statement")
        else:
            text.append(f"patch matches, {patch.id[7:23]}")
    if a.base:
        before = tensor_table(Checkpoint.open(a.base))
        if root_of(before) != p["base"]["weights_root"]:
            problems.append("base weights root differs from the statement")
        else:
            text.append("base weights root matches")
        if a.patch and check_patch(a.patch, a.base) != "ok":
            problems.append("patch does not apply to this base")
    if a.result:
        code = at.bind_model(opened.statement, a.result)
        if code != "ok":
            problems.append(code)
        else:
            text.append("result weights root and file digests match")
        if a.base and p["extent"] is not None:
            counted = at.measure(Checkpoint.open(a.base), Checkpoint.open(a.result), before,
                                 tensor_table(Checkpoint.open(a.result)))
            if counted != p["extent"]:
                problems.append("extent differs from what the models show")
    text += [f"problem: {x}" for x in problems]
    text.append("not measured: " + ", ".join(p["not_measured"]) + ". A signature shows who said this, not that it is true")
    emit(a, {"kind": p["kind"], "grade": p["grade"], "signed": opened.signed, "problems": problems}, "\n".join(text))
    return FAILED if problems else OK


def verify_status(a, doc: dict, public: bytes | None) -> int:
    opened = at.open_status(doc, public)
    p = opened.statement["predicate"]
    text = [f"{p['action']} record well formed, {'signature verified' if opened.signed else 'NOT signed'}",
            f"target {p['target'][7:23]}, reason: {p['reason']}"]
    emit(a, {"action": p["action"], "target": p["target"], "signed": opened.signed}, "\n".join(text))
    return OK


def load_statement(path: str) -> at.Opened:
    doc = strict_loads(Path(path).read_text(encoding="utf-8"))
    statement, payload = decode_envelope(doc)
    return at.Opened(statement, bool(doc.get("signatures")), payload)


def write_status(a, action: str, replacement: at.Opened | None = None) -> int:
    target = load_statement(a.target)
    root = at.weights_root_of(target.statement)
    if root is None:
        raise Fail(BAD_INPUT, "the target is neither a transformation statement nor a receipt")
    record = at.status(action=action, target=at.statement_digest(target.payload), weights_root=root,
                       reason=a.reason, meaning=a.meaning, time=a.time,
                       replacement=at.statement_digest(replacement.payload) if replacement else None)
    doc = envelope(record, read_private(a.key) if a.key else None)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    emit(a, {"path": a.out, "action": action, "target": record["predicate"]["target"]},
         f"{action} record for {record['predicate']['target'][7:23]} -> {a.out}\n"
         "the original statement is untouched; the record says nothing unless a reader trusts its key")
    return OK


def cmd_supersede(a) -> int:
    return write_status(a, "supersede", load_statement(a.replacement))


def cmd_withdraw(a) -> int:
    return write_status(a, "withdraw")


def cmd_status(a) -> int:
    target = load_statement(a.target)
    trusted = [read_public(k) for k in a.key]
    records, ignored = [], 0
    for path in a.records:
        doc = strict_loads(Path(path).read_text(encoding="utf-8"))
        for public in trusted:
            try:
                records.append(at.open_status(doc, public))
                break
            except ProtocolError:
                continue
        else:
            ignored += 1
    answer = at.resolve(target, records)
    text = [f"{answer['state']}"]
    text += [f"  withdrawn: {w['reason']} ({w['time']})" for w in answer["withdrawn"]]
    text += [f"  superseded by {r[7:23]}" for r in answer["superseded_by"]]
    if ignored:
        text.append(f"{ignored} records were not signed by a trusted key, or were malformed, and were ignored")
    emit(a, {**answer, "ignored": ignored}, "\n".join(text))
    return OK if answer["state"] == "current" else FINDINGS


def cmd_keygen(a) -> int:
    key = new_key(a.out, a.algorithm)
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
    if a.story == "edit":
        return edit_demo()
    if a.story == "files":
        return demo_files(a.dir)
    from .demo import run

    if run():
        return OK
    print("neyra: the demo did not behave as designed, which is a bug; please report it", file=sys.stderr)
    return FAILED


def demo_files(target: str | None) -> int:
    """Write the demo's toy models to disk, so every command can be tried by hand."""
    from .demo import build

    if not target:
        raise Fail(USAGE, "name a folder to write them to: neyra demo files DIR")
    out = Path(target)
    if out.exists() and any(out.iterdir()):
        raise Fail(BAD_INPUT, f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    build(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote base, merged, forgot-to-merge and adapter to {out}, {size / 1000:.0f} KB in all\n\ntry:\n"
          f"  neyra check {out}/base {out}/forgot-to-merge --adapter {out}/adapter\n"
          f"  neyra check {out}/base {out}/merged --adapter {out}/adapter\n"
          f"  neyra diff {out}/base {out}/merged")
    return OK


def edit_demo() -> int:
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


def vectors_dir(given: str | None) -> Path:
    """The conformance vectors: the folder named, the copy shipped in the wheel, or the repo's own."""
    if given:
        return Path(given).expanduser()
    here = Path(__file__).resolve().parent
    for candidate in (here / "vectors", here.parents[1] / "spec" / "vectors"):
        if (candidate / "cases.json").is_file():
            return candidate
    raise Fail(ENVIRONMENT, "the conformance vectors are not in this install; pass the spec/vectors folder")


def cmd_conformance(a) -> int:
    from .conformance import run_vectors

    root = vectors_dir(a.vectors)
    if not (root / "cases.json").is_file():
        raise Fail(BAD_INPUT, f"{root}: no cases.json, so this is not a vectors folder")
    total = len(json.loads((root / "cases.json").read_text(encoding="utf-8")))
    failures = run_vectors(root)
    text = [f"FAIL {case}: want {want}, got {got}" for case, want, got in failures]
    text.append(f"{total - len(failures)}/{total} conformance vectors passed")
    emit(a, {"total": total, "failed": [{"case": c, "want": w, "got": g} for c, w, g in failures]}, "\n".join(text))
    return FINDINGS if failures else OK


def cmd_mcp(a) -> int:
    from .lab import Lab, Task
    from .mcp_server import build
    from .runtime import open_runtime

    path = local_path(a.model)
    lab = Lab(Path(a.root), path, Task.load(a.task), open_runtime(path, a.backend), a.passes, a.submissions, a.allow)
    build(lab).run()
    return OK


def parser(prog: str = "neyra") -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine readable output")
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def command(name, fn, help):
        p = sub.add_parser(name, help=help, parents=[common])
        p.set_defaults(fn=fn)
        return p

    p = command("demo", cmd_demo, "catch a merge that never took its adapter, on a toy model, with no downloads")
    p.add_argument("story", nargs="?", default="merge", choices=["merge", "edit", "files"],
                   help="merge (default); edit: plant an edit in a tiny model and recover it from the weights; "
                        "files DIR: write the toy models out to try the commands on")
    p.add_argument("dir", nargs="?", help="folder for `demo files`")

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
    p.add_argument("--method", default="sum", choices=["sum", *DEFAULTS], help="sum adds the deltas with unit weights")
    p.add_argument("--weights", type=float, nargs="+", help="one weight per patch, in the order the patches are listed")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="a method setting, such as density=0.5 or seed=7")

    p = command("recompute", cmd_recompute, "rebuild a merge from the recipe it carries and compare it bit for bit")
    p.add_argument("base")
    p.add_argument("merged", help="the merged patch")
    p.add_argument("parents", nargs="+", help="the patches it was merged from, in any order")

    p = command("check", cmd_check, "decide whether a merged model matches its base and adapter")
    p.add_argument("base")
    p.add_argument("merged", help="the merged model, or with --parents the merged patch")
    p.add_argument("--adapter", help="PEFT LoRA adapter directory that was merged into the base")
    p.add_argument("--ulps", type=int, default=None,
                   help="steps of the dtype an entry may differ by and still count as rounding (default 1 with an adapter, "
                        "0 without; at most 2%% of a tensor's entries may be off)")
    p.add_argument("--also", action="append", default=[], metavar="BASE",
                   help="another candidate base; with it, say which candidate the merge was made from (reads a sample)")
    p.add_argument("--sample", type=int, default=12, help="edited tensors to read per candidate with --also")
    p.add_argument("--parents", nargs="+", help="recompute a merged patch from these patches and the recipe it carries")

    p = command("identity", cmd_identity, "say what a model is relative to its declared base, from a few megabytes of tensors")
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--adapter", help="a PEFT adapter to try as the transform")
    p.add_argument("--full", action="store_true", help="hash every tensor to confirm identical weights, not a sample")

    p = command("hop", cmd_hop, "find the first hop that lost a fine-tune: base, adapter, merged model, GGUF")
    p.add_argument("artifacts", nargs="+", metavar="ARTIFACT", help="base, then the adapter if any, then each later artifact in order")
    p.add_argument("--ulps", type=int, default=1, help="steps of the dtype two tensors may differ by and still count as rounding")
    p.add_argument("--predict", metavar="TYPES", help="with a base and an adapter: forecast survival in these GGUF types, e.g. q8_0,q4_0")

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
    p.add_argument("requests", help='a prompt with the subject in brackets, "The capital of [France] is", then the '
                   "target; or a JSON file of {subject, prompt with {}, target} or CounterFact records")
    p.add_argument("target", nargs="?", help="the new answer, when the first argument is a prompt")
    p.add_argument("--neighbour", action="append", default=[], metavar="PROMPT",
                   help="a prompt whose answer must not change, for a single fact")
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
    p.add_argument("--clamp", type=float, help="cap on how far the value descent may move the output, in its norm")
    p.add_argument("--max-rank", type=int, dest="max_rank")
    p.add_argument("--seed", type=int)
    p.add_argument("--track", action="store_true", help="measure after every batch, to see sequential decay")
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

    p = command("import", cmd_import, "read an edit made by another tool as a Neyra patch")
    p.add_argument("base")
    p.add_argument("source", help="a file or folder in the other tool's format")
    p.add_argument("--format", choices=formats_names(), help="default: guessed from the contents, except ballast")
    p.add_argument("--identity", help="delta-league: the parent's identity string, to verify its content hash")
    p.add_argument("--merge", default="peft", help="peft: the merge path whose rounding to follow")
    p.add_argument("-o", "--out", required=True)

    p = command("export", cmd_export, "write a patch as a LoRA, or in another tool's edit format")
    p.add_argument("base")
    p.add_argument("patch")
    p.add_argument("--format", choices=["peft", "gguf", "delta-league", "openwsp", "weightspace", "ballast", "wvc"],
                   required=True)
    p.add_argument("--arch", help="GGUF architecture name, for --format gguf")
    p.add_argument("--parent-meta", help="delta-league: the delta-model.json of the parent")
    p.add_argument("--parent-id")
    p.add_argument("--parent-hash")
    p.add_argument("--model-id", help="delta-league: the model config id. openwsp: the base model's name")
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

    p = command("attest", cmd_attest, "sign a statement of what changed between two models")
    p.add_argument("base")
    p.add_argument("result")
    p.add_argument("-o", "--out", default="change", help="directory for the statement, CycloneDX file and one page summary")
    p.add_argument("--patch", help="the patch that makes the result from the base; gives the exact grade")
    p.add_argument("--receipt", help="a lab or edit receipt for the result, recorded as evidence")
    p.add_argument("--kind", choices=at.KINDS)
    p.add_argument("--meaning", choices=sorted(at.MEANINGS), default="authored")
    p.add_argument("--attempts", type=int, metavar="N",
                   help="how many results you made and tried before this one; left out, the statement says nobody counted")
    p.add_argument("--recipe", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--oms", metavar="MODEL.SIG", help="the OMS signature of the result; refuse if it describes other files")
    p.add_argument("--name", help="model name for the OMS subject; default the directory name")
    p.add_argument("--ignore", action="append", default=[], metavar="PATH", help="a path OMS was told to ignore")
    p.add_argument("--allow-symlinks", action="store_true")
    p.add_argument("--no-files", action="store_true", help="leave file digests out: no join to OMS, no whole file read")
    p.add_argument("--key", help="private key file, Ed25519 as neyra keygen writes it or a PEM P-256 key")
    p.add_argument("--time", help="RFC 3339 UTC time to record; default now")

    p = command("render", cmd_render, "derive the CycloneDX file or the one page summary from a statement")
    p.add_argument("statement")
    p.add_argument("--format", choices=["cdx", "page"], default="page")
    p.add_argument("-o", "--out")

    for name, fn, text in (("supersede", cmd_supersede, "record that a statement is replaced by another"),
                           ("withdraw", cmd_withdraw, "record that the signer no longer stands behind a statement")):
        p = command(name, fn, text)
        p.add_argument("target", help="the statement or receipt this concerns; it is read, never changed")
        if name == "supersede":
            p.add_argument("--replacement", required=True)
        p.add_argument("--reason", required=True)
        p.add_argument("--meaning", choices=sorted(at.MEANINGS), default="authored")
        p.add_argument("--key")
        p.add_argument("--time")
        p.add_argument("-o", "--out", required=True)

    p = command("status", cmd_status, "say whether a statement is current, superseded or withdrawn, by records you trust")
    p.add_argument("target")
    p.add_argument("records", nargs="*")
    p.add_argument("--key", action="append", required=True, metavar="PUB", help="a public key whose records count")

    p = command("verify", cmd_verify, "check a receipt, a transformation statement, or a row proof")
    p.add_argument("receipt", help="receipt or statement, or a row proof from neyra prove")
    p.add_argument("--base", help="base model, for a transformation statement")
    p.add_argument("--result", help="result model, for a transformation statement")
    p.add_argument("model", nargs="?", help="model to read the rows from, for a row proof")
    p.add_argument("--link", help="signed link from neyra index --sign, for a row proof")
    p.add_argument("--ledger")
    p.add_argument("--patch")
    p.add_argument("--key", help="public key file")

    p = command("clock", cmd_clock, "claim that two checkpoints hold one scalar relation on rows training never fed (it can be planted or scrubbed)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--rows", metavar="A:B[,C:D]", help="rows to compare, such as the reserved token ids")
    p.add_argument("--tensor", help="default: the input embedding")
    p.add_argument("--a-index")
    p.add_argument("--b-index")
    p.add_argument("--check", metavar="CLAIM", help="reproduce a claim from the bytes instead of making one")
    p.add_argument("-o", "--out", default="clock.claim.json")

    p = command("keygen", cmd_keygen, "create a key for signing receipts and statements")
    p.add_argument("-o", "--out", default="neyra.key")
    p.add_argument("--algorithm", choices=["ed25519", "ecdsa-p256"], default="ed25519",
                   help="ecdsa-p256 writes PEM files, the kind OMS key signing reads")

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

    p = command("conformance", cmd_conformance, "run the protocol's conformance vectors against this install")
    p.add_argument("vectors", nargs="?", help="a spec/vectors folder; default: the copy that ships with neyra")

    p = command("stats", cmd_stats, "anonymous stats for tune verdict: status, show the last report, on, off")
    p.add_argument("action", nargs="?", choices=["show", "on", "off"],
                   help="show: print the last report sent; on: ask after the next verdict; off: never ask or send")

    tune = command("tune", None, "check a tuning run before it starts and judge it afterwards")
    tune_sub = tune.add_subparsers(dest="tune_cmd", required=True)
    p = tune_sub.add_parser("lint", help="the token ids and loss mask a trainer will build from your data, and what is wrong with them",
                            parents=[common])
    p.set_defaults(fn=cmd_tune_lint)
    p.add_argument("data", help="a jsonl file: messages, prompt and completion, text, alpaca or sharegpt rows")
    p.add_argument("model", help="a model folder or Hugging Face repo; only tokenizer and config files are read")
    p.add_argument("--trainer", default="trl", choices=["trl", "unsloth", "axolotl", "mlx", "llama-factory"])
    p.add_argument("--gguf", metavar="FILE", help="compare the chat template and tokenizer with this GGUF file")
    p.add_argument("--val", metavar="FILE", help="a validation jsonl, checked for rows that also appear in DATA")
    p.add_argument("--max-length", type=int, metavar="N", help="the trainer's sequence length (default: its own default)")
    p.add_argument("--mask-prompt", action=argparse.BooleanOptionalAction, default=None,
                   help="whether the loss skips the prompt (default: what the trainer does out of the box)")
    p.add_argument("--show", type=int, default=0, metavar="N", help="print the first N rows with trained tokens marked")

    p = tune_sub.add_parser("verdict", help="after fine-tuning: did it work, and did you need it, on your own cases",
                            parents=[common])
    p.set_defaults(fn=cmd_tune_verdict)
    p.add_argument("base", help="the base model: a folder or Hugging Face repo")
    p.add_argument("tuned", help="the tuned model: a PEFT or mlx-lm adapter folder, or a folder of merged weights")
    p.add_argument("cases", help="a jsonl file of 30 to 200 held-out cases with an input and an expected output")
    p.add_argument("--examples", required=True, metavar="FILE",
                   help="a jsonl file the few-shot prompt takes its examples from, such as your training file")
    p.add_argument("--gguf", nargs="+", default=[], metavar="FILE", help="quantized files of the tuned model to test too")
    p.add_argument("--scorer", default="exact", help="exact, contains, regex:PATTERN or python:FILE (default: exact)")
    p.add_argument("--shots", type=int, default=8, metavar="N", help="examples in the few-shot prompt (default: 8)")
    p.add_argument("--seed", type=int, default=0, help="which examples the prompt gets")
    p.add_argument("--format", default="auto", choices=["auto", "chat", "plain"],
                   help="chat template or plain text prompts (default: chat when the tokenizer has a template)")
    p.add_argument("--template", default="{input}", help="plain format only: how an input becomes a prompt, "
                   "for example 'Q: {input}\\nA:'")
    p.add_argument("--system", default="", help="a system prompt for cases that carry none")
    p.add_argument("--stop", action="append", default=[], metavar="TEXT",
                   help="cut every answer at this text (plain format defaults to a newline)")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--margin", type=float, default=0.05,
                   help="two arms count as equal when the interval of their difference stays inside this (default: 0.05)")
    p.add_argument("--threads", type=int, default=None, help="threads for llama-server")
    p.add_argument("--save", metavar="FILE", help="also write the JSON document here, whatever the output format")
    p = tune_sub.add_parser("train", parents=[common], formatter_class=argparse.RawDescriptionHelpFormatter,
                            help="fine-tune a local model with mlx-lm's LoRA trainer and keep the result as an exact patch",
                            description=TRAIN_HELP)
    p.set_defaults(fn=cmd_tune_train)
    p.add_argument("model", help="a full precision model folder or Hugging Face repo")
    p.add_argument("data", help="a jsonl file: messages, prompt and completion, text, alpaca or sharegpt rows")
    p.add_argument("-o", "--out", required=True, help="a new folder for the adapter, the patch and run.json")
    p.add_argument("--iters", type=int, metavar="N", help="training steps (default: 3 passes over the data, 60 to 1000)")
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8)")
    p.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    p.add_argument("--val", metavar="FILE", help="validation jsonl (default: a tenth of the data is held out)")
    p.add_argument("--mask-prompt", action="store_true", help="train on the answers only")
    p.add_argument("--force", action="store_true", help="train even when tune lint reports errors")
    p.add_argument("--batch-size", type=int, default=4, help="rows per step (default: 4)")
    p.add_argument("--max-length", type=int, default=512, metavar="N",
                   help="longest row in tokens; longer rows are cut (default: 512)")
    p.add_argument("--memory-gb", type=float, metavar="GB", help="memory limit for MLX (default: 60%% of this Mac's RAM)")
    p.add_argument("--seed", type=int, default=0)

    command("doctor", cmd_doctor, "check this machine and what it can run")
    return ap


def formats_names() -> tuple[str, ...]:
    from .formats import NAMES

    return NAMES


def hub_error(e: urllib.error.HTTPError) -> str:
    repo = re.search(r"/api/models/(.+?)/revision/", e.url or "")
    name = f" for {repo[1]}" if repo else ""
    if e.code in (401, 403, 404):
        return (f"Hugging Face returned {e.code}{name}: no such repo, or it is gated (set HF_TOKEN). "
                "A path that does not exist on disk ends up here too")
    return f"Hugging Face returned {e.code}{name}; try again"


def main(argv: list[str] | None = None, prog: str = "neyra") -> int:
    args = parser(prog).parse_args(argv)
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
    except (FormatError, Unsupported, json.JSONDecodeError) as e:
        print(f"neyra: bad input: {e}", file=sys.stderr)
        return BAD_INPUT
    except (PatchMismatch, ProtocolError) as e:
        print(f"neyra: {e}", file=sys.stderr)
        return FAILED
    except urllib.error.HTTPError as e:
        print(f"neyra: {hub_error(e)}", file=sys.stderr)
        return BAD_INPUT
    except urllib.error.URLError as e:
        print(f"neyra: cannot reach Hugging Face ({e.reason}); check the connection, or use a folder on disk",
              file=sys.stderr)
        return ENVIRONMENT
    except FileNotFoundError as e:
        print(f"neyra: {e.filename}: no such file", file=sys.stderr)
        return BAD_INPUT
    except UnicodeDecodeError:
        print("neyra: bad input: expected a text file but found binary data; is this the right file?", file=sys.stderr)
        return BAD_INPUT
    except OSError as e:
        why = e.strerror or (str(e).splitlines() or ["failed"])[0]
        print(f"neyra: {e.filename or 'I/O error'}: {why}", file=sys.stderr)
        return BAD_INPUT
    except KeyboardInterrupt:
        print("neyra: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
