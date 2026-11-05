"""Transformation statements: the signed record of what changed between two models.

The normative text is Annex C of spec/PROTOCOL.md. The statement is the single source of truth. The
CycloneDX fragment and the one page summary are derived from it and from nothing else, so a reader who holds
the statement can regenerate both and compare.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .protocol import (
    STATEMENT_TYPE,
    ProtocolError,
    check_signature,
    decode_envelope,
    digest,
    jcs,
    now,
    root_of,
    tensor_table,
)
from .store import Checkpoint

PREDICATE_TYPE = "https://neyra.run/transformation/v1"
STATUS_TYPE = "https://neyra.run/status/v1"
WEIGHTS_KEY = "neyra_weights_v1"
KINDS = ("patch", "merge", "adapter", "quantize", "edit", "unknown")
GRADES = ("exact", "observed", "sampled", "declared")
MEANINGS = {
    "authored": "made this change",
    "reviewed": "reviewed this change and did not make it",
    "approved-for-deploy": "approves this exact result for deployment, on the strength of the checks listed here only",
    "responsible-for-merge": "is responsible for combining the inputs listed here",
}
NOT_MEASURED = ("training compute", "training data", "behaviour")
OMS_IGNORE = (".git", ".gitattributes", ".github", ".gitignore")
EXTENT = ("tensors_base", "tensors_result", "changed", "added", "removed", "entries_changed", "entries_total")
ACTIONS = ("supersede", "withdraw")

PREDICATE_MEMBERS = {"type", "kind", "grade", "base", "result", "recipe", "extent", "coverage", "basis", "attempts",
                     "rederive", "meaning", "not_measured", "evidence", "time", "tool"}
STATUS_MEMBERS = {"type", "action", "target", "replacement", "weights_root", "reason", "meaning", "time", "tool"}
SHA256 = re.compile(r"[0-9a-f]{64}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def bad(message: str) -> ProtocolError:
    return ProtocolError("bad_transform", message)


def statement_digest(payload: bytes) -> str:
    return digest("statement", payload)


def path_order(path: str) -> tuple[str, ...]:
    """The order OMS lists files in: component by component, as pathlib compares them."""
    return tuple(path.split("/"))


def file_digests(root: str | Path, ignore: tuple[str, ...] = (), allow_symlinks: bool = False) -> list[dict]:
    """sha256 of every file the way OMS serialises a model directory (method `files`)."""
    root = Path(root)
    skipped = [Path(s).parts for s in (*OMS_IGNORE, *ignore)]
    found = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(rel.parts[:len(s)] == s for s in skipped):
            continue
        if path.is_symlink() and not allow_symlinks:
            raise ValueError(f"{rel} is a symlink, which OMS refuses by default")
        if path.is_file():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while block := f.read(1 << 20):
                    h.update(block)
            found.append({"path": rel.as_posix(), "sha256": h.hexdigest()})
    return sorted(found, key=lambda f: path_order(f["path"]))


def oms_root(files: list[dict]) -> str:
    """The subject digest OMS gives a model: sha256 over the file digests in manifest order."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: path_order(f["path"])):
        h.update(bytes.fromhex(f["sha256"]))
    return h.hexdigest()


def oms_files(bundle: dict) -> tuple[list[dict], str]:
    """The files and subject digest an OMS signature declares. The signature itself is OMS's to verify."""
    st = decode_oms(bundle)
    files = [{"path": r["name"], "sha256": r["digest"]} for r in st["predicate"]["resources"]]
    if st["predicate"]["serialization"]["method"] != "files":
        raise ValueError("only OMS serialisation by files can be joined to a Neyra statement")
    return sorted(files, key=lambda f: path_order(f["path"])), st["subject"][0]["digest"]["sha256"]


def decode_oms(bundle: dict) -> dict:
    return json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))


def measure(base: Checkpoint, result: Checkpoint, before: dict, after: dict) -> dict:
    """Count what differs, reading only the tensors whose digests differ."""
    changed = removed = entries = total = 0
    for name, ref in base.tensors.items():
        other = result.tensors.get(name)
        if other is None:
            removed += 1
        elif before[name]["digest"] != after[name]["digest"]:
            changed += 1
            if other.dtype == ref.dtype and other.shape == ref.shape:
                a, b = ref.read(), other.read()
                entries += int(np.count_nonzero(a != b))
                total += a.size
            else:
                entries += other.numel
                total += other.numel
    return {"tensors_base": len(base.tensors), "tensors_result": len(result.tensors), "changed": changed,
            "added": len(set(result.tensors) - set(base.tensors)), "removed": removed,
            "entries_changed": entries, "entries_total": total}


def basis_for(grade: str) -> dict:
    """What a reader can recompute from the models and what rests on the signer's word, fixed by the grade.

    Written into the statement so that nobody reads an unchecked field as a checked one. A verifier compares it
    with this function, so a signer cannot move a field from one list to the other.
    """
    checkable = ["base.weights_root", "result.weights_root", "result.files", "extent"]
    recomputable = checkable + ["patch_application"] if grade == "exact" else checkable if grade == "observed" else []
    asserted = ["base.files", "kind", "recipe", "meaning", "time", "tool", "evidence", "attempts"]
    return {"recomputable": recomputable, "asserted": asserted + ([] if recomputable else checkable)}


def rederive_steps(kind: str, grade: str, recipe: dict, base_root: str, result_root: str) -> list[str]:
    if grade == "exact":
        return [f"Take a model whose weights root is {base_root}.",
                f"Apply patch {recipe['patch_id']} to it with `neyra apply`.",
                f"The result must have weights root {result_root}.",
                ("Check the whole statement with `neyra verify STATEMENT --key KEY --base BASE --result RESULT "
                 "--patch PATCH`.")]
    if grade == "observed":
        return ["Hash both models tensor by tensor with `neyra verify STATEMENT --key KEY --base BASE --result RESULT`.",
                f"The weights roots must be {base_root} and {result_root}, and the counts under extent must match."]
    if grade == "sampled":
        return ["Re-run the same sampled comparison. Counts are lower bounds: spans that were not read may differ."]
    return ["The signer recomputed nothing. Kind and recipe are the signer's statement, not a measurement."]


def transformation(*, kind: str, grade: str, meaning: str, base_root: str, result_root: str, base_files: list[dict],
                   result_files: list[dict], result_name: str, recipe: dict, extent: dict | None,
                   coverage: dict, attempts: int | None = None, ignore: tuple[str, ...] = (),
                   evidence: list[dict] | None = None, time: str | None = None) -> dict:
    """An in-toto statement. The subjects are derived from the predicate, never chosen separately."""
    predicate = {
        "type": "neyra.transformation/1", "kind": kind, "grade": grade,
        "base": {"weights_root": base_root, "files": base_files},
        "result": {"weights_root": result_root, "files": result_files, "name": result_name, "ignore": list(ignore)},
        "recipe": recipe, "extent": extent, "coverage": coverage, "basis": basis_for(grade), "attempts": attempts,
        "rederive": rederive_steps(kind, grade, recipe, base_root, result_root),
        "meaning": meaning, "not_measured": list(NOT_MEASURED), "evidence": evidence or [],
        "time": time or now(), "tool": {"name": "neyra", "version": __version__},
    }
    return {"_type": STATEMENT_TYPE, "subject": subjects_of(predicate), "predicateType": PREDICATE_TYPE,
            "predicate": predicate}


def subjects_of(predicate: dict) -> list[dict]:
    result = predicate["result"]
    subjects = [{"name": "result-weights", "digest": {WEIGHTS_KEY: result["weights_root"][7:]}}]
    subjects += [{"name": f["path"], "digest": {"sha256": f["sha256"]}} for f in result["files"]]
    if result["files"]:
        subjects.append({"name": result["name"], "digest": {"sha256": oms_root(result["files"])}})
    return subjects


def check_members(obj: dict, want: set[str], where: str, extensible: bool = False) -> None:
    extra = {k for k in obj if k.startswith("x-")} if extensible else set()
    if set(obj) - want - extra or want - set(obj):
        raise bad(f"{where} has the wrong members")


def check_files(files, where: str) -> None:
    if not isinstance(files, list):
        raise bad(f"{where}.files is not a list")
    for f in files:
        if not isinstance(f, dict) or set(f) != {"path", "sha256"} or not isinstance(f["path"], str) \
                or not isinstance(f["sha256"], str) or not SHA256.fullmatch(f["sha256"]):
            raise bad(f"{where}.files holds a malformed entry")
        if any(part in ("", ".", "..") for part in f["path"].split("/")) or "\\" in f["path"]:
            raise bad(f"{where}.files holds the path {f['path']!r}")
    keys = [path_order(f["path"]) for f in files]
    if keys != sorted(set(keys)):
        raise bad(f"{where}.files is not sorted without repeats")


def check_predicate(p) -> None:
    if not isinstance(p, dict):
        raise bad("the predicate is not an object")
    check_members(p, PREDICATE_MEMBERS, "predicate", extensible=True)
    if p["type"] != "neyra.transformation/1":
        raise bad("unknown predicate type")
    if p["kind"] not in KINDS or p["grade"] not in GRADES or p["meaning"] not in MEANINGS:
        raise bad("kind, grade or meaning is outside its vocabulary")
    for side in ("base", "result"):
        part = p[side]
        if not isinstance(part, dict):
            raise bad(f"{side} is not an object")
        check_members(part, {"weights_root", "files"} | ({"name", "ignore"} if side == "result" else set()), side)
        if not isinstance(part["weights_root"], str) or not DIGEST.fullmatch(part["weights_root"]):
            raise bad(f"{side}.weights_root is not a digest")
        check_files(part["files"], side)
    if not isinstance(p["result"]["name"], str) or (p["result"]["files"] and not p["result"]["name"]):
        raise bad("result.name must name the model when files are listed")
    skipped = p["result"]["ignore"]
    if not isinstance(skipped, list) or not all(isinstance(s, str) and s for s in skipped):
        raise bad("result.ignore must be a list of paths")
    recipe = p["recipe"]
    if not isinstance(recipe, dict):
        raise bad("recipe is not an object")
    named = isinstance(recipe.get("patch_id"), str) and DIGEST.fullmatch(recipe["patch_id"])
    if p["grade"] == "exact" and p["kind"] in ("patch", "edit", "merge") and not named:
        raise bad("an exact patch, edit or merge names its patch_id in the recipe")
    extent = p["extent"]
    if extent is None:
        if p["grade"] != "declared":
            raise bad("only a declared change may leave extent empty")
    else:
        if not isinstance(extent, dict) or set(extent) != set(EXTENT):
            raise bad("extent has the wrong members")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in extent.values()):
            raise bad("extent holds a value that is not a count")
    cover = p["coverage"]
    if not isinstance(cover, dict) or set(cover) != {"bytes_read", "bytes_total"} or not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in cover.values()):
        raise bad("coverage must count bytes_read and bytes_total")
    read, total = cover["bytes_read"], cover["bytes_total"]
    whole = {"exact": read == total, "observed": read == total, "sampled": read < total, "declared": read == 0}
    if not whole[p["grade"]]:
        raise bad(f"coverage does not fit a {p['grade']} grade")
    if p["basis"] != basis_for(p["grade"]):
        raise bad("basis is not the one this grade fixes")
    tries = p["attempts"]
    if tries is not None and (not isinstance(tries, int) or isinstance(tries, bool) or tries < 1):
        raise bad("attempts is a count of at least 1, or null when nobody counted")
    steps = p["rederive"]
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s for s in steps):
        raise bad("rederive must be a non empty list of steps")
    unmeasured = p["not_measured"]
    if not isinstance(unmeasured, list) or not all(isinstance(s, str) for s in unmeasured) \
            or not set(NOT_MEASURED) <= set(unmeasured):
        raise bad("not_measured must say that " + ", ".join(NOT_MEASURED) + " were not measured")
    ev = p["evidence"]
    if not isinstance(ev, list) or any(
            not isinstance(e, dict) or set(e) != {"role", "digest"} or not isinstance(e["role"], str)
            or not isinstance(e["digest"], str) or not DIGEST.fullmatch(e["digest"]) for e in ev):
        raise bad("evidence is malformed")
    if not isinstance(p["time"], str) or not TIME.fullmatch(p["time"]):
        raise bad("time is not an RFC 3339 UTC string")
    tool = p["tool"]
    if not isinstance(tool, dict) or set(tool) != {"name", "version"} or not all(isinstance(v, str) for v in tool.values()):
        raise bad("tool is malformed")
    if p["kind"] == "unknown" and p["grade"] == "exact":
        raise bad("a change of unknown kind cannot be exact")


@dataclass
class Opened:
    statement: dict
    signed: bool
    payload: bytes


def open_statement(doc: dict, public: bytes | None, predicate_type: str, check) -> Opened:
    """The shared steps of section 8: envelope, canonical payload, types, then the predicate's own checks."""
    statement, payload = decode_envelope(doc)
    if statement.get("_type") != STATEMENT_TYPE or statement.get("predicateType") != predicate_type:
        raise ProtocolError("bad_type", "wrong statement or predicate type")
    check(statement)
    if public is None:
        return Opened(statement, False, payload)
    check_signature(doc, payload, public)
    return Opened(statement, True, payload)


def check_transformation(statement: dict) -> None:
    check_predicate(statement.get("predicate"))
    want = sorted(jcs_text(s) for s in subjects_of(statement["predicate"]))
    got = statement.get("subject")
    if not isinstance(got, list) or sorted(jcs_text(s) for s in got) != want:
        raise ProtocolError("subject_mismatch", "subjects must be exactly those the predicate implies")


def jcs_text(value) -> str:
    return jcs(value).decode()


def open_transformation(doc: dict, public: bytes | None = None) -> Opened:
    return open_statement(doc, public, PREDICATE_TYPE, check_transformation)


def bind_patch(statement: dict, patch) -> str:
    """The patch the statement names must be the patch given, and agree on both roots."""
    p = statement["predicate"]
    if p["recipe"].get("patch_id") != patch.id:
        return "binding_mismatch"
    if patch.base_root != p["base"]["weights_root"] or patch.result_root != p["result"]["weights_root"]:
        return "binding_mismatch"
    return "ok"


def bind_model(statement: dict, model: str | Path) -> str:
    """Hash a result directory two ways and compare: the weights root, then the OMS file digests."""
    p = statement["predicate"]
    root = root_of(tensor_table(Checkpoint.open(str(model))))
    if root != p["result"]["weights_root"]:
        return "result_mismatch"
    if not p["result"]["files"]:
        return "ok"
    try:
        listed = file_digests(model, tuple(p["result"]["ignore"]))
    except ValueError:
        return "files_mismatch"
    return "ok" if listed == p["result"]["files"] else "files_mismatch"


def status(*, action: str, target: str, weights_root: str, reason: str, meaning: str,
           replacement: str | None = None, time: str | None = None) -> dict:
    predicate = {"type": "neyra.status/1", "action": action, "target": target, "weights_root": weights_root,
                 "reason": reason, "meaning": meaning, "time": time or now(),
                 "tool": {"name": "neyra", "version": __version__}}
    if replacement is not None:
        predicate["replacement"] = replacement
    return {"_type": STATEMENT_TYPE, "subject": [{"name": "result-weights", "digest": {WEIGHTS_KEY: weights_root[7:]}}],
            "predicateType": STATUS_TYPE, "predicate": predicate}


def check_status(statement: dict) -> None:
    p = statement.get("predicate")
    if not isinstance(p, dict):
        raise ProtocolError("bad_status", "the predicate is not an object")
    if p.get("action") not in ACTIONS:
        raise ProtocolError("bad_status", "action is not supersede or withdraw")
    want = STATUS_MEMBERS - ({"replacement"} if p["action"] == "withdraw" else set())
    if set(p) - want - {k for k in p if k.startswith("x-")} or want - set(p) or p["type"] != "neyra.status/1":
        raise ProtocolError("bad_status", "the predicate has the wrong members")
    for key in ("target", "weights_root", *(["replacement"] if "replacement" in want else [])):
        if not isinstance(p[key], str) or not DIGEST.fullmatch(p[key]):
            raise ProtocolError("bad_status", f"{key} is not a digest")
    if not isinstance(p["reason"], str) or not p["reason"] or p["meaning"] not in MEANINGS:
        raise ProtocolError("bad_status", "reason or meaning is missing")
    if not isinstance(p["time"], str) or not TIME.fullmatch(p["time"]):
        raise ProtocolError("bad_status", "time is not an RFC 3339 UTC string")
    tool = p["tool"]
    if not isinstance(tool, dict) or set(tool) != {"name", "version"} or not all(isinstance(v, str) for v in tool.values()):
        raise ProtocolError("bad_status", "tool is malformed")
    if p["action"] == "supersede" and p["replacement"] == p["target"]:
        raise ProtocolError("bad_status", "a statement cannot supersede itself")
    if statement.get("subject") != [{"name": "result-weights", "digest": {WEIGHTS_KEY: p["weights_root"][7:]}}]:
        raise ProtocolError("subject_mismatch", "the subject must be the weights root the record concerns")


def open_status(doc: dict, public: bytes | None = None) -> Opened:
    return open_statement(doc, public, STATUS_TYPE, check_status)


def weights_root_of(statement: dict) -> str | None:
    """The result weights root a transformation statement or a v1 receipt is about."""
    p = statement.get("predicate", {})
    if statement.get("predicateType") == PREDICATE_TYPE:
        return p["result"]["weights_root"]
    claim = p.get("claim") if isinstance(p, dict) else None
    return claim.get("result_root") if isinstance(claim, dict) else None


def bind_target(record: Opened, target: Opened) -> str:
    """A status record applies to a statement when it names the digest of its payload and the same weights."""
    p = record.statement["predicate"]
    ok = p["target"] == statement_digest(target.payload) and p["weights_root"] == weights_root_of(target.statement)
    return "ok" if ok else "binding_mismatch"


def resolve(target: Opened, records: list[Opened]) -> dict:
    """What the records the caller trusts say about a statement. Nothing is ever edited; a withdrawal wins."""
    mine = [r for r in records if bind_target(r, target) == "ok"]
    withdrawn = [r.statement["predicate"] for r in mine if r.statement["predicate"]["action"] == "withdraw"]
    superseded = [r.statement["predicate"] for r in mine if r.statement["predicate"]["action"] == "supersede"]
    state = "withdrawn" if withdrawn else "superseded" if superseded else "current"
    return {"state": state, "withdrawn": withdrawn, "superseded_by": [p["replacement"] for p in superseded]}


def short(value: str, n: int = 12) -> str:
    return value.split(":", 1)[-1][:n]


def to_cyclonedx(doc: dict) -> dict:
    """A CycloneDX 1.7 BOM with the result as a machine-learning-model component and the base as its ancestor.

    Properties use the `neyra:` namespace. Hashes appear only on the file components, where they are hashes of
    files. The weights roots are not file hashes and travel as properties.
    """
    statement, _ = decode_envelope(doc)
    p = statement["predicate"]
    base, result = p["base"], p["result"]

    def files(owner: str, listing: list[dict]) -> list[dict]:
        return [{"type": "file", "bom-ref": f"{owner}:{f['path']}", "name": f["path"],
                 "hashes": [{"alg": "SHA-256", "content": f["sha256"]}]} for f in listing]

    ancestor = {"type": "machine-learning-model", "bom-ref": "base", "name": "base model",
                "properties": [{"name": "neyra:weights-root", "value": base["weights_root"]}]}
    if base["files"]:
        ancestor["components"] = files("base", base["files"])
    pedigree = {"ancestors": [ancestor],
                "notes": f"{p['kind']} ({p['grade']}). Not measured: {', '.join(p['not_measured'])}."}
    if "patch_id" in p["recipe"]:
        pedigree["patches"] = [{"type": "unofficial", "diff": {"url": "urn:neyra:patch:" + p["recipe"]["patch_id"][7:]}}]
    props = [("neyra:kind", p["kind"]), ("neyra:evidence-grade", p["grade"]), ("neyra:meaning", p["meaning"]),
             ("neyra:weights-root", result["weights_root"]), ("neyra:base-weights-root", base["weights_root"])]
    if p["extent"]:
        props += [(f"neyra:extent:{k.replace('_', '-')}", str(v)) for k, v in p["extent"].items()]
    props += [("neyra:coverage:bytes-read", str(p["coverage"]["bytes_read"])),
              ("neyra:coverage:bytes-total", str(p["coverage"]["bytes_total"]))]
    props += [("neyra:attempts", str(p["attempts"]) if p["attempts"] else "not stated")]
    props += [("neyra:recomputable", s) for s in p["basis"]["recomputable"]]
    props += [("neyra:asserted", s) for s in p["basis"]["asserted"]]
    props += [("neyra:not-measured", s) for s in p["not_measured"]]
    props += [("neyra:" + k.replace("_", "-"), str(v)) for k, v in sorted(p["recipe"].items())]
    model = {"type": "machine-learning-model", "bom-ref": "result", "name": result["name"] or "result model",
             "pedigree": pedigree, "properties": [{"name": n, "value": v} for n, v in props],
             "externalReferences": [{"type": "attestation", "url": "change.statement.json",
                                     "comment": "DSSE envelope holding the signed neyra transformation statement",
                                     "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(
                                         base64.b64decode(doc["payload"])).hexdigest()}]}]}
    if result["files"]:
        model["components"] = files("result", result["files"])
    return {"bomFormat": "CycloneDX", "specVersion": "1.7", "version": 1,
            "metadata": {"timestamp": p["time"],
                         "tools": {"components": [{"type": "application", "name": p["tool"]["name"],
                                                   "version": p["tool"]["version"]}]}},
            "components": [model]}


def to_page(doc: dict, signed_by: str | None = None) -> str:
    """One page for a person. It leads with what the record cannot tell them."""
    statement, payload = decode_envelope(doc)
    p = statement["predicate"]
    base, result, extent = p["base"], p["result"], p["extent"]
    signer = signed_by or (doc["signatures"][0]["keyid"] if doc.get("signatures") else None)
    lines = ["# Model change statement", "",
             f"Made {p['time']} with {p['tool']['name']} {p['tool']['version']}. "
             + (f"Signed by key {short(signer, 16)}. Meaning of the signature: {p['meaning']}, the signer {MEANINGS[p['meaning']]}."
                if signer else "Not signed: an unauthenticated record."), "",
             "## What changed", "",
             f"- Result: {result['name'] or 'unnamed'}, weights root {result['weights_root']}",
             f"- Base: weights root {base['weights_root']}",
             f"- Kind: {p['kind']}. Evidence: {p['grade']} ({GRADE_TEXT[p['grade']]})."]
    if extent:
        pct = 100 * extent["entries_changed"] / extent["entries_total"] if extent["entries_total"] else 0
        lines.append(f"- Extent: {extent['changed']} of {extent['tensors_base']} tensors changed, {extent['added']} added, "
                     f"{extent['removed']} removed. In the changed tensors {extent['entries_changed']:,} of "
                     f"{extent['entries_total']:,} entries differ ({pct:.3f}%).")
    else:
        lines.append("- Extent: not measured by the signer.")
    if "patch_id" in p["recipe"]:
        lines.append(f"- Patch: {p['recipe']['patch_id']}")
    if result["files"]:
        lines.append(f"- Result files: {len(result['files'])}, joined to an OpenSSF model signature by sha256 "
                     f"(subject digest {oms_root(result['files'])}).")
    lines.append(f"- Reading: {p['coverage']['bytes_read']:,} of {p['coverage']['bytes_total']:,} weight bytes were read.")
    tries = p["attempts"]
    lines.append(f"- Attempts: the signer says {tries}." if tries else
                 "- Attempts: not stated. Nothing here shows how many results were tried and put aside.")
    lines += ["", "## What anyone can recompute, and what is only the signer's word", ""]
    lines.append("Recomputable from the models: " + (", ".join(p["basis"]["recomputable"]) or "nothing") + ".")
    lines.append("Only the signer's word: " + ", ".join(p["basis"]["asserted"]) + ".")
    lines += ["", "## How to check it", ""] + [f"{i}. {s}" for i, s in enumerate(p["rederive"], 1)]
    lines += ["", "## What this does not measure", "",
              "- Training compute. No weight comparison can say how much was spent making these weights.",
              "- Training data. Nothing here says what the model was trained or tuned on.",
              "- Behaviour. Which bytes differ does not say what the model now does, or whether it is safe.",
              "", ("This record shows which bytes differ. It is not a compute estimate, a safety evaluation or a "
                   "legal determination. A signature shows which key said this, not when, not that the signer is who they say, "
                   "and not that anything above is true beyond what the list of recomputable fields lets you check."),
              "", f"Statement digest {statement_digest(payload)}", ""]
    return "\n".join(lines)


GRADE_TEXT = {"exact": "the signer applied the named patch and got this result bit for bit",
              "observed": "the signer hashed both models in full and compared every tensor",
              "sampled": "only parts of the models were read; counts are lower bounds",
              "declared": "the signer computed nothing and this is their word"}
