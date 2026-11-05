import base64
import json
from pathlib import Path

import numpy as np
import pytest

from neyra import attest as at
from neyra.cli import main
from neyra.patch import Patch, make_patch, write_safetensors
from neyra.protocol import (
    ProtocolError,
    decode_envelope,
    envelope,
    key_id,
    new_key,
    open_envelope,
    read_private,
    read_public,
    root_of,
    statement_for,
    tensor_table,
)
from neyra.store import Checkpoint

FIXTURES = Path(__file__).parent / "fixtures" / "cyclonedx"
ED = bytes(range(32))


def bits(rng, *shape):
    return rng.integers(0, 2**15, size=shape, dtype=np.uint16)


@pytest.fixture
def world(tmp_path, rng):
    """A base model, a result made by a real patch, and the statement about it."""
    tensors = {f"layers.{i}.w": bits(rng, 8, 8) for i in range(4)}
    edited = {**tensors, "layers.1.w": bits(rng, 8, 8)}

    def write(name, data, split=1):
        d = tmp_path / name
        d.mkdir()
        names = sorted(data)
        for i in range(split):
            write_safetensors(d / f"part-{i}.safetensors", {n: data[n] for n in names[i::split]}, None, "BF16")
        (d / "config.json").write_text('{"hidden_size": 8}')
        return d

    base, result = write("base", tensors), write("result", edited)
    patch = make_patch(Checkpoint.open(str(base)), Checkpoint.open(str(result)), {"method": "closed form edit"})
    patch.save(tmp_path / "edit.neyra")
    return {"tmp": tmp_path, "tensors": edited, "original": tensors, "write": write, "base": base, "result": result, "patch": patch}


def statement_of(world, **over):
    base, result = Checkpoint.open(str(world["base"])), Checkpoint.open(str(world["result"]))
    before, after = tensor_table(base), tensor_table(result)
    size = sum(r.nbytes for ck in (base, result) for r in ck.tensors.values())
    args = {"kind": "edit", "grade": "exact", "meaning": "authored", "base_root": root_of(before),
            "result_root": root_of(after), "base_files": at.file_digests(world["base"]),
            "result_files": at.file_digests(world["result"]), "result_name": "result",
            "recipe": {"patch_id": world["patch"].id}, "extent": at.measure(base, result, before, after),
            "coverage": {"bytes_read": size, "bytes_total": size}, "time": "2026-11-05T12:00:00Z"}
    if over.get("grade") == "declared" and "coverage" not in over:
        args["coverage"] = {"bytes_read": 0, "bytes_total": size}
    return at.transformation(**{**args, **over})


def test_oms_digests_match_what_model_signing_wrote(tmp_path):
    # Values produced by model-signing 1.1.1 `sign key` over this directory: nested, upper case and non ASCII names.
    for name, body in (("a/b", "1"), ("a.txt", "2"), ("B/x", "3"), ("é", "4"), ("Z", "5")):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(body + "\n")
    files = at.file_digests(tmp_path)
    assert [f["path"] for f in files] == ["B/x", "Z", "a/b", "a.txt", "é"]
    assert at.oms_root(files) == "bb95a887e6314422fae33423a71ce013580f1cdae723f37678dac24db6c14286"


def test_ignored_paths_are_left_out(tmp_path):
    (tmp_path / "w.bin").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "model.sig").write_text("{}")
    assert [f["path"] for f in at.file_digests(tmp_path, ("model.sig",))] == ["w.bin"]
    assert len(at.file_digests(tmp_path)) == 2


def test_symlinks_are_refused_like_oms(tmp_path):
    (tmp_path / "real").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValueError, match="symlink"):
        at.file_digests(tmp_path)


@pytest.mark.parametrize("algorithm", ["ed25519", "ecdsa-p256"])
def test_statement_signs_and_verifies(world, tmp_path, algorithm):
    key = tmp_path / "k"
    new_key(key, algorithm)
    doc = envelope(statement_of(world), read_private(key))
    opened = at.open_transformation(doc, read_public(key.with_suffix(".pub")))
    assert opened.signed and opened.statement["predicate"]["grade"] == "exact"
    other = tmp_path / "other"
    new_key(other, algorithm)
    with pytest.raises(ProtocolError) as e:
        at.open_transformation(doc, read_public(other.with_suffix(".pub")))
    assert e.value.code == "bad_signature"


def test_p256_signs_the_old_receipt_too(world, tmp_path):
    key = tmp_path / "k"
    new_key(key, "ecdsa-p256")
    claim = {"type": "neyra.claim/1", "base_root": "sha256:" + "1" * 64, "result_root": "sha256:" + "2" * 64,
             "patch_id": "sha256:" + "3" * 64, "task": {}, "eval_env": {}, "statements": [], "gates": {},
             "verdict": "PASS", "limits": {}}
    doc = envelope(statement_for(claim, claim["patch_id"], "sha256:" + "4" * 64, 1), read_private(key))
    assert open_envelope(doc, read_public(key.with_suffix(".pub"))).signed
    tampered = {**doc, "signatures": [{**doc["signatures"][0], "sig": doc["signatures"][0]["sig"][:-4] + "AAAA"}]}
    with pytest.raises(ProtocolError):
        open_envelope(tampered, read_public(key.with_suffix(".pub")))


def test_keyid_is_the_digest_of_the_public_key(tmp_path):
    key = tmp_path / "k"
    assert new_key(key, "ecdsa-p256") == key_id(read_public(key.with_suffix(".pub")))
    assert len(read_public(key.with_suffix(".pub"))) != 32


def reopen(statement, **kw):
    return at.open_transformation(envelope(statement), **kw)


def test_subjects_follow_from_the_predicate(world):
    s = statement_of(world)
    names = [x["name"] for x in s["subject"]]
    assert names[0] == "result-weights" and "config.json" in names and "result" in names
    assert s["subject"][0]["digest"] == {at.WEIGHTS_KEY: s["predicate"]["result"]["weights_root"][7:]}
    reopen(s)
    s["subject"][1]["digest"]["sha256"] = "0" * 64
    with pytest.raises(ProtocolError) as e:
        reopen(s)
    assert e.value.code == "subject_mismatch"


def test_a_subject_nobody_asked_for_is_refused(world):
    s = statement_of(world)
    s["subject"].append({"name": "extra", "digest": {"sha256": "1" * 64}})
    with pytest.raises(ProtocolError) as e:
        reopen(s)
    assert e.value.code == "subject_mismatch"


@pytest.mark.parametrize("change", [
    lambda p: p.update(kind="rebase"),
    lambda p: p.update(grade="certain"),
    lambda p: p.update(meaning="blessed"),
    lambda p: p.update(not_measured=["behaviour"]),
    lambda p: p.update(extent=None),
    lambda p: p.update(extent={"changed": 1}),
    lambda p: p.update(recipe={}),
    lambda p: p.update(rederive=[]),
    lambda p: p.update(time="yesterday"),
    lambda p: p.update(surprise=1),
    lambda p: p["base"].update(weights_root="sha256:xyz"),
    lambda p: p["result"].update(files=[{"path": "../x", "sha256": "0" * 64}]),
    lambda p: p["result"].update(files=list(reversed(p["result"]["files"]))),
    lambda p: p["tool"].pop("version"),
    lambda p: p["basis"]["recomputable"].remove("extent"),
    lambda p: p["basis"]["asserted"].remove("attempts"),
    lambda p: p.update(basis={"recomputable": [], "asserted": p["basis"]["asserted"]}),
    lambda p: p["coverage"].update(bytes_read=p["coverage"]["bytes_total"] - 1),
    lambda p: p["coverage"].update(bytes_read=-1),
    lambda p: p.update(attempts=0),
    lambda p: p.update(attempts="3"),
    lambda p: p.pop("coverage"),
])
def test_malformed_predicates_are_refused(world, change):
    s = statement_of(world)
    change(s["predicate"])
    s["subject"] = at.subjects_of(s["predicate"]) if "files" in s["predicate"]["result"] else s["subject"]
    with pytest.raises(ProtocolError) as e:
        reopen(s)
    assert e.value.code == "bad_transform"


def test_coverage_has_to_fit_the_grade(world):
    total = statement_of(world)["predicate"]["coverage"]["bytes_total"]
    reopen(statement_of(world, grade="sampled", kind="unknown", recipe={}, coverage={"bytes_read": total - 8, "bytes_total": total}))
    reopen(statement_of(world, grade="declared", kind="quantize", extent=None, recipe={},
                        coverage={"bytes_read": 0, "bytes_total": total}))
    for grade, read in (("sampled", total), ("declared", 1)):
        with pytest.raises(ProtocolError) as e:
            reopen(statement_of(world, grade=grade, kind="unknown", extent=None, recipe={},
                                coverage={"bytes_read": read, "bytes_total": total}))
        assert e.value.code == "bad_transform"


def test_basis_follows_the_grade_and_says_what_is_only_asserted():
    exact, declared = at.basis_for("exact"), at.basis_for("declared")
    assert "patch_application" in exact["recomputable"] and "patch_application" not in at.basis_for("observed")["recomputable"]
    assert declared["recomputable"] == [] and {"result.weights_root", "extent"} <= set(declared["asserted"])
    assert "kind" in exact["asserted"] and "attempts" in exact["asserted"]
    assert not set(exact["recomputable"]) & set(exact["asserted"])


def test_attempts_are_null_unless_the_signer_counts(world):
    assert statement_of(world)["predicate"]["attempts"] is None
    assert "Attempts: not stated" in at.to_page(envelope(statement_of(world)))
    counted = statement_of(world, attempts=7)
    assert reopen(counted).statement["predicate"]["attempts"] == 7
    assert "the signer says 7" in at.to_page(envelope(counted))


def test_unknown_x_members_are_kept_and_allowed(world):
    s = statement_of(world)
    s["predicate"]["x-note"] = "fine"
    assert reopen(s).statement["predicate"]["x-note"] == "fine"


def test_declared_change_may_leave_extent_empty_but_not_claim_exact(world):
    reopen(statement_of(world, kind="quantize", grade="declared", extent=None, recipe={"scheme": "q4"}))
    with pytest.raises(ProtocolError) as e:
        reopen(statement_of(world, kind="unknown", grade="exact"))
    assert e.value.code == "bad_transform"


def test_wrong_types_are_told_apart_from_bad_predicates(world):
    s = statement_of(world)
    s["predicateType"] = "https://neyra.run/receipt/v1"
    with pytest.raises(ProtocolError) as e:
        reopen(s)
    assert e.value.code == "bad_type"


def test_reshard_keeps_the_weights_root_and_loses_the_oms_join(world):
    again = world["write"]("result-resharded", world["tensors"], split=3)
    s = statement_of(world)
    twin = statement_of({**world, "result": again})
    assert s["predicate"]["result"]["weights_root"] == twin["predicate"]["result"]["weights_root"]
    assert at.oms_root(s["predicate"]["result"]["files"]) != at.oms_root(twin["predicate"]["result"]["files"])
    assert s["subject"][0] == twin["subject"][0]
    assert at.bind_model(s, world["result"]) == "ok"
    assert at.bind_model(s, again) == "files_mismatch"


def test_ignored_paths_are_recorded_so_a_reader_lists_the_same_files(world):
    (world["result"] / "model.sig").write_text("{}")
    s = statement_of(world, result_files=at.file_digests(world["result"], ("model.sig",)), ignore=("model.sig",))
    assert at.bind_model(s, world["result"]) == "ok"
    assert at.bind_model({**s, "predicate": {**s["predicate"], "result": {**s["predicate"]["result"], "ignore": []}}},
                         world["result"]) == "files_mismatch"


def test_binding_to_a_model_directory(world):
    s = statement_of(world)
    assert at.bind_model(s, world["result"]) == "ok"
    (world["result"] / "config.json").write_text('{"hidden_size": 9}')
    assert at.bind_model(s, world["result"]) == "files_mismatch"
    assert at.bind_model(s, world["base"]) == "result_mismatch"


def test_binding_to_a_patch(world):
    s = statement_of(world)
    assert at.bind_patch(s, world["patch"]) == "ok"
    other = Patch.load(world["tmp"] / "edit.neyra")
    other.result_root = "sha256:" + "9" * 64
    assert at.bind_patch(s, other) == "binding_mismatch"
    s["predicate"]["recipe"]["patch_id"] = "sha256:" + "8" * 64
    assert at.bind_patch(s, world["patch"]) == "binding_mismatch"


def test_extent_counts_what_differs(world):
    s = statement_of(world)
    e = s["predicate"]["extent"]
    assert (e["tensors_base"], e["changed"], e["added"], e["removed"]) == (4, 1, 0, 0)
    differ = int(np.count_nonzero(world["original"]["layers.1.w"] != world["tensors"]["layers.1.w"]))
    assert (e["entries_changed"], e["entries_total"]) == (differ, 64) and differ > 0


def test_cyclonedx_fragment_validates_against_the_1_7_schema(world):
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource

    schemas = {}
    for f in FIXTURES.glob("*.json"):
        doc = json.loads(f.read_text())
        schemas[doc["$id"]] = doc
    registry = Registry().with_resources([(k, Resource.from_contents(v)) for k, v in schemas.items()])
    validator = jsonschema.Draft7Validator(schemas["http://cyclonedx.org/schema/bom-1.7.schema.json"], registry=registry)
    bom = at.to_cyclonedx(envelope(statement_of(world)))
    assert list(validator.iter_errors(bom)) == []
    model = bom["components"][0]
    assert model["type"] == "machine-learning-model"
    assert {h["alg"] for c in model["components"] for h in c["hashes"]} == {"SHA-256"}
    assert model["pedigree"]["ancestors"][0]["properties"][0]["name"] == "neyra:weights-root"
    assert {p["name"] for p in model["properties"]} >= {"neyra:kind", "neyra:evidence-grade", "neyra:meaning"}
    bom["components"][0]["type"] = "model"
    assert list(validator.iter_errors(bom)) != []


def test_cyclonedx_and_page_come_from_the_statement_alone(world):
    doc = envelope(statement_of(world))
    assert at.to_cyclonedx(doc) == at.to_cyclonedx(json.loads(json.dumps(doc)))
    assert at.to_page(doc) == at.to_page(json.loads(json.dumps(doc)))
    ref = at.to_cyclonedx(doc)["components"][0]["externalReferences"][0]
    assert ref["hashes"][0]["content"] == __import__("hashlib").sha256(base64.b64decode(doc["payload"])).hexdigest()


def test_page_says_what_is_not_measured(world):
    page = at.to_page(envelope(statement_of(world)))
    for phrase in ("Training compute", "Training data", "Behaviour", "not a compute estimate", "Not signed",
                   "Only the signer's word", "not when", "Recomputable from the models"):
        assert phrase in page
    assert "1 of 4 tensors changed" in page


def test_page_for_a_declared_change_does_not_invent_numbers(world):
    page = at.to_page(envelope(statement_of(world, kind="quantize", grade="declared", extent=None, recipe={})))
    assert "not measured by the signer" in page and "tensors changed" not in page


def record(world, action="withdraw", **over):
    s = statement_of(world)
    doc = envelope(s)
    _, payload = decode_envelope(doc)
    args = {"action": action, "target": at.statement_digest(payload),
            "weights_root": s["predicate"]["result"]["weights_root"], "reason": "wrong base",
            "meaning": "authored", "time": "2026-11-06T09:00:00Z"}
    if action == "supersede":
        args["replacement"] = "sha256:" + "5" * 64
    return doc, at.status(**{**args, **over})


def test_status_records_bind_by_payload_digest(world):
    doc, rec = record(world)
    target = at.Opened(*decode_envelope(doc)[:1], False, decode_envelope(doc)[1])
    opened = at.open_status(envelope(rec))
    assert at.bind_target(opened, target) == "ok"
    _, wrong = record(world, target="sha256:" + "6" * 64)
    assert at.bind_target(at.open_status(envelope(wrong)), target) == "binding_mismatch"
    assert at.resolve(target, [at.open_status(envelope(wrong))])["state"] == "current"


def test_withdrawal_wins_over_supersession_and_nothing_is_edited(world):
    doc, withdraw = record(world)
    _, supersede = record(world, "supersede")
    target = at.Opened(decode_envelope(doc)[0], False, decode_envelope(doc)[1])
    both = [at.open_status(envelope(x)) for x in (supersede, withdraw)]
    assert at.resolve(target, [both[0]])["state"] == "superseded"
    assert at.resolve(target, both)["state"] == "withdrawn"
    assert at.resolve(target, [])["state"] == "current"
    assert decode_envelope(doc)[1] == target.payload


@pytest.mark.parametrize("change,code", [
    (lambda p: p.update(replacement="sha256:" + "5" * 64), "bad_status"),
    (lambda p: p.update(action="revoke"), "bad_status"),
    (lambda p: p.update(reason=""), "bad_status"),
    (lambda p: p.update(meaning="blessed"), "bad_status"),
    (lambda p: p.update(target="nope"), "bad_status"),
])
def test_malformed_status_records(world, change, code):
    _, rec = record(world)
    change(rec["predicate"])
    with pytest.raises(ProtocolError) as e:
        at.open_status(envelope(rec))
    assert e.value.code == code


def test_supersession_needs_a_replacement_and_cannot_point_at_itself(world):
    _, rec = record(world, "supersede")
    del rec["predicate"]["replacement"]
    with pytest.raises(ProtocolError):
        at.open_status(envelope(rec))
    _, rec = record(world, "supersede")
    rec["predicate"]["replacement"] = rec["predicate"]["target"]
    with pytest.raises(ProtocolError):
        at.open_status(envelope(rec))


def test_status_subject_must_be_the_weights_it_concerns(world):
    _, rec = record(world)
    rec["subject"][0]["digest"][at.WEIGHTS_KEY] = "0" * 64
    with pytest.raises(ProtocolError) as e:
        at.open_status(envelope(rec))
    assert e.value.code == "subject_mismatch"


def run(*argv):
    return main([str(a) for a in argv])


def test_cli_attest_verify_supersede_withdraw(world, tmp_path, capsys):
    key = tmp_path / "neyra.key"
    assert run("keygen", "-o", key, "--algorithm", "ecdsa-p256") == 0
    out = tmp_path / "change"
    assert run("attest", world["base"], world["result"], "--patch", tmp_path / "edit.neyra", "--key", key,
               "--time", "2026-11-05T12:00:00Z", "-o", out) == 0
    assert {p.name for p in out.iterdir()} == {"change.statement.json", "change.cdx.json", "CHANGE.md"}
    stmt, pub = out / "change.statement.json", key.with_suffix(".pub")
    capsys.readouterr()
    args = ["verify", stmt, "--key", pub, "--base", world["base"], "--result", world["result"], "--patch",
            tmp_path / "edit.neyra"]
    assert run(*args) == 0
    assert "signature verified" in capsys.readouterr().out
    assert run("render", stmt, "--format", "cdx") == 0
    assert json.loads(capsys.readouterr().out) == json.loads((out / "change.cdx.json").read_text())

    (world["result"] / "config.json").write_text("{}")
    assert run(*args) != 0
    assert "files_mismatch" in capsys.readouterr().out
    (world["result"] / "config.json").write_text('{"hidden_size": 8}')

    withdrawn = tmp_path / "withdrawn.json"
    assert run("withdraw", stmt, "--reason", "wrong base", "--key", key, "-o", withdrawn) == 0
    capsys.readouterr()
    assert run("status", stmt, withdrawn, "--key", pub) == 1
    assert "withdrawn" in capsys.readouterr().out
    stranger = tmp_path / "stranger"
    run("keygen", "-o", stranger)
    assert run("status", stmt, withdrawn, "--key", stranger.with_suffix(".pub")) == 0
    assert "ignored" in capsys.readouterr().out

    again = tmp_path / "again"
    assert run("attest", world["base"], world["result"], "--patch", tmp_path / "edit.neyra", "--key", key,
               "--meaning", "reviewed", "-o", again) == 0
    superseded = tmp_path / "superseded.json"
    assert run("supersede", stmt, "--replacement", again / "change.statement.json", "--reason", "reviewed",
               "--key", key, "-o", superseded) == 0
    capsys.readouterr()
    assert run("status", stmt, superseded, "--key", pub) == 1
    assert "superseded by" in capsys.readouterr().out


def test_cli_attest_refuses_a_patch_for_other_models(world, tmp_path):
    other = world["write"]("other", {**world["tensors"], "layers.0.w": world["tensors"]["layers.2.w"]})
    assert run("attest", world["base"], other, "--patch", tmp_path / "edit.neyra", "-o", tmp_path / "x") == 3


def test_cli_attest_can_leave_files_out(world, tmp_path):
    assert run("attest", world["base"], world["result"], "--no-files", "-o", tmp_path / "bare") == 0
    doc = json.loads((tmp_path / "bare" / "change.statement.json").read_text())
    p = at.open_transformation(doc).statement["predicate"]
    assert p["grade"] == "observed" and p["result"]["files"] == [] and p["kind"] == "unknown"
    assert doc["signatures"] == []


def test_cli_attest_checks_the_oms_signature_it_is_given(world, tmp_path):
    files = at.file_digests(world["result"])
    payload = {"subject": [{"name": "result", "digest": {"sha256": at.oms_root(files)}}],
               "predicate": {"serialization": {"method": "files"},
                             "resources": [{"name": f["path"], "algorithm": "sha256", "digest": f["sha256"]} for f in files]}}
    bundle = {"dsseEnvelope": {"payload": base64.b64encode(json.dumps(payload).encode()).decode()}}
    sig = tmp_path / "model.sig"
    sig.write_text(json.dumps(bundle))
    assert run("attest", world["base"], world["result"], "--oms", sig, "-o", tmp_path / "a") == 0
    (world["result"] / "README.md").write_text("added after signing")
    assert run("attest", world["base"], world["result"], "--oms", sig, "-o", tmp_path / "b") == 3
