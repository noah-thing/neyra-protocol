"""Check Neyra output with other people's tools. Runs in an environment that has model-signing, sigstore,
in-toto-attestation, securesystemslib and jsonschema, and nothing from this repository.

    python attestation_stock.py WORKDIR SCHEMADIR
"""

import base64
import hashlib
import json
import pathlib
import sys

import jsonschema
from cryptography.hazmat.primitives import serialization
from in_toto_attestation.v1.resource_descriptor import ResourceDescriptor
from in_toto_attestation.v1.statement import Statement
from model_signing._signing import sign_ec_key, sign_sigstore_pb
from referencing import Registry, Resource
from securesystemslib.dsse import Envelope
from securesystemslib.exceptions import VerificationError
from securesystemslib.signer import SSlibKey
from sigstore_models import intoto as intoto_pb
from sigstore_models.bundle import v1 as bundle_pb
from sigstore_models.common import v1 as common_pb

work, schemas = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
results = {}


def load(path):
    return json.loads((work / path).read_text())


def statement_of(doc):
    return json.loads(base64.b64decode(doc["payload"]))


def in_toto_accepts(st):
    subjects = [ResourceDescriptor(name=s["name"], digest=s["digest"]).pb for s in st["subject"]]
    Statement(subjects, st["predicateType"], st["predicate"]).validate()


# 1. in-toto-attestation parses and validates both documents Neyra writes.
receipt, change = load("edit.receipt.json"), load("change/change.statement.json")
for name, doc in (("receipt", receipt), ("transformation statement", change)):
    in_toto_accepts(statement_of(doc))
    results[f"in-toto-attestation validates the {name}"] = "accepted"

# 2. securesystemslib verifies the Ed25519 receipt, once the key object carries the keyid Neyra wrote.
pub = bytes.fromhex((work / "neyra.pub").read_text().strip())
keyid = receipt["signatures"][0]["keyid"]
env = Envelope.from_dict(json.loads(json.dumps(receipt)))
env.verify([SSlibKey(keyid, "ed25519", "ed25519", {"public": pub.hex()})], 1)
results["securesystemslib verifies the Ed25519 receipt"] = "accepted with the keyid Neyra wrote"
try:
    Envelope.from_dict(json.loads(json.dumps(receipt))).verify([SSlibKey("x", "ed25519", "ed25519", {"public": pub.hex()})], 1)
    results["securesystemslib with its own keyid"] = "accepted"
except VerificationError as e:
    results["securesystemslib with its own keyid"] = f"rejected ({type(e).__name__}): it matches signatures by keyid"

# 3. The OMS key verifier accepts the Neyra statement, signed with the same P-256 key as the model.
pem = (work / "oms.pub").read_bytes()
hint = hashlib.sha256(serialization.load_pem_public_key(pem).public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()
bundle = bundle_pb.Bundle(
    media_type=sign_sigstore_pb._BUNDLE_MEDIA_TYPE,
    verification_material=bundle_pb.VerificationMaterial(public_key=common_pb.PublicKeyIdentifier(hint=hint),
                                                         tlog_entries=[]),
    dsse_envelope=intoto_pb.Envelope(
        payload=change["payload"].encode(), payload_type=change["payloadType"],
        signatures=[intoto_pb.Signature(sig=change["signatures"][0]["sig"].encode(), keyid="")]))
verifier = sign_ec_key.Verifier(work / "oms.pub")
payload_type, payload = verifier._verify_bundle(bundle)
results["OMS ec_key.Verifier checks the Neyra statement's signature"] = f"accepted ({payload_type}, {len(payload)} bytes)"
try:
    verifier.verify(sign_sigstore_pb.Signature(bundle))
except ValueError as e:
    results["OMS Verifier.verify on the Neyra statement"] = f"refused as a model signature: {e}"

# 4. The join: Neyra's sha256 subjects against the OMS resources and subject of the same directory.
oms = statement_of(json.loads((work / "result.sig").read_text())["dsseEnvelope"])
oms_files = {r["name"]: r["digest"] for r in oms["predicate"]["resources"]}
st = statement_of(change)
joined = {s["name"]: s["digest"]["sha256"] for s in st["subject"] if "sha256" in s["digest"]}
root = oms["subject"][0]
assert joined.pop(root["name"]) == root["digest"]["sha256"], "OMS subject digest not found among Neyra subjects"
assert joined == oms_files, "file subjects differ from OMS resources"
results["join to the OMS signature"] = f"{len(joined)} file digests and the OMS subject digest equal, no file read"

# 5. CycloneDX 1.7 schema.
docs = {json.loads(p.read_text())["$id"]: json.loads(p.read_text()) for p in schemas.glob("*.json")}
registry = Registry().with_resources([(k, Resource.from_contents(v)) for k, v in docs.items()])
validator = jsonschema.Draft7Validator(docs["http://cyclonedx.org/schema/bom-1.7.schema.json"], registry=registry)
errors = list(validator.iter_errors(load("change/change.cdx.json")))
assert not errors, errors[:3]
results["CycloneDX 1.7 schema (jsonschema)"] = "valid"

print(json.dumps(results, indent=1))
