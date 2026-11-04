# Neyra Protocol, version 1

Status: draft. Frozen only after a second independent implementation passes every vector.

Neyra Protocol is a format for making checkable claims about edits to neural network weights.
A claim says: this patch, applied to exactly this model, gives exactly that model, and here is
what was measured. Anyone can check the first part with no GPU and no model runtime.

The key words MUST, MUST NOT, SHOULD and MAY are used as in BCP 14 (RFC 2119, RFC 8174).

## 1. Two classes of claim

Exact claims are bit level: "patch P applied to weights root R gives weights root R2". Every
conforming verifier reaches the same answer. They are the subject of conformance.

Statistical claims are measurements: "on this holdout, under this runtime, the effect was d".
They are signed statements, recorded here but not reproducible bit for bit, since runtimes differ.

A signature shows who made a claim. It does not show that the claim is true.

## 2. Canonical JSON

Every hashed object is encoded with the JSON Canonicalization Scheme, RFC 8785, as UTF-8, with
these restrictions:

- Numbers MUST be integers with magnitude at most 2^53 - 1. Fractions, exponents and floats MUST NOT
  appear. A measurement is a decimal string such as `"0.012300"`.
- Object keys are sorted by their UTF-16 code units.
- Duplicate keys, lone surrogates, NaN and Infinity MUST be rejected with `bad_json`.
- Strings escape only what RFC 8785 requires: `"`, `\`, and U+0000 to U+001F, using `\b \t \n \f \r`
  where those exist and lowercase `\u00xx` otherwise. Everything else is literal UTF-8.

## 3. Digests

A digest is written `sha256:` followed by 64 lowercase hex characters.

    digest(domain, bytes) = "sha256:" + hex(SHA-256("neyra/" + domain + "/v1" + 0x00 + bytes))

Domains in use: `payload`, `patch`, `weights`, `ledger`, `holdout`, `key`.

Tensor digest. `dtype` is the safetensors dtype name, `shape` the dimensions, and `raw` the tensor
bytes exactly as stored, little endian:

    SHA-256("neyra/tensor/v1" + 0x00 + dtype + 0x00 + ndim as one byte
            + each dimension as u64 little endian + raw)

Weights root. Form an array with one object per tensor, `{"name", "dtype", "shape", "digest"}`,
sorted by `name` in UTF-16 code unit order, then:

    weights_root = digest("weights", JCS(array))

The root covers every tensor and ignores how tensors are split across files, so two shardings of
the same weights have the same root. It does not cover tokenizers, configs or chat templates.

## 4. Reading safetensors

A file is an 8 byte little endian header length N, then N bytes of JSON, then tensor data. A
verifier MUST reject with `bad_safetensors` if: the file is under 8 bytes; N exceeds 100,000,000 or
runs past the end of the file; the header is not a JSON object; an entry lacks a known dtype, a shape
of non negative integers, or two non negative offsets; the byte span differs from what dtype and
shape require; spans are not contiguous from zero when sorted; or bytes remain after the last span.

The member `__metadata__`, if present, MUST map strings to strings.

Item sizes: `BOOL U8 I8 F8_E4M3 F8_E5M2 F8_E8M0` are 1, `U16 I16 F16 BF16` are 2, `U32 I32 F32` are
4, and `U64 I64 F64` are 8.

## 5. Patch

A patch is one safetensors file. Its `__metadata__` holds `neyra.manifest`, a string containing the
JCS encoding of the manifest. It MAY hold `neyra.meta`, which is free text outside the patch identity.

Manifest members, all REQUIRED:

- `type`: `"neyra.patch/1"`
- `base_root`, `result_root`: weights roots
- `requires`: array of extension names, empty if none
- `tensors`: array sorted by `name` in UTF-16 code unit order, each with `name`, `dtype`, `shape`,
  `base_digest`, `result_digest`, `idx_digest`, `val_digest`

Tensor number `i` in that array has two payload tensors in the file: `{i}.idx`, dtype I64, flat
element indices into the tensor; and `{i}.val`, the new elements as raw bits of the same width.

    idx_digest = digest("payload", raw bytes of {i}.idx)
    val_digest = digest("payload", raw bytes of {i}.val)
    patch id   = digest("patch", JCS(manifest))

Members whose names start with `x-` MUST be ignored and preserved. Any other unknown member is
`unknown_member`. An extension in `requires` that the verifier does not implement is
`unknown_extension`.

### Verification order

A verifier MUST run these steps in order and report the first failure:

1. Read the file as safetensors. `bad_safetensors`
2. Require `neyra.manifest` and parse it strictly. `bad_type` if absent, `bad_json` if malformed.
3. The string MUST equal the JCS encoding of what it parsed to. `not_canonical`
4. `type` MUST be `neyra.patch/1`. `bad_type`
5. Check members, then `requires`. `unknown_member`, `unknown_extension`
6. Check each payload digest. `payload_mismatch`
7. Each `{i}.idx` MUST be strictly increasing and inside the tensor. `bad_index`

With a base model available:

8. The base's weights root MUST equal `base_root`. `base_mismatch`
9. For each tensor, replace elements of the base bytes at `idx` with `val`. The tensor digest of the
   result MUST equal `result_digest`. `result_mismatch`
10. The weights root after substitution MUST equal `result_root`. `result_mismatch`

## 6. Ledger

A ledger is a file of lines, each the JCS encoding of one entry followed by 0x0A.

Entry members, all REQUIRED and no others: `seq` (integer from 0), `prev` (digest of the previous
entry, or `sha256:` and 64 zeros for the first), `time` (RFC 3339 UTC string), `kind` (string), and
`body` (object).

    entry digest = digest("ledger", JCS(entry))

The head is the digest of the last entry. A verifier MUST report `bad_chain` if a line is not
canonical, if members differ, if `seq` is out of order, if `prev` is wrong, or if the file does not
end with 0x0A. A writer recovering from a crash truncates to the last complete line and appends an
entry of kind `recovered`.

## 7. Claim

A claim is an object with `type` set to `"neyra.claim/1"`, plus `base_root`, `result_root`,
`patch_id`, `task`, `eval_env`, `statements`, `gates`, `verdict` and `limits`. A claim MAY include
further informative members.

`task.holdout_commit` is `digest("holdout", salt + JCS(items))` with a salt of at least 32 random
bytes, so the sealed items can be revealed later and checked, and cannot be guessed from the hash.

Each statement has `metric`, `estimate`, `ci_low`, `ci_high` (decimal strings), `n` (integer) and
`method` (string). `verdict` is `"PASS"` or `"FAIL"`.

## 8. Receipt

A receipt is a DSSE envelope, as JSON: `payloadType` set to `application/vnd.in-toto+json`, `payload`
(standard base64 with padding), and `signatures`, an array of `{keyid, sig}`.

The payload is the JCS encoding of an in-toto Statement v1:

- `_type`: `https://in-toto.io/Statement/v1`
- `subject`: two entries, `{"name": "result-weights", "digest": {"sha256": hex of result_root}}` and
  `{"name": "patch", "digest": {"sha256": hex of patch_id}}`
- `predicateType`: `https://neyra.run/receipt/v1`
- `predicate`: `claim`, `digest_algorithm` set to `"neyra/weights/v1"`, `ledger_head`, `ledger_len`

A signature is Ed25519 over the DSSE pre authentication encoding:

    "DSSEv1 " + len(payloadType) + " " + payloadType + " " + len(payload) + " " + payload

where lengths are decimal byte counts and `payload` is the decoded bytes. `sig` is standard base64.
`keyid` is `digest("key", raw 32 byte public key)`.

### Verification order

1. Parse the envelope. `bad_json`
2. `payloadType` MUST match. `bad_type`
3. Decode the payload, parse it strictly. `bad_json`
4. It MUST be canonical. `not_canonical`
5. `_type` and `predicateType` MUST match. `bad_type`
6. The two subjects MUST match `result_root` and `patch_id` in the claim. `subject_mismatch`
7. With a public key: a signature with its `keyid` MUST exist and verify. `bad_signature`

An envelope with no signatures is an unauthenticated record, and tools MUST say so.

## 9. Error codes

`ok`, `bad_safetensors`, `bad_json`, `not_canonical`, `bad_type`, `unknown_member`,
`unknown_extension`, `payload_mismatch`, `bad_index`, `base_mismatch`, `result_mismatch`, `bad_chain`,
`subject_mismatch`, `bad_signature`.

These are normative, so two implementations can be compared line by line.

## 10. Conformance

A C1 verifier gives the expected result on every case in `spec/vectors/cases.json`. A C2 producer
writes patches, ledgers and receipts that a C1 verifier from another code base accepts. Re-running
statistical claims is informative and not part of conformance.

An implementation counts as independent only if it is in another language and shares no code.

## 11. Versioning

The integer in `type` and the predicate URI is the major version. Changing a hash construction or
the canonical form is a new major version with a new domain string. Optional additions need none.

## Annex A: extension `low-rank/1`

Dense low rank edits would be huge as index lists. With this extension a patch tensor may also have
`{i}.left` and `{i}.right`, F64 payloads of shape [rows, r] and [r, columns], and manifest members
`left_digest` and `right_digest`. Applying it starts from the base converted to float64, adds each
outer product in order k = 0 to r - 1, casts to float32, rounds to the tensor dtype with round to
nearest even, then substitutes `idx` and `val`, which correct any remaining entries. Fused multiply
add MUST NOT be used. Only BF16, F16 and F32 tensors may use it. Verifiers need not implement it.
