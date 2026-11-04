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
- Input follows the RFC 8259 grammar. Invalid UTF-8, leading zeros, duplicate keys, lone surrogates,
  NaN, Infinity and nesting deeper than 128 levels MUST be rejected with `bad_json`. The outermost
  array or object is level 1, and a bare scalar is level 0.
- `-0` parses to the integer 0, and so its canonical form is `0`.
- Strings escape only what RFC 8785 requires: `"`, `\`, and U+0000 to U+001F, using `\b \t \n \f \r`
  where those exist and lowercase `\u00xx` otherwise. Everything else is literal UTF-8.

## 3. Digests

A digest is always written `sha256:` followed by 64 lowercase hex characters, in every object and
in every array hashed below, including tensor digests.

    digest(domain, bytes) = "sha256:" + hex(SHA-256("neyra/" + domain + "/v1" + 0x00 + bytes))

Domains in use: `payload`, `patch`, `weights`, `ledger`, `holdout`, `key`.

Tensor digest. `dtype` is the safetensors dtype name, `shape` the dimensions, and `raw` the tensor
bytes exactly as stored, little endian:

    "sha256:" + hex(SHA-256("neyra/tensor/v1" + 0x00 + dtype + 0x00 + ndim as one byte
                            + each dimension as u64 little endian + raw))

Tensors with more than 255 dimensions are not supported.

Weights root. Form an array with one object per tensor, `{"name", "dtype", "shape", "digest"}`,
sorted by `name` in UTF-16 code unit order, then:

    weights_root = digest("weights", JCS(array))

The root covers every tensor and ignores how tensors are split across files, so two shardings of
the same weights have the same root. It does not cover tokenizers, configs or chat templates.

A model is the union of the tensors in its weight files. In a directory those are the regular top
level files whose names end in `.safetensors`, compared case sensitively, in any order, skipping
names that start with a dot. The same tensor name in two files is `bad_safetensors`, as is
a model with no weight files.

## 4. Reading safetensors

A file is an 8 byte little endian header length N, then N bytes of JSON, then tensor data. A
verifier MUST reject with `bad_safetensors` if: the file is under 8 bytes; N exceeds 100,000,000 or
runs past the end of the file; the header is not a JSON object; an entry lacks a known dtype, a shape
of non negative integers, or two non negative offsets; the byte span differs from what dtype and
shape require; spans are not contiguous from zero when sorted; or bytes remain after the last span.
A header that fails to parse is `bad_safetensors`, not `bad_json`.

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
element indices into the tensor; and `{i}.val`, the new elements as raw bits. `{i}.val` has dtype
`U8`, `U16`, `U32` or `U64`, whichever matches the item size of the tensor, and not the tensor's own
dtype, so a verifier never interprets them. Both are one dimensional and of equal length.

    idx_digest = digest("payload", raw bytes of {i}.idx)
    val_digest = digest("payload", raw bytes of {i}.val)
    patch id   = digest("patch", JCS(manifest))

In the manifest and in each tensor entry, members whose names start with `x-` MUST be ignored and
preserved, and any other unknown member is `unknown_member`. An extension in `requires` that the
verifier does not implement is `unknown_extension`. Members that an implemented extension defines
are known members.

Step checks run over the whole patch before the next step starts, and within a step tensors are
taken in array order, so the first failure is well defined.

### Verification order

A verifier MUST run these steps in order and report the first failure:

1. Read the file as safetensors. `bad_safetensors`
2. Require `neyra.manifest` and parse it strictly. `bad_type` if absent, `bad_json` if malformed.
3. The string MUST equal the JCS encoding of what it parsed to. `not_canonical`
4. `type` MUST be `neyra.patch/1`. `bad_type`
5. Check for unknown members at the top level, then in tensor entries. `unknown_member`
6. A required member that is missing or of the wrong JSON type, a digest that is not well formed, or
   a `tensors` array that is unsorted or repeats a name, is `bad_manifest`.
7. Check `requires`. `unknown_extension`
8. Each payload tensor MUST exist with the right dtype, the two lengths MUST agree, and the file MUST
   hold no other tensors. `bad_manifest`
9. Check each payload digest. `payload_mismatch`
10. Each `{i}.idx` MUST be strictly increasing and inside the tensor. `bad_index`

With a base model available:

11. The base's weights root MUST equal `base_root`, and each patched tensor MUST exist in the base with
    the manifest's dtype, shape and `base_digest`. `base_mismatch`
12. For each tensor, replace elements of the base bytes at `idx` with `val`. The tensor digest of the
    result MUST equal `result_digest`. `result_mismatch`
13. The weights root after substitution MUST equal `result_root`. `result_mismatch`

## 6. Ledger

A ledger is a file of lines, each the JCS encoding of one entry followed by 0x0A.

Entry members, all REQUIRED and no others: `seq` (integer from 0), `prev` (digest of the previous
entry, or `sha256:` and 64 zeros for the first), `time` (RFC 3339 UTC string), `kind` (string), and
`body` (object).

    entry digest = digest("ledger", JCS(entry))

The head is the digest of the last entry. A verifier MUST report `bad_chain` if a line is not
canonical or does not parse, if members differ, if `seq` is out of order, if `prev` is wrong, if
`time` does not match the pattern `YYYY-MM-DDThh:mm:ssZ`, or if a non empty file does not end with
0x0A. The check is on the pattern alone, not on whether the date exists. An empty file is a valid
ledger of no entries, with the all zero digest as its head. A writer recovering from a crash truncates to the last complete line and appends an
entry of kind `recovered`.

## 7. Claim

A claim is an object with `type` set to `"neyra.claim/1"`, plus `base_root`, `result_root`,
`patch_id`, `task`, `eval_env`, `statements`, `gates`, `verdict` and `limits`. Any further member is
informative and MUST be preserved.

`task.holdout_commit` is `digest("holdout", salt + JCS(items))` with a salt of at least 32 random
bytes, so the sealed items can be revealed later and checked, and cannot be guessed from the hash.

Each statement has `metric`, `estimate`, `ci_low`, `ci_high` (decimal strings), `n` (integer) and
`method` (string). `verdict` is `"PASS"` or `"FAIL"`.

## 8. Receipt

A receipt is a DSSE envelope, as JSON, which need not itself be canonical. Base64 here is strict:
standard alphabet, required padding, no whitespace. Members are `payloadType`, set to `application/vnd.in-toto+json`, `payload`
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
6. The predicate MUST have a claim with `type` set to `neyra.claim/1` and every member of section 7,
   plus `ledger_head`, `ledger_len`, and `digest_algorithm` equal to `neyra/weights/v1`. Only presence
   and those two values are checked here, not the types of other claim members. `bad_claim`
7. `subject` MUST hold exactly the two named entries, in either order, and they MUST match
   `result_root` and `patch_id` in the claim. `subject_mismatch`
8. With a public key: some entry in `signatures` MUST have its `keyid` and a signature that verifies.
   Malformed base64 in `sig` is a verification failure. `bad_signature`

An envelope with no signatures is an unauthenticated record, and tools MUST say so. A public key
that is not 32 bytes is a usage error, outside this protocol.

Binding a receipt to its files is a separate step: the patch's id and result root MUST equal the
claim's, `ledger_head` MUST be the digest of an entry in the ledger, and `ledger_len` MUST equal that
entry's `seq` plus one. Revealing the salt and items lets anyone check `holdout_commit`.

## 9. Error codes

`ok`, `bad_safetensors`, `bad_json`, `not_canonical`, `bad_type`, `bad_manifest`, `unknown_member`,
`unknown_extension`, `payload_mismatch`, `bad_index`, `base_mismatch`, `result_mismatch`, `bad_chain`,
`bad_claim`, `subject_mismatch`, `bad_signature`.

These are normative, so two implementations can be compared line by line.

## 10. Conformance

A C1 verifier gives the expected result on every case in `spec/vectors/cases.json`. A C2 producer
writes patches, ledgers and receipts that a C1 verifier from another code base accepts. Re-running
statistical claims is informative and not part of conformance.

An implementation counts as independent only if it is in another language and shares no code.

`spec/vectors/cases.json` is an array. Every case has `kind`, `id` and `expect`. By kind:

| kind | input members | expect |
|---|---|---|
| `jcs` | `input_hex`, bytes to parse | `{code, hex}` with the canonical bytes, or `{code}` |
| `digest` | `domain`, `data_hex` | `{digest}` |
| `tensor` | `dtype`, `shape`, `raw_hex` | `{digest}` |
| `root` | `model`, a directory | `{root}` or `{code}` |
| `patch` | `file`, optional `base` directory | `{code}` |
| `ledger` | `file` | `{code, head}` or `{code}` |
| `receipt` | `file`, optional `public_hex` | `{code}` |

Paths are relative to the vectors directory.

## 11. Versioning

The integer in `type` and the predicate URI is the major version. Changing a hash construction or
the canonical form is a new major version with a new domain string. Optional additions need none.

## Annex A: extension `low-rank/1`

Dense low rank edits would be huge as index lists. With this extension a patch tensor may also have
`{i}.left` and `{i}.right`, F64 payloads of shape [rows, r] and [r, columns], and manifest members
`left_digest` and `right_digest`. Applying it starts from the base converted to float64, adds each
outer product in order k = 0 to r - 1, casts to float32, rounds to the tensor dtype with round to
nearest even, then substitutes `idx` and `val`, which correct any remaining entries. Fused multiply
add MUST NOT be used. Only BF16, F16 and F32 tensors may use it. Verifiers need not implement it, and
one that does not MUST report `unknown_extension`. No conformance vector requires it.
