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

Domains in use: `payload`, `patch`, `weights`, `ledger`, `holdout`, `key`, and, from Annex B, `rowleaf`,
`rownode`, `rowtensor`.

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
names that start with a dot. Directories, even ones named `x.safetensors`, are not files and are skipped.
The same tensor name in two files is `bad_safetensors`, as is a model with no weight files. No file is
skipped for its name alone: `consolidated.safetensors` is a shard like any other. A tool that reads a Hub
repository may leave out such duplicate copies as a convenience, and then it is not computing this root.

## 4. Reading safetensors

A file is an 8 byte little endian header length N, then N bytes of JSON, then tensor data. A
verifier MUST reject with `bad_safetensors` if: the file is under 8 bytes; N exceeds 100,000,000 or
runs past the end of the file; the header is not a JSON object; an entry lacks a known dtype, a shape
of non negative integers, or two non negative offsets; the byte span differs from what dtype and
shape require; spans are not contiguous from zero when sorted; or bytes remain after the last span.
A header that fails to parse is `bad_safetensors`, not `bad_json`.

The member `__metadata__`, if present, MUST be an object that maps strings to strings. Null, an array, a
string or a number there is `bad_safetensors`, not the same as absent. A key that appears twice in the header
object, or in any object inside it, is `bad_safetensors`: two readers must not see different tensors under
one name.

Item sizes: `BOOL U8 I8 F8_E4M3 F8_E5M2 F8_E8M0` are 1, `U16 I16 F16 BF16` are 2, `U32 I32 F32` are
4, and `U64 I64 F64` are 8.

## 5. Patch

A patch is one safetensors file. Its `__metadata__` holds `neyra.manifest`, a string containing the
JCS encoding of the manifest. It MAY hold `neyra.meta`, which is free text outside the patch identity and
which a verifier MUST NOT parse or reject.

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
4. The manifest MUST be an object whose `type` is `neyra.patch/1`. Any other JSON value that passed step 3 is
   `bad_type`.
5. Check for unknown members at the top level, then in tensor entries. `unknown_member`
6. A required member that is missing or of the wrong JSON type (`name` and `dtype` are strings, `shape` is a
   list of non negative integers and not booleans), a digest that is not well formed, or
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
0x0A. The check is on the pattern alone, not on whether the date exists, and every digit is ASCII 0 to 9.
`seq` is a JSON integer, not a boolean, `kind` a string and `body` an object. An empty file is a valid
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
(standard base64 with padding), and `signatures`, an array of `{keyid, sig}`, where `keyid` and `sig` are
strings. Base64 is also canonical: the unused bits of the last character MUST be zero (RFC 4648 section 3.5),
so one byte string has one encoding. An envelope that is not an object, lacks a member, or has a member or
signature entry of the wrong type is `bad_json`, before anything else is looked at.

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

1. Parse the envelope, including the shape described above. `bad_json`
2. `payloadType` MUST match. `bad_type`
3. Decode the payload, parse it strictly. `bad_json`
4. It MUST be canonical. `not_canonical`
5. `_type` and `predicateType` MUST match. `bad_type`
6. The predicate MUST have a claim with `type` set to `neyra.claim/1` and every member of section 7,
   plus `ledger_head`, `ledger_len`, and `digest_algorithm` equal to `neyra/weights/v1`. Only presence
   and those two values are checked here, not the types of other claim members. `bad_claim`
7. `subject` MUST hold exactly the two named entries, in either order, and they MUST match
   `result_root` and `patch_id` in the claim. Those two claim members MUST be well formed digests, and a
   subject digest is compared with the hex after `sha256:`. `subject_mismatch`
8. With a public key: some entry in `signatures` MUST have its `keyid` and a signature that verifies. An
   entry that has the key but fails does not stop a later one from succeeding. Malformed base64 in `sig`
   is a verification failure of that entry. `bad_signature`

An envelope with no signatures is an unauthenticated record, and tools MUST say so. A public key
that is not 32 bytes is a usage error, outside this protocol.

### Binding a receipt to its evidence

Given a receipt, its ledger and its patch, a verifier MUST first verify them alone, in that order,
and report the first failure. It MUST then check in this order and report `binding_mismatch` on the
first failure, which includes `ledger_head` or `ledger_len` being of the wrong type:

1. The patch id and the patch's `result_root` MUST equal those in the claim.
2. `ledger_head` MUST be the digest of an entry in the ledger.
3. `ledger_len` MUST be a JSON integer, not a boolean, equal to that entry's `seq` plus one.

Revealing the salt and items lets anyone check `holdout_commit`, which is outside conformance.

## 9. Error codes

`ok`, `bad_safetensors`, `bad_json`, `not_canonical`, `bad_type`, `bad_manifest`, `unknown_member`,
`unknown_extension`, `payload_mismatch`, `bad_index`, `base_mismatch`, `result_mismatch`, `bad_chain`,
`bad_claim`, `subject_mismatch`, `bad_signature`, `binding_mismatch`, and from Annex B `bad_proof` and
`proof_mismatch`.

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
| `binding` | `receipt`, `ledger`, `patch`, all files | `{code}` |
| `rowtree` | `leaves_hex`, a list of 32 byte hashes | `{root}` |
| `rowroot` | `model`, a directory | `{root}` or `{code}` |
| `rowproof` | `proof` and `data`, files | `{code}` |

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

## Annex B: extension `rows/1`

The weights root hashes each tensor whole, so a claim about a few rows of a tensor cannot be checked
against it without reading all of that tensor. This annex defines a second root over the same tensors,
built so that a few rows and a short proof are enough. It changes no digest above. A verifier that does
not implement it loses nothing, and it needs no entry in `requires`. Conformance for it is the cases of
kinds `rowtree`, `rowroot` and `rowproof`.

### B.1 Tree

Hashes are 32 raw bytes here, the bytes behind the `sha256:` form. `H(domain, bytes)` is the SHA-256
construction of section 3 with the digest left as raw bytes.

    leaf(b)     = H("rowleaf", b)
    node(l, r)  = H("rownode", l + r)

A tree over n hashes in order is, by RFC 6962: for n = 0, `H("rownode", "")`; for n = 1 the hash itself; for
n above 1, `node(T(first k), T(rest))` with k the largest power of two below n. A tree is never padded or
duplicated, so no two sequences share a root.

### B.2 Rows and leaves

A tensor of shape `[d0, d1, ...]` has `d0` rows, each `d1 * ... * itemsize` bytes, the product being 1 for
`[d0]`. A tensor of shape `[]` has one row of `itemsize` bytes. Let `width` be the bytes in a row.

If the tensor has no bytes it has no leaves. Otherwise a leaf holds `g = ceil(4096 / width)` consecutive
rows, so every leaf but the last is at least 4096 bytes, and the last holds what is left. Leaf `k` is
`leaf(raw bytes of rows k*g to min(rows, (k+1)*g))`. The tensor tree is the tree over its leaves. Leaves
depend only on dtype and shape, so a verifier needs nothing else to find where a row sits.

A tensor root is the tree root of its leaf hashes. The tensor's entry in the model tree is

    entry = H("rowtensor", JCS({"name", "dtype", "shape", "tree": digest of the tensor root}))

so the name, dtype and shape are part of what a proof proves. The rows root is the tree over the entries
of all tensors, sorted by `name` in UTF-16 code unit order, written as a digest. It covers the same tensors
as the weights root and ignores sharding in the same way.

### B.3 Row proof

A row proof is a JSON object with exactly these members: `type` set to `"neyra.rowproof/1"`;
`rows_root`; `tensor`, an object with `name`, `dtype` and `shape`; `index` and `count`, the position of the
tensor among the `count` tensors; `ranges`, a non empty list of `[start, end)` row pairs, each non empty,
sorted and not overlapping, inside the tensor; `tree`, a list of digests; and `path`, a list of digests.

The opened leaves are every leaf that holds a row of some range, in increasing order. The data is their
bytes joined in that order. `tree` holds, left to right, the hash of each maximal subtree of the tensor
tree that contains no opened leaf. `path` holds the same for the model tree and the single leaf `index`.

A verifier, given a proof and the data, MUST report the first of these, in this order:

1. The proof does not have the members and types above, a range is out of order or outside the tensor, the
   tensor has no leaves, `index` is not below `count`, or a digest is malformed. `bad_proof`
2. The data is not exactly the length of the opened leaves. `bad_proof`
3. `tree` or `path` has fewer hashes than the walk consumes, or more. `bad_proof`
4. Rebuilding the tensor root from the data and `tree`, then the root from the entry and `path`, does not
   give `rows_root`. `proof_mismatch`

A proof that is not JSON is `bad_proof`. Rows are those the ranges name, cut out of the data. A proof for one range in a
tensor of a quarter million leaves holds at most about 46 hashes, 1.5 KB as raw hashes.

### B.4 What a proof does and does not show

A proof shows that rows belong to a tensor of a given name, dtype and shape in a model with a given rows
root. It says nothing about the weights root. That tie cannot be made from the rows, because the weights
root hashes each tensor whole and SHA-256 gives no way to show a part of a message to be a part of the
hashed one. Someone who has read the whole model must make it.

### B.5 Index and link

An index is a safetensors file whose `__metadata__` holds `neyra.rows`, the JCS encoding of
`{"type": "neyra.rows/1", "weights_root", "rows_root", "tensors"}`. `tensors` has, for each tensor in
name order, `name`, `dtype`, `shape`, `digest` (the tensor digest of section 3) and `tree` (the tensor
root). Tensor number `i` has the payload tensor `{i}`, U8 of shape `[leaves, 32]`, holding its leaf hashes.
Anyone can rebuild both roots from an index, and check them against a model by hashing the model once.
Publishing the index for a model one does not own is allowed, and worth exactly as much as the link.

A link is a DSSE envelope as in section 8, signed in the same way, over an in-toto Statement v1 with
`predicateType` `https://neyra.run/rows/v1`, subjects `weights` and `rows` in that order holding the two
roots, and a predicate of `digest_algorithm` `"neyra/rows/v1"`, `weights_root`, `rows_root` and `tensors`.
A signature on a link says that the signer read the whole model and found its two roots to be these. A
verifier that trusts the key, or several independent keys, can then accept rows against the weights root.
Links have no conformance cases in this version.
