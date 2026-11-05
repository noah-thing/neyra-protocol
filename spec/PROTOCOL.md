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

A signature shows that a key signed these bytes. It does not show that the claim is true. In particular it does not show:

- when. Times in a ledger or a statement are the signer's own clock, and a ledger can be written after the fact
  with any dates, in a hash chain that verifies. Nothing here anchors a time; a transparency log or a trusted
  timestamp would, and none is required.
- that a holdout was sealed before anything was tuned against it. `holdout_commit` binds the items to a hash.
  It does not bind the moment, and a signer who saw the items can commit them afterwards.
- that the figures in a statistical claim came from a run. They are typed into the claim by the signer, and a
  forger can write a PASS with invented intervals under a fresh key that verifies.
- how many tries lie behind a result. A fresh ledger per try leaves no trace. The count a receipt carries
  (section 7) is the signer's, read from one ledger, and can be too low. With many sessions a null edit will pass
  gates that a single session would fail, so a pass without a trustworthy count is weak evidence.
- who the signer is. A key is not an identity; binding keys to people is the job of the system around it.

What a verifier does establish is exact: digests, roots, canonical forms, a patch turning one set of weights
into another, a ledger's internal chain. Records say which of their fields are exact and which are asserted
(`basis`, sections 7 and C.1) so that a reader does not take one for the other.

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

Domains in use: `payload`, `patch`, `weights`, `ledger`, `holdout`, `key`, from Annex B `rowleaf`,
`rownode`, `rowtensor`, and from Annex C `statement`.

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

Two further members are informative and every producer in this repository writes them. `attempts` is
`{counted, unit, counted_in, complete}`: `counted` is how many sessions (`unit`) the ledger the receipt is bound to
holds, counting this one, and `complete` is always `false`, because a signer can open a new ledger for each try.
`basis` is `{exact, asserted}`, two lists of plain sentences saying which parts of the claim a verifier can
check and which rest on the signer.

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

A signature is over the DSSE pre authentication encoding:

    "DSSEv1 " + len(payloadType) + " " + payloadType + " " + len(payload) + " " + payload

where lengths are decimal byte counts and `payload` is the decoded bytes. `sig` is standard base64.
Two algorithms are defined, told apart by the length of the public key:

- Ed25519, with a 32 byte raw public key. `sig` is the 64 byte signature.
- ECDSA over P-256 with SHA-256, with the DER encoded SubjectPublicKeyInfo as public key (91 bytes for the
  uncompressed point form). `sig` is the ASN.1 DER signature. This is the key type that OpenSSF model
  signing (OMS) accepts, so one key can sign a model and the statements about it. Any valid DER
  signature is accepted, malleable ones included; nothing here depends on signatures being unique.

`keyid` is `digest("key", public key bytes)`, the raw 32 bytes or the DER, so keys of the two kinds never share one.

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
that is neither 32 bytes nor a P-256 SubjectPublicKeyInfo is a usage error, outside this protocol.

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
`bad_claim`, `subject_mismatch`, `bad_signature`, `binding_mismatch`, from Annex B `bad_proof` and
`proof_mismatch`, and from Annex C `bad_transform`, `bad_status` and `files_mismatch`.

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
| `transform` | `file`, optional `public_hex` (Ed25519 or DER) | `{code}` |
| `transformbinding` | `statement`, `patch`, files | `{code}` |
| `transformfiles` | `statement`, a file, and `model`, a directory | `{code}` |
| `status` | `file`, optional `public_hex` | `{code}` |
| `statusbinding` | `status`, `target`, files | `{code}` |

Paths are relative to the vectors directory.

## 11. Versioning

The integer in `type` and the predicate URI is the major version. Changing a hash construction or
the canonical form is a new major version with a new domain string. Optional additions need none.

## Annex A: extension `low-rank/1`

Dense low-rank edits would be huge as index lists. With this extension a patch tensor may also have
`{i}.left` and `{i}.right`, F64 payloads, and manifest members `left_digest` and `right_digest`, digests of
those payloads as in section 5. `neyra edit` writes patches of this kind, so a C1 verifier MUST implement
the extension. Conformance for it is the cases of kind `patch` whose ids begin `lowrank`.

Structure. A tensor entry has both `left_digest` and `right_digest` or neither, and `requires` MUST then name
`low-rank/1`. Only tensors of dtype BF16, F16 or F32 with two dimensions `[rows, cols]` may use it. Any other
combination is `bad_manifest`, reported with the other structure checks of step 6. `{i}.left` has shape
`[rows, r]` and `{i}.right` shape `[r, cols]`, both F64, with `r` at least 1. A mismatch, a missing payload or
a payload tensor the manifest does not account for is `bad_manifest` at step 8. A payload that does not hash
to its digest is `payload_mismatch` at step 9, together with `idx` and `val`.

Arithmetic. Applying the tensor starts from the base converted to float64 (BF16 is the high half of a float32,
F16 widens exactly). For k from 0 to r - 1 in order, `w[i][j] = w[i][j] + (left[i][k] * right[k][j])`, each product
rounded to float64 on its own and then added. Fused multiply add MUST NOT be used, and a language that may fuse
must separate the multiplication explicitly. The result is cast to float32 by round to nearest even, then
rounded to the tensor dtype by round to nearest even, with overflow to infinity and gradual underflow for F16,
and BF16 as the float32 bits plus `0x7FFF` and the lowest kept bit, shifted right 16. Then `val` replaces the
entries at `idx`, and the tensor digest MUST equal `result_digest`. A result that depends on NaN is outside
conformance; infinities are inside it.

A verifier that does not implement the extension reports `unknown_extension`, and is not C1.

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

## Annex C: transformation statements

A receipt (section 8) says what was measured about an edit. It does not say, in a form other tools can
read, which model came from which, by what kind of change, how a reader checks that, or what the signer's
signature is supposed to mean. This annex adds a second statement type for that, and records that retire or
replace a statement. It changes nothing above: receipts, patches and ledgers verify as they did. A verifier that
does not implement this annex loses nothing and needs no entry in `requires`. Conformance for it is the cases
of kinds `transform`, `transformbinding`, `transformfiles`, `status` and `statusbinding`, and the P-256 cases of
kind `receipt`.

The statement is made to sit beside OpenSSF model signing (OMS, model-signing v1.x, `files` serialisation). It
names the files of the result by plain sha256, and the digest OMS gives the model as a whole, as in-toto
subjects, so a policy engine that already holds a verified OMS signature can join the two by comparing strings.
Nothing is read at join time. It also keeps the weights root as a subject, because file digests change when a
model is resharded, re-serialised or has its safetensors header reordered, and the weights root does not.

### C.1 Transformation statement

The envelope is the one in section 8 with the same verification, and it is signed in the same ways. The
statement is an in-toto Statement v1 with `predicateType` `https://neyra.run/transformation/v1`.

`subject` MUST hold exactly the following, in any order, and nothing else:

- `{"name": "result-weights", "digest": {"neyra_weights_v1": hex of result.weights_root}}`. The digest key is not
  `sha256`, because this value is a domain-separated hash of tensor digests and is not the sha256 of any file.
- For each entry of `result.files`: `{"name": path, "digest": {"sha256": sha256}}`.
- If `result.files` is not empty: `{"name": result.name, "digest": {"sha256": oms_root(result.files)}}`.

`predicate` has these members, all REQUIRED, and no others except `x-` members, which MUST be ignored and preserved:

- `type`: `"neyra.transformation/1"`.
- `kind`: one of `patch`, `merge`, `adapter`, `quantize`, `edit`, `unknown`. It is what the signer says the change was.
- `grade`: how the signer knows. `exact`: the signer applied the patch named in `recipe.patch_id` to the base and
  got the result bit for bit. `observed`: the signer hashed both models in full and compared them tensor by
  tensor, with no recipe re-derived. `sampled`: only parts were read, and `extent` counts are lower bounds.
  `declared`: the signer recomputed nothing. A change of kind `unknown` MUST NOT be `exact`.
- `base`: `{weights_root, files}`. `weights_root` is a digest. `files` lists the base's files as below, and is empty
  if the signer did not list them.
- `result`: `{weights_root, files, name, ignore}`. `name` is the model's name as OMS records it (the directory
  name), non empty if `files` is not. `ignore` is a list of paths left out of `files` beyond the four defaults.
- `recipe`: an object with no required members, except that an `exact` `patch`, `edit` or `merge` MUST have
  `patch_id`, a digest. Producers put the facts a reader needs to repeat the change here: for an adapter, the
  merge path `neyra check` matched; for a quantisation, the scheme and bit width. Values are strings, integers
  and booleans, never floats.
- `extent`: counts of integers, `tensors_base`, `tensors_result`, `changed`, `added`, `removed`,
  `entries_changed`, `entries_total`, or `null`. `changed` counts tensors present in both with different
  digests, `entries_changed` the elements that differ in those, and `entries_total` all elements of those
  tensors. A tensor that changed dtype or shape counts all its elements as changed. `null` is allowed only when
  `grade` is `declared`.
- `coverage`: `{bytes_read, bytes_total}`, integers. `bytes_total` is the size of the tensor data of the base and
  the result together, `bytes_read` how much of it the signer hashed or compared. For `exact` and `observed` the
  two MUST be equal, for `sampled` `bytes_read` MUST be smaller, and for `declared` it MUST be 0.
- `basis`: `{recomputable, asserted}`, lists of field names. The grade fixes them, and a verifier MUST reject any
  other value. `asserted` always holds `base.files`, `kind`, `recipe`, `meaning`, `time`, `tool`, `evidence` and
  `attempts`. For `exact`, `recomputable` is `base.weights_root`, `result.weights_root`, `result.files`, `extent`,
  `patch_application`; for `observed` the same without `patch_application`; for `sampled` and `declared` it is
  empty and the first four move to `asserted`, after the fixed eight, in that order.
- `attempts`: an integer of at least 1, or `null`. It is how many results the signer made and tried before this
  one, counting it. `null` means nobody counted, and says nothing about whether tries were thrown away. It is
  always in `asserted`.
- `rederive`: a non empty array of strings, the steps by which a reader repeats the check. Text for people.
- `meaning`: what the signature means, one of `authored` (the signer made the change), `reviewed` (the signer
  reviewed it and did not make it), `approved-for-deploy` (the signer approves this exact result for deployment
  on the strength of the checks listed here only) and `responsible-for-merge` (the signer is responsible for
  combining the inputs listed here). A signature does not carry more meaning than the word it signs.
- `not_measured`: an array of strings that MUST contain `training compute`, `training data` and `behaviour`.
  The statement cannot be made without saying so.
- `evidence`: an array of `{role, digest}`, possibly empty. Roles in use are `patch` (a patch id) and `receipt`
  (the `statement` digest, below, of a receipt's payload).
- `time`: an RFC 3339 UTC string in the pattern of section 6. It is the signer's clock, not a timestamp.
- `tool`: `{name, version}`, strings.

A file entry is `{path, sha256}`: `path` relative to the model directory with `/` separators, no empty,
`.` or `..` component and no `\`; `sha256` 64 lowercase hex characters, of the file's bytes. The array is
sorted by path component by component, comparing code points, as OMS orders them, with no repeats. So `a/b`
comes before `a.txt`.

The files of a directory are its regular files at any depth, except a path equal to one of `.git`,
`.gitattributes`, `.github`, `.gitignore` or an `ignore` path of the statement, or below one followed by `/`.
A sibling that only shares leading characters stays listed. A symbolic link makes
the directory unlistable. `oms_root(files)` is the hex of the SHA-256 over the raw 32 byte file digests,
concatenated in that order. For the same files OMS 1.1.1 writes this value as the subject digest of its
statement, and the same file digests as `predicate.resources`.

A statement is a claim about the signer's own reading of the two models. It is not a measure of training compute,
data or behaviour, and says so. Signing it does not show when it was signed, who the signer is, or that the
`asserted` fields are true; section 1 lists what no signature here proves.

Reading of the byte counts. `coverage` is how much the signer says they read. A reader who holds the models
can recompute `result.files`, `result.weights_root`, `base.weights_root` and `extent` (C.3) and does not need to
believe it.

### C.2 Verification order

1. Steps 1 to 4 of section 8. `bad_json`, `bad_type`, `not_canonical`.
2. `_type` MUST be Statement v1 and `predicateType` the one above. `bad_type`
3. The predicate MUST be well formed as C.1 describes. Any failure is `bad_transform`, so the order within this
   step does not matter.
4. `subject` MUST be exactly what C.1 derives from the predicate. `subject_mismatch`
5. With a public key, a signature as in section 8. `bad_signature`

An unsigned statement is an unauthenticated record, and tools MUST say so.

### C.3 Binding to evidence

Given a statement and a patch, a verifier MUST first verify each alone, the statement and then the patch, and
report the first failure. It then reports `binding_mismatch` if `recipe.patch_id` is not the patch's id, or the
patch's `base_root` or `result_root` differs from `base.weights_root` or `result.weights_root`.

Given a statement and a model directory, a verifier verifies the statement alone, then computes the weights root
of the directory (section 3) and reports `result_mismatch` if it is not `result.weights_root`. If `result.files`
is not empty it then lists the directory as C.1 describes and reports `files_mismatch` unless the listing equals
`result.files`. A directory with no weight files is `bad_safetensors`.

Both bindings need a full read of the model. The weights root has never been cheaper than that, and a file
digest cannot be sampled. What costs nothing is joining to an OMS signature that is already verified: compare
`result.files` with its `predicate.resources` and `oms_root(result.files)` with its subject digest.

### C.4 Status records

A statement is never edited or deleted. A later signed record says that it is replaced, or that the signer no
longer stands behind it. The statement keeps verifying as what it was.

A status record is an in-toto Statement v1 with `predicateType` `https://neyra.run/status/v1`, one subject,
`{"name": "result-weights", "digest": {"neyra_weights_v1": hex of weights_root}}`, and a predicate with these
members, no others except `x-`:

- `type`: `"neyra.status/1"`.
- `action`: `supersede` or `withdraw`.
- `target`: the `statement` digest of the payload of the envelope concerned,
  `digest("statement", payload bytes)`. The target may be a transformation statement or a receipt.
- `replacement`: the same digest of the payload that takes its place. REQUIRED for `supersede`, and MUST NOT
  appear for `withdraw`. It MUST differ from `target`.
- `weights_root`: the result weights root of the target.
- `reason`: a non empty string.
- `meaning`, `time`, `tool`: as in C.1.

Verification: steps 1 and 2 of C.2 with this predicate type, then the members above. `bad_status`. The subject
MUST be the one named. `subject_mismatch`. Then the signature, `bad_signature`.

Binding. Given a record and its target envelope, a verifier verifies each alone, the record and then the target
by its own predicate type, and reports `binding_mismatch` if `target` is not the digest of the target's payload
or `weights_root` is not the target's result root (`result.weights_root`, or `claim.result_root` in a receipt).

Meaning. A record carries weight only when signed by a key the reader already trusts for the target. A reader
that gathers records MUST ignore those it cannot attribute to a trusted key. Of the records that bind, a
withdrawal makes the target `withdrawn`, else a supersession makes it `superseded` by the replacement digests
named, else it is `current`. The reader is told who withdrew it, why and when. Withdrawing a replacement does
not make the replaced statement current again.

A withdrawal is not a revocation of a key, and says nothing about other statements by the same signer.

### C.5 Relation to other attestations

A pipeline attestation (an Atlas CLI or SLSA record of what ran, with which inputs, on which machine) says what a
system reports having done. A transformation statement says what two sets of bytes show. Neither implies the other.
They link by digest and do not need to know each other's format. A statement may carry a pipeline record in
`evidence` with a role such as `c2pa-manifest` or `slsa-provenance` and the digest of that record. The role is a
free string, readers that do not know it ignore it, and Neyra checks nothing about what the digest points to.
The other direction is a pipeline record that lists the `result-weights` subject, or the OMS root, among its outputs.

OMS signs the files of a model and says nothing about how it relates to another. This statement does not replace
that signature and holds no signature of OMS: it names the same files and the same OMS subject digest, and
whoever trusts both signatures can join them. A signer who wants one key for both signs with P-256 (section 8).
`model_signing verify` reads only its own predicate, so it will report a mismatch on a transformation statement; the
signature itself verifies with the OMS key verifier, as the demonstration in the repository shows.
