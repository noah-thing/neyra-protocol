# CycloneDX properties used by `neyra render`

`neyra attest` and `neyra render --format cdx` write a CycloneDX 1.7 file whose `machine-learning-model` component
carries these properties. The namespace `neyra:` is not registered in the CycloneDX property taxonomy. Nothing has
been submitted there; a reader that does not know it ignores the properties, and the file validates against
`bom-1.7.schema.json` either way. Every value is a string, and each is a copy of a field of the signed statement
(Annex C of the protocol), which is the source of truth.

| Property | Value |
|---|---|
| `neyra:kind` | `patch`, `merge`, `adapter`, `quantize`, `edit` or `unknown` |
| `neyra:evidence-grade` | `exact`, `observed`, `sampled` or `declared` |
| `neyra:meaning` | what the signature means: `authored`, `reviewed`, `approved-for-deploy`, `responsible-for-merge` |
| `neyra:weights-root` | weights root of the result, `sha256:` form. Not the hash of any file |
| `neyra:base-weights-root` | the same for the base, also carried on the ancestor |
| `neyra:extent:*` | the counts of `extent`: `tensors-base`, `tensors-result`, `changed`, `added`, `removed`, `entries-changed`, `entries-total` |
| `neyra:coverage:bytes-read`, `neyra:coverage:bytes-total` | how much tensor data the signer read |
| `neyra:attempts` | the signer's count of tries, or `not stated` |
| `neyra:recomputable` (repeated) | a field a reader can recompute from the models |
| `neyra:asserted` (repeated) | a field that rests on the signer's word |
| `neyra:not-measured` (repeated) | `training compute`, `training data`, `behaviour`, and anything the signer adds |
| `neyra:<name>` | each member of the statement's `recipe`, for example `neyra:patch-id` |

The component's `hashes` are on its file components only, since the weights root is not a file hash. The
`externalReferences` entry of type `attestation` points at `change.statement.json` and carries the SHA-256 of the
statement's payload. `pedigree.patches` holds the patch id as a `urn:neyra:patch:` reference with type `unofficial`
when the recipe names one.
