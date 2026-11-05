# Signing what changed

```sh
neyra keygen -o signer.key --algorithm ecdsa-p256      # the key type OpenSSF model signing reads
neyra attest BASE RESULT --patch edit.neyra --receipt edit.receipt.json --oms result.sig --key signer.key -o change
neyra verify change/change.statement.json --key signer.pub --base BASE --result RESULT --patch edit.neyra
```

`attest` writes three files from one signed source. `change.statement.json` is an in-toto statement in a DSSE
envelope with a new predicate, `https://neyra.run/transformation/v1`: base and result weights roots, the kind of
change (patch, merge, adapter, quantize, edit or unknown), the recipe, how a reader repeats the check, what the
signature means (authored, reviewed, approved for deploy, responsible for merge), and which fields a reader can
recompute and which are only the signer's word. `change.cdx.json` is a CycloneDX 1.7 file with the result as a
`machine-learning-model` and the base as its ancestor. `CHANGE.md` is one page for a person, and says first what it
does not measure: training compute, training data and behaviour.

The statement also names the result's files by sha256, and the digest OMS gives the model, as subjects. Anyone who
holds a verified OMS signature can join the two by comparing strings. A resharded model gets new OMS digests and
keeps its weights root, so the statement is tied to the weights and the file subjects are one serialisation of them.
A later `supersede` or `withdraw` record points at the statement by digest and leaves it as it was.
[`experiments/attestation_demo.py`](../experiments/attestation_demo.py) runs the whole thing on a real edit and checks it
with model-signing, in-toto-attestation, securesystemslib and the CycloneDX schema. A signature does not show when,
who, or that the signer's figures are true; [section 1 of the spec](../spec/PROTOCOL.md) lists what it leaves open.
