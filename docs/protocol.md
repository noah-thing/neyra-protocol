# The protocol

[`spec/PROTOCOL.md`](../spec/PROTOCOL.md) defines the patch, claim, ledger and receipt formats, with RFC 8785
canonical JSON, domain separated digests, and a weights root that covers every tensor and ignores how
files are sharded. [`spec/vectors`](../spec/vectors) holds 312 conformance cases.

[`impl/go`](../impl/go) is a second verifier in Go. It was written from the spec and vectors alone, with no
access to the Python source, and it passes the first 103. The problems it exposed in the spec were fixed. The row
proof cases of Annex B, the cases from differential fuzzing, the low rank patches of Annex A, the P-256 signatures and
the Annex C statements were added later by the Python author, with the Python source open, so the Go verifier passes
all 312 cases but is independent only for the first 103. It does read the patches `neyra edit` writes.

[`fuzz/differential.py`](../fuzz/differential.py) mutates valid patches, ledgers, receipts, row proofs and models,
runs both verifiers on every mutant, and ranks disagreements with wrongly accepted input first. It is the first
piece of a public arena where independent verifiers are scored on the invalid claims they wrongly accept. Its
first run found 17 points of disagreement, listed in [`CHANGELOG.md`](../CHANGELOG.md); each is now a vector.

Annex B adds a second root over the same tensors, so a few rows of a remote checkpoint can be checked with
the rows and a proof of about a kilobyte. `neyra index` builds it, `neyra prove` and `neyra verify` use it.
A proof binds rows to a rows root. Tying that root to the weights root needs someone who read the whole
model to sign a link, and the annex says why nothing shorter can do it.
