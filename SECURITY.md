# Security

## Reporting

Report a vulnerability through GitHub's private vulnerability reporting on this repository (the Security tab,
"Report a vulnerability"). Please do not open a public issue for one. Expect an answer within a week.

## What counts

Neyra reads files from strangers: safetensors checkpoints, patches, ledgers, receipts and row proofs. These
are in scope:

- A verifier that accepts input the spec says to reject: a patch, ledger, receipt, binding or row proof that
  `neyra verify`, `neyra conformance` or the Go verifier in `impl/go` calls valid when it is not. This is the
  most serious class. Send the file.
- A crash, hang or unbounded memory use on a crafted file in any command that reads one.
- Code execution from opening a patch or checkpoint. Neither format can hold code, and `neyra` never loads
  pickles.
- A way for `neyra mcp` to let an agent read the sealed holdout or write outside the directories it was given
  (see the limits below).

## What does not

- A signature on a receipt shows who made a claim, not that it is true. A receipt for a false claim from a
  key you do not trust is working as designed.
- An agent with shell access to the machine that runs `neyra mcp` can read the holdout from disk. The README
  says to run the server on another host for adversarial use.
- Results of `neyra edit`, `bisect` and the other model commands being wrong or poor. File those as bugs.

## Supported versions

The latest release. The formats may change until a v1 freeze, and the changelog says when they do.

## Hardening already in place

`fuzz/differential.py` mutates every protocol object and runs the Python and Go verifiers against each other;
it is how most of the 0.2.0 verifier bugs were found. Its output for a new bug is a ready-made test vector.
