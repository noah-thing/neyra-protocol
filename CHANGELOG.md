# Changelog

## 0.2.0, first release

Not yet published. 0.1.x was the private working version.

### Install and first run
- `neyra demo` shows a merge that never took its LoRA adapter, caught by `check`, then the same merge done
  the way PEFT does it, passing. It builds the toy model in memory, so it needs no download and runs in 0.1 s.
  The earlier edit recovery demo is `neyra demo edit`; `neyra demo files DIR` writes the toy models out.
- `neyra conformance` runs the protocol vectors against the installed copy, which ships in the wheel.
- The sdist no longer includes `.hypothesis/`. The package version lives in `src/neyra/__init__.py` only.
- `action.yml`: a composite GitHub Action that runs `neyra check` and fails the job when a merge is wrong.
- `extensions/hf-neyra`: an `hf` CLI extension, `hf neyra ...`.

### Errors
- Wrong input ends in a one line message and an exit code. Fixed tracebacks for: a directory given as a file,
  a binary file given as JSON, row ranges that are not numbers, an edit file with no requests or the wrong
  shape, a public key file that is not a key, a claim file that is not a claim, an unwritable output path,
  an unreachable Hub, and a Hub repo that does not exist when `huggingface_hub` is installed.

### Protocol fixes found by `fuzz/differential.py`
The Python and Go verifiers disagreed on these. The spec text decided each one, and each is now a vector
(138 became 195). Where one verifier accepted input the other rejected, the accepting one was wrong.

Python accepted what it should have rejected:
1. A ledger entry whose `kind` is not a string or whose `body` is not an object.
2. A ledger entry whose `seq` is `true` (Python's `True == 1`).
3. A ledger `time` written in non-ASCII digits (`\d` matches every script).
4. A receipt signature or payload whose base64 has non-zero unused bits in the last character.
5. A patch manifest whose `shape` holds lists or booleans, or whose `name` or `dtype` is not a string.
6. A safetensors header with a repeated key (the last one won, so two readers could see different tensors).
7. A safetensors `__metadata__` that is `null`, `[]`, `0` or `""` (coerced to empty).
8. A binding whose `ledger_len` is `true` for a ledger of one entry.
9. A receipt whose claim digests are numbers or have no `sha256:` prefix, matched by string coercion.

Python rejected, crashed on, or computed differently from the spec:
10. Dot files counted as weight files in a folder, which changed the weights root.
11. A directory named `x.safetensors` crashed it. `consolidated*.safetensors` was silently left out of a
    folder's root; that Hub convention now applies only to Hub repos and CLI reads.
12. A later valid signature was ignored when an earlier one with the same key id failed.
13. `neyra.meta` that is not JSON crashed loading a patch.
14. A manifest that parses to a non-object gave `bad_json` and skipped the canonical check; the spec says
    `not_canonical`, then `bad_type`.
15. A receipt whose `signatures` is missing or not a list gave the wrong code; the spec says `bad_json`.
16. Receipts and ledgers with invalid UTF-8, a non-object envelope, or malformed signature entries crashed with
    an exception instead of returning a code.

Go accepted what it should have rejected:
17. A signature entry that is not an object, or lacks a string `sig`, with no key supplied.

Not a disagreement: patches that require `low-rank/1` are declined by Go, which the spec allows.

After the fixes, 119,458 mutants across nine kinds produced no disagreement.

### Spec
`spec/PROTOCOL.md` now says, in sections 3, 4, 5, 6 and 8: dot files and directories are skipped and
`consolidated.safetensors` is an ordinary shard; repeated header keys and non-object `__metadata__` are
`bad_safetensors`; `neyra.meta` is never parsed; a non-object manifest is `bad_type`; `shape`, `name` and
`dtype` types; ASCII digits and types in a ledger entry; envelope shape and canonical base64; every
signature with the key is tried; claim digests must be well formed; `ledger_len` is an integer.
