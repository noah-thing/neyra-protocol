# Changelog

## Unreleased

- `neyra tune lint DATA MODEL` builds the token ids and loss mask a fine-tuning trainer would use (TRL, mlx-lm, Unsloth, Axolotl, LLaMA-Factory) and reports BOS and end token problems, prompt text in the loss, truncation, template mismatches, duplicates and leakage. Needs `neyra[tune]`.
- `neyra hop` checks Q4_K, Q5_K and Q6_K tensors against the interval each stored code allows instead of an error ceiling. A value outside it refutes the candidate; `docs/source.md` has the limits. Exact decoding of these types is checked bit for bit against llama.cpp by `bench/cells_vs_llamacpp.py`.
- Removed `neyra.kquant`, a numpy Q4_K quantizer that did not reproduce llama.cpp's bytes.

## 0.2.2

- `neyra edit` runs on torch, so Linux, Windows and CPU work as well as Apple Silicon. Benchmark numbers still come from MLX.
- `neyra edit` scores live decoding as well as logits.
- Tests run on Windows.
- `neyra import` reads Delta League, OpenWSP, weightspace, PEFT and ballast edits into Neyra patches, and `export` writes them back.
- `neyra merge --method` adds task arithmetic, TIES, DARE, SLERP, Model Stock and a per-tensor length match, with exact sums and a stored recipe. `neyra check --parents` recomputes one.

## 0.2.1

- `neyra edit MODEL "The capital of [France] is" Rome` takes a single-fact on the command line.
- First release on PyPI.

## 0.2.0, first release

0.1.x was the private working version.

### Install and first run
- `neyra demo` shows a merge that never took its LoRA adapter, caught by `check`, then the same merge done
  the way PEFT does it, passing. It builds the toy model in memory, so it needs no download and runs in 0.1 s.
  The earlier edit recovery demo is `neyra demo edit`; `neyra demo files DIR` writes the toy models out.
- `neyra conformance` runs the protocol vectors against the installed copy, which ships in the wheel.
- The sdist no longer includes `.hypothesis/`. The package version lives in `src/neyra/__init__.py` only.
- `action.yml`: a composite GitHub Action that runs `neyra check` and fails the job when a merge is wrong.
- `extensions/hf-neyra`: an `hf` CLI extension, `hf neyra ...`.

### Errors
- Wrong input ends in a one-line message and an exit code. Fixed tracebacks for: a directory given as a file,
  a binary file given as JSON, row ranges that are not numbers, an edit file with no requests or the wrong
  shape, a public key file that is not a key, a claim file that is not a claim, an unwritable output path,
  an unreachable Hub, and a Hub repo that does not exist when `huggingface_hub` is installed.

### Protocol fixes

`fuzz/differential.py` found 17 disagreements between the Python and Go verifiers, mostly malformed input that one of them accepted. All are fixed and kept as conformance vectors (138 became 195).

### Spec
`spec/PROTOCOL.md` now says, in sections 3, 4, 5, 6 and 8: dot files and directories are skipped and
`consolidated.safetensors` is an ordinary shard; repeated header keys and non-object `__metadata__` are
`bad_safetensors`; `neyra.meta` is never parsed; a non-object manifest is `bad_type`; `shape`, `name` and
`dtype` types; ASCII digits and types in a ledger entry; envelope shape and canonical base64; every
signature with the key is tried; claim digests must be well formed; `ledger_len` is an integer.
