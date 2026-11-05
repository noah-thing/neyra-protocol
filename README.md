<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="Neyra" width="260">
  </picture>
</p>

<p align="center"><strong>Edit what a local model does, directly in its weights.</strong><br>Change it, measure it, undo it, and prove what changed.</p>

<p align="center">
  <a href="https://github.com/noah-thing/neyra-protocol/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/noah-thing/neyra-protocol/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache_2.0-blue"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <a href="spec/PROTOCOL.md"><img alt="312 conformance vectors" src="https://img.shields.io/badge/conformance-312_vectors-brightgreen"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#evidence">Evidence</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="spec/PROTOCOL.md">Specification</a>
</p>

---

Neyra is a workbench for the inside of a model. Instead of prompting a local model or retraining it, you find the
directions and neurons behind a behaviour and change them: rewrite a fact, remove a reasoning loop, write a change
into the layers that carry it. Agents can do this too, under a fixed budget, against a test they cannot see.

Editing weights is only useful if you can trust the result, so every change is a small exact patch you can undo bit
for bit. Each one is measured against what it might have damaged, and refused if it did harm. Neyra also reads
changes made by anyone else, so you can see what a fine-tune, a merge or a conversion really did to the tensors.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/terminal-dark.svg">
  <img src="docs/assets/terminal-light.svg" alt="neyra diff recovering a published weight edit" width="100%">
</picture>

That is a weight edit an agent made to stop a model looping, from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/),
recovered with no hints by reading 4% of the two checkpoints over HTTP range requests. The authors' notes give the
same formula with a strength of 6.

## Quick start

```sh
git clone https://github.com/noah-thing/neyra-protocol && cd neyra-protocol
uv tool install .        # or: pip install .
neyra demo               # about 0.1 s, no GPU, nothing downloaded
```

`neyra demo` builds a toy model and a LoRA in memory, shows a merge that never took its adapter being caught, then a
correct merge passing. Python 3.10 or later is all it needs. To run models, install `neyra[mlx]` on Apple Silicon or
`neyra[torch]` elsewhere. `neyra doctor` shows what your machine can do.

Then point it at real files, local or on the Hugging Face Hub:

```sh
neyra probe MODEL                             # what goes wrong: loops, runaway thinking, missing answers
neyra fix MODEL -o fix.neyra                  # find a loop direction and write a candidate patch
neyra edit MODEL facts.json -o edit.neyra     # write facts into the model, measured and revertible
neyra apply MODEL edit.neyra -o edited-model  # rebuild the edited model, bit for bit

neyra diff BASE OTHER --sample 4              # what any change did, from a few percent of the bytes
neyra hop BASE ADAPTER MERGED model.gguf      # which hop lost my fine-tune
```

## Commands

**Edit**

| Command | What it does |
|---|---|
| [`edit`](docs/edit.md) | Write facts into a model as a low rank patch, and refuse to call it a success if it did harm |
| `fix` | Find the direction behind a loop and write a candidate patch that removes it |
| `probe` | Run a model through a fixed battery and report loops, runaway thinking and missing answers |
| `patch`, `apply`, `show` | Store any edit as a small exact file and rebuild it bit for bit |
| [`merge`](docs/merge.md) | Combine patches exactly, with the same bytes in any order |
| [`bisect`](docs/bisect.md) | Find the smallest part of an edit that keeps its effect |
| `export` | Turn a patch into a LoRA for PEFT or stock llama.cpp |

**Inspect**

| Command | What it does |
|---|---|
| [`diff`](docs/accuracy.md) | What changed between two checkpoints, and the form of the change |
| [`check`](docs/check.md) | Whether a merge matches its parts, recomputed the way PEFT and mlx-lm do it |
| [`hop`](docs/hop.md) | The first hop that differs across base, adapter, merged model and GGUF, in one sentence |
| [`identity`](docs/identity.md) | Same weights, a named transform, a derivative, or unrelated |

**Prove**

| Command | What it does |
|---|---|
| [`attest`](docs/signing.md), `render`, `supersede`, `withdraw`, `status` | Sign what changed, in formats OpenSSF model signing and CycloneDX tools read |
| `verify`, `log`, `keygen` | Check signed receipts and tamper evident ledgers |
| [`index`](docs/protocol.md), `prove` | Prove a few rows of a tensor against a root, reading only those rows |
| `conformance` | Run the protocol's test vectors against your install |

**Automate**

| Command | What it does |
|---|---|
| [`mcp`](docs/lab.md) | A measured lab for agents: fixed budget, sealed holdout, hash chained ledger |
| [`action.yml`](action.yml) | Fail a CI job when a merge is not what its adapter says |
| [`hf-neyra`](extensions/hf-neyra) | `hf neyra diff` and `hf neyra check` from the Hugging Face command line |

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hops-dark.svg">
  <img src="docs/assets/hops-light.svg" alt="A fine-tune passing through merge and conversion, with the first differing hop marked" width="100%">
</picture>

Every claim Neyra makes is one of two kinds, and it keeps them apart.

**Exact claims** can be recomputed by anyone. A patch is a safetensors file, so opening one cannot execute code. It
names the bits it starts from and the bits it ends on, and applying its inverse restores the base bit for bit. A
weights root covers every tensor and ignores how files are sharded, so a resharded model keeps its identity. Merges
add deltas as exact integers and round once, because bf16 addition does not associate: at one entry with value
1 - 2^-8, three patches that each add 2^-8 give 1.0 in pairs and 1 + 2^-7 together.

**Statistical claims** say how they were measured. Sampled reads say they are sampled and never call themselves
proof. A "derived" verdict is judged against a null measured on unrelated sibling models. An edit is held to limits
fixed before the run, and the signed receipt records a failure just as it records a success. A signature shows who
made a claim, not that it is true, so receipts name everything needed to rerun it.

## Evidence

| Claim | Result | Reproduce |
|---|---|---|
| `check` recognises real merges bit for bit | Merges from PEFT 0.21.2 and mlx-lm 0.32.2 in fp32, bf16 and fp16 | [`tests/test_check_real.py`](tests/test_check_real.py) |
| `diff` recovers planted edits | 12 of 14 conditions at 100% form recovery, no false alarms in 250 noise trials | [`bench/forensics.py`](bench/forensics.py) |
| `edit` writes facts | Qwen3-0.6B, 200 unseen CounterFact cases: efficacy 1.00, paraphrase 0.95, neighbours 0.70 (0.745 before) | [`bench/edit_counterfact.py`](bench/edit_counterfact.py) |
| `merge` is independent of order | Checked against brute force rounding and all 24 orders of four patches | [`tests/test_merge.py`](tests/test_merge.py) |
| The verifiers agree | Python and a second verifier in Go on 312 vectors and every mutant of a differential fuzzer | [`fuzz/differential.py`](fuzz/differential.py) |

Each of these has limits, stated with it. Weight edits can rewrite a fact or remove an obstacle such as a loop; they
have not been shown to add ability. `edit` is not claimed to beat ROME, which led on the 20 development cases compared
so far. Everything has been tested on models under 1B parameters. [What Neyra does not do](docs/limits.md)
lists the rest and is worth reading before you rely on any of this.

## Documentation

- **Guides:** [checking a merge](docs/check.md) · [which hop lost it](docs/hop.md) · [identity](docs/identity.md) · [editing facts](docs/edit.md) · [merging patches](docs/merge.md) · [what bisect measures](docs/bisect.md) · [the lab](docs/lab.md)
- **Protocol:** [specification](spec/PROTOCOL.md) · [overview](docs/protocol.md) · [signing what changed](docs/signing.md) · [conformance vectors](spec/vectors)
- **Reference:** [measured accuracy](docs/accuracy.md) · [limits](docs/limits.md) · [prior work](docs/prior-work.md) · [changelog](CHANGELOG.md)

## Status

Neyra is early. Formats may change until a v1 freeze, and it is not yet on PyPI.

## Contributing

Issues and pull requests are welcome, and [CONTRIBUTING.md](CONTRIBUTING.md) covers the workflow. A second verifier
written by someone outside this project would be the most valuable contribution of all: the
[specification](spec/PROTOCOL.md) and [vectors](spec/vectors) are written for that. Please report vulnerabilities as
described in [SECURITY.md](SECURITY.md).

## License

[Apache 2.0](LICENSE).
