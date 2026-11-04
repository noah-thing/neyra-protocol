# Neyra Protocol

Version control for what a neural network does.

Neyra compares model checkpoints, works out what was changed inside them, and records edits as small
files anyone can verify without running the model. Agents can work through it over MCP, under a fixed
budget, against a test they cannot see.

```console
$ neyra diff Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming --sample 4

488 tensors paired, 2 changed, no difference found in the other 486 (sampled spans only, which is not proof they match)
130,188 entries differ, 1.774% of the changed tensors
read 144.2 MB of 3,493.9 MB in 172s

model.language_model.layers.15.mlp.down_proj.weight
    W' = W - 6.000 u(uᵀW) on 64 of 3584 columns
model.language_model.layers.19.mlp.down_proj.weight
    W' = W - 6.000 u(uᵀW) on 1023 of 1024 rows, 64 of 3584 columns
```

That is the published edit from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/),
found with no hints by reading 4% of the two checkpoints over HTTP range requests. The authors' run
notes give the same formula with a strength of 6.

## Try it

```sh
uvx neyra demo          # plants an edit in a tiny model and recovers it, no download, under a second
uvx neyra diff BASE OTHER --sample 4
```

Both work on any machine with Python 3.10 or later, with no GPU. To run models, install
`neyra[mlx]` on Apple Silicon or `neyra[torch]` elsewhere. `neyra doctor` shows what your machine can do.

## What it does

| Command | Purpose |
|---|---|
| `neyra diff` | What changed between two checkpoints, local or on the Hub, including chat templates, removed parts and new code files |
| `neyra patch`, `apply`, `show` | Store an edit as a small exact file, rebuild it bit for bit, inspect it |
| `neyra export` | Turn a patch into a LoRA for PEFT or stock llama.cpp |
| `neyra probe` | Run a model through a fixed battery and report loops, runaway thinking and missing answers |
| `neyra verify`, `log`, `keygen` | Check signed receipts and tamper evident ledgers |
| `neyra mcp` | Serve a measured lab to any agent |

A patch is a safetensors file, so opening one cannot execute code. The published 0.8B edit is a 1.3 MB
patch that rebuilds exactly against the published hashes, and a 37 KB rank 1 LoRA that stock
`llama-server` loads.

## The lab

`neyra mcp` gives an agent tools to sample, find directions, propose edits, run controls and search
for the part of an edit that matters. It treats the agent as an untrusted party:

- Measurements come from a fixed budget.
- The holdout stays in the server. Its salted hash is committed when the session opens.
- Every action lands in a hash chained ledger.
- The verdict comes from fixed gates: a real effect, beating a size matched random edit, low drift on
  neutral text, and beating free alternatives such as a system prompt or turning thinking off.

A receipt is an in-toto statement in a DSSE envelope, signed with Ed25519. A signature shows who made
a claim. It does not show the claim is true, so receipts name everything needed to rerun it.

## The protocol

[`spec/PROTOCOL.md`](spec/PROTOCOL.md) defines the patch, claim, ledger and receipt formats, with RFC 8785
canonical JSON, domain separated digests, and a weights root that covers every tensor and ignores how
files are sharded. [`spec/vectors`](spec/vectors) holds 103 conformance cases.

[`impl/go`](impl/go) is a second verifier in Go. It was written from the spec and vectors alone, with no
access to the Python source, and it passes all 103. The problems it exposed in the spec were fixed.

## Measured accuracy of `diff`

We planted known edits in real Qwen3.5-0.8B tensors and scored recovery. 50 trials per condition.
[`benchmarks/forensics.py`](benchmarks/forensics.py) reproduces this table.

| Planted edit | Form correct | Direction cosine | Strength error | Column F1 | Rank exact |
|---|---|---|---|---|---|
| writer projection k=64 a=6 | 100% | 1.000 | 0.0000 | 1.000 |  |
| writer projection k=64 a=1 | 100% | 1.000 | 0.0000 | 1.000 |  |
| writer projection k=8 a=2 | 100% | 1.000 | 0.0001 | 1.000 |  |
| writer projection k=512 a=1 | 100% | 1.000 | 0.0000 | 1.000 |  |
| random projection k=64 a=6 | 100% | 1.000 | 0.0002 | 1.000 |  |
| random projection k=64 a=1 | 100% | 1.000 | 0.0009 | 1.000 |  |
| random projection k=512 a=1 | 100% | 1.000 | 0.0002 | 1.000 |  |
| full projection a=1 | 80% | 1.000 | 0.0001 | 1.000 |  |
| input projection a=1 | 68% |  | 0.0003 |  |  |
| additive rank 1 | 100% |  |  |  | 100% |
| additive rank 2 | 100% |  |  |  | 100% |
| additive rank 8 | 100% |  |  |  | 100% |
| zeroed 256 columns | 100% |  |  | 1.000 |  |
| scaled 0.9 | 100% |  | 0.0000 |  |  |

Rounding noise produced no false alarms in 250 trials. The weakest rows are dense edits
along random directions, which sit close to bf16 rounding and are easy to miss.

## What it does not do

- It does not make a model smarter. Weight edits can remove an obstacle, such as a loop. They do not
  add ability, and the Weight-Space Programming authors found the same on a model that did not loop.
- A sampled diff cannot prove two tensors match. An edit confined to a few entries can slip through.
  Without `--sample`, every byte is compared.
- `diff` needs the base model. Finding an edit with no base is a much harder problem.
- Structure recovery describes weights, not intent, and does not detect every backdoor.
- Edits to expert tensors cannot yet be exported to GGUF, and quantized weights are not analysed.
- Models with several residual streams, such as DeepSeek V4, or a norm after each write, such as
  Gemma 4, are recognised, but direction edits on them are untested.
- Everything here has been tested on small Qwen3.5 models. Larger models are untested.
- The lab keeps honest agents honest. An agent with shell access to the same machine could read the
  holdout from disk, so adversarial use needs the server on another host.

## Status

Early. The formats may change until a v1 freeze. Results on edit search and loop repair are in progress
and are not claimed here.

## Prior work

Neyra builds on Heretic, gguf-neuron-explorer, Abliterlitics, the diffing toolkit, MEMIT and AlphaEdit,
Steer2Edit, CHiPS, and the Weight-Space Programming logs, which made an exact check possible at all. The
direction and column edits here are established methods, not new ones.

## License

Apache 2.0.
