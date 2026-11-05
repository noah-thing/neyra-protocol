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
git clone https://github.com/noah-thing/neyra-protocol && cd neyra-protocol
uv tool install .       # or: pip install .
neyra demo              # a merge that never took its adapter, caught; then a correct one. No download, 0.1 s
neyra diff BASE OTHER --sample 4
```

<!-- On the first PyPI release, replace the three install lines above with: uvx neyra demo / uv tool install neyra
     / pip install neyra, and the extras below keep working as written. RELEASING.md lists every such place. -->

This works on any machine with Python 3.10 or later, with no GPU. `neyra demo files DIR` writes the toy models it
uses (about 210 KB) so you can run `check`, `diff` and `patch` on them by hand, and `neyra conformance` runs the
protocol's 312 test vectors against your install. To run models, install `neyra[mlx]` on Apple Silicon or
`neyra[torch]` elsewhere. `neyra doctor` shows what your machine can do.

To check merges in CI, `uses: noah-thing/neyra-protocol@main` runs `neyra check` and fails the job when the
adapter was never applied; the inputs are in [`action.yml`](action.yml). From the `hf` command line, the
extension in [`extensions/hf-neyra`](extensions/hf-neyra) gives `hf neyra diff` and `hf neyra check`.

## What it does

| Command | Purpose |
|---|---|
| `neyra diff` | What changed between two checkpoints, local or on the Hub, including chat templates, removed parts and new code files |
| `neyra patch`, `apply`, `show` | Store an edit as a small exact file, rebuild it bit for bit, inspect it |
| `neyra merge` | Combine several patches made on one base into one patch, with the same bytes in any order |
| `neyra hop` | Give it the base, the adapter, the merged model and a GGUF in order and get one sentence: the first hop that differs from what the hops before it say it should be, across weights, tokenizer, chat template, special tokens, config and generation config |
| `neyra identity` | Say what a model is relative to the base it declares: the same files, the same weights, a re-runnable transform, a derivative, or unrelated, from a few megabytes of tensors |
| `neyra check` | Decide whether a merged model matches its base and its LoRA adapter, and say if the adapter was never applied |
| `neyra edit` | Write facts into a model as a low rank patch, measure the result, and refuse to call it a success if it hurt neighbours or neutral text |
| `neyra bisect` | Find the smallest part of a patch that keeps its effect on a behavior |
| `neyra export` | Turn a patch into a LoRA for PEFT or stock llama.cpp |
| `neyra probe` | Run a model through a fixed battery and report loops, runaway thinking and missing answers |
| `neyra verify`, `log`, `keygen` | Check signed receipts and tamper evident ledgers |
| `neyra attest`, `render`, `supersede`, `withdraw`, `status` | Sign a statement of what changed between two models, in the formats other tools read, and retire it later without editing it |
| `neyra index`, `prove` | Prove a few rows of a tensor against a rows root, reading only those rows |
| `neyra clock` | Claim that two checkpoints share a scalar on rows training never fed, bound to their rows roots |
| `neyra mcp` | Serve a measured lab to any agent |
| `neyra demo`, `conformance` | Try it on a toy model, and run the protocol's test vectors against this install |

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

## Editing facts

```sh
neyra edit MODEL requests.json -o edit.neyra --key neyra.key
neyra verify edit.receipt.json --ledger edit.ledger.jsonl --patch edit.neyra --key neyra.pub
neyra apply MODEL edit.neyra -o edited-model
```

`requests.json` is a list of `{"subject", "prompt" (with {} for the subject), "target"}`, or CounterFact
records as ROME and MEMIT ship them. Add `true` (the object the model gives today), `paraphrases` and
`neighbours` (prompts about other subjects whose answer is `true`) and the report can say whether the edit
spread and whether it stayed where it belongs.

Each fact becomes a key, the input to the last MLP projection at the subject's final token, and a value,
found by a short descent on the MLP output there. A closed form solve then shares the change over several
layers. The solve is MEMIT's, with the keys already written added to the neutral text covariance it must not
disturb. Over 400 sequential edits that kept 1.7 points of efficacy and 12 of exact match against plain MEMIT
updates, with no better overall score. Which layers to use is measured, by
causal tracing on the model's own facts, not assumed. The weights are solved and measured in the rounded
dtype the patch produces.

The patch is an ordinary low rank patch, so `apply`, `merge`, `bisect` and `export` take it, and applying the
inverse restores the base bit for bit. The receipt is signed and holds the requests, layers, settings and
before and after measures. `neyra edit` exits 1 and prints FAIL when efficacy, specificity, damage on neutral
text or fluency miss their limits, and when specificity was not measured at all. The limits are fixed before a
run and written into the receipt. Editing needs gradients through the model, so it runs on the MLX backend.
Only dense MLP layers are edited. Hybrid stacks work with a memory budget for the gradient; on Qwen3.5-0.8B
edits had to sit in the first five layers.

Measured on Qwen3-0.6B with `--layers 5,6,7,8,9 --lam 100`, one CounterFact edit at a time over 200 cases the
settings never saw: efficacy 1.00, paraphrase 0.95, neighbourhood 0.70 (0.745 before), fluency unchanged, 3.2 seconds
and a 169 KB patch per edit. ROME at its best layer scores higher on the 20 dev cases where the two were compared,
and the test split has not been run for it, so this is not a claim of better edits. A batch of 20 facts at the same
settings failed its own specificity gate (67.0% to 59.5%). [`benchmarks/edit_counterfact.py`](benchmarks/edit_counterfact.py) reruns it. Neutral text damage grows with
the number of edits, 0.04 to 0.1 nats per token per 100, and nothing past 400 sequential edits has been measured.

## Merging patches

```sh
neyra merge BASE a.neyra b.neyra c.neyra -o merged.neyra
neyra apply BASE merged.neyra -o merged-model
```

Merging published models one after another gives a different answer for each order, because every step
rounds to bf16. At one entry with value 1 - 2^-8, three patches that each add 2^-8 give 1.0 when merged
in pairs and 1 + 2^-7 when added together. `merge` adds all the deltas exactly, as integers, and rounds
once to the tensor dtype, so any order writes the same file. [`tests/test_merge.py`](tests/test_merge.py)
checks this against a brute force rounding and against all 24 orders of four patches.

Entries that only one patch touches are copied. Where patches overlap, `merge` reports how many entries
they share and how many of those got different values, and exits with 1. It does not judge whether the
result is a good model. Two patches that make the same change apply it twice, since deltas add.

## Checking a merge

```sh
neyra check BASE MERGED --adapter ADAPTER_DIR
neyra check BASE OTHER
```

Given a base, a LoRA adapter (PEFT, or mlx-lm's own layout) and the merged safetensors, `check` recomputes
every edited tensor the way each real tool merges it and compares bits. The tools round differently, so a
merged file that matches one of them exactly is reported by that name:

| Path | What it does to a bf16 or fp16 weight |
|---|---|
| `peft` | `merge_and_unload()`. The adapter is float32 (PEFT casts 16 bit adapters up on load), `weight += delta` adds in float32 and rounds once. By their source, llama.cpp `export-lora` and unsloth `merged_16bit` do the same; neither was run |
| `peft-safe` | `merge_and_unload(safe_merge=True)`. The delta is rounded to the model dtype first, then added |
| `peft-lowp` | An adapter loaded with `autocast_adapter_dtype=False` on a CPU. A and B are rounded to the model dtype, the product is float32, and the delta is rounded like `peft-safe` |
| `mlx-fuse` | `mlx_lm fuse`. The scale multiplies B before the product, and the product is rounded to the adapter dtype and then to the model dtype |
| `exact` | One rounding of the exact sum. No tool does this, so a match here is reported as equal under rounding |

For a float32 model `peft` and `peft-safe` coincide. A GPU runs the product in the adapter's dtype, and that
is not modelled.

Each tensor comes out as identical (the adapter does not touch it), merged (equals a named path exactly),
equal under rounding (within `--ulps` steps of a path, which absorbs a different order of the float32
product), or different. A tensor the adapter edits but that still equals the base is reported as never
applied, and if that is every edited tensor the report says so first. A different tensor shows how much of
the delta is present, so a doubled or halved scale is visible. A correct merge that kept under half of the
delta to rounding, as happens when a tiny adapter lands in fp16, passes with a warning. A LoRA on a tied `lm_head` is compared on the
embedding it lands on. Leftover `lora_` tensors in the merged file also fail the check. The exit code is 0
when the models are equivalent and 1 otherwise. DoRA and embedding adapters are not supported, and a
quantized base is not modelled.

## Which hop lost it

```sh
neyra hop BASE ADAPTER MERGED model.gguf
```

Artifacts go in the order the fine-tune passed through them. Each one is compared with what the earlier ones say
it should be, and the first sentence names the first that is not, ready to paste into a bug report:

```
The first difference is at the merged model: the adapter was never applied, all 6 tensors it edits equal the base. Intact there: tokenizer, chat template, BOS and special tokens, config and generation config.
```

`--json` gives the same verdict for machines, and the exit code is 1 when a hop has a problem. When nothing differs
the sentence says the hops are clear and that a remaining difference is not in these files: weights and text can
rule a hop out, they cannot find a runtime or sampling bug.

Weights of a merged model are checked with `check`. A GGUF is read with the `gguf` package (`neyra[gguf]`): F32,
F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0 and Q5_1 are rebuilt from the merged weights and compared as bytes. Every other
type, K-quants included, is dequantized and compared within a stated error, and the output says so. The share of the
fine-tune that is still in each hop is the projection of the stored change on the adapter's change, with its error;
`neyra hop BASE ADAPTER --predict q8_0,q4_0` forecasts it before converting. Rounding is unbiased, so this number
stays near 100% for a light adapter even in Q4_0, where the stored change is mostly noise. The cosine shows that.
`python bench/hop_suite.py` scores the command on planted failures, and `--today` runs `check` and `diff` on the same files.

### Naming the real base of a merge

```sh
neyra check BASE MERGED --adapter ADAPTER_DIR --also MIRROR_OR_OTHER_BASE ...
```

With `--also`, `check` reads a sample of the edited tensors against every candidate base and reports which one
plus the adapter gives the merged tensors back. A card, an adapter config and a mirror can each name a different
base. Candidates that agree on every sampled span are listed as sharing weights, because no merge can tell them
apart. Without an adapter two checkpoints are compared with no tolerance, and with one an entry may be a step
off only if at most 2% of the tensor is.

## Identity

```sh
neyra identity BASE MODEL [--adapter ADAPTER_DIR] [--full]
```

One verdict, from a sample of 24 matrices read over HTTP ranges (a megabyte or two), or from every byte with `--full`:

| Verdict | Meaning |
|---|---|
| bit identical | the safetensors files have the same SHA-256 |
| same weights in a different file | every tensor hashes the same wherever it is stored, including a fused projection against its split parts and a head against the embedding it is tied to. Only `--full` says this |
| no difference found in a sample | the sampled spans are equal. Not proof: an edit outside them is missed |
| derived by a named transform | a cast to another dtype, or a LoRA merge reproduced bit for bit by a named path |
| derived with unknown transform | far more alike than independently trained models are |
| unrelated | no more alike than independently trained siblings |
| inconclusive, not comparable | between the two ranges, or the files lay their tensors out differently |

"Derived" is judged against a null measured on unrelated siblings, never against a number chosen for the occasion;
the panel, thresholds and held-out error rates are in `src/neyra/identity_null.json`, rebuilt by
`benchmarks/identity_panel.py`. `tools/weights_identity_check.py` is a numpy-only version for a benchmark judge: it
takes a candidate, the allowed base and the forbidden instruct model.

## What bisect measures

"Blame" names four different measurements, and they can disagree:

| Name | Question |
|---|---|
| movement | how much did the behavior change at this step of the history |
| ablation | how much is lost removing this part from the finished model |
| insertion | how much comes back adding this part alone to the base |
| replay | what happens if the history is run again without it |

`neyra bisect` reports insertion. Every trial applies a subset of the patch to the base model and measures
the behavior, and the answer is the smallest subset that keeps the chosen share of the full effect. It
does not say that removing those units from the full patch would remove the behavior, and it says nothing
about training history. Results carry `"blame": "insertion"` in the command's JSON, the lab result, the
ledger entry and the metadata of the patch it writes. Patch identity and the protocol vectors do not
depend on any of these.

## Signing what changed

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
[`experiments/attestation_demo.py`](experiments/attestation_demo.py) runs the whole thing on a real edit and checks it
with model-signing, in-toto-attestation, securesystemslib and the CycloneDX schema. A signature does not show when,
who, or that the signer's figures are true; [section 1 of the spec](spec/PROTOCOL.md) lists what it leaves open.

## The protocol

[`spec/PROTOCOL.md`](spec/PROTOCOL.md) defines the patch, claim, ledger and receipt formats, with RFC 8785
canonical JSON, domain separated digests, and a weights root that covers every tensor and ignores how
files are sharded. [`spec/vectors`](spec/vectors) holds 312 conformance cases.

[`impl/go`](impl/go) is a second verifier in Go. It was written from the spec and vectors alone, with no
access to the Python source, and it passes the first 103. The problems it exposed in the spec were fixed. The row
proof cases of Annex B, the cases from differential fuzzing, the low rank patches of Annex A, the P-256 signatures and
the Annex C statements were added later by the Python author, with the Python source open, so the Go verifier passes
all 312 cases but is independent only for the first 103. It does read the patches `neyra edit` writes.

[`fuzz/differential.py`](fuzz/differential.py) mutates valid patches, ledgers, receipts, row proofs and models,
runs both verifiers on every mutant, and ranks disagreements with wrongly accepted input first. It is the first
piece of a public arena where independent verifiers are scored on the invalid claims they wrongly accept. Its
first run found 17 points of disagreement, listed in [`CHANGELOG.md`](CHANGELOG.md); each is now a vector.

Annex B adds a second root over the same tensors, so a few rows of a remote checkpoint can be checked with
the rows and a proof of about a kilobyte. `neyra index` builds it, `neyra prove` and `neyra verify` use it.
A proof binds rows to a rows root. Tying that root to the weights root needs someone who read the whole
model to sign a link, and the annex says why nothing shorter can do it.

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
- Everything here has been tested on small Qwen3.5 and Qwen3 models. Larger models are untested.
- `neyra edit` needs MLX, edits dense MLP layers only, and its AlphaEdit style `--solver null` lost to the default in every run.
- `neyra merge` of two edit patches is exact but returns a dense patch, since Annex A has no way to merge low rank factors.
- The lab keeps honest agents honest. An agent with shell access to the same machine could read the
  holdout from disk, so adversarial use needs the server on another host.

## Status

Early. The formats may change until a v1 freeze. Results on edit search and loop repair are in progress
and are not claimed here.

## Prior work

Neyra builds on Heretic, gguf-neuron-explorer, Abliterlitics, the diffing toolkit, MEMIT and AlphaEdit,
Steer2Edit, CHiPS, and the Weight-Space Programming logs, which made an exact check possible at all. The
direction and column edits here are established methods, not new ones.

`neyra edit` takes its update from ROME and MEMIT and its protection of earlier edits from AlphaEdit and
RLSEdit. For merging, mergekit has far more methods; `neyra merge` is a unit weight delta sum whose point is that
it is exact and independent of order. `check` models the merge paths of PEFT and mlx-lm. Git-Theta versions model
parameters in git and Xet deduplicates them on the Hub. OpenSSF Model Signing signs model files with Sigstore, and
Neyra's receipts use the same DSSE and in-toto shapes. modelDNA answers a different question, which model a
checkpoint descends from, and it has a reference database that Neyra does not.

## License

Apache 2.0.
