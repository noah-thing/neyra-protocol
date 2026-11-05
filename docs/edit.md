# Editing facts

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
found by a short descent on the MLP output there. A closed-form solve then shares the change over several
layers. The solve is MEMIT's, with the keys already written added to the neutral text covariance it must not
disturb. Over 400 sequential edits that kept 1.7 points of efficacy and 12 of exact match against plain MEMIT
updates, with no better overall score. Which layers to use is measured, by
causal tracing on the model's own facts. The weights are solved and measured in the rounded
dtype the patch produces.

The patch is an ordinary low-rank patch, so `apply`, `merge`, `bisect` and `export` take it, and applying the
inverse restores the base bit for bit. The receipt is signed and holds the requests, layers, settings and
before and after measures. `neyra edit` exits 1 and prints FAIL when efficacy, specificity, damage on neutral
text or fluency miss their limits, and when specificity was not measured at all. The limits are fixed before a
run and written into the receipt. Only dense MLP layers are edited. Hybrid stacks work with a memory budget for the gradient; on Qwen3.5-0.8B
edits had to sit in the first five layers.

## Backends

`neyra edit` runs on MLX and on torch. `--backend auto` takes MLX on Apple Silicon when it is installed and torch
everywhere else; `--backend torch` forces torch. On torch the gradient is taken with autograd through a vector added to
one layer's MLP output, and the layers after it are recomputed on the backward pass to keep memory down. Torch picks
CUDA, then MPS, then the CPU; `NEYRA_DEVICE=cpu` overrides. On the CPU the stored bf16 weights are widened to float32
inside each matrix product, because bf16 products run about six times slower there; set `NEYRA_WIDE=0` to compute in
bf16 as an accelerator does.

What has run where. The engine's tests run on tiny trained models on both backends, and on Linux CPU torch in CI. On
identical weights the two backends agree: in float32 the largest difference in activations, logits, loss and gradient is
about 1e-6 of the largest value, in bf16 under 1e-2. On a real model, only one thing has run on torch: the one line form
`neyra edit Qwen/Qwen3-0.6B "The capital of [France] is" Rome` on an Apple CPU. It took 368 seconds and 4.5 GB, the
efficacy check went from 0% to 100% and greedy decoding from 0% to 100%, neutral text KL was 0.0006 nats per token, and the
loaded weights were bit for bit what they had been once the edit was undone. It exits 1 because one fact has no
neighbours to check, which is the gate working. The CounterFact figures below are MLX figures. CUDA, MPS, hybrid stacks
such as Qwen3.5 and any batch on a real model are untested on torch.

Measured on Qwen3-0.6B with `--layers 5,6,7,8,9 --lam 100`, one CounterFact edit at a time over 200 cases the
settings never saw: efficacy 1.00, paraphrase 0.95, neighbourhood 0.70 (0.745 before), fluency unchanged, 3.2 seconds
and a 169 KB patch per edit. ROME at its best layer scores higher on the 20 dev cases where the two were compared,
and the test split has not been run for it, so this is not a claim of better edits. A batch of 20 facts at the same
settings failed its own specificity gate (67.0% to 59.5%). [`bench/edit_counterfact.py`](../bench/edit_counterfact.py) reruns it. Neutral text damage grows with
the number of edits, 0.04 to 0.1 nats per token per 100, and nothing past 400 sequential edits has been measured.
