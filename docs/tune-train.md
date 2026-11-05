# tune train

```sh
neyra tune train MODEL data.jsonl -o run
neyra tune train Qwen/Qwen3-0.6B data.jsonl -o run --iters 120 --rank 16 --val val.jsonl --mask-prompt
```

One command on an Apple silicon Mac: lint the data, tune with mlx-lm's LoRA trainer, and keep the change as
an exact Neyra patch. Install with `uv tool install 'neyra[tune,mlx]'`, on Python 3.11 or later. Neyra has no training loop of its own; mlx-lm
trains and fuses, and Neyra checks what came out.

## What it does

1. `tune lint` for the mlx trainer. An error stops the run with the lint report and exit code 1. `--force` trains anyway.
2. `mlx_lm` LoRA training, in process, so the memory limit applies and the validation losses can be read. mlx-lm's own
   progress display prints each step.
3. A validation loss before the first step and after the last one.
4. The patch is written from the adapter's factors, the way `neyra edit` writes its patch: per trained tensor, `left` is
   B times the scale and `right` is A, both float64, with no stored entries. The patch's own low rank rule defines
   the tuned weights (base plus the exact delta, rounded once to the model dtype). It is a few megabytes where a
   diff against a fused model is larger than the model, because the delta of a small adapter is below bf16 rounding
   in most entries and every such entry would be stored.
5. `neyra check` logic on what the patch builds, against base plus adapter, tensor by tensor. An adapter that changed
   no weight, or a tensor that is not base plus the adapter within one rounding step, ends the run with nothing kept.
   The inverse of the patch is applied and compared with the base.
6. A short summary.

```
run/
  adapter/            adapters.safetensors and adapter_config.json, as mlx_lm loads them
  tune.neyra          the patch: the adapter's low-rank factors, bound to the base
  run.json            settings, losses, time, memory, patch id, and a manifest of what went in
```

The manifest holds the command, the digests of your data files, the model's revision, config, tokenizer and chat template
digests, the seed, and the versions of Neyra, Python, mlx-lm and the rest. The patch already binds the base weights by
digest.

Everything is built in a hidden folder next to `run` and moved into place at the end. Ctrl-C, an error or a failed
check removes it, so `run` either holds a finished result or does not exist. An existing filled `run` is refused.
No merged model is written: `neyra apply` rebuilds the tuned model from the base and the patch.

## Defaults

From mlx-lm's LoRA guide, with two changes for short runs on small files.

| Setting | Value | Why |
|---|---|---|
| `--rank` | 8, scale 20, dropout 0 | the guide's defaults |
| layers | last 16 (all if fewer) | the guide's default |
| `--lr` | 2e-5 | the guide's 1e-5 moves a run of under 100 steps very little |
| `--iters` | 3 passes over the training rows, from 60 to 1000 | enough to learn a style, short enough to stop before memorising |
| `--batch-size` | 4 | |
| `--max-length` | 512 | longer rows are cut, as lint reports |
| validation | a tenth of the rows, at least one batch, or `--val FILE` | needed for the loss before and after |
| memory | MLX limited to 60% of RAM (`--memory-gb`); gradient checkpointing when the weights take over a quarter of that | a 16 GB Mac stays usable |

Refusals: a model whose weights exceed 80% of the memory limit, an adapter that changed no weight, a quantized model (merging into 4 bit weights rounds
again, so the patch would not be exact), fewer than 12 rows without `--val`, and any machine that is not Apple silicon.

## How this differs from `mlx_lm fuse`

`mlx_lm fuse` scales B, multiplies in the adapter dtype and rounds the product before adding it. The patch adds the
exact product to the weight and rounds once. The two models differ by rounding, and the run prints how much: on
Qwen3-0.6B (bf16 adapter) 11% of the entries differ, by at most a small share of the largest change in their tensor.
`neyra check BASE MODEL --adapter run/adapter` on the applied model reports "rounding: within 1 step of exact", not
`mlx-fuse`. To get the fused model instead, run `mlx_lm fuse`; `neyra check` then matches `mlx-fuse`.

## After the run

```sh
neyra apply MODEL run/tune.neyra -o run/model
mlx_lm generate --model run/model --prompt "..."
neyra show run/tune.neyra
neyra export MODEL run/tune.neyra --format peft -o run/peft
```

To undo, delete `run/model`. The base is only read. The inverse of the patch restores the base bit for bit, and the run
checks that before it finishes.

## Limits

- Float models only, and LoRA only (no DoRA or full tuning).
- The loss after training is on the test rows; it says nothing about the questions you care about. Compare
  generations, and run `neyra tune verdict`.
- One process, one machine. The memory limit is a guideline MLX follows. The system does not enforce it.

## Learning rate

The default is 2e-5. On Qwen3-0.6B with 66 rows, a style task, every rate learned the style. General questions, which the base answered at 90.0%, scored 53.3% after training at 1e-4 and 86.7% at 2e-5. That is one small model and one easy task, on 30 questions, so a harder task may need a higher rate. Run `neyra tune verdict` to see what your run cost.
