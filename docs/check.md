# Checking a merge

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

## Naming the real base of a merge

```sh
neyra check BASE MERGED --adapter ADAPTER_DIR --also MIRROR_OR_OTHER_BASE ...
```

With `--also`, `check` reads a sample of the edited tensors against every candidate base and reports which one
plus the adapter gives the merged tensors back. A card, an adapter config and a mirror can each name a different
base. Candidates that agree on every sampled span are listed as sharing weights, because no merge can tell them
apart. Without an adapter two checkpoints are compared with no tolerance, and with one an entry may be a step
off only if at most 2% of the tensor is.
