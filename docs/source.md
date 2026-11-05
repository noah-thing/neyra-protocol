# Could this file have been made from that model?

`neyra hop` answers one question about a K-quant GGUF that a tolerance cannot: is the model you think you converted
the one that went in? It does this by reading the file, not by re-running the quantizer.

## The idea

A block quantizer picks a scale for each block of weights and stores, for every weight, the nearest integer code to
`weight / scale`. The scale is stored in the file. So every code carries a statement about its source: the weight was
within half a scale step of the value the code decodes to. Call that interval the weight's cell.

To test a candidate, decode the file, line the candidate up with it and count the weights outside their cells. One
weight outside is enough: that candidate was not what the quantizer was given. Zero outside means the candidate is
consistent with the file. It does not mean the file was made from it (see Limits).

Nothing here depends on how llama.cpp found the scales. K-quants search for them, with or without an importance
matrix, and the search differs between versions; the last step does not, because the codes are computed from the
scales that end up stored. That last step is what the cell relies on, and `bench/cells_vs_llamacpp.py` checks it
against llama.cpp's own `ggml-quants.c`, compiled, with and without an importance matrix.

## What you get in `neyra hop`

```
neyra hop Qwen/Qwen3-0.6B my-adapter merged/ model-Q4_K_M.gguf
```

For Q4_K, Q5_K and Q6_K tensors the weights step used to compare dequantized values against a rounding ceiling. It now
checks cells. A tensor is either consistent with the model it should come from, or the output says how many of its
weights lie outside the range their codes allow. When the merged model is refuted but the base is consistent, the GGUF
holds the base: the adapter was lost in conversion. That is the case the old ceiling could not separate from "a light
adapter that Q4_K rounded away".

The route is tried in three ways, because `llama-quantize` may have read an F16 or BF16 GGUF rather than your float32
weights: the candidate as is, rounded to float16, rounded to bfloat16. The one that fits best is used.

## What it was measured on

Qwen3-0.6B, every quantized tensor of each file, read from the Hub by range and never stored. The protocol was written
and hashed before the first file was measured: `bench/cells_protocol.md`. The planted failures are applied to the true
source: `lora` adds a rank 8 update whose size is 0.2 percent of the tensor's rms; `ulp1` moves 1 percent of the
entries one bfloat16 step; `step1` moves 1 percent of the entries by one cell width.

| file | importance matrix | tensors | weights | true source refuted | base | lora | ulp1 | step1 | unrelated |
|---|---|---|---|---|---|---|---|---|---|
| unsloth Q4_K_M | yes | 197 | 596M | **0** | 197/197 | 197/197 | 197/197 | 197/197 | 197/197 |
| bartowski Q4_K_M | yes | 198 | 752M | **0** | 197/197 | 198/198 | 198/198 | 198/198 | 198/198 |
| unsloth Q5_K_M | yes | 197 | 596M | **0** | 197/197 | 197/197 | 197/197 | 197/197 | 197/197 |
| bartowski Q5_K_M | yes | 198 | 752M | **0** | 197/197 | 198/198 | 198/198 | 198/198 | 198/198 |
| unsloth Q6_K | yes | 197 | 596M | **0** | 197/197 | 197/197 | 197/197 | 197/197 | 197/197 |
| bartowski Q6_K | yes | 198 | 752M | **0** | 197/197 | 198/198 | 198/198 | 198/198 | 198/198 |
| unsloth Q8_0 | yes | 197 | 596M | **0** | 197/197 | 197/197 | 197/197 | 197/197 | 197/197 |

Each cell is tensors refuted over tensors checked; every file also holds F32 tensors (norms), which matched the true source exactly
(113 of 113). "Weights" is the number of quantized weights checked. The base is Qwen3-0.6B-Base; bartowski's files keep a
separate output tensor, so there are 198 and the base has no counterpart for it.

How many weights lay outside their cells (true source: zero in every row):

| file | lora | ulp1 | step1 | base |
|---|---|---|---|---|
| unsloth Q4_K_M | 4.28M | 104k | 5.57M | 296M |
| unsloth Q5_K_M | 6.03M | 155k | 5.62M | 386M |
| unsloth Q6_K | 8.66M | 206k | 5.68M | 456M |
| unsloth Q8_0 | 38.5M | 1.27M | 5.75M | 551M |
| bartowski Q4_K_M | 4.64M | 113k | 7.01M | 229M |
| bartowski Q5_K_M | 7.11M | 187k | 7.07M | 356M |
| bartowski Q6_K | 11.1M | 265k | 7.17M | 456M |

`ulp1` is the hardest: 1 percent of the weights, 6 million of them, each moved by well under one percent of its value,
yet 104 thousand to 1.3 million fell outside their cells and every tensor was refuted. A refutation needs one weight; a tensor
holds a million or more.

Two things in the table that went wrong, kept in:

- The first version of the Q8_0 cells refuted the true source in 15 of 197 tensors of unsloth's Q8_0 (47 weights of 596M).
  The cause is in the stored scale: see Q8_0 below. It is fixed, tested with a block that fails without the fix, and the file
  now passes. It is the kind of failure that 100,000 random test values do not find.
- Qwen's own Qwen3-0.6B-Q8_0.gguf is refuted against the Qwen/Qwen3-0.6B weights in 196 of 197 tensors (23 percent of the weights),
  and Qwen's Qwen2.5-0.5B-Instruct Q4_K_M, Q5_K_M and even its fp16 file do not match the Qwen2.5-0.5B-Instruct weights
  (48 of 121 F32 tensors differ; cosine 0.90 to 0.996 on the norms). Widening the candidate by half a bfloat16 step, as if
  the file had been made from a float32 master, leaves 20 percent outside. So these files were made from different values
  than the published weights. The cell test cannot say why, and nothing here is a statement about the uploads: the model
  cards name those repositories as the base, and an earlier or internal checkpoint is one explanation among several.

Controls, llama.cpp's own quantizer (commit 10d4846a572a) on every tensor of Qwen3-0.6B (198 tensors, 752M weights; the tied
embedding is counted twice), plain and with a random importance matrix: Q4_K, Q5_K, Q6_K, Q8_0, **0 weights outside in each of the 8
runs**. Requantizing: Q8_0 values rounded again to Q4_K, then asked about the bf16 source: 198 of 198 tensors refuted (6.2M weights),
and about the Q8_0 values themselves: 0 outside.

GPTQ-Int4 and AWQ files of Qwen2.5-0.5B-Instruct against that model: 168 of 168 tensors refuted in each; 12 and 53 percent of the weights outside.

Cosine between the dequantized file and each candidate, over the quantized tensors. Where it cannot tell candidates apart, the rows are equal to the
printed digits:

| file | true | lora | ulp1 | step1 | base | unrelated |
|---|---|---|---|---|---|---|
| unsloth Q4_K_M | 0.998238 | 0.998236 | 0.998238 | 0.997991 | 0.990064 | 0.034262 |
| unsloth Q6_K | 0.999838 | 0.999836 | 0.999837 | 0.999815 | 0.991640 | 0.034246 |
| unsloth Q8_0 | 0.999985 | 0.999983 | 0.999985 | 0.999983 | 0.991784 | 0.034232 |

The cosine ties the true source with `lora` and `ulp1` to four to six digits; the spread of the true source's own cosine across the four formats
is two orders of magnitude larger than the difference, so no threshold separates them. The relative L2 difference ties the same way
(unsloth Q4_K_M: 0.05940, 0.05943, 0.05940). It separates `base` and `unrelated`, as does a cell test, but only because those are
different models; a fine-tune near the base would sit where `lora` does. A file hash cannot relate a candidate to a GGUF
at all: it has no answer for any row.

## What holds, by type

- **Q4_K, Q5_K, Q6_K, Q8_0.** The cell is the interval around the decoded value, one half of the block's step
  wide on each side, open at the two end codes (a weight beyond the largest code was clamped). Q6_K scales are signed,
  so for a negative scale the open end flips; getting that wrong produced hundreds of false refutations in the first
  attempt and is covered by a test. Q8_0 picks its code with the unrounded scale and stores the float16 one, so its
  cells are widened by the rounding error of that scale. Below 2^-14 float16 is subnormal and the error is absolute, not
  relative; that was the false refutation above. Q8_0 was measured on one real file with a source, and `neyra hop` still
  compares Q8_0 by rebuilding the bytes, as before. Every cell is widened by 2^-13 of its step for float32 arithmetic in the
  quantizer. On 102,400 random values per type and setting, llama.cpp's K-quant output needed none of it and Q8_0
  needed the float16 part, with either compiler setting for fused multiply-add.
- **Legacy types, Q4_0, Q4_1, Q5_0, Q5_1.** Not covered, on purpose. With an importance matrix, llama.cpp picks their
  codes with a weighted search for the scale and not by rounding to the stored one. The harness shows cells built on
  rounding refuting the true source for 100 to 300 blocks per 400. A file does not say whether it used an importance
  matrix, so the whole family is left to the existing tolerance check. Qwen2.5-0.5B has 896 columns, which is not a
  multiple of 256, so llama.cpp stores part of such a model in these types; the benchmark counts those weights as
  not covered.
- **Q2_K, Q3_K, IQ types.** No cell model has been written or checked.
- **Tensors kept in F32, F16, BF16.** Compared for equality after the same routes. That check is exact.
- **GPTQ and AWQ.** They do not round to the nearest code of a fixed scale. GPTQ adjusts the remaining weights to
  cancel each rounding error; AWQ scales the weights by activation statistics before rounding. Their dequantized
  values are not cell centres of the source, and cells refute the true source in every tensor (168 of 168 tensors for GPTQ-Int4 and for AWQ). The
  test does not claim anything about them.

## Limits

- **Consistent is not the source.** Inside the cells is a necessary condition. The cell of a Q4_K weight is one sixteenth
  of its block's range wide, so many different tensors fit. A fine-tune that moves every weight by
  less than the distance to the cell edge is consistent with the file, and so is its base; the check cannot tell them
  apart and does not try. The `ulp1` case shows how fine it can see: one moved weight in a hundred, a quarter of one
  percent of a weight each, was caught in every tensor. How often the weights of a particular fine-tune cross an edge
  decides what it can see, and a light adapter on a coarse type will often cross none of them.
- **A refutation is about the pair, not the upload.** Another model, another conversion, a different row order,
  a changed tokenizer-linked tensor (embedding resized) all refute. Before saying a published file is mislabelled,
  read its model card: many state the base, the conversion script, or that the file was requantized.
- **Requantized files refute their original source.** A GGUF made from a Q8_0 or an F16 GGUF that is not an exact
  copy of the weights has cells around the intermediate values. The original float32 weights fall outside some of them.
  Candidate = the intermediate file's dequantized values would be consistent. The routes above cover F16 and BF16
  intermediates, not a quantized intermediate.
- **Conversion rewrites.** Where the converter reorders or transforms a tensor (the Llama q/k permutation, the
  Qwen3.5 linear attention tensors), `neyra hop` applies the conversion it knows and skips the tensors it does not
  model. A tensor that is wrong in the file because the converter rewrote it differently from the model in this code
  would read as a refutation; the output says when a tensor belongs to a family it cannot model.
- **Sound, not complete.** The slack is chosen so that a true source is not refuted, which a few hundred million real
  weights and the llama.cpp harness support. It is not proved for every version of every quantizer.

## Reproduce

```
PYTHONPATH=src python bench/cells_vs_llamacpp.py /path/to/llama.cpp/ggml          # decode and cells against ggml-quants.c
python -m bench.cells_real.whole_file unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf Qwen/Qwen3-0.6B base=Qwen/Qwen3-0.6B-Base
python -m bench.cells_real.controls Q4_K
```

A whole-file run reads about 1.8 GB over the network and takes about seven minutes with 3 GB of memory.
`tests/test_cells.py` holds llama.cpp's bytes and values for each type in `tests/fixtures/ggml_cells.npz`, so the decode
and the true-source property are tested without llama.cpp.

## Prior art

The cell of a quantized weight is the object Egashira et al. use in "Exploiting LLM Quantization" (arXiv 2505.23786),
to hide behaviour inside the interval, not to test provenance. Tools that compare a GGUF with its source (wvc,
modelDNA, ggufkit, llm-forensics) dequantize and apply a threshold, or re-run `llama-quantize` and compare bytes
(ggufpacker). Double JPEG detection has used the same idea on DCT coefficients for twenty years. We found no tool that
counts source weights outside their cells, which is the part that is new here, and it is a small part.
