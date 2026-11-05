# Cell test, frozen protocol

Written before any real file was measured. The hash of this file is recorded in the benchmark output.

## Claim under test

For a GGUF tensor stored as Q4_K, Q5_K, Q6_K or Q8_0, the dequantized value and the quantizer's scale define an
interval around every weight, and the tensor that was quantized lies inside it (neyra.cells). Counting candidate
values outside their intervals decides, with no quantizer run, whether the candidate can be the source.

## Files

Qwen3-0.6B (hidden 1024, every 2-D tensor is a multiple of 256 wide) and Qwen2.5-0.5B-Instruct (hidden 896, so llama.cpp
falls back to legacy types for some tensors). GGUFs read by range over HTTP, never stored: unsloth/Qwen3-0.6B-GGUF,
bartowski/Qwen_Qwen3-0.6B-GGUF (importance matrix), Qwen/Qwen3-0.6B-GGUF, Qwen/Qwen2.5-0.5B-Instruct-GGUF; formats
Q4_K_M, Q5_K_M, Q6_K, Q8_0 where the repository has them. Every tensor of a supported type is checked, whole.

## Candidates

true      the repository named in the GGUF repository's card as its base (Qwen/Qwen3-0.6B, Qwen/Qwen2.5-0.5B-Instruct)
base      Qwen/Qwen3-0.6B-Base for the Qwen3 files, Qwen/Qwen2.5-0.5B for the Qwen2.5 files
finetune  a real public fine-tune of the same architecture when one is cached on the mini
lora      true source plus a planted rank-8 update with per-tensor rms equal to 0.2 percent of the tensor's rms (seed 11)
ulp1      true source with 1 percent of entries (seed 12) moved one bfloat16 step up
step1     true source with 1 percent of entries (seed 13) moved by one full cell width of their stored code
unrelated the true source's tensor from another layer of the same kind (layer i+1, or 0 for the last)
requant   the true source against a file made by this llama.cpp's own quantizer from a Q8_0 file's dequantized values

Controls on whole real tensors with llama.cpp's ggml-quants.c compiled on the mini: every supported tensor of
Qwen3-0.6B, quantized to each of the four types, with and without a random importance matrix, then checked against the
true source.

## Rule

A tensor is refuted when at least one live element of the candidate is outside its cell, under the best of the routes
direct, via float16, via bfloat16. A file is refuted when any checked tensor is. Tensors of other types are listed
as not covered. Stored float tensors are compared for equality after the same routes.

## Predictions, written first

1. True source: zero tensors refuted, for every file and format. One refutation of a true source in any control or any
   real file breaks the claim for that format.
2. base, unrelated, step1: refuted in every file. lora and ulp1: refuted in every file, because the effect is first order
   in the change and there are hundreds of millions of values. requant: refuted for Q4_K, Q5_K and Q6_K; consistent for Q8_0.
3. Cosine and relative norm of the difference do not separate true from lora or ulp1: the spread across formats is
   larger than the change. They do separate true from unrelated. A file hash relates no candidate to the file and
   cannot decide any case.
4. Importance-matrix files (bartowski) show no violations of the true source for the four types.
5. GPTQ-Int4 and AWQ files of Qwen2.5-0.5B-Instruct are refuted against their own true source (not nearest rounding,
   or scaled before rounding). If they are not, the test measures something other than rounding and is dropped.
6. Legacy types are not covered by the claim. llama.cpp's importance-matrix path for them picks codes by a weighted
   search; the harness shows violations of the true source there.

Report order: false refutations of true sources, then detection, then baselines.
