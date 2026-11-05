# Merge study protocol, frozen before any model was scored (2026-11-05)

The sha256 of this file and of the scorer (c4/proofs/mm_eval.py) is in merge_protocol.sha256, written before the first job ran.

## Models (all public, bf16, apache-2.0, identical tensor layout, 494,032,768 parameters)
base Qwen/Qwen2.5-0.5B. Parents: coder = Qwen/Qwen2.5-Coder-0.5B, grpo = jaygala24/Qwen2.5-0.5B-GRPO-math-reasoning, instruct = Qwen/Qwen2.5-0.5B-Instruct.
Merge sets: M2 = {coder, grpo} (a code and a maths fine-tune). M3 = {coder, instruct, grpo}.

## Scoring
GSM8K (openai/grade-school-math test, 4 fixed train shots, completion style, greedy, 256 new tokens, last "#### n" or last number) and
MBPP (Google mbpp.jsonl task ids 11 to 510, 3 fixed prompt-split shots, "[BEGIN]/[DONE]" format, greedy, 200 new tokens, the asserts are the test).
No chat template for any model. Items are ordered by sha256("neyra-merge-prereg-1/<task>/<id>"); dev = first 80 of each task, test = next 250 (GSM8K) and 200 (MBPP). Test items are never scored before the configuration is fixed on dev.
Score = macro mean of the two task accuracies. Held-out NLL (150 GSM8K train solutions, 150 MBPP solutions) is used only in the kill test.

## Kill test
Task arithmetic of coder, instruct, grpo with weights 0.5, 0.25, 0.25 (dyadic, so every tool uses the same coefficients), base Qwen2.5-0.5B.
Variants: Neyra exact; float32 torch in two orders; mergekit (commit 0f1423e) default bfloat16 in two orders; mergekit dtype float32 with bfloat16 output.
Measured: entries and ulps that differ between every pair. Then dev and test scores and NLL of the variants that differ most.
Rule: if no pair of rounding variants differs by more than the 95% interval of the paired bootstrap on any score, rounding and order do not change quality, and exactness is reported as an auditing property.

## Methods and hyperparameter grids (tuned on dev macro only, ties broken by the first listed)
Equal weight per parent unless stated. Coefficients are quantised to 2^-16, accumulation is exact, rounding once (neyra.methods).
- task (plain task arithmetic): w in {0.5, 0.75, 1.0} for M2, {0.375, 0.5, 0.75} for M3.
- ties: density in {0.2, 0.5}, weight 1, normalize, lam 1, sign by mass.
- dare (M2 only): density in {0.5, 0.8}, seed 0, rescale 1/density, weights = the dev best task weight, no consensus.
- model_stock: no hyperparameter.
- slerp (M2 only): t = 0.5, first = coder. If every tensor takes the linear branch it is task arithmetic at (0.5, 0.5) and is not scored separately.
- matched (my own idea, defined here before any score): per tensor, lambda_T = min(cap, kappa * (sum_i |w_i| |d_i| / sum_i |w_i|) / |sum_i w_i d_i|), d_i the exact delta of parent i, norms over the whole tensor, coefficient w_i * lambda_T. kappa in {0.7, 1.0, 1.4}, cap 1.0, weights 1.
Reference points: base, each parent alone, the best single parent on dev, and dev-tuned task arithmetic.

## Reporting
For the dev-chosen configuration of each method and merge set: test accuracy per task and macro, with 95% intervals from 10,000 paired bootstrap resamples over items (resampled within task). Differences against dev-tuned task arithmetic and against the best single parent (chosen on dev) use the same resamples.
"Better than X" is claimed only if the lower end of the interval of the difference is above zero. "Best" is claimed only for a method whose interval for the difference against every other method and the best parent is above zero. Anything else is reported as not distinguished.
Failures and null results are reported as they come. Configurations are never added after seeing test scores.
