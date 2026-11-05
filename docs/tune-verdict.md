# tune verdict

```sh
neyra tune verdict Qwen/Qwen2.5-0.5B-Instruct ./adapter cases.jsonl --examples train.jsonl
neyra tune verdict BASE ./merged-model cases.jsonl --examples train.jsonl --gguf model-Q8_0.gguf model-Q4_K_M.gguf
neyra tune verdict BASE ADAPTER cases.jsonl --examples train.jsonl --scorer 'regex:^ANSWER: {expected}$' --format plain
neyra tune verdict BASE ADAPTER cases.jsonl --examples train.jsonl --json --save verdict.json
```

Run it after a tune. It answers two questions on your own test cases: did the tune help, and could a prompt
have done the same. Install the extra with `uv tool install 'neyra[torch,tune]'`. On a Mac, `neyra[tune,mlx]` runs it without torch. `--backend` picks the engine, and the default is torch when it is installed. The MLX engine has run the banking77 case below end to end, with the tuned model at the same 98.5%, but it has not been compared with torch case by case, so treat small differences between the two as unexplained. Quantized files need llama.cpp's
`llama-server` on the PATH (or in `NEYRA_LLAMA_SERVER`). Without it the files are listed as not run, the first line
says so, and the exit code is 1.

## Input

`CASES` is a jsonl file of 30 to 200 cases the tune never saw. Each line has an input and the expected output:
`{"input": "...", "output": "..."}`. `expected`, `prompt`/`completion` and a `messages` row with one user and one
assistant message are read too. An optional `system` is passed to every arm, and `id` names the case in the report.
Ids must be unique. Under 10 cases the command refuses to run; under 30 it warns.

`--examples FILE` is where the prompt with examples takes its examples. It is read in the same shapes, so your training file
works. Examples that are also cases are dropped. `--shots N` (default 8) picks N of them with `--seed`. When the
answers repeat, as class labels do, the picks are spread over the distinct answers.

`--scorer` is one of `exact` (trimmed, case folded), `contains`, `regex:PATTERN` (`{expected}` stands for the escaped
expected output) or `python:FILE`. The file defines `score(output, expected, case)` and returns a bool or a number from
0 to 1. `case` is a dict with `id`, `input` and `system`.

`--format chat` renders prompts with the tokenizer's chat template, which is what most tunes were trained on. `--format
plain` sends `--template` with `{input}` filled in; with plain prompts every answer is cut at the first newline unless
you give `--stop`. All arms get the same format, the same stop rules and greedy decoding for at most `--max-tokens`.

`TUNED` is a PEFT or mlx-lm adapter folder (merged into the base the way PEFT's `merge_and_unload` rounds, using the
same code as `neyra check`) or a folder of merged weights. An adapter that also stores full tensors
(`modules_to_save`) has to be merged first.

`--save FILE` and `--json` include a manifest: the command, the digests of the cases, the examples and every quantized file,
the base and tuned model records, the seed and the tool versions.

## What it compares

| Arm | Model | Prompt |
|---|---|---|
| base | base weights | the case alone |
| few-shot | base weights | the case after `--shots` examples |
| tuned | adapter or merged weights | the case alone |
| each `--gguf` file | the quantized file | the case alone |

Every arm answers the same cases. For each pair that matters the report gives the difference in points, a 95%
bootstrap interval over cases (10,000 resamples, fixed seed) and the number of cases won and lost. Two arms count as
equal only when the whole interval sits inside `--margin` (default 5 points). An interval that includes zero but is
wider than the margin is "cannot tell", and the sentence says roughly how many cases would settle it.

## The first line

One sentence, chosen in this order:

1. The tuned model is worse than the base, or equal to it: **it did not work**.
2. It beats the base but is unclear against the prompt arm: **cannot tell** how many cases it would take.
3. It beats the base, and the prompt arm is equal or ahead: **a prompt does the same, you did not need to train**.
4. It beats both free arms, and a quantized file is equal to or below one of them: **the gain is lost at Q4_K_M**. A
   file whose interval cannot tell is reported as cannot tell.
5. It beats both and survives every file: **this worked**. If the general check below shows a drop it says so in the
   same sentence.

Exit code 0 means the tuned model and every quantized file given beat both free arms, and the general check found no
drop. A file that could not run counts against it: nothing was measured. Otherwise it is 1. Bad input is 3.

## Damage

- Thirty general questions (capitals, arithmetic, simple facts) asked of the base and the tuned model, scored by
  substring, paired. A drop whose interval stays below zero is damage; a point drop of 10 or more that thirty
  questions cannot confirm is reported as possible damage.
- KL(base || tuned) on eight short neutral paragraphs, in nats per token. No threshold is applied: nobody has a
  calibrated one, so compare the number between your own runs.
- The cases the base got right and the tuned model got wrong, with both outputs.

## Limits

- Only what was tested is reported: your cases, your scorer, the prompt format and examples shown, greedy decoding.
- The prompt arm is one prompt. A better prompt could beat it, so "a prompt does the same" holds for that prompt.
- Quantized files are compared with free arms that ran in full precision. Their prompts go in as token ids from the
  base tokenizer, so a GGUF whose vocabulary differs gives meaningless results; `neyra hop` and `tune lint --gguf` check that.
- The general questions and the KL texts are small. They show large damage and cannot show its absence.
- Backends differ by rounding. A quantized file run through llama.cpp is compared with torch on CPU in bf16, and a
  small number of flips between them is noise.

## Checked on

Run with torch for the models and llama.cpp build 11960 (`8642541b2`) for the files. Every case file, log
and the scripts that built the files are in `~/weights-research/c4/` (`proofs/tv_*`, `results/tv_*`). The numbers are
one run each: greedy decoding is deterministic, and the intervals are over cases.

- A public tune, [booleai/qwen2.5-0.5b-banking77-10intent-lora](https://huggingface.co/booleai/qwen2.5-0.5b-banking77-10intent-lora)
  on Qwen2.5-0.5B-Instruct, 200 of the BANKING77 test cases for its 10 intents, 40 training examples in the prompt.
  The model card reports 60.75% to 97.75% on 400 cases and says a prompt with examples was not tried. The command gave base 62.5%,
  40-shot prompt 90.0%, tuned 98.5%, and "This worked ... +8.5 points ahead of the best free option, few-shot
  (interval +5.0 to +12.5)". With 10 examples the prompt reached 70% on the first 100 cases. F16, Q8_0 and Q4_K_M files
  scored the same 98.5% as the adapter in torch. Q3_K_S and Q2_K scored 97.5%, two cases fewer. Those two names flatter the result: Qwen2.5-0.5B has a hidden size of 896, K quants need rows in multiples of 256, and llama-quantize falls back for 144 of 290 tensors, so its Q2_K file holds mostly `q4_0` and `q5_0` at 5.48 bits per weight.
- A planted case where a prompt matches: a LoRA trained for 40 steps on 32 country and capital pairs to answer in the form
  `ANSWER: <capital>`, tested on 90 other countries, on SmolLM2-360M-Instruct. It can only have learned the format.
  Base 0%, 8-shot prompt 93.3%, tuned 94.4%. At the default margin of 5 points the interval (-3.3 to +5.6) does not fit, and the
  command says it cannot tell and that about 90 cases would settle it. With `--margin 0.1` it says "A prompt does the same, you did not need to train".
- A planted loss in quantization: the same adapter and cases, with one file converted from a merge that dropped the adapter
  (alpha 0). It said "The gain is lost at Q4_K_M ... 98.5% before quantizing and 57.0% in that file ... It keeps the gain at Q8_0."
- A tune that does nothing (the adapter at alpha 0): "It did not work", exit 1.
- The same adapter at a quarter of its strength scored 84.5% and fell behind the 40-shot prompt. Its Q2_K file lost 5 points
  (interval -10.5 to +0.5). A gain that dies in quantization did not appear in any real file tested.
