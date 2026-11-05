# tune lint

```sh
neyra tune lint data.jsonl Qwen/Qwen3-0.6B --trainer trl
neyra tune lint data.jsonl ./my-model --trainer mlx --max-length 4096 --val val.jsonl --gguf model-Q4_K_M.gguf
neyra tune lint data.jsonl MODEL --show 3      # three rows with trained tokens marked
neyra tune lint data.jsonl MODEL --json
```

Install the extra with `uv tool install 'neyra[tune]'`. It adds `transformers` (used only for its tokenizers and chat
templates, no torch), `tokenizers` and `jinja2`. `--gguf` also needs `neyra[gguf]`.

The command rebuilds, row by row, the token ids and the loss mask the trainer would use, then looks for what is wrong.
No model weights are read. For a Hub name only the tokenizer and config files are downloaded, into
`~/.cache/neyra/tokenizers`. It exits 1 when any finding is an error, so it can gate a run.

The first line is one sentence to act on. Below it come the settings it assumed (trainer, sequence length, which tokens
are in the loss) and the findings, worst first. Each has a count and one decoded row. Trained runs are inside `[[ ]]`.

```
Do not train yet: 4,812 of 5,000 rows end without an end token, so the model will not learn to stop.

1. error: no end token, 4,812 of 5,000 (96%)
   ...
   row 12, 214 tokens, 186 trained:
   ### Instruction:\nName a color.\n\n### Response:\n[[Blue]]
```

## Input

One JSON object per line. The shape of each line is read from its keys.

| Shape | Keys |
|---|---|
| chat | `messages` (`role`, `content`; content may be a list of `{"type": "text", "text": ...}`) |
| sharegpt | `conversations` (`from`: human, gpt, system; `value`) |
| prompt and completion | `prompt`, `completion`, as strings or as lists of messages |
| alpaca | `instruction`, optional `input`, `output`, optional `system`. Read as a user and an assistant message |
| text | `text` |

A `system` key beside `messages` is added as the first message. `tools` is passed to the template.

## Trainers

`--trainer` picks the pipeline. The defaults are what each trainer does with no extra settings. `--max-length` and
`--mask-prompt` / `--no-mask-prompt` override them.

| Trainer | Tokens | Loss mask | Default length | Default loss |
|---|---|---|---|---|
| `trl` | exact | exact | `max_length` 1024 | chat rows: every token. prompt and completion rows: the completion. `--mask-prompt` is `assistant_only_loss`, which needs `{% generation %}` markers in the template |
| `mlx` | exact | exact | `max_seq_length` 2048 | every token; `--mask-prompt` is mlx-lm's `mask_prompt` |
| `unsloth` | follows the notebook flow: render, add eos, tokenize | rebuilt from `train_on_responses_only`'s rules | `max_seq_length` 2048 | answers only |
| `axolotl` | the tokenizer's chat template | approximate | `sequence_len` 2048, long rows are dropped | assistant turns and the eos token |
| `llama-factory` | the tokenizer's chat template | approximate | `cutoff_len` 2048 | assistant turns |

How exact, and against what (see `tests/test_tune_agreement.py`):

- **TRL 1.17.1** (transformers 5.22.1). Real `SFTTrainer` on a tiny random model: ids and labels equal on chat, prompt
  and completion, conversational prompt and completion, and text rows, and under truncation, for 9 real tokenizers
  (Qwen2.5, Qwen2.5 base, Qwen3, Llama 3.2, SmolLM2, TinyLlama, Gemma 2, Mistral 7B v0.3, Phi 3). The first token is
  never counted as trained: nothing predicts it.
- **mlx-lm 0.32.2**. The real `ChatDataset`, `CompletionsDataset` and `TextDataset`: ids and prompt offset equal on the
  same 9 tokenizers. The mask follows `default_loss`. mlx-lm also counts one padding target per padded row in the loss
  average; that is not modelled.
- **Unsloth 2026.11.2 / unsloth_zoo 2026.11.2**. The real `train_on_responses_only` mask function, run on the ids this
  tool builds, with the markers found from the chat template: equal. The tokenization step (BOS handling and the eos that
  TRL appends to text) is rebuilt from reading unsloth's source and was not run against unsloth, which needs a GPU. Chat
  templates whose markers tokenize differently inside a conversation than alone are refused.
- **Axolotl 0.21.1**. The real `chat_template` strategy with its defaults: ids equal on all tokenizers tested, mask equal
  on all but Qwen3 (Axolotl leaves the empty thinking block out of the trained span) and Mistral v0.3. Treated as
  approximate because its turn locator and options are many. Text rows are refused.
- **LLaMA-Factory 0.9.5**. Its own named templates are not read. With the `qwen` template and a Qwen2.5 tokenizer ids
  and mask are equal. With `llama3` they are not: LLaMA-Factory leaves out the date header that the Llama 3.2 template
  adds. Treat the result as the tokenizer's template, which is what the model is served with. Text rows are refused.

## What it reports

Errors make the exit code 1. A share of rows below 1% turns most errors into warnings. `note` never fails.

| Finding | Level | Claim |
|---|---|---|
| no end token | error | the last tokens of the row are not an end token (the tokenizer's eos, the generation config's, or the template's end of turn) |
| end token not trained | error | the end token is there but its label is ignored. The same finding names the Axolotl default where only the eos token is trained |
| double BOS | error | two BOS tokens in a row at the start |
| missing BOS | error for text rows, warning for chat rows | the tokenizer adds BOS to plain text and the row has none. Chat templates that leave it out are the author's choice |
| empty loss | error | no token of the row is in the loss, after truncation |
| truncated | warning, error when answers lose their end | rows longer than the sequence length, how many lose their end token, and the share of tokens cut. Axolotl drops such rows instead, shown as `rows dropped` |
| prompt text in the loss | warning | system or user tokens are trained. Silenced by `--no-mask-prompt` |
| template applied twice | error | message content already holds this model's chat markers |
| another chat format | error | text rows hold another model's markers (ChatML, `[INST]`, Llama 3 headers and similar) that this tokenizer does not know as single tokens |
| prompt is not a prefix | error | the template renders the question with `add_generation_prompt` differently from the start of the full chat, so training and serving see different contexts. This is how a template that drops the system prompt from full chats shows up |
| thinking removed | error | the answer holds `<think>...</think>` or `reasoning_content` and the template leaves it out of the trained text |
| rows the template rejects | error | the template raises on the row, for example a system turn on Gemma 2 |
| system prompt in answers | warning, error from 5% | an answer repeats a sentence of its own row's system prompt |
| duplicate rows | warning, error from 10% | exact repeats and near repeats (word 3-grams, estimated Jaccard of 0.9 or more) |
| validation rows also in training | warning, error from 2% | `--val FILE` rows with an exact or near match in the training file |
| GGUF template, tokenizer | error | `--gguf`: the embedded template renders chats differently from the tokenizer's (four test chats), or the vocabulary, BOS or eos settings differ |
| pad token is the end token | note | the modelled trainers build labels from the attention mask, so the end token stays in the loss. A collator that masks every pad id, such as the transformers `DataCollatorForLanguageModeling`, would remove it. The note does not claim the loss changed |
| unreadable rows, mixed shapes | error | lines that are not rows, rows not in the shape of the first |
| mask approximate | note | chat rows whose turns could not be found by prefix, treated as fully trained |

Everything is counted from the ids and mask that were built. Where a trainer is not reproduced exactly the finding says
`(approximate)`.

## Limits

- It does not know the trainer's version or your config beyond the flags above. LoRA settings, packing, `eot_tokens`,
  custom collators and chat templates passed on the trainer's command line are not read.
- Packing and multimodal rows are not modelled.
- A base model that never learned its end token is a problem in the weights. Lint reads files, so it does not check that.
- Near duplicates are found by word overlap, so paraphrases are not.
- Speed: 100,000 chat rows of about 320 tokens take about 30 seconds on an Apple Silicon laptop.
