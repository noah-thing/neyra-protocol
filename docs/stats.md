# Anonymous stats

Neyra can send a small anonymous report after `tune verdict`, `tune lint`, `tune train` and `hop`. It is off until you
answer a question, and it asks once. The numbers feed the public page at the receiver's `/v1/stats` address, such as how
many tunes beat a prompt.

## The question

After a command finishes in an interactive terminal, Neyra prints the complete report as JSON and asks
`Send this report, and the same kind of report after later runs? [Y/n]`. Enter or `y` means yes. `n` means no. Either
answer is saved in `stats.json` in `~/.config/neyra/` (`$XDG_CONFIG_HOME/neyra/`, or `$NEYRA_CONFIG_DIR`) and the question is not asked
again. Later runs that send print one line saying so and how to stop.

Nothing is asked or sent when the command ran with `--json`, when stdin or stdout is not a terminal, when `CI` is set,
or when `NEYRA_NO_STATS` or `DO_NOT_TRACK` is set to anything but empty or `0`. Those variables win over a saved yes.

## Turn it off

- `neyra stats off` saves a no. `neyra stats on` clears the answer, so the question comes back after the next result.
- `export NEYRA_NO_STATS=1` or `DO_NOT_TRACK=1` turns it off for that shell.
- `neyra stats` shows the current state. `neyra stats show` prints exactly what the last report held.
- Delete `stats.json` to reset.

## What is collected

Every report has `schema` (1), `kind`, `neyra` (the version), and `os` (macos, linux, windows or other). Then, by kind:

| kind | fields |
|---|---|
| verdict | `backend`; `family`; `size`; `verdict` (one of eight labels); `cases` (10-29, 30-49, 50-99, 100-199, 200+); `gain_over_base` and `gain_over_prompt` in points, rounded to a multiple of 5; `quants`: for each GGUF format tested, whether the gain held |
| lint | `trainer`; `family`; `size`; `rows` (<100, 100-999, 1k-9.9k, 10k-99k, 100k+); `blocked` (an error stopped training); `problems`: the type and level of each problem found |
| train | `family`; `size`; `rows`; `outcome` (trained or blocked_by_lint); `loss_improved` (yes, no, unknown); `revert_exact`; `problems` |
| hop | `family`; `size`; `hops` (2, 3, 4+); `ok`; `first_part` and `first_cause` of the first break, if any |

`family` is one of qwen, llama, gemma, mistral, deepseek, phi, smollm or other. It is read from the model name or its
`config.json` on your machine, and only the matching word leaves. `size` is a bucket: <1B, 1-3B, 3-8B, 8-30B, >30B or
unknown. Problem types, quantization formats, parts and causes come from fixed lists in the code; anything else is
sent as `other`.

A sample verdict report:

```json
{
  "schema": 1,
  "kind": "verdict",
  "neyra": "0.3.1",
  "os": "macos",
  "backend": "torch",
  "family": "qwen",
  "size": "<1B",
  "verdict": "worked",
  "cases": "100-199",
  "gain_over_base": 30,
  "gain_over_prompt": 20,
  "quants": [{"format": "Q4_K_M", "held": true}, {"format": "Q8_0", "held": false}]
}
```

## What is never collected

Prompts, cases, outputs, training data, file paths, model names, repository names, adapter names, hostnames, user
names, hashes of any of these, exact timestamps, and any user or device identifier. Neyra stores no identifier on your
machine, so two reports from you cannot be linked.

The receiver reads your IP address only to apply a rate limit, in memory, and does not store or log it. It stores no
headers. It keeps the report fields and the month the report arrived. It rejects any request with a field outside the
lists above, so a changed client cannot send more.

## The public numbers

The receiver's `/v1/stats` returns, from all reports:

- reports received by kind, and tunes checked (verdict plus train reports)
- share of tunes that beat a prompt: verdicts where the tuned model beat both the base and a prompt with examples
- share where a prompt would have done the job: verdicts `prompt_enough` or `prompt_better`
- median gain over the base and over the prompt, in points
- share that lost the gain in quantization, among tunes that beat the prompt and had a GGUF file tested
- the same verdict numbers by model family, by size, and by both
- problems found before training, counted by type, and how many runs an error blocked
- share of training runs where validation loss improved and where the patch reverted exactly
- how often `hop` found a break, and in which part

Groups with fewer than five runs show only a count. A "prompt" here is one prompt with examples made from the user's own
examples; a better prompt could do more. The numbers describe people who chose to share, on their own cases.

## Where the code is

- Client: `src/neyra/tune/share.py`. The endpoint is empty in a build that has no receiver, and then nothing is asked.
- Receiver: `stats/`. `src/schema.mjs` holds the accepted fields, `src/handler.mjs` the request handling,
  `src/aggregate.mjs` the numbers above, `src/store.mjs` the storage.
- Tests: `tests/test_tune_share.py` and `stats/test/`. They fail when a field is added.

The client posts with a three second timeout, using only the standard library, and ignores any failure.

## Running the receiver

It is a Cloudflare Worker with a D1 database. The free plan needs no payment card and covers this traffic. Create the
database with `wrangler d1 create neyra-stats`, put its id in `stats/wrangler.toml`, run
`wrangler d1 execute neyra-stats --remote --file=schema.sql`, then `wrangler deploy`. The handler takes a standard
`Request` and a store, so it can also run on other hosts.
