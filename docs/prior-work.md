# Prior work

Neyra builds on Heretic, gguf-neuron-explorer, Abliterlitics, the diffing toolkit, MEMIT and AlphaEdit,
Steer2Edit, CHiPS, and the Weight-Space Programming logs, which made an exact check possible at all. The
direction and column edits here are established methods.

For tuning, TRL, Unsloth, Axolotl, LLaMA-Factory and mlx-lm do the training, and `tune train` calls mlx-lm instead
of replacing it. Unsloth Studio and Transformer Lab cover the whole job in an app, with far more models and methods.
`tune lint` models the token pipelines of those trainers and `tune verdict` reads their adapters.

`neyra edit` takes its update from ROME and MEMIT and its protection of earlier edits from AlphaEdit and
RLSEdit. For merging, mergekit has far more methods; `neyra merge` is a unit weight delta sum whose point is that
it is exact and independent of order. `check` models the merge paths of PEFT and mlx-lm. Git-Theta versions model
parameters in git and Xet deduplicates them on the Hub. OpenSSF Model Signing signs model files with Sigstore, and
Neyra's receipts use the same DSSE and in-toto shapes. modelDNA answers a different question, which model a
checkpoint descends from, and it has a reference database that Neyra does not.
