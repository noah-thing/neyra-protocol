# Limits

- Direct weight edits do not make a model smarter. They can remove an obstacle, such as a loop. They do not
  add ability, and the Weight-Space Programming authors found the same on a model that did not loop. Tuning can
  raise a task score, and `tune verdict` measures that on your cases.
- `tune train` needs Apple Silicon and a full precision base, and it has run on one style task.
  mlx-lm does the training. Neyra adds the lint before it and the patch after it. Settings are defaults, with no search.
- `tune verdict` needs torch even for a tune trained with MLX, and llama.cpp's `llama-server` for GGUF files. You bring
  the test cases. Its prompt arm is one prompt, so a better prompt could do more. The damage check is 30 questions
  and eight texts: it shows large damage and cannot show its absence.
- `tune lint` reads your data and tokenizer, and none of your trainer's config beyond its flags. Packing and custom
  collators are not modelled. It finds format mistakes and cannot judge whether your labels are right.
- A sampled diff cannot prove two tensors match. An edit confined to a few entries can slip through.
  Without `--sample`, every byte is compared.
- `diff` needs the base model. Finding an edit with no base is a much harder problem.
- Structure recovery describes the weights. It cannot tell intent and does not detect every backdoor.
- Edits to expert tensors cannot yet be exported to GGUF, and quantized weights are not analysed.
- Models with several residual streams, such as DeepSeek V4, or a norm after each write, such as
  Gemma 4, are recognised, but direction edits on them are untested.
- Testing so far has been on smaller models, from the Qwen3.5, Qwen3, Qwen2.5 and SmolLM2 families.
- `neyra edit` edits dense MLP layers only, and its AlphaEdit style `--solver null` lost to the default in every run. It runs on MLX and on torch,
  but the benchmark numbers in [the edit guide](edit.md) come from MLX. On torch one fact has been edited on one real model
  (Qwen3-0.6B, on a CPU, 368 s). CUDA, MPS, hybrid stacks such as Qwen3.5 and batches of facts on a real model are untested there.
- `neyra merge` of two edit patches is exact but returns a dense patch, since Annex A has no way to merge low rank factors.
- The lab keeps honest agents honest. An agent with shell access to the same machine could read the
  holdout from disk, so adversarial use needs the server on another host.
