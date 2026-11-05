# Limits

- It does not make a model smarter. Weight edits can remove an obstacle, such as a loop. They do not
  add ability, and the Weight-Space Programming authors found the same on a model that did not loop.
- A sampled diff cannot prove two tensors match. An edit confined to a few entries can slip through.
  Without `--sample`, every byte is compared.
- `diff` needs the base model. Finding an edit with no base is a much harder problem.
- Structure recovery describes the weights. It cannot tell intent and does not detect every backdoor.
- Edits to expert tensors cannot yet be exported to GGUF, and quantized weights are not analysed.
- Models with several residual streams, such as DeepSeek V4, or a norm after each write, such as
  Gemma 4, are recognised, but direction edits on them are untested.
- Everything here has been tested on small Qwen3.5 and Qwen3 models. Larger models are untested.
- `neyra edit` needs MLX, edits dense MLP layers only, and its AlphaEdit style `--solver null` lost to the default in every run.
- `neyra merge` of two edit patches is exact but returns a dense patch, since Annex A has no way to merge low-rank factors.
- The lab keeps honest agents honest. An agent with shell access to the same machine could read the
  holdout from disk, so adversarial use needs the server on another host.
