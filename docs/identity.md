# Identity

```sh
neyra identity BASE MODEL [--adapter ADAPTER_DIR] [--full]
```

One verdict, from a sample of 24 matrices read over HTTP ranges (a megabyte or two), or from every byte with `--full`:

| Verdict | Meaning |
|---|---|
| bit identical | the safetensors files have the same SHA-256 |
| same weights in a different file | every tensor hashes the same wherever it is stored, including a fused projection against its split parts and a head against the embedding it is tied to. Only `--full` says this |
| no difference found in a sample | the sampled spans are equal. Not proof: an edit outside them is missed |
| derived by a named transform | a cast to another dtype, or a LoRA merge reproduced bit for bit by a named path |
| derived with unknown transform | far more alike than independently trained models are |
| unrelated | no more alike than independently trained siblings |
| inconclusive, not comparable | between the two ranges, or the files lay their tensors out differently |

"Derived" is judged against a null measured on unrelated siblings, never against a number chosen for the occasion;
the panel, thresholds and held-out error rates are in `src/neyra/identity_null.json`, rebuilt by
`bench/identity_panel.py`. `tools/weights_identity_check.py` is a numpy-only version for a benchmark judge: it
takes a candidate, the allowed base and the forbidden instruct model.
