# Which hop lost it

```sh
neyra hop BASE ADAPTER MERGED model.gguf
```

Artifacts go in the order the tune passed through them. Each one is compared with what the earlier ones say
it should be, and the first sentence names the first that is not, ready to paste into a bug report:

```
The first difference is at the merged model: the adapter was never applied, all 6 tensors it edits equal the base. Intact there: tokenizer, chat template, BOS and special tokens, config and generation config.
```

`--json` gives the same verdict for machines, and the exit code is 1 when a hop has a problem. A correct merge where
rounding thinned a light adapter is a note, as it is in `check`; it fails when an edited tensor ended up equal to the
base, or when under half of the change survived across the edited tensors. When nothing differs
the sentence says the hops are clear and that a remaining difference is not in these files: weights and text can
rule a hop out, they cannot find a runtime or sampling bug.

Weights of a merged model are checked with `check`. A GGUF is read with the `gguf` package (`neyra[gguf]`): F32,
F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0 and Q5_1 are rebuilt from the merged weights and compared as bytes. Q4_K, Q5_K and
Q6_K cannot be rebuilt, so each stored weight is checked against the interval its code allows (see `docs/source.md`):
a value outside one was not what the quantizer saw. Every other type is dequantized and compared within a stated error,
and the output says so. The share of the
tune that is still in each hop is the projection of the stored change on the adapter's change, with its error;
`neyra hop BASE ADAPTER --predict q8_0,q4_0` forecasts it before converting. Rounding is unbiased, so this number
stays near 100% for a light adapter even in Q4_0, where the stored change is mostly noise. The cosine shows that.
`python bench/hop_suite.py` scores the command on planted failures, and `--today` runs `check` and `diff` on the same files.
