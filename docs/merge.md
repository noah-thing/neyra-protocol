# Merging patches

```sh
neyra merge BASE a.neyra b.neyra c.neyra -o merged.neyra
neyra apply BASE merged.neyra -o merged-model
```

Merging published models one after another gives a different answer for each order, because every step
rounds to bf16. At one entry with value 1 - 2^-8, three patches that each add 2^-8 give 1.0 when merged
in pairs and 1 + 2^-7 when added together. `merge` adds all the deltas exactly, as integers, and rounds
once to the tensor dtype, so any order writes the same file. [`tests/test_merge.py`](../tests/test_merge.py)
checks this against a brute force rounding and against all 24 orders of four patches.

Entries that only one patch touches are copied. Where patches overlap, `merge` reports how many entries
they share and how many of those got different values, and exits with 1. It does not judge whether the
result is a good model. Two patches that make the same change apply it twice, since deltas add.
