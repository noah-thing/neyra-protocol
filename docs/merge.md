# Merging patches

```sh
neyra merge BASE a.neyra b.neyra c.neyra -o merged.neyra
neyra apply BASE merged.neyra -o merged-model
```

Merging published models one after another gives a different answer for each order, because every step
rounds to bf16. At one entry with value 1 - 2^-8, three patches that each add 2^-8 give 1.0 when merged
in pairs and 1 + 2^-7 when added together. `merge` adds all the deltas exactly, as integers, and rounds
once to the tensor dtype, so any order writes the same file. [`tests/test_merge.py`](../tests/test_merge.py)
checks this against a brute-force rounding and against all 24 orders of four patches.

Entries that only one patch touches are copied. Where patches overlap, `merge` reports how many entries
they share and how many of those got different values, and exits with 1. It does not judge whether the
result is a good model. Two patches that make the same change apply it twice, since deltas add.

## Methods

```sh
neyra merge BASE a.neyra b.neyra --method ties --weights 1 1 --set density=0.5 -o merged.neyra
neyra check BASE merged.neyra --parents a.neyra b.neyra
```

| `--method` | What it does |
|---|---|
| `sum` | Unit weights, the default |
| `task` | Weighted task arithmetic |
| `ties` | Trim each delta, elect a sign by weighted mass, average the parents that agree |
| `dare` | Drop entries with a hash of seed, tensor, parent and index, then rescale, so a run repeats anywhere |
| `slerp` | Spherical interpolation of two parents |
| `model_stock` | Weight the mean of the parents by the cosine between their deltas |
| `matched` | Scale each tensor's merged delta to the mean length of its parents |

Coefficients sit on a grid of 2^-16 and sums stay exact, so input order does not change the bytes. A merged patch
carries its recipe, and `check --parents` merges again and compares every tensor. [`tests/test_methods.py`](../tests/test_methods.py)
tests each method against a brute-force oracle written from its paper.

These are arithmetic results. How well each method scores on a task has not been measured here. The study protocol is
in [`bench/merge_methods`](../bench/merge_methods). mergekit has more methods and a far longer track record.

