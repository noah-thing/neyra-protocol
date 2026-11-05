# Measured accuracy of `diff`

We planted known edits in real Qwen3.5-0.8B tensors and scored recovery. 50 trials per condition.
[`bench/forensics.py`](../bench/forensics.py) reproduces this table.

| Planted edit | Form correct | Direction cosine | Strength error | Column F1 | Rank exact |
|---|---|---|---|---|---|
| writer projection k=64 a=6 | 100% | 1.000 | 0.0000 | 1.000 |  |
| writer projection k=64 a=1 | 100% | 1.000 | 0.0000 | 1.000 |  |
| writer projection k=8 a=2 | 100% | 1.000 | 0.0001 | 1.000 |  |
| writer projection k=512 a=1 | 100% | 1.000 | 0.0000 | 1.000 |  |
| random projection k=64 a=6 | 100% | 1.000 | 0.0002 | 1.000 |  |
| random projection k=64 a=1 | 100% | 1.000 | 0.0009 | 1.000 |  |
| random projection k=512 a=1 | 100% | 1.000 | 0.0002 | 1.000 |  |
| full projection a=1 | 80% | 1.000 | 0.0001 | 1.000 |  |
| input projection a=1 | 68% |  | 0.0003 |  |  |
| additive rank 1 | 100% |  |  |  | 100% |
| additive rank 2 | 100% |  |  |  | 100% |
| additive rank 8 | 100% |  |  |  | 100% |
| zeroed 256 columns | 100% |  |  | 1.000 |  |
| scaled 0.9 | 100% |  | 0.0000 |  |  |

Rounding noise produced no false alarms in 250 trials. The weakest rows are dense edits
along random directions, which sit close to bf16 rounding and are easy to miss.
