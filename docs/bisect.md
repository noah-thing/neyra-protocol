# What bisect measures

"Blame" names four different measurements, and they can disagree:

| Name | Question |
|---|---|
| movement | how much did the behavior change at this step of the history |
| ablation | how much is lost removing this part from the finished model |
| insertion | how much comes back adding this part alone to the base |
| replay | what happens if the history is run again without it |

`neyra bisect` reports insertion. Every trial applies a subset of the patch to the base model and measures
the behavior, and the answer is the smallest subset that keeps the chosen share of the full effect. It
does not say that removing those units from the full patch would remove the behavior, and it says nothing
about training history. Results carry `"blame": "insertion"` in the command's JSON, the lab result, the
ledger entry and the metadata of the patch it writes. Patch identity and the protocol vectors do not
depend on any of these.
