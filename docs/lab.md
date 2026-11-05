# The lab

`neyra mcp` gives an agent tools to sample, find directions, propose edits, run controls and search
for the part of an edit that matters. It treats the agent as an untrusted party:

- Measurements come from a fixed budget.
- The holdout stays in the server. Its salted hash is committed when the session opens.
- Every action lands in a hash-chained ledger.
- The verdict comes from fixed gates: a real effect, beating a size-matched random edit, low drift on
  neutral text, and beating free alternatives such as a system prompt or turning thinking off.

A receipt is an in-toto statement in a DSSE envelope, signed with Ed25519. A signature shows who made
a claim. It does not show the claim is true, so receipts name everything needed to rerun it.
