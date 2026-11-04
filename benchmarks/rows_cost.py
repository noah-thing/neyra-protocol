"""What a row index costs: time to build, size on disk, and the size of a proof.

Runs on a synthetic checkpoint with the row shapes of a 7B model, scaled down to fit in memory.
The ratios carry to a real model; the seconds depend on the disk and the CPU.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from neyra.patch import write_safetensors
from neyra.protocol import weights_root
from neyra.rows import Index, build_index, fetch, prove, verify_proof
from neyra.store import Checkpoint

rng = np.random.default_rng(0)
d, layers = 1024, 1
tensors = {"embed": rng.integers(0, 65535, (12000, d), dtype=np.uint16)}
for i in range(layers):
    for n, shape in (("q", (d, d)), ("up", (4 * d, d)), ("down", (d, 4 * d)), ("norm", (d,))):
        tensors[f"l{i}.{n}"] = rng.integers(0, 65535, shape, dtype=np.uint16)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_safetensors(root / "m.safetensors", tensors, None, "BF16")
    ck = Checkpoint.open(str(root))
    t = time.time()
    flat = weights_root(ck)
    t_flat = time.time() - t
    t = time.time()
    index = build_index(ck)
    t_index = time.time() - t
    index.save(root / "m.rows")
    assert Index.load(root / "m.rows").weights_root == flat == index.weights_root
    t = time.time()
    proof = prove(index, "embed", [(100, 104)])
    t_prove = time.time() - t
    rows = fetch(ck.tensors["embed"], proof)
    verify_proof(proof, rows)
    print(json.dumps({
        "model_mb": round(ck.size / 1e6, 1), "tensors": len(tensors), "weights_root_seconds": round(t_flat, 2),
        "index_seconds": round(t_index, 2), "index_mb": round((root / "m.rows").stat().st_size / 1e6, 2),
        "index_share": round((root / "m.rows").stat().st_size / ck.size, 4), "prove_seconds": round(t_prove, 3),
        "proof_bytes": len(json.dumps(proof)), "rows_read_bytes": len(rows),
    }, indent=1))
