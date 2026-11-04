from pathlib import Path

import numpy as np

from neyra.store import LocalBlob, parse_header


def read(path):
    refs, _ = parse_header(LocalBlob(Path(path)))
    return {k: r.read().view(np.float32) for k, r in refs.items()}
