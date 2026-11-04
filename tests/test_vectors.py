from pathlib import Path

from neyra.conformance import run_vectors

VECTORS = Path(__file__).parent.parent / "spec" / "vectors"


def test_reference_passes_every_vector():
    assert run_vectors(VECTORS) == []
