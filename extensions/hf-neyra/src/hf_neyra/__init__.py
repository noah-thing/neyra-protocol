"""`hf neyra ...` runs `neyra ...`. The hf CLI hands every argument after the name to this script."""

import sys

from neyra.cli import main as neyra


def main() -> int:
    return neyra(sys.argv[1:], prog="hf neyra")
