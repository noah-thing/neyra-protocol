# Working on Neyra

- `uv venv && uv pip install -e '.[dev]'`, then `pytest -q` and `ruff check src tests`.
- Anything that changes hashing or a format must update `spec/PROTOCOL.md` and rerun `spec/vectors/generate.py`.
  The Go verifier in `impl/go` must still pass: `go run . vectors ../../spec/vectors`.
- Tests marked `model` need real weights and do not run by default.
- A claimed result needs a script in `benchmarks` or `experiments` that reproduces it.
