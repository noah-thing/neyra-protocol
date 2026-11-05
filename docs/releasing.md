# Releasing

1. `pytest -q`, `ruff check src tests`, and `go run . vectors ../../spec/vectors` from `impl/go`.
2. `uv build`, then install the wheel in a fresh environment and run `python scripts/smoke_installed.py`.
3. Update `CHANGELOG.md` and the version in `src/neyra/__init__.py`.
4. Tag `vX.Y.Z`, publish to TestPyPI, check the install, then publish to PyPI.
5. Pin the action example to the tag.

On the first PyPI release, change the install lines in the README to `uv tool install neyra`, and create the public
repository for `extensions/hf-neyra`, whose name must start with `hf-` for `hf extensions install` to find it.
