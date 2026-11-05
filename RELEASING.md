# Releasing Neyra

Everything below the line "Needs Noah" cannot be done by an agent: it makes something public, spends an
account, or is a decision. Nothing here has been done. The repo is private and `neyra` is not on PyPI.

## Before, checked on the ship branch

- `PYTHONPATH=src python -m pytest -q` and `ruff check src tests experiments scripts extensions fuzz` pass.
- `cd impl/go && go run . vectors ../../spec/vectors` prints 195/195.
- `uv build`, then in a fresh venv `pip install dist/neyra-*.whl && python scripts/smoke_installed.py`.
- `python fuzz/differential.py --cases 20000 --batches 5` finds nothing.

## Needs Noah

1. Decide to go public. Until then the repo stays private and no step below is taken.
2. Merge the `ship` branch and read the diff of `README.md` against what you want public. The pitch and tagline
   were left as they were.
3. Create a PyPI account (or use the existing one), enable 2FA, and reserve `neyra`. Checked read only on
   2026-11-05: `https://pypi.org/pypi/neyra/json` returns 404, so the name is free.
4. Prefer trusted publishing: on PyPI add a pending publisher for the repo (owner `noah-thing`, repository
   `neyra-protocol`, workflow `release.yml`) and add that workflow, which runs `uv build` and
   `pypa/gh-action-pypi-publish` on a tag. The alternative is `uv publish` with an API token from your machine.
5. Try it on TestPyPI first: `uv build && uv publish --publish-url https://test.pypi.org/legacy/`, then
   `uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ neyra demo`.
6. Tag `v0.2.0` and publish to PyPI. Then run `uvx neyra demo` on a machine that has never seen the repo.
7. Make the GitHub repo public. In Settings turn on private vulnerability reporting (SECURITY.md points
   there), and add the topic `hf-extension` only to the extension repo below.
8. README: replace the three install lines inside the `<!-- On the first PyPI release ... -->` comment with
   `uvx neyra demo`, `uv tool install neyra`, `pip install neyra`, and delete the comment. The `uses:
   noah-thing/neyra-protocol@main` line works only once the repo is public; pin it to the tag.
9. `pyproject.toml` already points `Source`, `Changelog` and `Issues` at `noah-thing/neyra-protocol`; check the
   name is the final one.
10. GitHub Action: no marketplace listing is needed to use it. To list it, publish a release from the GitHub UI
    with "Publish this Action to the GitHub Marketplace" ticked (needs the 2FA and terms acceptance only you
    can give). The `package` input defaults to `neyra` from PyPI, so it works only after step 6.
11. `hf` extension: `hf extensions install` clones nothing private; it needs a PUBLIC GitHub repo whose name
    starts with `hf-`, with the extension's files at the repo root. So: create `noah-thing/hf-neyra` as a new
    public repo from the contents of `extensions/hf-neyra/`, add the topic `hf-extension`, and only then run
    `hf extensions install noah-thing/hf-neyra` on a clean machine. This lane is prepared and has NOT been
    tested end to end: `extensions/hf-neyra/scripts/local_install.py` ran hf's real install code with only the
    network faked. Its `pyproject.toml` depends on `neyra>=0.2.0`, so step 6 must come first.
12. Email nobody until a release exists. The outreach (PostTrainBench, mergekit, hf-mem and the like) is a
    separate decision.

## After

Delete the TestPyPI project if you made one, and bump to `0.2.1` in `src/neyra/__init__.py` for the next change.
