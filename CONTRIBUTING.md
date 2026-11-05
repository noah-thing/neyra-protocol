# Contributing

Thanks for taking the time. Small, focused pull requests are easiest to review.

## Setup

```sh
uv venv && uv pip install -e '.[dev]'
pytest -q
ruff check src tests
```

Tests marked `model` need real weights and do not run by default.

## Changing a format

Anything that changes hashing or a format must update [`spec/PROTOCOL.md`](spec/PROTOCOL.md) and rerun
`spec/vectors/generate.py`. The Go verifier must still pass:

```sh
cd impl/go && go run . vectors ../../spec/vectors
```

## Claims

A result claimed in the documentation needs a script in `bench` or `experiments` that reproduces it. If it has
limits, state them next to the claim. Failed results are welcome.

## Writing a verifier

An independent implementation is the best test the protocol can get. Work from the specification and the vectors
alone, and please open an issue about anything they leave unclear.
