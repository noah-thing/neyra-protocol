# hf-neyra

Neyra for the `hf` command line.

```sh
hf extensions install noah-thing/hf-neyra
hf neyra demo
hf neyra diff BASE_REPO OTHER_REPO --sample 4
hf neyra check BASE_REPO MERGED_REPO --adapter ADAPTER_REPO
```

Every `hf neyra` command is the `neyra` command of the same name, so
[the Neyra README](https://github.com/noah-thing/neyra-protocol#readme) documents all of them. Models named
`owner/name` are read from the Hub with range requests, and `HF_TOKEN` is used for gated ones.

## How hf installs this

`hf extensions install owner/hf-name` looks for a file called `hf-name` at the root of the GitHub repo. This
repo has none, so hf creates a virtualenv under `~/.local/share/hf/extensions/hf-neyra`, runs
`pip install https://github.com/owner/hf-neyra/archive/HEAD.zip` in it, and links the console script
`hf-neyra` that `pyproject.toml` declares. The repo has to be public, its name has to start with `hf-`, and
the topic `hf-extension` is what makes `hf extensions search` list it.
