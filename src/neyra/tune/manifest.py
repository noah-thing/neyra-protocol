"""What a tuning run has to record for someone else to rerun it: inputs by digest, tools by version."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import sys
from importlib import metadata
from pathlib import Path

from .. import __version__

PACKAGES = ("mlx", "mlx-lm", "torch", "transformers", "tokenizers", "peft", "gguf")


def digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def file_record(path: str | Path) -> dict:
    p = Path(path).expanduser()
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return {"path": str(path), "bytes": p.stat().st_size, "sha256": f"sha256:{h.hexdigest()}"}


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"neyra": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def command() -> str:
    return shlex.join(["neyra", *sys.argv[1:]])


def local_folder(spec: str) -> Path | None:
    """The folder of a model: the path itself, or the cached snapshot of a Hub name. None when it is in neither."""
    folder = Path(spec).expanduser()
    if folder.is_dir():
        return folder
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    found = try_to_load_from_cache(spec, "config.json")
    return Path(found).parent if isinstance(found, str) else None


def chat_template(folder: Path) -> str | None:
    separate = folder / "chat_template.jinja"
    if separate.exists():
        return separate.read_text(encoding="utf-8")
    config = folder / "tokenizer_config.json"
    if config.exists():
        template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        return template if isinstance(template, str) else None
    return None


def model_record(spec: str) -> dict:
    """Names the model and what shapes its prompts. Weights are bound by digest in the patch, not here."""
    out: dict = {"name": spec}
    folder = local_folder(spec)
    if folder is None:
        return out
    resolved = folder.resolve().parts
    if "snapshots" in resolved[:-1]:
        out["revision"] = resolved[resolved.index("snapshots") + 1]
    for name in ("config.json", "tokenizer.json", "adapter_config.json", "adapter_model.safetensors",
                 "adapters.safetensors"):
        if (folder / name).exists():
            out[name] = file_record(folder / name)["sha256"]
    template = chat_template(folder)
    if template is not None:
        out["chat_template"] = digest(template.encode())
    return out
