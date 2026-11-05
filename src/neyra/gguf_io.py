"""Read the tensors and metadata of a GGUF file and line them up with a Hugging Face checkpoint.

This needs the optional `gguf` package, which is also where the dequantizers come from. A value read
out of a quantized tensor is a dequantized one and every report that uses it says so. Types that
gguf-py can also quantize (F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1) are reproduced from the source
weights and compared as bytes, which is exact. K-quants and the IQ types can only be dequantized, so
nothing claims them bit for bit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .families import v_head_order
from .store import FormatError

# What gguf-py can write from float32, and so what a comparison can rebuild byte for byte.
REPRODUCIBLE = ("F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1")
# The relative error that a type is expected to leave on Gaussian weights, with room to spare.
# Anything past it is not rounding.
ROUNDING_CEILING = {"F32": 1e-6, "F16": 1e-3, "BF16": 1e-2, "Q8_0": 0.02, "Q6_K": 0.03, "Q5_1": 0.06, "Q5_0": 0.06,
                    "Q5_K": 0.06, "Q4_1": 0.15, "Q4_0": 0.2, "Q4_K": 0.15, "Q3_K": 0.3, "Q2_K": 0.6}
DEFAULT_CEILING = 0.5
# Converters that interleave the two rotary halves of each head in q and k.
PERMUTED = {"llama"}
# Families whose converter reorders tensors beyond the ones modelled here.
UNMODELLED = re.compile(r"linear_attn\.(?:in_proj|norm)")
LINEAR_ARCHS = ("qwen35", "qwen3next", "qwen35moe")
DT_BIAS = re.compile(r"layers\.(\d+)\.linear_attn\.dt_bias$")


def need_gguf():
    try:
        import gguf
    except ModuleNotFoundError as e:
        raise FormatError("reading GGUF needs the gguf package: uv tool install 'neyra[gguf]'") from e
    return gguf


def permute(x: np.ndarray, heads: int) -> np.ndarray:
    """convert_hf_to_gguf's LlamaModel.permute."""
    return x.reshape(heads, 2, x.shape[0] // heads // 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)


@dataclass(frozen=True)
class GTensor:
    name: str
    kind: str
    shape: tuple[int, ...]
    raw: np.ndarray
    qtype: object

    def values(self) -> np.ndarray:
        """Float32, dequantized when the type is a quantization."""
        gguf = need_gguf()
        if self.kind == "F32":
            return np.asarray(self.raw, dtype=np.float32)
        if self.kind == "F16":
            return np.asarray(self.raw).astype(np.float32)
        out = gguf.quants.dequantize(np.asarray(self.raw), self.qtype)
        return out.reshape(self.shape)


class GGUFFile:
    def __init__(self, path: str | Path):
        gguf = need_gguf()
        self.spec = str(path)
        try:
            reader = gguf.GGUFReader(self.spec)
        except (ValueError, OSError) as e:
            raise FormatError(f"{path}: not a readable GGUF file ({e})") from e
        self._reader = reader
        self.meta = {k: _contents(f) for k, f in reader.fields.items() if not k.startswith("GGUF.")}
        self.arch = self.meta.get("general.architecture", "")
        self.tensors = {
            t.name: GTensor(t.name, t.tensor_type.name, tuple(int(d) for d in reversed(t.shape)), t.data, t.tensor_type)
            for t in reader.tensors
        }
        self.size = Path(path).stat().st_size

    def key(self, suffix: str):
        return self.meta.get(f"{self.arch}.{suffix}")

    @property
    def layers(self) -> int | None:
        return self.key("block_count")

    def types(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tensors.values():
            out[t.kind] = out.get(t.kind, 0) + 1
        return out

    def lookup(self):
        """HF tensor name -> GGUF tensor name, from gguf-py's own table for this architecture."""
        gguf = need_gguf()
        arch = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}.get(self.arch)
        if arch is None or self.layers is None:
            raise FormatError(f"{self.spec}: architecture {self.arch!r} is not known to this gguf package")
        table = gguf.get_tensor_name_map(arch, self.layers)

        def find(name: str) -> str | None:
            bias = DT_BIAS.search(name)
            if bias and self.arch in LINEAR_ARCHS:
                return f"blk.{bias[1]}.ssm_dt.bias"
            # Multimodal checkpoints nest the text model; the converter drops the wrapper before it maps names.
            for candidate in (name, name.replace("model.language_model.", "model.", 1),
                              name.replace("language_model.model.", "model.", 1), name.replace("language_model.", "", 1)):
                hit = table.get_name(candidate, try_suffixes=(".weight", ".bias"))
                if hit:
                    return hit
            return None

        return find


def _contents(field):
    try:
        value = field.contents()
    except Exception:  # noqa: BLE001 - gguf-py raises bare errors on exotic field types
        return None
    return value.tolist() if isinstance(value, np.ndarray) else value


def to_gguf_layout(name: str, values: np.ndarray, arch: str, config: dict) -> np.ndarray | None:
    """The tensor the way convert_hf_to_gguf stores it, or None when this converter's rewrite is not modelled."""
    if arch in PERMUTED and name.endswith(("q_proj.weight", "k_proj.weight")):
        heads = config.get("num_attention_heads") if name.endswith("q_proj.weight") else config.get("num_key_value_heads")
        return permute(values, heads or config["num_attention_heads"])
    if arch in LINEAR_ARCHS and name.endswith(".A_log"):
        return -np.exp(values)
    if arch in LINEAR_ARCHS and "linear_attn.conv1d.weight" in name:
        return values.reshape(values.shape[0], values.shape[-1])
    if "linear_attn.out_proj.weight" in name:
        nk, nv = config.get("linear_num_key_heads"), config.get("linear_num_value_heads")
        if nk and nv and nk != nv:
            return values[:, v_head_order(values.shape[1], nk, nv)]
        return values
    nk, nv = config.get("linear_num_key_heads"), config.get("linear_num_value_heads")
    if arch in LINEAR_ARCHS and nk != nv and UNMODELLED.search(name):
        return None
    return values


def alternates(name: str, values: np.ndarray, arch: str) -> list[np.ndarray]:
    """Other forms the converter is known to write a 1-D norm weight in: these families store the weight plus one."""
    if values.ndim == 1 and name.endswith("norm.weight") and arch in LINEAR_ARCHS + ("gemma2", "gemma3", "gemma3n"):
        return [values + 1.0]
    return []


def quantize(values: np.ndarray, kind: str) -> np.ndarray:
    gguf = need_gguf()
    qtype = getattr(gguf.GGMLQuantizationType, kind)
    return gguf.quants.quantize(np.ascontiguousarray(values, dtype=np.float32), qtype)


def as_f16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).astype(np.float32)


def as_bf16(values: np.ndarray) -> np.ndarray:
    from .numerics import from_float, to_float

    return to_float(from_float(values, "BF16"), "BF16")


# llama-quantize reads an F16 or BF16 GGUF, so a Q8_0 file may have been made from either, not from the
# original float32 values. Each is a way the same bytes can come about.
SOURCES = {"direct": lambda v: v, "via F16": as_f16, "via BF16": as_bf16}


def rebuild(values: np.ndarray, kind: str, raw: np.ndarray, qtype) -> tuple[str, float] | None:
    """How `values` become `raw` in a reproducible type: the route, and the share of blocks that match."""
    if kind not in REPRODUCIBLE:
        return None
    best: tuple[str, float] | None = None
    for route, prepare in SOURCES.items():
        made = encode(prepare(values), kind)
        if made.size * made.itemsize != np.asarray(raw).size * np.asarray(raw).itemsize:
            return None
        share = matching_blocks(made, raw, kind, qtype)
        if best is None or share > best[1]:
            best = (route, share)
        if share == 1.0:
            break
    return best


def encode(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "F32":
        return values.astype(np.float32)
    if kind == "F16":
        return values.astype(np.float16)
    return quantize(values, kind)


def matching_blocks(made: np.ndarray, raw: np.ndarray, kind: str, qtype) -> float:
    gguf = need_gguf()
    width = gguf.GGML_QUANT_SIZES[qtype][1]
    a = np.ascontiguousarray(made).view(np.uint8).reshape(-1, width)
    b = np.ascontiguousarray(raw).view(np.uint8).reshape(-1, width)
    return float((a == b).all(axis=1).mean()) if len(a) else 1.0
