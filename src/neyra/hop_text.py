"""The parts of a model that are not weights: tokenizer, chat template, special tokens, config.

These are read from a model directory or from the metadata of a GGUF file into one shape, so the
same comparison works between any two hops.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .gguf_io import GGUFFile
from .store import Checkpoint

# Keys that mean the same in a Hugging Face config and in GGUF metadata.
SHAPE = ("hidden", "layers", "heads", "kv_heads", "ff", "context", "rope_theta", "rms_eps")
HF_SHAPE = {"hidden_size": "hidden", "num_hidden_layers": "layers", "num_attention_heads": "heads",
            "num_key_value_heads": "kv_heads", "intermediate_size": "ff", "max_position_embeddings": "context",
            "rope_theta": "rope_theta", "rms_norm_eps": "rms_eps"}
GGUF_SHAPE = {"hidden": "embedding_length", "layers": "block_count", "heads": "attention.head_count",
              "kv_heads": "attention.head_count_kv", "ff": "feed_forward_length", "context": "context_length",
              "rope_theta": "rope.freq_base", "rms_eps": "attention.layer_norm_rms_epsilon"}
# Fields that change on every save and say nothing about the model.
NOISE = {"_name_or_path", "transformers_version", "use_cache", "_commit_hash", "unsloth_version", "unsloth_fixed"}
DTYPE_KEYS = {"torch_dtype", "dtype"}
QUANT_KEYS = {"quantization_config", "quantization"}
CONTEXT = {"context", "rope_theta"}


@dataclass
class Finding:
    hop: int
    part: str
    cause: str
    problem: bool
    text: str


@dataclass
class Text:
    tokens: list[str] | None = None
    merges: str | None = None
    tokenizer_digest: str | None = None
    template: str | None = None
    special: dict[str, str | None] = field(default_factory=dict)
    add_bos: bool | None = None
    eos_ids: set[int] = field(default_factory=set)
    shape: dict = field(default_factory=dict)
    config: dict | None = None
    generation: dict | None = None
    has_config: bool = False
    flags: dict = field(default_factory=dict)

    @property
    def has_tokenizer(self) -> bool:
        return self.tokens is not None or self.tokenizer_digest is not None


def _json(raw: str | None):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _token_text(value) -> str | None:
    return value.get("content") if isinstance(value, dict) else value


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _merge_text(merges) -> str:
    return "\n".join(m if isinstance(m, str) else " ".join(m) for m in merges)


def _vocabulary(tokenizer: dict) -> list[str] | None:
    model = tokenizer.get("model") or {}
    vocab = model.get("vocab")
    table: dict[int, str] = {}
    if isinstance(vocab, dict):
        table = {i: t for t, i in vocab.items()}
    elif isinstance(vocab, list):
        table = {i: (t[0] if isinstance(t, list) else t) for i, t in enumerate(vocab)}
    for added in tokenizer.get("added_tokens") or []:
        table[added["id"]] = added["content"]
    if not table:
        return None
    return [table.get(i, "") for i in range(max(table) + 1)]


def read_dir(ck: Checkpoint) -> Text:
    out = Text()
    config = _json(ck.text("config.json"))
    if config is not None:
        out.has_config = True
        flat = {**config, **(config.get("text_config") or {})}
        out.shape = {HF_SHAPE[k]: flat[k] for k in HF_SHAPE if k in flat}
        out.config = {k: v for k, v in config.items() if k not in NOISE}
    gen = _json(ck.text("generation_config.json"))
    if gen is not None:
        out.generation = {k: v for k, v in gen.items() if k not in NOISE}
        ids = gen.get("eos_token_id")
        out.eos_ids = set(ids if isinstance(ids, list) else [] if ids is None else [ids])
    tok = _json(ck.text("tokenizer.json"))
    if tok is not None:
        out.tokens = _vocabulary(tok)
        merges = (tok.get("model") or {}).get("merges")
        out.merges = _merge_text(merges) if merges else None
        out.tokenizer_digest = _digest(tok.get("normalizer"), tok.get("pre_tokenizer"), tok.get("decoder"))
    else:
        for name in ("tokenizer.model", "vocab.json"):
            raw = ck.raw(name)
            if raw is not None:
                out.tokenizer_digest = hashlib.sha256(raw).hexdigest()[:16]
                break
    cfg = _json(ck.text("tokenizer_config.json")) or {}
    template = ck.text("chat_template.jinja")
    if template is None:
        template = cfg.get("chat_template")
    if isinstance(template, list):
        template = json.dumps(template, sort_keys=True)
    out.template = template
    out.special = {k: _token_text(cfg.get(f"{k}_token")) for k in ("bos", "eos", "pad", "unk") if f"{k}_token" in cfg}
    if isinstance(cfg.get("add_bos_token"), bool):
        out.add_bos = cfg["add_bos_token"]
    out.flags = {k: cfg[k] for k in ("add_eos_token", "legacy", "add_prefix_space") if isinstance(cfg.get(k), bool)}
    sp = ck.raw("tokenizer.model")
    if sp is not None:
        out.tokenizer_digest = _digest(out.tokenizer_digest, hashlib.sha256(sp).hexdigest()[:16])
    return out


def read_gguf(g: GGUFFile) -> Text:
    out = Text(has_config=True)
    tokens = g.meta.get("tokenizer.ggml.tokens")
    if tokens:
        out.tokens = list(tokens)
        out.tokenizer_digest = "gguf:" + str(g.meta.get("tokenizer.ggml.model"))
    merges = g.meta.get("tokenizer.ggml.merges")
    out.merges = _merge_text(merges) if merges else None
    out.template = g.meta.get("tokenizer.chat_template")
    if isinstance(g.meta.get("tokenizer.ggml.add_bos_token"), bool):
        out.add_bos = g.meta["tokenizer.ggml.add_bos_token"]
    for key, label in (("bos", "bos"), ("eos", "eos"), ("padding", "pad"), ("unknown", "unk")):
        i = g.meta.get(f"tokenizer.ggml.{key}_token_id")
        if i is not None and tokens and i < len(tokens):
            out.special[label] = tokens[i]
    if g.meta.get("tokenizer.ggml.eos_token_id") is not None:
        out.eos_ids = {g.meta["tokenizer.ggml.eos_token_id"]}
    for k, suffix in GGUF_SHAPE.items():
        value = g.key(suffix)
        if value is not None:
            out.shape[k] = value
    return out


def compared(before: Text, after: Text) -> set[str]:
    """The parts that both texts carry, so a part that nobody has is shown as absent rather than as matching."""
    out = set()
    if before.has_tokenizer and after.has_tokenizer:
        out.add("tokenizer")
    if before.template is not None and after.template is not None:
        out.add("template")
    if (before.add_bos is not None and after.add_bos is not None) or set(before.flags) & set(after.flags) or set(before.special) & set(after.special):
        out.add("bos")
    if before.has_config and after.has_config:
        out.add("config")
    if before.generation is not None and (after.generation is not None or after.eos_ids):
        out.add("generation")
    return out


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-12)
    return a == b


def _short(value, n: int = 60) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= n else text[:n - 1] + "..."


def _tokens(hop: int, before: list[str], after: list[str], padded: bool) -> list[Finding]:
    """A converter pads the vocabulary up to the embedding size, so a longer list from a GGUF is not a change."""
    n = min(len(before), len(after))
    padding = padded or all(t.startswith("[PAD") for t in after[n:])
    bad = [i for i in range(n) if before[i] != after[i] and before[i] != ""]
    if len(after) != len(before) and not (padding and len(after) > len(before)):
        out = [Finding(hop, "tokenizer", "tokenizer", True, f"the vocabulary has {len(after):,} tokens where the one before has {len(before):,}")]
    else:
        out = []
    if bad:
        i = bad[0]
        out.append(Finding(hop, "tokenizer", "tokenizer", True,
                           f"{len(bad):,} tokens differ from the one before, the first is id {i}: "
                           f"{_short(before[i], 24)!r} became {_short(after[i], 24)!r}"))
    return out


def compare(hop: int, before: Text, after: Text, model: bool = True) -> list[Finding]:
    """What changed between the nearest earlier artifact that carries each part and this one.

    `model` is false for an adapter, which is allowed to carry none of these files.
    """
    out: list[Finding] = []

    if before.has_tokenizer and not after.has_tokenizer:
        if model:
            out.append(Finding(hop, "tokenizer", "tokenizer", True, "no tokenizer files, the one before has them"))
    elif after.has_tokenizer and before.has_tokenizer:
        if before.tokens and after.tokens:
            out += _tokens(hop, before.tokens, after.tokens, (after.tokenizer_digest or "").startswith("gguf:"))
        if before.merges and after.merges and before.merges != after.merges:
            out.append(Finding(hop, "tokenizer", "tokenizer", True, "the BPE merge list differs from the one before"))
        elif before.tokenizer_digest and after.tokenizer_digest and before.tokenizer_digest != after.tokenizer_digest \
                and not (before.tokenizer_digest + after.tokenizer_digest).count("gguf:"):
            out.append(Finding(hop, "tokenizer", "tokenizer", True,
                               "the tokenizer's rules or sentencepiece model differ from the one before"))

    if before.template is not None and after.template is None:
        if model:
            out.append(Finding(hop, "template", "template", True, "the chat template is gone"))
    elif before.template is not None and after.template is not None \
            and before.template.replace("\r\n", "\n") != after.template.replace("\r\n", "\n"):
        out.append(Finding(hop, "template", "template", True, "the chat template differs from the one before"))

    if before.add_bos is not None and after.add_bos is not None and before.add_bos != after.add_bos:
        out.append(Finding(hop, "bos", "bos", True, f"add_bos_token went from {before.add_bos} to {after.add_bos}"))
    for key in ("bos", "eos"):
        a, b = before.special.get(key), after.special.get(key)
        if a is not None and b is not None and a != b:
            out.append(Finding(hop, "bos", "bos", True, f"the {key.upper()} token went from {a!r} to {b!r}"))
    for key in sorted(set(before.flags) & set(after.flags)):
        if before.flags[key] != after.flags[key]:
            out.append(Finding(hop, "bos", "bos", True, f"{key} went from {before.flags[key]} to {after.flags[key]}"))
    pad_a, pad_b = before.special.get("pad"), after.special.get("pad")
    if pad_a is not None and pad_b is not None and pad_a != pad_b:
        out.append(Finding(hop, "bos", "bos", False, f"the pad token went from {pad_a!r} to {pad_b!r}"))

    out += _config(hop, before, after, model)
    out += _generation(hop, before, after)
    return out


def _config(hop: int, before: Text, after: Text, model: bool) -> list[Finding]:
    out = []
    for key in SHAPE:
        a, b = before.shape.get(key), after.shape.get(key)
        if a is not None and b is not None and not _close(a, b):
            out.append(Finding(hop, "config", "config", key not in CONTEXT, f"{key} is {b} where the one before has {a}"))
    if before.config is not None and after.config is not None:
        for key in sorted((set(before.config) | set(after.config)) - NOISE):
            a, b = before.config.get(key), after.config.get(key)
            if a == b or key in HF_SHAPE:
                continue
            if key in QUANT_KEYS and b is not None:
                out.append(Finding(hop, "config", "quant_config", True,
                                   f"config.json still carries {key} ({_short(b, 50)}), which the weights do not match"))
            elif key in QUANT_KEYS:
                continue
            elif key in DTYPE_KEYS:
                out.append(Finding(hop, "config", "dtype", False, f"{key} went from {a} to {b}"))
            else:
                out.append(Finding(hop, "config", "config", False, f"{key}: {_short(a)} became {_short(b)}"))
    elif before.has_config and not after.has_config and model:
        out.append(Finding(hop, "config", "config", True, "no config.json, the one before has it"))
    return out


def _generation(hop: int, before: Text, after: Text) -> list[Finding]:
    out = []
    if before.generation is not None and after.generation is not None:
        for key in sorted(set(before.generation) | set(after.generation)):
            a, b = before.generation.get(key), after.generation.get(key)
            if a != b:
                stop = key in ("eos_token_id", "bos_token_id", "pad_token_id")
                out.append(Finding(hop, "generation", "generation", stop, f"generation_config {key}: {_short(a)} became {_short(b)}"))
    elif before.generation is not None and after.eos_ids and before.eos_ids and not after.eos_ids <= before.eos_ids:
        out.append(Finding(hop, "generation", "generation", True,
                           f"stops on token {sorted(after.eos_ids)}, generation_config stops on {sorted(before.eos_ids)}"))
    return out
