"""Planted failures between a fine-tune and the model that runs it, on a 2 layer random Llama.

Each case rebuilds a failure that people reported as a bug: the merge wrote the base, a quantization
config stayed behind, the chat template changed on the way. The case says which artifact is the first
that differs and what the cause is. `bench/hop_suite.py` scores a tool against all of them.

The merge arithmetic here is written out by hand and shares no code with neyra, so a bug in neyra's
merge recipes cannot hide in the fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neyra.numerics import from_float, to_float
from neyra.patch import write_safetensors

HIDDEN, FF, HEADS, KV_HEADS, LAYERS, VOCAB, RANK = 32, 64, 4, 2, 2, 64, 4
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
OTHER_TEMPLATE = ("{% for m in messages %}[{{ m['role'] | upper }}] {{ m['content'] }}\n{% endfor %}"
                  "{% if add_generation_prompt %}[ASSISTANT] {% endif %}")
EDITED = ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")


def bf16(x) -> np.ndarray:
    return from_float(np.asarray(x, dtype=np.float32), "BF16")


def shapes() -> dict[str, tuple[int, ...]]:
    kv = HIDDEN // HEADS * KV_HEADS
    out = {"model.embed_tokens.weight": (VOCAB, HIDDEN), "lm_head.weight": (VOCAB, HIDDEN),
           "model.norm.weight": (HIDDEN,)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        out |= {p + "input_layernorm.weight": (HIDDEN,), p + "post_attention_layernorm.weight": (HIDDEN,),
                p + "self_attn.q_proj.weight": (HIDDEN, HIDDEN), p + "self_attn.k_proj.weight": (kv, HIDDEN),
                p + "self_attn.v_proj.weight": (kv, HIDDEN), p + "self_attn.o_proj.weight": (HIDDEN, HIDDEN),
                p + "mlp.gate_proj.weight": (FF, HIDDEN), p + "mlp.up_proj.weight": (FF, HIDDEN),
                p + "mlp.down_proj.weight": (HIDDEN, FF)}
    return out


def vocab() -> list[str]:
    return ["<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"] + [f"t{i}" for i in range(VOCAB - 5)]


def tokenizer_json(tokens: list[str]) -> str:
    special = {"<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"}
    return json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(tokens)},
                                                   "merges": []},
                       "added_tokens": [{"id": i, "content": t, "special": True}
                                        for i, t in enumerate(tokens) if t in special]})


def text_files() -> dict[str, str]:
    return {
        "config.json": json.dumps({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": HIDDEN,
            "intermediate_size": FF, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": False,
            "torch_dtype": "bfloat16", "bos_token_id": 0, "eos_token_id": 1}),
        "generation_config.json": json.dumps({"bos_token_id": 0, "eos_token_id": 1, "do_sample": True,
                                              "temperature": 0.7}),
        "tokenizer.json": tokenizer_json(vocab()),
        "tokenizer_config.json": json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
                                             "add_bos_token": True, "chat_template": TEMPLATE}),
    }


@dataclass
class World:
    """Everything a fine-tune pipeline leaves behind, before any hop goes wrong."""

    rng: np.random.Generator
    dtype: str = "BF16"
    strength: float = 0.05
    base: dict[str, np.ndarray] = field(default_factory=dict)
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    alpha: float = 8.0
    text: dict[str, str] = field(default_factory=text_files)
    adapter_extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for name, shape in shapes().items():
            scale = 1.0 if "norm" in name else 0.5
            self.base[name] = bf16(self.rng.standard_normal(shape) * scale)
        for i in range(LAYERS):
            for mod in EDITED:
                name = f"model.layers.{i}.{mod}.weight"
                out, inn = shapes()[name]
                self.pairs[name] = (self.rng.standard_normal((RANK, inn)).astype(np.float32) * 0.3,
                                    self.rng.standard_normal((out, RANK)).astype(np.float32) * self.strength)

    def delta(self, name: str) -> np.ndarray:
        a, b = self.pairs[name]
        return (b @ a) * np.float32(self.alpha / RANK)

    def merged(self, names=None, under: dict | None = None) -> dict[str, np.ndarray]:
        """base plus adapter, one float32 add and one rounding. `under` swaps in other starting weights."""
        out = {}
        for name, bits in self.base.items():
            start = to_float(bits, "BF16") if under is None else under[name]
            if name in self.pairs and (names is None or name in names):
                start = start + self.delta(name)
            out[name] = start.astype(np.float32) if self.dtype == "F32" else from_float(start, self.dtype)
        return out


def write_model(root: Path, tensors: dict[str, np.ndarray], files: dict[str, str], dtype: str = "BF16") -> Path:
    root.mkdir(parents=True)
    write_safetensors(root / "model.safetensors", tensors, {"format": "pt"}, dtype)
    for path, body in files.items():
        (root / path).write_text(body)
    return root


def write_adapter(root: Path, w: World, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True)
    key = "base_model.model."
    tensors = {}
    for name, (a, b) in w.pairs.items():
        module = name.removesuffix(".weight")
        tensors[f"{key}{module}.lora_A.weight"], tensors[f"{key}{module}.lora_B.weight"] = a, b
    tensors |= w.adapter_extra
    write_safetensors(root / "adapter_model.safetensors", tensors)
    (root / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": RANK, "lora_alpha": w.alpha, "base_model_name_or_path": "tiny/base",
        "target_modules": ["q_proj", "v_proj", "down_proj"]}))
    for path, body in (w.text if files is None else files).items():
        if path in ("tokenizer.json", "tokenizer_config.json"):
            (root / path).write_text(body)
    return root


def permute(x: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's convert_hf_to_gguf LlamaModel.permute: interleave the two rotary halves of each head."""
    return x.reshape(heads, 2, x.shape[0] // heads // 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)


GGUF_NAMES = {
    "model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",
    "model.norm.weight": "output_norm.weight",
    "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def gguf_name(name: str) -> str:
    if name in GGUF_NAMES:
        return GGUF_NAMES[name]
    layer, rest = name.removeprefix("model.layers.").split(".", 1)
    return f"blk.{layer}.{GGUF_NAMES[rest.removesuffix('.weight')]}.weight"


def write_gguf(path: Path, tensors: dict[str, np.ndarray], files: dict[str, str], kind: str = "Q8_0",
               template: str | None = ..., add_bos: bool | None = True, tokens: list[str] | None = None,
               only_dtype: str = "BF16") -> Path:
    """What convert_hf_to_gguf (plus llama-quantize) writes for the tiny Llama. `tensors` are bf16 or f16 bits."""
    import gguf

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(LAYERS)
    writer.add_embedding_length(HIDDEN)
    writer.add_feed_forward_length(FF)
    writer.add_context_length(256)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(VOCAB)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens or vocab())
    writer.add_token_types([1] * VOCAB)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    if add_bos is not None:
        writer.add_add_bos_token(add_bos)
    chat = json.loads(files["tokenizer_config.json"]).get("chat_template") if template is ... else template
    if chat is not None:
        writer.add_chat_template(chat)
    types = gguf.GGMLQuantizationType
    for name, bits in tensors.items():
        values = to_float(bits, only_dtype).astype(np.float32)
        if name.endswith("q_proj.weight"):
            values = permute(values, HEADS)
        elif name.endswith("k_proj.weight"):
            values = permute(values, KV_HEADS)
        small = values.ndim == 1
        want = types.F32 if small else getattr(types, kind)
        if want == types.F32:
            writer.add_tensor(gguf_name(name), values)
        elif want == types.F16:
            writer.add_tensor(gguf_name(name), values.astype(np.float16))
        else:
            writer.add_tensor(gguf_name(name), gguf.quants.quantize(values, want), raw_dtype=want)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@dataclass
class Case:
    name: str
    thread: str
    why: str
    first: str | None  # label of the first artifact that differs, None for a control
    cause: str | None
    build: Callable[[Path, np.random.Generator], list[Path]]
    kill_set: bool = True  # one of the 14 the bet counts


def standard(tmp: Path, w: World, merged=None, merged_files=None, gguf=None) -> list[Path]:
    """base, adapter, merged, in hop order, plus a GGUF when asked."""
    paths = [write_model(tmp / "base", w.base, w.text), write_adapter(tmp / "adapter", w)]
    files = w.text if merged_files is None else merged_files
    tensors = w.merged() if merged is None else merged
    paths.append(write_model(tmp / "merged", tensors, files, w.dtype))
    if gguf:
        paths.append(gguf(tmp, tensors, files))
    return paths


def without(files: dict[str, str], *drop: str) -> dict[str, str]:
    return {k: v for k, v in files.items() if k not in drop}


def edit_config(files: dict[str, str], **changes) -> dict[str, str]:
    config = json.loads(files["config.json"]) | changes
    return files | {"config.json": json.dumps(config)}


def edit_tokenizer_config(files: dict[str, str], **changes) -> dict[str, str]:
    config = {k: v for k, v in (json.loads(files["tokenizer_config.json"]) | changes).items() if v is not ...}
    return files | {"tokenizer_config.json": json.dumps(config)}


def dequantized_4bit(values: np.ndarray, block: int = 64) -> np.ndarray:
    """A 4 bit copy of a weight read back as floats: per block absmax, 15 levels."""
    flat = values.reshape(-1, block)
    top = np.abs(flat).max(axis=1, keepdims=True) / 7
    return (np.round(flat / np.maximum(top, 1e-12)) * top).reshape(values.shape)


# ---- the cases ----------------------------------------------------------------------------------

def merge_wrote_base(tmp, rng):
    w = World(rng)
    return standard(tmp, w, merged=dict(w.base))


def delta_rounds_away(tmp, rng):
    """unsloth 2142: a tiny adapter merged into fp16. The merge is right and changes nothing."""
    w = World(rng, dtype="F16", strength=1e-6)
    return standard(tmp, w)


def partial_merge(tmp, rng):
    w = World(rng)
    done = {n for n in w.pairs if n.startswith("model.layers.0.")}
    return standard(tmp, w, merged=w.merged(names=done))


def quant_config_left(tmp, rng):
    w = World(rng)
    files = edit_config(w.text, quantization_config={"quant_method": "bitsandbytes", "load_in_4bit": True,
                                                     "bnb_4bit_quant_type": "nf4"})
    return standard(tmp, w, merged_files=files)


def fresh_random_head(tmp, rng):
    """peft 2289: the adapter trained a new head, and the merged model has a freshly initialised one."""
    w = World(rng)
    head = bf16(rng.standard_normal((3, HIDDEN)) * 0.5)
    w.adapter_extra = {"base_model.model.score.modules_to_save.default.weight": head}
    merged = w.merged() | {"score.weight": bf16(rng.standard_normal((3, HIDDEN)) * 0.5)}
    return standard(tmp, w, merged=merged)


def wrong_layer_range(tmp, rng):
    """mergekit 632: a layer range that is too long, so the merged model has a third layer."""
    w = World(rng)
    merged = w.merged()
    for name in shapes():
        if name.startswith("model.layers.1."):
            merged[name.replace("layers.1.", "layers.2.")] = merged[name]
    return standard(tmp, w, merged=merged, merged_files=edit_config(w.text, num_hidden_layers=LAYERS + 1))


def dtype_ignored(tmp, rng):
    """mergekit 604: the merged model is written in float32 though the base is bfloat16."""
    w = World(rng, dtype="F32")
    return standard(tmp, w, merged_files=edit_config(w.text, torch_dtype="float32"))


def truncated(tmp, rng):
    """mergekit 562: a checkpoint a quarter of the size, the last layer and the head never written."""
    w = World(rng)
    merged = {n: t for n, t in w.merged().items() if "layers.1." not in n and n != "lm_head.weight"}
    return standard(tmp, w, merged=merged)


def template_missing(tmp, rng):
    w = World(rng)
    return standard(tmp, w, merged_files=edit_tokenizer_config(w.text, chat_template=...))


def lm_head_lost(tmp, rng):
    """mergekit 521: the output head is not in the merged model, and the config says it is not tied."""
    w = World(rng)
    merged = {n: t for n, t in w.merged().items() if n != "lm_head.weight"}
    return standard(tmp, w, merged=merged)


def merged_into_4bit_base(tmp, rng):
    """unsloth 2374: the adapter was merged into a dequantized 4 bit copy of the base."""
    w = World(rng)
    under = {n: (dequantized_4bit(to_float(b, "BF16")) if b.ndim == 2 else to_float(b, "BF16"))
             for n, b in w.base.items()}
    return standard(tmp, w, merged=w.merged(under=under))


def template_changed(tmp, rng):
    w = World(rng)
    return standard(tmp, w, merged_files=edit_tokenizer_config(w.text, chat_template=OTHER_TEMPLATE))


def tokenizer_changed(tmp, rng):
    w = World(rng)
    tokens = vocab()
    tokens[10], tokens[11] = tokens[11], tokens[10]
    return standard(tmp, w, merged_files=w.text | {"tokenizer.json": tokenizer_json(tokens)})


def bos_missing(tmp, rng):
    w = World(rng)
    return standard(tmp, w, merged_files=edit_tokenizer_config(w.text, add_bos_token=False))


def eos_changed(tmp, rng):
    """The eos_token in tokenizer_config.json is not the one the base and adapter were trained with."""
    w = World(rng)
    return standard(tmp, w, merged_files=edit_tokenizer_config(w.text, eos_token="<|im_end|>"))


def clean(tmp, rng):
    return standard(tmp, World(rng))


def clean_fp16(tmp, rng):
    return standard(tmp, World(rng, dtype="F16"), merged_files=None)


# GGUF hops. The merged model is right in each, and the file that llama.cpp reads is not.
def gguf_clean(kind):
    def build(tmp, rng):
        w = World(rng)
        return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", m, f, kind))
    return build


def gguf_from_base(tmp, rng):
    w = World(rng)
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", w.base, f, "Q8_0"))


def gguf_light_q4(tmp, rng):
    """The adapter is real but small, and Q4_0 steps are much larger than the change."""
    w = World(rng, strength=0.004)
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", m, f, "Q4_0"))


def gguf_template_gone(tmp, rng):
    w = World(rng)
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", m, f, "Q8_0", template=None))


def gguf_other_template(tmp, rng):
    w = World(rng)
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", m, f, "Q8_0", template=OTHER_TEMPLATE))


def gguf_no_bos(tmp, rng):
    w = World(rng)
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", m, f, "Q8_0", add_bos=False))


def gguf_other_vocab(tmp, rng):
    w = World(rng)
    tokens = vocab()
    tokens[20] = "changed"
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", m, f, "Q8_0", tokens=tokens))


def gguf_one_layer_stale(tmp, rng):
    """One layer of the GGUF still holds the base weights."""
    w = World(rng)
    stale = {n: (w.base[n] if n.startswith("model.layers.1.") else t) for n, t in w.merged().items()}
    return standard(tmp, w, gguf=lambda t, m, f: write_gguf(t / "model.gguf", stale, f, "Q8_0"))


CASES = [
    Case("merge_wrote_base", "unsloth 2009", "the merged model is the base", "merged", "never_applied", merge_wrote_base),
    Case("delta_rounds_away", "unsloth 2142", "a correct fp16 merge of an adapter too small for fp16", "merged",
         "rounded_away", delta_rounds_away),
    Case("partial_merge", "unsloth 1378", "one layer of adapter edits was not merged", "merged", "partly_applied",
         partial_merge),
    Case("quant_config_left", "mlx-lm 223", "quantization_config left on an fp16 model", "merged", "quant_config",
         quant_config_left),
    Case("fresh_random_head", "peft 2289", "the trained head is not the one in the merged file", "merged", "head",
         fresh_random_head),
    Case("wrong_layer_range", "mergekit 632", "the merged model has an extra layer", "merged", "extra_tensors",
         wrong_layer_range),
    Case("dtype_ignored", "mergekit 604", "merged in float32, base is bfloat16", "merged", "dtype", dtype_ignored),
    Case("truncated", "mergekit 562", "a layer and the head were never written", "merged", "missing_tensors", truncated),
    Case("template_missing", "mergekit 589", "the chat template is gone", "merged", "template", template_missing),
    Case("lm_head_lost", "mergekit 521", "no lm_head in an untied model", "merged", "missing_tensors", lm_head_lost),
    Case("merged_into_4bit_base", "unsloth 2374", "merged into a dequantized 4 bit base", "merged", "other_base",
         merged_into_4bit_base),
    Case("template_changed", "unsloth 1519", "a different chat template", "merged", "template", template_changed),
    Case("tokenizer_changed", "unsloth 1073", "a different vocabulary", "merged", "tokenizer", tokenizer_changed),
    Case("bos_missing", "llama.cpp 7062", "add_bos_token turned off", "merged", "bos", bos_missing),
    Case("eos_changed", "llama.cpp 7062", "eos_token changed in tokenizer_config.json", "merged", "bos", eos_changed, False),
    Case("gguf_from_base", "unsloth 2009", "the GGUF holds the base weights", "gguf", "lost_in_conversion",
         gguf_from_base, False),
    Case("gguf_stale_layer", "unsloth 1378", "one layer of the GGUF is the base", "gguf", "partly_applied",
         gguf_one_layer_stale, False),
    Case("gguf_template_gone", "unsloth 1073", "the GGUF has no chat template", "gguf", "template",
         gguf_template_gone, False),
    Case("gguf_other_template", "unsloth 1519", "the GGUF has another chat template", "gguf", "template",
         gguf_other_template, False),
    Case("gguf_no_bos", "llama.cpp 7062", "add_bos_token is off in the GGUF", "gguf", "bos", gguf_no_bos, False),
    Case("gguf_other_vocab", "llama.cpp 7062", "a token differs in the GGUF vocabulary", "gguf", "tokenizer",
         gguf_other_vocab, False),
]
CONTROLS = [
    Case("clean_bf16", "", "nothing wrong", None, None, clean, False),
    Case("clean_fp16", "", "fp16 merge, nothing wrong", None, None, clean_fp16, False),
    Case("clean_gguf_q8", "", "Q8_0 of the merged model", None, None, gguf_clean("Q8_0"), False),
    Case("clean_gguf_f16", "", "F16 of the merged model", None, None, gguf_clean("F16"), False),
    Case("clean_gguf_bf16", "", "BF16 of the merged model", None, None, gguf_clean("BF16"), False),
    Case("clean_gguf_q4_0", "", "Q4_0 of the merged model", None, None, gguf_clean("Q4_0"), False),
    Case("light_adapter_q4_0", "unsloth 976", "a light adapter in Q4_0: right on average, mostly noise", None, None,
         gguf_light_q4, False),
]
