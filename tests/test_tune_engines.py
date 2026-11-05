"""The torch engine on a tiny random model: an adapter merges the way a merged folder does, and the arms can fail."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import tinytok

from neyra.check import Adapter, peft, targets
from neyra.patch import write_safetensors
from neyra.store import Checkpoint
from neyra.tune import cases as cases_mod
from neyra.tune import damage, prompts, verdict
from neyra.tune.engines import TorchEngine, open_backends
from neyra.tune.prompts import Format


def bf16(t):
    return t.detach().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root = tmp_path_factory.mktemp("engine")
    base = root / "base"
    tinytok.make(base, tinytok.CHATML)
    torch.manual_seed(5)
    config = Qwen3Config(hidden_size=32, num_hidden_layers=2, intermediate_size=64, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, vocab_size=300, max_position_embeddings=512,
                         tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config)
    write_safetensors(base / "model.safetensors", {k: bf16(v) for k, v in model.state_dict().items()},
                      {"format": "pt"}, "BF16")
    (base / "config.json").write_text(config.to_json_string())

    def adapter(name, scale):
        rng = np.random.default_rng(3)
        d = root / name
        d.mkdir()
        tensors = {}
        for layer in range(2):
            for proj, (out, inn) in {"self_attn.q_proj": (32, 32), "mlp.down_proj": (32, 64)}.items():
                m = f"base_model.model.model.layers.{layer}.{proj}"
                tensors[f"{m}.lora_A.weight"] = rng.standard_normal((4, inn)).astype(np.float32) * 0.5
                tensors[f"{m}.lora_B.weight"] = rng.standard_normal((out, 4)).astype(np.float32) * scale
        write_safetensors(d / "adapter_model.safetensors", tensors)
        (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 4, "lora_alpha": 8,
                                                          "target_modules": ["q_proj", "down_proj"]}))
        return str(d)

    return {"base": str(base), "big": adapter("big", 1.0), "zero": adapter("zero", 0.0), "root": root}


def case_prompts(n=12):
    fmt = Format(True)
    return [prompts.render(fmt, cases_mod.Case(str(i), f"tides and bread {i}", "x")) for i in range(n)]


def test_adapter_and_merged_folder_give_the_same_text(world):
    base, big, root = world["base"], world["big"], world["root"]
    ck, ad = Checkpoint.open(base), Checkpoint.open(big)
    adapter = Adapter(ad)
    merged = {n: (peft(ck.tensors[n].dtype, ck.tensors[n].read(), adapter.term(t))) for n, t in targets(adapter, ck).items()}
    folder = root / "merged"
    folder.mkdir(exist_ok=True)
    write_safetensors(folder / "model.safetensors", {n: (merged[n] if n in merged else r.read())
                                                     for n, r in ck.tensors.items()}, {"format": "pt"}, "BF16")
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (folder / f).write_bytes((root / "base" / f).read_bytes())
    a, b = TorchEngine(base, big, 6), TorchEngine(base, str(folder), 6)
    ps = case_prompts()
    assert a.generator("tuned")(ps) == b.generator("tuned")(ps)
    assert a.generator("tuned")(ps) != a.generator("base")(ps), "the adapter changed nothing"


def test_switching_back_restores_the_base(world):
    e = TorchEngine(world["base"], world["big"], 6)
    ps = case_prompts()
    first = e.generator("base")(ps)
    e.generator("tuned")(ps)
    assert e.generator("base")(ps) == first


def test_kl_is_zero_for_an_adapter_that_does_nothing_and_positive_for_one_that_does(world):
    texts = damage.NEUTRAL[:3]
    assert max(TorchEngine(world["base"], world["zero"], 4).kl(texts)) < 1e-6
    assert min(TorchEngine(world["base"], world["big"], 4).kl(texts)) > 1e-4


def test_a_model_that_equals_the_base_has_nothing_to_test(world):
    with pytest.raises(Exception, match="nothing to test"):
        TorchEngine(world["base"], world["base"], 4)


def test_verdict_end_to_end_on_the_tiny_model(world):
    b = open_backends(world["base"], world["big"], [], 5)
    fmt = Format(True)
    ps_cases = [cases_mod.Case(str(i), f"tides and bread {i}", "") for i in range(40)]
    ps = [prompts.render(fmt, c) for c in ps_cases]
    tuned_says = [prompts.cut(o, ()) for o in b.generate["tuned"](ps)]
    base_says = [prompts.cut(o, ()) for o in b.generate["base"](ps)]
    assert tuned_says != base_says

    def run(expected):
        cs = [cases_mod.Case(c.id, c.input, e) for c, e in zip(ps_cases, expected)]
        return verdict.run(cs, b, fmt, cases_mod.scorer("exact"), "exact", cs[:2], 0.05, "tiny")

    try:
        right = run(tuned_says)
        wrong = run(base_says)
    finally:
        b.close()
    assert right.scores["tuned"] == [1.0] * 40 and right.verdict in ("worked", "prompt_enough", "cannot_tell")
    assert wrong.scores["base"] == [1.0] * 40 and wrong.verdict in ("worse_than_base", "did_not_work")
    assert not wrong.passed
