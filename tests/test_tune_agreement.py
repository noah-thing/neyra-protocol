"""Token ids and loss masks against the real trainers, on a few small rows and real tokenizers.

Each test skips unless the trainer is installed and the tokenizer files can be fetched, so the default suite stays
offline. The versions matched are in docs/tune-lint.md.
"""

from __future__ import annotations

import importlib.metadata as meta
import json

import pytest
from jinja2 import TemplateError

pytest.importorskip("transformers")

from neyra.tune import data
from neyra.tune.tok import Tok
from neyra.tune.trainers import TRAINERS, Session, Unsupported

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "unsloth/Llama-3.2-1B-Instruct",
          "HuggingFaceTB/SmolLM2-135M-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "unsloth/gemma-2-2b-it",
          "unsloth/mistral-7b-instruct-v0.3", "microsoft/Phi-3-mini-4k-instruct"]

CHATS = [
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of Sweden?"},
                  {"role": "assistant", "content": "Stockholm."}]},
    {"messages": [{"role": "user", "content": "Say hej, åäö."}, {"role": "assistant", "content": "Hej! 你好"}]},
    {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"},
                  {"role": "user", "content": "and 3+3?"}, {"role": "assistant", "content": "6"}]},
]
PAIRS = [{"prompt": "What is 2+2?", "completion": "4"}, {"prompt": "Name a color.\n", "completion": " Blue"}]
CONVO_PAIRS = [{"prompt": [{"role": "user", "content": "Hi"}], "completion": [{"role": "assistant", "content": "Hello"}]}]
TEXTS = [{"text": "plain words with no template"}, {"text": "ends with newline\n"}]


def rows_of(objs):
    return [data.parse(o, i + 1) for i, o in enumerate(objs)]


def accepted(tok, objs):
    """The rows this template renders; some refuse a system turn."""
    out = []
    for o in objs:
        try:
            if "messages" in o:
                tok.render(o["messages"])
            out.append(o)
        except TemplateError:
            pass
    return out


@pytest.fixture(params=MODELS)
def tok(request):
    try:
        return Tok(request.param)
    except Exception as e:  # noqa: BLE001  no network, gated repo, or a tokenizer this transformers cannot load
        pytest.skip(f"{request.param}: {e}")


def have(name):
    try:
        return meta.version(name)
    except meta.PackageNotFoundError:
        return None


class TestMlx:
    @pytest.fixture(autouse=True)
    def lib(self):
        if not have("mlx-lm"):
            pytest.skip("mlx-lm is not installed")

    @pytest.mark.parametrize("mask", [False, True])
    def test_chat_and_pairs(self, tok, mask):
        from mlx_lm.tuner.datasets import ChatDataset, CompletionsDataset, TextDataset

        if not tok.template:
            pytest.skip("no chat template")
        objs = accepted(tok, CHATS) + PAIRS + TEXTS
        mine = Session(TRAINERS["mlx"], tok, 10**6, mask).build(rows_of(objs))
        for obj, b in zip(objs, mine):
            if "messages" in obj:
                real = ChatDataset([obj], tok.t, mask_prompt=mask).process(obj)
            elif "prompt" in obj:
                real = CompletionsDataset([obj], tok.t, "prompt", "completion", mask).process(obj)
            else:
                real = TextDataset([obj], tok.t).process(obj)
            ids, offset = real
            assert b.ids == list(ids)
            expect = [offset <= i and i >= 1 for i in range(len(ids))]
            if "text" in obj:
                expect = [i >= 1 for i in range(len(ids))]
            assert b.trained == expect


class TestTrl:
    @pytest.fixture(autouse=True)
    def lib(self):
        if not have("trl"):
            pytest.skip("trl is not installed")

    def real(self, tok, objs, tmp_path, **config):
        import torch
        from datasets import Dataset
        from transformers import LlamaConfig, LlamaForCausalLM
        from trl import SFTConfig, SFTTrainer

        size = len(tok.t)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=size, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                              num_attention_heads=2, num_key_value_heads=2,
                                              pad_token_id=tok.t.pad_token_id, eos_token_id=tok.eos_id,
                                              bos_token_id=tok.bos_id))
        torch.manual_seed(0)
        args = SFTConfig(output_dir=str(tmp_path), report_to=[], use_cpu=True, **{"max_length": 10**5, **config})
        trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(objs), processing_class=tok.t)
        return [(r["input_ids"], r["labels"]) for r in trainer.train_dataset]

    @staticmethod
    def same(mine, real):
        for b, (ids, labels) in zip(mine, real):
            assert b.ids == list(ids)
            assert b.trained[1:] == [x != -100 for x in labels][1:]

    @pytest.mark.parametrize("shape", ["chat", "pair", "text", "convo-pair"])
    def test_defaults(self, tok, tmp_path, shape):
        objs = {"chat": CHATS, "pair": PAIRS, "text": TEXTS, "convo-pair": CONVO_PAIRS}[shape]
        if shape in ("chat", "convo-pair") and not tok.template:
            pytest.skip("no chat template")
        objs = accepted(tok, objs)
        mine = Session(TRAINERS["trl"], tok, 10**5).build(rows_of(objs))
        self.same(mine, self.real(tok, objs, tmp_path))

    def test_assistant_only_when_template_marks_generation(self, tok, tmp_path):
        if not tok.has_generation_markers:
            pytest.skip("this template has no generation markers")
        objs = accepted(tok, CHATS)
        mine = Session(TRAINERS["trl"], tok, 10**5, True).build(rows_of(objs))
        self.same(mine, self.real(tok, objs, tmp_path, assistant_only_loss=True))

    def test_truncation(self, tok, tmp_path):
        if not tok.template:
            pytest.skip("no chat template")
        objs = accepted(tok, CHATS)[:1]
        mine = Session(TRAINERS["trl"], tok, 12).build(rows_of(objs))
        real = self.real(tok, objs, tmp_path, max_length=12)
        assert [b.kept for b in mine] == [len(r[0]) for r in real]
        assert mine[0].ids[:12] == list(real[0][0])


class TestUnsloth:
    @pytest.fixture(autouse=True)
    def lib(self):
        if not have("unsloth_zoo"):
            pytest.skip("unsloth_zoo is not installed")

    @pytest.fixture
    def responses_only(self):
        import importlib.util
        import os
        import sys
        import types

        root = os.path.join(os.path.dirname(meta.distribution("unsloth_zoo").locate_file("unsloth_zoo/__init__.py")))
        pkg = types.ModuleType("unsloth_zoo")
        pkg.__path__ = [root]
        sys.modules["unsloth_zoo"] = pkg
        spec = importlib.util.spec_from_file_location("unsloth_zoo.dataset_utils", os.path.join(root, "dataset_utils.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["unsloth_zoo.dataset_utils"] = mod
        spec.loader.exec_module(mod)
        return mod.train_on_responses_only

    def test_mask(self, tok, responses_only):
        if not (tok.template and tok.response_part and tok.instruction_part):
            pytest.skip("markers not found")
        try:
            mine = Session(TRAINERS["unsloth"], tok, 10**5).build(rows_of(accepted(tok, CHATS) + CONVO_PAIRS))
        except Unsupported:
            pytest.skip("markers do not survive tokenization in context; the lint refuses this template")
        fn = responses_only(None, tok.instruction_part, tok.response_part, tokenizer=tok.t, return_function=True)
        for b in mine:
            real = fn({"input_ids": [b.ids]})["labels"][0]
            assert b.trained == [x != -100 for x in real]


class TestAxolotl:
    @pytest.fixture(autouse=True)
    def lib(self):
        if not have("axolotl"):
            pytest.skip("axolotl is not installed")

    def test_chat_template_defaults(self, tok):
        from axolotl.prompt_strategies import chat_template
        from axolotl.utils.dict import DictDefault

        if not tok.template:
            pytest.skip("no chat template")
        if "<think>" in tok.template:
            pytest.skip("axolotl splits the empty thinking block out of the trained span, which is not modelled")
        try:
            tok.render(CHATS[0]["messages"])
        except TemplateError:
            pytest.skip("the template rejects a system turn, which axolotl probes with")
        if "Be brief" not in tok.render(CHATS[0]["messages"]):
            pytest.skip("the template drops the system prompt from a full conversation; the lint reports that")
        cfg = DictDefault({"train_on_inputs": False, "sequence_len": 2048, "tokenizer_use_mistral_common": False,
                           "chat_template": "tokenizer_default"})
        strategy = chat_template.load(tok.t, cfg, {"type": "chat_template"})
        objs = accepted(tok, CHATS)
        mine = Session(TRAINERS["axolotl"], tok).build(rows_of(objs))
        for obj, b in zip(objs, mine):
            real = strategy.tokenize_prompt(obj)
            assert b.ids == real["input_ids"]
            assert b.trained[1:] == [x != -100 for x in real["labels"]][1:]


class TestLlamaFactory:
    """LLaMA-Factory renders with its own named templates, so only a model whose tokenizer template equals its template agrees."""

    @pytest.fixture(autouse=True)
    def lib(self):
        if not have("llamafactory"):
            pytest.skip("llamafactory is not installed")
        import os

        os.environ["DISABLE_VERSION_CHECK"] = "1"

    def test_chatml_qwen(self):
        from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments

        try:
            tok = Tok("Qwen/Qwen2.5-0.5B-Instruct")
        except Exception as e:  # noqa: BLE001
            pytest.skip(str(e))
        args = DataArguments(template="qwen", cutoff_len=2048)
        template = get_template_and_fix_tokenizer(tok.t, args)
        processor = SupervisedDatasetProcessor(template=template, tokenizer=tok.t, processor=None, data_args=args)
        mine = Session(TRAINERS["llama-factory"], tok).build(rows_of(CHATS))
        for obj, b in zip(CHATS, mine):
            m = obj["messages"]
            system = m[0]["content"] if m[0]["role"] == "system" else None
            m = m[1:] if system else m
            ids, labels = processor._encode_data_example(m[:-1], m[-1:], system, None, [], [], [])
            assert b.ids == ids
            assert b.trained[1:] == [x != -100 for x in labels][1:]


def test_versions_recorded(capsys):
    seen = {n: have(n) for n in ("trl", "transformers", "mlx-lm", "unsloth-zoo", "unsloth", "axolotl", "llamafactory")}
    print(json.dumps(seen))
