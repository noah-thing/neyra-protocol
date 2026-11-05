"""Small tokenizers with chosen chat templates, trained on the spot so tests need no downloads."""

from __future__ import annotations

import json
from pathlib import Path

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LLAMA = (
    "{{ bos_token }}{% for m in messages %}<|start|>{{ m['role'] }}<|stop|>\n{{ m['content'] }}<|eot|>"
    "{% endfor %}{% if add_generation_prompt %}<|start|>assistant<|stop|>\n{% endif %}")
LLAMA_NO_BOS = LLAMA.replace("{{ bos_token }}", "")
ALPACA = (
    "{% for m in messages %}{% if m['role'] == 'user' %}### Instruction:\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'assistant' %}### Response:\n{{ m['content'] }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}### Response:\n{% endif %}")
GENMARK = (
    "{% for m in messages %}{% if m['role'] == 'assistant' %}<|im_start|>assistant\n{% generation %}{{ m['content'] }}"
    "{% endgeneration %}<|im_end|>\n{% else %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
STRIPTHINK = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['role'] == 'assistant' and '</think>' in m['content'] %}"
    "{{ m['content'].split('</think>')[-1] }}{% else %}{{ m['content'] }}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
OPENTHINK = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}")

CORPUS = [
    "the quick brown fox jumps over the lazy dog", "question answer explain number details works steps",
    "hello world please tell me about tides bread python chess rain coffee trains maps violins glass",
    "you are a helpful tutor who answers briefly and kindly", "system user assistant response instruction",
] * 20


def make(path: Path, template: str, bos: bool = False, end: str = "<|im_end|>", pad: str | None = None,
         extra: tuple[str, ...] = ()) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import AutoTokenizer

    specials = ["<unk>", "<s>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|start|>", "<|stop|>", "<|eot|>",
                "<think>", "</think>", *extra]
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=300, special_tokens=specials,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    if bos:
        t.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B",
                                                         special_tokens=[("<s>", t.token_to_id("<s>"))])
    path.mkdir(parents=True, exist_ok=True)
    t.save(str(path / "tokenizer.json"))
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": end, "unk_token": "<unk>",
              "chat_template": template, "model_max_length": 4096}
    if bos:
        config["bos_token"] = "<s>"
    if pad:
        config["pad_token"] = pad
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    AutoTokenizer.from_pretrained(str(path))
    return str(path)
