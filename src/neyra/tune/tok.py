"""The tokenizer a trainer would load, and what its chat template does."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from jinja2 import TemplateError

from ..store import opener

FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
         "chat_template.json", "added_tokens.json", "generation_config.json", "config.json", "vocab.json",
         "merges.txt", "tokenizer.model", "spiece.model")
Q, A, Q2 = "PROBEQ", "PROBEA", "PROBEQTWO"


class TokenizerError(Exception):
    pass


def resolve(spec: str) -> Path:
    """A folder as given, or the tokenizer and config files of a Hub repo, cached."""
    path = Path(spec).expanduser()
    if path.exists():
        return path
    repo, _, rev = spec.partition("@")
    rev = rev or "main"
    cache = Path(os.environ.get("NEYRA_CACHE", Path.home() / ".cache" / "neyra")) / "tokenizers" / f"{repo.replace('/', '--')}@{rev}"
    if (cache / ".done").exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "neyra"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    got = 0
    for name in FILES:
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", headers=headers)
        try:
            with opener.open(req, timeout=60) as r:
                (cache / name).write_bytes(r.read())
            got += 1
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and name == FILES[0]:
                raise TokenizerError(f"{spec}: the Hub refused the request (HTTP {e.code}); gated repos need HF_TOKEN") from e
            if e.code != 404:
                raise
    if not got or not any((cache / n).exists() for n in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise TokenizerError(f"{spec} is not a folder on disk and the Hub has no tokenizer files under that name")
    (cache / ".done").write_text("")
    return cache


class Tok:
    def __init__(self, spec: str):
        from transformers import AutoTokenizer

        self.spec = spec
        self.path = resolve(spec)
        self.t = AutoTokenizer.from_pretrained(str(self.path))
        self.bos_id, self.eos_id, self.pad_id = self.t.bos_token_id, self.t.eos_token_id, self.t.pad_token_id
        self.template = self.t.chat_template if isinstance(self.t.chat_template, str) else None
        gen = {}
        try:
            gen = json.loads((self.path / "generation_config.json").read_text())
        except (OSError, ValueError):
            pass
        stops = gen.get("eos_token_id", [])
        self.eos_ids = {i for i in ([self.eos_id] + (stops if isinstance(stops, list) else [stops])) if i is not None}
        self.special = set(self.t.all_special_ids) | {
            i for i, a in (getattr(self.t, "added_tokens_decoder", None) or {}).items() if getattr(a, "special", False)}
        self._piece: dict[int, str] = {}
        self.end_ids: set[int] = set(self.eos_ids)
        self.turn_end = ""
        self.instruction_part = self.response_part = None
        self.adds_bos = bool(self.bos_id is not None and self.encode("hello")[:1] == [self.bos_id])
        if self.template:
            self._probe()

    def encode(self, text: str, special: bool = True) -> list[int]:
        return self.t(text, add_special_tokens=special)["input_ids"]

    def encode_many(self, texts: list[str], special: bool, offsets: bool = False):
        out = self.t(texts, add_special_tokens=special, return_offsets_mapping=offsets, truncation=False)
        return (out["input_ids"], out["offset_mapping"]) if offsets else (out["input_ids"], None)

    def render(self, messages: list[dict], gen: bool = False, tools=None) -> str:
        return self.t.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=gen)

    def piece(self, i: int) -> str:
        p = self._piece.get(i)
        if p is None:
            p = self.t.decode([i], skip_special_tokens=False)
            raw = self.t.convert_ids_to_tokens(i)
            if isinstance(raw, str) and raw.startswith("\u2581") and not p.startswith(" "):
                p = " " + p  # sentencepiece pieces lose their leading space when decoded alone
            self._piece[i] = p
        return p

    def blank(self, i: int) -> bool:
        return not self.piece(i).strip()

    def _probe(self):
        user, answer = {"role": "user", "content": Q}, {"role": "assistant", "content": A}
        try:
            text = self.render([user, answer])
            at = text.rfind(A)
            self.turn_end = text[at + len(A):]
            for i in self.encode(self.turn_end, False):
                if i in self.special:
                    self.end_ids.add(i)
                    break
            first = self.render([user])
            with_prompt = self.render([user], gen=True)
            if with_prompt.startswith(first) and len(with_prompt) > len(first):
                self.response_part = with_prompt[len(first):]
            two = self.render([user, answer, {"role": "user", "content": Q2}])
            a, b = two.find(A) + len(A) + len(self.turn_end), two.find(Q2)
            if 0 < a <= b:
                self.instruction_part = two[a:b]
        except (TemplateError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            pass  # a template that rejects the probe leaves the markers unknown

    @property
    def has_generation_markers(self) -> bool:
        return bool(self.template and "generation" in self.template and "endgeneration" in self.template)
