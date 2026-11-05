"""Compare the tokenizer the data is built with against the one inside the GGUF file the model will be served from."""

from __future__ import annotations

from jinja2 import TemplateError

from .. import hop_text
from ..gguf_io import GGUFFile
from .checks import Finding
from .tok import Tok

CHATS = [
    [{"role": "user", "content": "How are you?"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "How are you?"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Bye"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
]


def _text(tok: Tok) -> hop_text.Text:
    vocab = tok.t.get_vocab()
    tokens = [""] * (max(vocab.values()) + 1)
    for piece, i in vocab.items():
        tokens[i] = piece
    out = hop_text.Text(tokens=tokens, template=tok.template, eos_ids=set(tok.eos_ids))
    out.add_bos = tok.adds_bos
    for key in ("bos", "eos", "pad"):
        value = getattr(tok.t, f"{key}_token", None)
        if value:
            out.special[key] = value
    return out


def renders(tok: Tok, template: str):
    for chat in CHATS:
        for gen in (False, True) if chat[-1]["role"] == "user" else (False,):
            try:
                yield chat, gen, tok.t.apply_chat_template(chat, chat_template=template, tokenize=False,
                                                           add_generation_prompt=gen)
            except (TemplateError, ValueError, TypeError) as e:
                yield chat, gen, f"<error: {e}>"


def compare(tok: Tok, path: str, total: int) -> list[Finding]:
    g = GGUFFile(path)
    theirs = hop_text.read_gguf(g)
    out: list[Finding] = []
    if tok.template and not theirs.template:
        out.append(Finding("template_gguf", "error", 1, 1, "GGUF has no chat template",
                           "the GGUF file has no chat template, but the tokenizer you train with has one",
                           "Runtimes will fall back to a default format that the model was not trained on."))
    elif tok.template and theirs.template:
        same = tok.template.replace("\r\n", "\n") == theirs.template.replace("\r\n", "\n")
        if not same:
            for (chat, gen, mine), (_, _, other) in zip(renders(tok, tok.template), renders(tok, theirs.template)):
                if mine != other:
                    out.append(Finding(
                        "template_gguf", "error", 1, 1, "chat template differs from the GGUF's",
                        "the GGUF file renders chats differently from the tokenizer you train with",
                        f"For {len(chat)} message(s)" + (" with a generation prompt" if gen else "") +
                        f", training sees {mine!r} and the GGUF gives {other!r}."))
                    break
    elif theirs.template and not tok.template:
        out.append(Finding("template_gguf", "warning", 1, 1, "tokenizer has no chat template",
                           "the GGUF file has a chat template but the tokenizer you train with does not", ""))
    for f in hop_text.compare(0, _text(tok), theirs):
        if f.problem and f.part in ("tokenizer", "bos", "generation"):
            out.append(Finding("tokenizer_gguf", "error", 1, 1, "tokenizer differs from the GGUF's",
                               f"the GGUF file's tokenizer settings differ: {f.text}", ""))
    return out
