"""Token ids and loss masks the way each trainer builds them.

TRL and mlx-lm are reproduced step for step from their source and compared with the real packages in
tests/test_tune_trainers.py. Unsloth's mask is rebuilt from the markers its auto-detection finds. Axolotl and
LLaMA-Factory render with their own template code, so theirs follow the tokenizer's template and are approximate.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

from jinja2 import TemplateError

from .data import Row
from .tok import Tok


class Unsupported(Exception):
    pass


PROBE = [{"role": "user", "content": "PROBEQ"}, {"role": "assistant", "content": "PROBEA"},
         {"role": "user", "content": "PROBEQ"}, {"role": "assistant", "content": "PROBEA"}]


@dataclass(frozen=True)
class Trainer:
    name: str
    label: str
    exact: bool
    max_length: int
    length_name: str
    mask_prompt: bool | None        # None: TRL decides by the row shape
    truncation: str                 # "cut" keeps the start, "drop" removes the row
    pad_masks_eos: bool = False


TRAINERS = {
    "trl": Trainer("trl", "TRL SFTTrainer", True, 1024, "max_length", None, "cut"),
    "unsloth": Trainer("unsloth", "Unsloth with train_on_responses_only", False, 2048, "max_seq_length", True, "cut"),
    "axolotl": Trainer("axolotl", "Axolotl chat_template", False, 2048, "sequence_len", True, "drop"),
    "mlx": Trainer("mlx", "mlx-lm lora", True, 2048, "max_seq_length", False, "cut"),
    "llama-factory": Trainer("llama-factory", "LLaMA-Factory sft", False, 2048, "cutoff_len", True, "cut"),
}


@dataclass
class Built:
    row: Row
    ids: list[int]
    trained: list[bool]
    answers: list[tuple[int, int]] | None = None
    text: str | None = None
    prompt_text: str | None = None
    kept: int = 0
    dropped: bool = False
    located: bool = True
    error: str | None = None

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass
class Plan:
    row: Row
    text: str
    special: bool
    aux: list[str] = field(default_factory=list)
    offsets: bool = False
    finish: object = None
    prompt_text: str | None = None
    rendered: bool = False


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _prompt_text(tok: Tok, row: Row, msgs: list[dict]) -> str | None:
    if len(msgs) > 1 and msgs[-1]["role"] == "assistant":
        return tok.render(msgs[:-1], gen=True, tools=row.tools)
    return None


def turn_spans(tok: Tok, row: Row, msgs: list[dict], text: str) -> list[tuple[int, int]] | None:
    """Character range of every assistant turn in the rendered chat, from the template's own prefixes.

    A template may append something only after the last message, so a turn ends where its own render stops
    agreeing with the full text, provided its words are inside.
    """
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if i == 0:
            return None
        head = tok.render(msgs[:i], gen=True, tools=row.tools)
        if not text.startswith(head):
            return None
        tail = text if i == len(msgs) - 1 else tok.render(msgs[:i + 1], tools=row.tools)
        end = 0
        for x, y in zip(tail, text):
            if x != y:
                break
            end += 1
        if end <= len(head) or m["content"].strip() not in text[len(head):end]:
            return None
        spans.append((len(head), end))
    return spans


def token_ranges(offsets, spans) -> list[tuple[int, int]]:
    starts = [s for s, _ in offsets]
    return [(bisect_left(starts, a), bisect_left(starts, b)) for a, b in spans]


def _mask(n: int, ranges, first: int = 1) -> list[bool]:
    out = [False] * n
    for a, b in ranges:
        for i in range(max(a, first), min(b, n)):
            out[i] = True
    return out


def _all(n: int) -> list[bool]:
    return [i >= 1 for i in range(n)]



def _content_and_eos(tok: Tok, ids: list[int], a: int, b: int) -> list[tuple[int, int]]:
    """A turn as Axolotl trains it: the words, then only the tokenizer's own eos token, wherever the template puts it."""
    stop = next((i for i in range(a, b) if ids[i] in tok.end_ids), b)
    out = [(a, stop)]
    for i in range(stop, min(b + 1, len(ids))):
        if ids[i] == tok.eos_id:
            out.append((i, i + 1))
    return out


class Session:
    """One trainer's settings for one run. Choices a trainer makes once, from the first row, are kept here."""

    def __init__(self, trainer: Trainer, tok: Tok, max_length: int | None = None, mask_prompt: bool | None = None):
        self.trainer, self.tok = trainer, tok
        self.limit = max_length or trainer.max_length
        self.mask = trainer.mask_prompt if mask_prompt is None else mask_prompt
        self.add_special: bool | None = None
        self.markers_checked = False

    def build(self, rows: list[Row]) -> list[Built]:
        tok, t = self.tok, self.trainer
        if any(r.shape != "text" for r in rows) and not tok.template:
            raise Unsupported("this tokenizer has no chat template, and the data has conversations or prompt and "
                              "completion pairs")
        one = {"trl": _trl, "mlx": _mlx, "unsloth": self._unsloth, "axolotl": _axolotl,
               "llama-factory": _llama_factory}[t.name]
        plans = []
        for r in rows:
            try:
                plans.append(one(tok, r, self.mask))
            except TemplateError as e:
                plans.append(Built(r, [], [], error=str(e)))
        built = _run(tok, plans)
        for b in built:
            if t.truncation == "drop" and b.n > self.limit:
                b.dropped, b.kept = True, 0
            else:
                b.kept = min(b.n, self.limit)
        return built

    def _unsloth(self, tok: Tok, r: Row, mask: bool | None):
        eos, bos = _eos_text(tok), tok.t.bos_token
        if r.shape == "text":
            text, prompt = r.text, None
        else:
            msgs = r.as_messages()
            text, prompt = tok.render(msgs, tools=r.tools), _prompt_text(tok, r, msgs)
        if self.add_special is None:
            self.add_special = not (bool(bos) and (text.startswith(bos) or bos in (tok.template or "")))
        if mask and (tok.response_part is None or tok.instruction_part is None):
            raise Unsupported("could not find where assistant turns start and end in this chat template")
        resp = tok.encode(tok.response_part, False) if mask else []
        inst = tok.encode(tok.instruction_part, False) if mask else []
        if mask and not self.markers_checked:
            probe = tok.encode(tok.render(PROBE), False)
            if not any(_find(probe, resp, i) for i in range(len(probe))) or \
                    not any(_find(probe, inst, i) for i in range(len(probe))):
                raise Unsupported("the assistant and user markers of this chat template do not tokenize the same way "
                                  "inside a conversation as alone, which the Unsloth mask depends on; not modelled")
            self.markers_checked = True
        text2 = text if not eos or text.endswith(eos) else text + eos

        def finish(ids, off, aux):
            if not mask:
                return _all(len(ids)), None, True
            spans = _response_spans(tok, ids, resp, inst)
            return _mask(len(ids), spans), spans, True
        return Plan(r, text2, self.add_special, rendered=r.shape != "text", prompt_text=prompt, finish=finish)


def _run(tok: Tok, plans: list) -> list[Built]:
    """Encode the planned texts in batches, then let each plan turn its ids into a mask."""
    out: list[Built | None] = [None] * len(plans)
    groups: dict[tuple[bool, bool], list[int]] = {}
    for i, p in enumerate(plans):
        if isinstance(p, Built):
            out[i] = p
        else:
            groups.setdefault((p.special, p.offsets), []).append(i)
    for (special, offsets), at in groups.items():
        for lo in range(0, len(at), 256):
            chunk = at[lo:lo + 256]
            texts = [plans[i].text for i in chunk]
            aux = [t for i in chunk for t in plans[i].aux]
            ids, offs = tok.encode_many(texts, special, offsets)
            aux_ids = tok.encode_many(aux, special)[0] if aux else []
            k = 0
            for j, i in enumerate(chunk):
                p = plans[i]
                mine = aux_ids[k:k + len(p.aux)]
                k += len(p.aux)
                trained, answers, located = p.finish(ids[j], offs[j] if offs else None, mine)
                out[i] = Built(p.row, ids[j], trained, answers, p.text if p.rendered else None, p.prompt_text, located=located)
    return out




def _eos_text(tok: Tok) -> str:
    return tok.t.eos_token or ""


def _trl_pair(only: bool):
    def finish(ids, off, aux):
        n = common_prefix(aux[0], ids)
        return (_mask(len(ids), [(n, len(ids))]) if only else _all(len(ids))), [(n, len(ids))], True
    return finish


def _trl(tok: Tok, r: Row, mask: bool | None):
    eos = _eos_text(tok)
    if r.shape == "text":
        text = r.text if not eos or r.text.endswith(eos) else r.text + eos
        return Plan(r, text, True, finish=lambda ids, off, aux: (_all(len(ids)), None, True))
    if r.shape == "messages":
        msgs = r.messages
        text = tok.render(msgs, tools=r.tools)
        prompt = _prompt_text(tok, r, msgs)
        if not mask:
            return Plan(r, text, False, offsets=True, rendered=True, prompt_text=prompt,
                        finish=_with_answers(tok, r, msgs, text))
        if not tok.has_generation_markers:
            raise Unsupported("assistant_only_loss needs {% generation %} markers in the chat template; TRL swaps in "
                              "its own template for some models, which this lint does not model")
        got = tok.t.apply_chat_template(msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                                        tools=r.tools or None)
        flags = list(got["assistant_masks"])
        runs, start = [], None
        for i, f in enumerate(flags + [0]):
            if f and start is None:
                start = i
            if not f and start is not None:
                runs.append((start, i))
                start = None
        return Built(r, list(got["input_ids"]), [bool(f) and i >= 1 for i, f in enumerate(flags)], runs, text, prompt)
    only = True if mask is None else mask
    if isinstance(r.prompt, list):
        prompt = tok.render(r.prompt, gen=True, tools=r.tools)
        text = tok.render(r.as_messages(), tools=r.tools)
        return Plan(r, text, False, aux=[prompt], rendered=True, prompt_text=prompt, finish=_trl_pair(only))
    completion = r.completion if not eos or r.completion.endswith(eos) else r.completion + eos
    return Plan(r, r.prompt + completion, True, aux=[r.prompt], finish=_trl_pair(only))


def _with_answers(tok: Tok, row: Row, msgs: list[dict], text: str):
    def finish(ids, offsets, aux):
        spans = turn_spans(tok, row, msgs, text)
        if spans is None:
            return _all(len(ids)), None, False
        return _all(len(ids)), token_ranges(offsets, spans), True
    return finish


def _mlx(tok: Tok, r: Row, mask: bool | None):
    if r.shape == "text":
        def finish(ids, off, aux):
            if ids[-1] != tok.eos_id:
                ids.append(tok.eos_id)
            return _all(len(ids)), None, True
        return Plan(r, r.text, True, finish=finish)
    msgs = r.as_messages()
    text = tok.render(msgs, tools=r.tools)
    prompt = tok.render(msgs[:-1], gen=True, tools=r.tools) if mask else None

    def finish(ids, off, aux):
        spans = turn_spans(tok, r, msgs, text)
        ranges = token_ranges(off, spans) if spans else None
        offset = len(aux[0]) if aux else 0
        return _mask(len(ids), [(offset, len(ids))]), ranges, spans is not None
    return Plan(r, text, False, aux=[prompt] if prompt is not None else [], offsets=True, rendered=True,
                prompt_text=_prompt_text(tok, r, msgs), finish=finish)


def _find(ids: list[int], pat: list[int], at: int) -> bool:
    return ids[at:at + len(pat)] == pat


def _response_spans(tok: Tok, ids: list[int], resp: list[int], inst: list[int]) -> list[tuple[int, int]]:
    """Each stretch after the response marker, up to the first end token, the next instruction marker, or the row end."""
    spans, j, n = [], 0, len(ids)
    while j < n:
        if not _find(ids, resp, j):
            j += 1
            continue
        start = k = j + len(resp)
        end = n
        while k < n:
            if ids[k] == tok.eos_id:
                end = k + 1
                break
            if _find(ids, inst, k) or _find(ids, resp, k) or (tok.bos_id is not None and ids[k] == tok.bos_id):
                end = k
                break
            k += 1
        spans.append((start, end))
        j = max(end, start + 1)
    return spans


def _located(tok: Tok, r: Row, mask: bool | None, whole: bool, text_ok: str):
    if r.shape == "text":
        raise Unsupported(text_ok)
    msgs = r.as_messages()
    text = tok.render(msgs, tools=r.tools)

    def finish(ids, off, aux):
        spans = turn_spans(tok, r, msgs, text)
        if spans is None:
            return _all(len(ids)), None, False
        ranges = token_ranges(off, spans)
        if not mask:
            return _all(len(ids)), ranges, True
        trained = ranges
        if not whole:
            trained = [piece for a, b in ranges for piece in _content_and_eos(tok, ids, a, b)]
        return _mask(len(ids), trained), ranges, True
    return Plan(r, text, False, offsets=True, rendered=True, prompt_text=_prompt_text(tok, r, msgs), finish=finish)


def _axolotl(tok: Tok, r: Row, mask: bool | None):
    return _located(tok, r, mask, False, "Axolotl reads text rows through a dataset type you choose in the config, which this lint does not model; use messages")


def _llama_factory(tok: Tok, r: Row, mask: bool | None):
    return _located(tok, r, mask, True, "this trainer reads a text column only for pretraining, not for supervised fine-tuning")
