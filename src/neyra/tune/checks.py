"""What is wrong with the rows a trainer would see. Every finding is counted from the built ids and mask."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from jinja2 import TemplateError

from . import similar
from .tok import Tok
from .trainers import Built, Trainer

THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
FOREIGN = ("<|im_start|>", "[INST]", "<|start_header_id|>", "<start_of_turn>", "<|user|>", "<|assistant|>",
           "### Instruction:", "<|system|>")
LEVELS = {"error": 0, "warning": 1, "note": 2}
HARD = 0.01


@dataclass
class Example:
    line: int
    ids: list[int]
    trained: list[bool]
    focus: int | None = None
    text: str = ""


@dataclass
class Finding:
    id: str
    level: str
    count: int
    of: int
    title: str
    headline: str
    detail: str
    example: Example | None = None
    approximate: bool = False

    @property
    def share(self) -> float:
        return self.count / self.of if self.of else 0.0


@dataclass
class Tally:
    count: int = 0
    example: Example | None = None
    extra: dict = field(default_factory=dict)

    def hit(self, b: Built, focus: int | None = None, **extra):
        self.count += 1
        if self.example is None:
            self.example = Example(b.row.line, b.ids[:6000], b.trained[:6000], focus)
        for k, v in extra.items():
            self.extra[k] = self.extra.get(k, 0) + v


def level(share: float, hard: float = HARD) -> str:
    return "error" if share >= hard else "warning"


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


class Rows:
    """Row-level checks, fed one built row at a time."""

    def __init__(self, tok: Tok, trainer: Trainer, limit: int, trains_prompt: bool = False):
        self.tok, self.trainer, self.limit = tok, trainer, limit
        self.trains_prompt = trains_prompt
        self.t: dict[str, Tally] = {k: Tally() for k in (
            "double_bos", "missing_bos", "missing_end", "end_masked", "empty", "truncated", "end_lost", "dropped",
            "prompt", "twice", "foreign", "prefix", "think", "system", "unlocated", "rejected", "dup_exact", "dup_near")}
        self.n = 0
        self.tokens = self.trained = self.cut_tokens = self.prompt_tokens = 0
        self.seen = similar.Seen()
        self.markers = self._markers()

    def _markers(self) -> list[str]:
        t = self.tok
        probe = ""
        if t.template:
            try:
                probe = t.render([{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                  {"role": "assistant", "content": "a"}])
            except (TemplateError, ValueError, TypeError):  # some templates reject a system turn
                probe = t.template
        added = getattr(t.t, "added_tokens_decoder", None) or {}
        return [a.content for a in added.values()
                if getattr(a, "special", False) and len(a.content) > 3 and a.content in probe
                and a.content not in (t.t.bos_token, t.t.pad_token, t.t.unk_token)]

    def tail_ends(self, ids: list[int]) -> list[int]:
        """Positions of end tokens in the run of end tokens and blanks that closes the row."""
        out, i = [], len(ids) - 1
        while i >= 0 and (ids[i] in self.tok.end_ids or self.tok.blank(ids[i])):
            if ids[i] in self.tok.end_ids:
                out.append(i)
            i -= 1
        return out

    def add(self, b: Built):
        tok, t = self.tok, self.t
        self.n += 1
        if b.error:
            t["rejected"].hit(b)
            t["rejected"].extra.setdefault("why", b.error)
            return
        ids, n, kept = b.ids, b.n, b.kept
        self.tokens += n
        if b.dropped:
            t["dropped"].hit(b)
            self.cut_tokens += n
            return
        cut = n - kept
        if cut > 0:
            t["truncated"].hit(b, focus=kept)
            self.cut_tokens += cut
        trained = sum(b.trained[:kept])
        self.trained += trained
        if not trained:
            t["empty"].hit(b)

        bos = tok.bos_id
        if bos is not None and n >= 2 and ids[0] == bos and ids[1] == bos:
            t["double_bos"].hit(b, focus=1)
        elif tok.adds_bos and ids[0] != bos:
            t["missing_bos"].hit(b, focus=0, templated=b.text is not None)

        ends = self.tail_ends(ids)
        if not ends:
            t["missing_end"].hit(b, focus=n - 1)
        else:
            if cut > 0 and min(ends) >= kept:
                t["end_lost"].hit(b, focus=kept)
            elif trained and not any(b.trained[i] for i in ends if i < kept):
                t["end_masked"].hit(b, focus=ends[-1])

        if not b.located:
            t["unlocated"].hit(b)
        elif b.answers is not None and trained and not self.trains_prompt:
            self._prompt(b, ends)
        self._template(b)
        self._words(b)

    def _prompt(self, b: Built, ends: list[int]):
        inside = bytearray(b.n)
        for lo, hi in b.answers:
            inside[lo:hi] = b"\x01" * (hi - lo)
        stop = max((hi for _, hi in b.answers), default=0)
        first = None
        count = 0
        for i in range(1, b.kept):
            if b.trained[i] and not inside[i] and not (i >= stop and (b.ids[i] in self.tok.end_ids or self.tok.blank(b.ids[i]))):
                count += 1
                first = i if first is None else first
        if count:
            self.t["prompt"].hit(b, focus=first, tokens=count)

    def _template(self, b: Built):
        row, tok = b.row, self.tok
        if b.text is not None and row.shape != "text":
            for m in row.as_messages():
                if any(mark in m["content"] for mark in self.markers):
                    self.t["twice"].hit(b)
                    break
        elif row.shape == "text":
            hit = next((m for m in FOREIGN if m in row.text and tok.encode(m, False) != [tok.t.convert_tokens_to_ids(m)]), None)
            if hit:
                self.t["foreign"].hit(b, **{"marker:" + hit: 1})
        if b.text is not None and b.prompt_text is not None and not b.text.startswith(b.prompt_text):
            self.t["prefix"].hit(b)

    def _words(self, b: Built):
        row = b.row
        if b.text is not None and row.shape != "text":
            last = [m for m in row.as_messages() if m["role"] == "assistant"][-1]
            body = (THINK.search(last["content"]) or [None, None])[1]
            body = body if body is not None else last.get("reasoning_content")
            if body and body.strip() and body.strip() not in b.text:
                self.t["think"].hit(b)
            system = row.system()
            if len(system) >= 30:
                chunks = [c.strip() for c in SENTENCE.split(system) if len(c.split()) >= 6] or [system.strip()]
                answers = [" ".join(a.split()) for a in row.answers()]
                if any(" ".join(c.split()) in a for c in chunks for a in answers):
                    self.t["system"].hit(b)
        text = row.plain()
        w = similar.words(text)
        if not w:
            return
        kind = self.seen.find(text, w, sig := similar.signature(w))
        if kind:
            self.t["dup_exact" if kind == "exact" else "dup_near"].hit(b)
        else:
            self.seen.add(text, w, sig)

    def findings(self, pad_eos: bool) -> list[Finding]:
        t, N, tr = self.t, self.n, self.trainer
        out: list[Finding] = []
        approx = not tr.exact

        def add(key, lvl, title, headline, detail, count=None, approximate=False):
            c = t[key].count if count is None else count
            if c:
                out.append(Finding(key, lvl, c, N, title, headline, detail, t[key].example, approximate))

        tk = self.tok
        eos = tk.t.eos_token
        add("double_bos", level(t["double_bos"].count / max(N, 1)), "double BOS",
            f"{plural(t['double_bos'].count, 'row')} start{'s' * (t['double_bos'].count == 1)} with two BOS tokens",
            "The text already carries the BOS token and the tokenizer adds another. Tokenize with add_special_tokens=False, "
            "or remove the BOS from the text.")
        # A template that leaves BOS out was written that way on purpose, so only plain text rows are certain errors.
        templated = t["missing_bos"].extra.get("templated", 0) == t["missing_bos"].count
        add("missing_bos", "warning" if templated else level(t["missing_bos"].count / max(N, 1)), "missing BOS",
            f"{plural(t['missing_bos'].count, 'row')} start{'s' * (t['missing_bos'].count == 1)} without the BOS token the tokenizer adds to plain text",
            "Serving normally adds a BOS token, but these rows do not begin with one"
            + (": the chat template does not include it." if templated else "."))
        add("missing_end", level(t["missing_end"].count / max(N, 1)), "no end token",
            f"{plural(t['missing_end'].count, 'row')} end{'s' * (t['missing_end'].count == 1)} without an end token, so the model will not learn to stop",
            f"The last tokens are not {eos or 'an end of turn token'} or the template's end of turn. Add it to the data, "
            "or use a trainer path that appends it.")
        add("end_masked", level(t["end_masked"].count / max(N, 1)), "end token not trained",
            f"{plural(t['end_masked'].count, 'row')} have the end token masked out of the loss",
            "The end token is in the row but its label is ignored, so the loss never asks the model to stop.", approximate=approx)
        add("empty", level(t["empty"].count / max(N, 1)), "empty loss",
            f"{plural(t['empty'].count, 'row')} have no token in the loss",
            "Nothing in these rows is trained on: the mask covers no tokens, or truncation cut them all.", approximate=approx)
        if tr.truncation == "drop":
            add("dropped", "warning", "rows dropped for length",
                f"{plural(t['dropped'].count, 'row')} are longer than {tr.length_name} {self.limit} and will be dropped",
                f"{self.cut_tokens / max(self.tokens, 1):.1%} of all tokens are in dropped rows. "
                f"Raise {tr.length_name} or shorten them.")
        else:
            lost = t["end_lost"].count
            share = self.cut_tokens / max(self.tokens, 1)
            add("truncated", "error" if lost / max(N, 1) >= HARD else "warning", "truncated",
                f"{plural(t['truncated'].count, 'row')} are longer than {tr.length_name} {self.limit}; "
                f"{plural(lost, 'answer')} lose{'s' * (lost == 1)} their end, {share:.1%} of tokens are cut",
                f"Raise {tr.length_name}, or shorten these rows. A cut answer ends mid sentence with no end token.",
                count=t["truncated"].count, approximate=tr.name == "llama-factory")
            if lost:
                t["truncated"].example = t["end_lost"].example or t["truncated"].example
        p = t["prompt"]
        if p.count:
            share = p.extra.get("tokens", 0) / max(self.trained, 1)
            out.append(Finding("prompt", "warning", p.count, N, "prompt text in the loss",
                               f"{plural(p.count, 'row')} train{'s' * (p.count == 1)} on prompt text, {share:.0%} of all trained tokens",
                               "System and user turns are inside the loss, so the model also learns to write them. Mask the "
                               "prompt (assistant only loss, completion only loss, or train on responses only).",
                               p.example, approx))
        add("twice", level(t["twice"].count / max(N, 1)), "template applied twice",
            f"{plural(t['twice'].count, 'row')} already contain this model's chat markers inside the messages",
            "The trainer will wrap them in the chat template again. Store plain text in the messages, or train on the "
            "text field.")
        if t["foreign"].count:
            marks = sorted((k[7:] for k in t["foreign"].extra if k.startswith("marker:")),
                           key=lambda m: -t["foreign"].extra["marker:" + m])
            add("foreign", level(t["foreign"].count / max(N, 1)), "another chat format",
                f"{plural(t['foreign'].count, 'row')} use{'s' * (t['foreign'].count == 1)} {marks[0]}, which this tokenizer does not read as a single token",
                "The text is formatted for a different model. Re-render it with this tokenizer's chat template.")
        add("prefix", level(t["prefix"].count / max(N, 1)), "prompt is not a prefix",
            f"{plural(t['prefix'].count, 'row')} train{'s' * (t['prefix'].count == 1)} on text that does not begin with the prompt the model will be given",
            "Rendering the question with add_generation_prompt gives different text from the start of the full "
            "conversation, so training and inference see different contexts.")
        add("think", level(t["think"].count / max(N, 1)), "thinking removed",
            f"{plural(t['think'].count, 'row')} have a reasoning block in the answer that the template drops",
            "The chat template removes the thinking from the training text, so the model trains on answers without it. "
            "Move it to reasoning_content, or use a template that keeps it.")
        add("system", level(t["system"].count / max(N, 1), 0.05), "system prompt in answers",
            f"{plural(t['system'].count, 'answer')} repeat{'s' * (t['system'].count == 1)} text from their own system prompt",
            "The model will learn to recite its instructions. Remove the copied sentences from the answers.")
        if t["rejected"].count:
            add("rejected", level(t["rejected"].count / max(N, 1)), "rows the template rejects",
                f"{plural(t['rejected'].count, 'row')} cannot be rendered by the chat template",
                f"The template says: {t['rejected'].extra.get('why', '')[:160]}")
        add("unlocated", "note", "mask approximate",
            f"{plural(t['unlocated'].count, 'row')} could not be split into turns, so their mask is approximate",
            "The chat template changes earlier turns when it renders later ones, so the turn boundaries could not be "
            "found by prefix. These rows are treated as fully trained.", approximate=True)
        dup, near = t["dup_exact"].count, t["dup_near"].count
        if dup + near:
            out.append(Finding("duplicates", level((dup + near) / max(N, 1), 0.1), dup + near, N, "duplicate rows",
                               f"{plural(dup + near, 'row')} repeat{'s' * (dup + near == 1)} an earlier row ({dup:,} exactly, {near:,} nearly)",
                               "Repeated rows are trained on several times and, if they straddle a split, leak into "
                               "evaluation. Keep one copy.", t["dup_exact"].example or t["dup_near"].example))
        if pad_eos:
            out.append(Finding("pad_eos", "note", 1, 1, "pad token is the end token",
                               f"the pad token is {eos}, the same as the end token",
                               "The trainers modelled here build labels from the attention mask, so the end token stays in "
                               "the loss. A collator that masks every pad id would remove it."))
        return sorted(out, key=lambda f: (LEVELS[f.level], -f.share, -f.count))
