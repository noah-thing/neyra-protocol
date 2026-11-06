"""Write facts into a model's MLP layers, as a patch that rebuilds and reverts bit for bit.

The update is the one Meng et al. (ROME 2022, MEMIT 2023) derive. A fact is a key, the input to
down_proj at the subject's last token, and a value, the MLP output the model would need there to
say the new object. Several layers share the change, each taking an equal part of what is still
missing. Sequential editing follows AlphaEdit (Fang et al., 2025) and RLSEdit: later batches must
not undo earlier ones, so the keys already written join the keys of neutral text as things the
update has to leave alone. Value vectors are capped near the size they had in the unedited model.

Solves run in float64 and everything is measured on the rounded weights the patch produces.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .families import writers
from .numerics import from_float, to_float
from .patch import Patch, TensorEdit
from .protocol import bits_digest, tensor_table
from .repair import down_proj_name
from .runtime import DeltaBatch, Hooks, Sampling
from .store import Checkpoint, FormatError


def on_stderr(message: str) -> None:
    import sys

    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Request:
    subject: str
    prompt: str  # holds one {} where the subject goes
    target: str
    true: str | None = None  # the object the model gives today, needed to score efficacy and neighbours
    paraphrases: tuple[str, ...] = ()
    neighbours: tuple[str, ...] = ()  # prompts about other subjects whose answer is `true`
    generation: tuple[str, ...] = ()

    def __post_init__(self):
        if self.prompt.count("{}") != 1:
            raise ValueError(f"the prompt needs exactly one {{}} for the subject: {self.prompt!r}")

    def text(self) -> str:
        return self.prompt.replace("{}", self.subject)


def _prompts(items) -> tuple[str, ...]:
    return tuple(i["prompt"] if isinstance(i, dict) else i for i in items or ())


def parse_request(raw: dict) -> Request:
    """A native request, or a CounterFact record as the ROME and MEMIT benchmarks ship it."""
    if "requested_rewrite" in raw:
        w = raw["requested_rewrite"]
        return Request(w["subject"], w["prompt"], w["target_new"]["str"], w.get("target_true", {}).get("str"),
                       _prompts(raw.get("paraphrase_prompts")), _prompts(raw.get("neighborhood_prompts")),
                       _prompts(raw.get("generation_prompts")))
    return Request(raw["subject"], raw["prompt"], raw["target"], raw.get("true"),
                   _prompts(raw.get("paraphrases")), _prompts(raw.get("neighbours")), _prompts(raw.get("generation")))


def one_fact(text: str, target: str) -> Request:
    """A fact from the command line, with the subject in brackets: "The capital of [France] is"."""
    hit = re.fullmatch(r"(.*)\[([^\[\]]+)\](.*)", text)
    if not hit:
        raise ValueError('put the subject in brackets: "The capital of [France] is"')
    return Request(hit[2], f"{hit[1]}{{}}{hit[3]}", target)


def what_it_says(rt, words: Words, request: Request) -> str:
    """The first word the unedited model gives, which becomes the object the edit replaces."""
    row = rt.generate([words.ids(request.text())], 8, Sampling(0.0), batch=1)[0]
    found = row.text.split()
    if not found:
        raise ValueError("the model gave no answer to this prompt, so there is nothing to replace")
    return found[0].strip(".,;:\"'")


def default_layers(ck: Checkpoint) -> list[int]:
    """Five layers where edits worked in testing: the first on a hybrid stack, a fifth of the way in otherwise.

    Tracing needs several facts to mean anything, so a single fact starts here.
    """
    found = writers(ck).values()
    dense = sorted({w.layer for w in found if w.kind == "mlp_down"})
    hybrid = any("linear_attn" in name for name in ck.tensors)
    start = 0 if hybrid else round(0.18 * len(dense))
    return dense[start:start + 5]


def load_requests(path: str | Path) -> list[Request]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc.get("requests") if isinstance(doc, dict) else doc
    if not isinstance(items, list) or not items:
        raise FormatError(f"{path}: expected a list of edit requests, or an object with a \"requests\" list")
    out = []
    for i, raw in enumerate(items):
        try:
            out.append(parse_request(raw))
        except (KeyError, TypeError, AttributeError):
            raise FormatError(f"{path}: request {i} needs a subject, a prompt and a target") from None
    return out


@dataclass
class Settings:
    layers: list[int] | None = None  # None: choose by causal tracing
    width: int = 5  # layers that share an edit when tracing chooses
    solver: str = "protect"  # "protect" keeps past edits, "memit" forgets them, "null" projects away neutral text
    lam: float = 1000.0  # weight of neutral text against the new keys, in tokens
    ridge: float = 1e-3
    null_energy: float = 0.99  # the null space drops directions holding this share of neutral variance
    batch: int = 100
    contexts: int = 2  # generated prefixes added to the plain prompt when learning keys and values
    spread: str = "mean"  # "mean": one key per fact, averaged over contexts. "each": every context writes its own key
    value_contexts: int | None = None  # fewer for the value descent, whose memory grows with rows. None: all
    steps: int = 20
    lr: float = 0.25  # of the typical size of one coordinate of the MLP output
    kl: float = 0.0625
    decay: float = 0.0015
    clamp: float = 4.0  # of the output's norm, per optimisation
    anchor: float = 1.5  # cap on the final value's norm, as a multiple of the unedited model's
    polish: int = 0  # extra passes on the last layer, to correct for rounding
    max_rank: int | None = None
    energy: float = 0.99999  # share of the change a rank cut must keep
    neutral: int = 192  # windows of neutral text for the covariance
    held: int = 48  # further windows to measure damage on
    window: int = 96  # tokens per window
    seed: int = 0


MIN_ROW = 16


class Words:
    """Prompts as token lists, with the position where the subject ends."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def target(self, prompt: str, target: str) -> list[int]:
        glue = "" if prompt[-1:].isspace() or target[:1].isspace() else " "
        return self.ids(glue + target)

    def lookup(self, request: Request, context: str = "") -> tuple[list[int], int]:
        head, _, tail = request.prompt.partition("{}")
        upto = self.ids(context + head + request.subject)
        full = self.ids(context + request.text())
        if full[:len(upto)] != upto:
            full = upto + self.ids(tail)
        return full, len(upto) - 1

    def span(self, request: Request) -> tuple[int, int]:
        first = len(self.ids(request.prompt.partition("{}")[0]))
        return first, self.lookup(request)[1]


def neutral_windows(rt, words: Words, source: str | None, count: int, length: int, seed: int) -> list[list[int]]:
    """Plain text windows: from a file if given, else sampled from the model itself."""
    rng = np.random.default_rng(seed)
    if source:
        ids = words.ids(Path(source).read_text(encoding="utf-8"))
        cuts = [ids[i:i + length] for i in range(0, len(ids) - length + 1, length)]
        if len(cuts) < count:
            raise ValueError(f"{source} holds {len(cuts)} windows of {length} tokens, {count} needed")
        return [cuts[i] for i in rng.permutation(len(cuts))[:count]]
    starts = ["The", "In", "A", "It", "When", "He", "She", "They", "We", "One", "After", "There", "This", "My", "Some"]
    prompts = [words.ids(starts[i % len(starts)]) for i in range(count)]
    rows = rt.generate(prompts, length - 1, Sampling(1.0, 0.95, 0, 0.0, seed), batch=32)
    return [(p + r.tokens)[:length] for p, r in zip(prompts, rows) if len(p) + len(r.tokens) >= length // 2]


def second_moments(rt, windows: list[list[int]], layers: list[int], skip: int = 1) -> dict[int, np.ndarray]:
    """E[a a^T] of the input to down_proj over neutral tokens. The first token is a sink with outsized activations."""
    sums = {l: 0.0 for l in layers}
    count = 0
    for s in range(0, len(windows), 16):
        part = windows[s:s + 16]
        keep = np.concatenate([np.arange(len(w)) >= skip for w in part])
        got = rt.gather(part, layers, "mlp")
        for l in layers:
            a = got[l][keep].astype(np.float64)
            sums[l] = sums[l] + a.T @ a
        count += int(keep.sum())
    return {l: s / count for l, s in sums.items()}


def solve_update(keys: np.ndarray, resid: np.ndarray, cov: np.ndarray, past: np.ndarray | None, lam: float,
                 ridge: float, basis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Factors (left, right) of the weight change D = left @ right with D k ~ resid.

    D = R K^T (lam C + P + K K^T + mu I)^-1, where C is the neutral covariance and P the keys already
    written. Inside a basis Q of directions neutral text barely uses, the same solve is AlphaEdit's
    null space projection, and D = D~ Q^T.
    """
    if basis is not None:
        keys, cov = basis.T @ keys, basis.T @ cov @ basis
        past = None if past is None else basis.T @ past @ basis
    m = lam * cov + keys @ keys.T
    if past is not None:
        m = m + past
    m = m + ridge * np.trace(m) / len(m) * np.eye(len(m))
    right = np.linalg.solve(m, keys).T
    return resid, right if basis is None else right @ basis.T


def null_basis(cov: np.ndarray, energy: float) -> np.ndarray:
    """Eigenvectors of the covariance left once the loudest ones holding `energy` of its variance are dropped."""
    vals, vecs = np.linalg.eigh(cov)
    share = np.cumsum(vals[::-1]) / vals.sum()
    drop = int(np.searchsorted(share, energy)) + 1
    return vecs[:, :len(vals) - drop]


def compress(left: np.ndarray, right: np.ndarray, energy: float, max_rank: int | None) -> tuple[np.ndarray, np.ndarray]:
    """left @ right at the smallest rank that keeps the given share of its energy."""
    q, r = np.linalg.qr(left)
    u, s, vh = np.linalg.svd(r @ right, full_matrices=False)
    kept = np.cumsum(s**2) / max((s**2).sum(), 1e-300)
    rank = min(int(np.searchsorted(kept, energy)) + 1, len(s), max_rank or len(s))
    return q @ (u[:, :rank] * s[:rank]), vh[:rank]


def adam(grad: np.ndarray, state: dict, lr: float, step: int) -> np.ndarray:
    """One Adam step. Gradients are rescaled by their first size, since models differ by orders of magnitude."""
    if "unit" not in state:
        state["unit"] = np.sqrt((grad**2).mean(axis=1, keepdims=True)) + 1e-30
    grad = grad / state["unit"]
    state["m"] = 0.9 * state.get("m", 0) + 0.1 * grad
    state["v"] = 0.999 * state.get("v", 0) + 0.001 * grad**2
    return lr * (state["m"] / (1 - 0.9**step)) / (np.sqrt(state["v"] / (1 - 0.999**step)) + 1e-8)


def check_layers(ck: Checkpoint, layers: list[int]) -> None:
    dense = {w.layer for w in writers(ck).values() if w.kind == "mlp_down"}
    for l in layers:
        if l not in dense:
            raise ValueError(f"layer {l} has no dense MLP down projection, so there is nothing here to edit")


class Stack:
    """The edited tensors: base weights, running change, and the factors that produce it."""

    def __init__(self, ck: Checkpoint, layers: list[int]):
        check_layers(ck, layers)
        self.names, self.bits, self.dtype, self.w0, self.delta = {}, {}, {}, {}, {}
        self.left: dict[int, list] = {}
        self.right: dict[int, list] = {}
        for l in layers:
            name = down_proj_name(ck, l)
            self.names[l] = name
            self.bits[l] = ck.tensors[name].read()
            self.dtype[l] = ck.tensors[name].dtype
            self.w0[l] = to_float(self.bits[l], self.dtype[l]).astype(np.float64)
            self.delta[l] = np.zeros_like(self.w0[l])
            self.left[l], self.right[l] = [], []

    def add(self, layer: int, left: np.ndarray, right: np.ndarray) -> None:
        self.left[layer].append(left)
        self.right[layer].append(right)
        self.delta[layer] += left @ right

    def tensors(self) -> dict[str, tuple[np.ndarray, str]]:
        return {self.names[l]: (from_float((self.w0[l] + self.delta[l]).astype(np.float32), self.dtype[l])
                                .reshape(self.bits[l].shape), self.dtype[l]) for l in self.names}

    def patch(self, ck: Checkpoint, settings: Settings, meta: dict) -> Patch:
        edits = []
        for l, name in self.names.items():
            if not self.left[l]:
                continue
            left, right = compress(np.hstack(self.left[l]), np.vstack(self.right[l]), settings.energy, settings.max_rank)
            dtype = self.dtype[l]
            empty = TensorEdit(name, dtype, self.bits[l].shape, bits_digest(dtype, self.bits[l]), "",
                               np.empty(0, np.int64), np.empty(0, self.bits[l].dtype), left, right)
            empty.result_digest = bits_digest(dtype, empty.apply(self.bits[l], verify=False))
            edits.append(empty)
        return Patch(edits, meta).bind(tensor_table(ck))


class Editor:
    def __init__(self, rt, ck: Checkpoint, settings: Settings, layers: list[int], cov: dict[int, np.ndarray],
                 say: Callable[[str], None] = on_stderr):
        self.rt, self.ck, self.s, self.layers, self.cov, self.say = rt, ck, settings, sorted(layers), cov, say
        self.words = Words(rt.tokenizer)
        self.stack = Stack(ck, self.layers)
        self.past = {l: np.zeros_like(cov[l]) for l in self.layers} if settings.solver != "memit" else {}
        self.basis = ({l: null_basis(cov[l], settings.null_energy).astype(np.float32) for l in self.layers}
                      if settings.solver == "null" else {})
        self.contexts = [""]
        self.written = 0

    def save(self, path: str | Path) -> None:
        """Everything a later run needs to carry on from here: the factors, the keys written, the contexts."""
        arrays = {"written": np.array(self.written), "contexts": np.array(json.dumps(self.contexts))}
        for l in self.layers:
            if self.stack.left[l]:
                arrays[f"left{l}"], arrays[f"right{l}"] = np.hstack(self.stack.left[l]), np.vstack(self.stack.right[l])
            if l in self.past:
                arrays[f"past{l}"] = self.past[l]
        np.savez(path, **arrays)

    def load(self, path: str | Path) -> None:
        got = np.load(path)
        self.written, self.contexts = int(got["written"]), json.loads(str(got["contexts"]))
        for l in self.layers:
            if f"left{l}" in got:
                self.stack.add(l, got[f"left{l}"], got[f"right{l}"])
            if f"past{l}" in got:
                self.past[l] = got[f"past{l}"]
        self.rt.set_tensors(self.stack.tensors())

    def prefixes(self) -> None:
        """A few generated openings, so a key learned from one phrasing also fires inside others."""
        if self.s.contexts <= 0 or len(self.contexts) > 1:
            return
        seeds = ["The", "Therefore", "Because", "I", "You", "However", "It"][:self.s.contexts]
        rows = self.rt.generate([self.words.ids(w) for w in seeds], 8, Sampling(0.8, 0.95, 20, 0.0, self.s.seed), batch=8)
        self.contexts += [(w + r.text).replace("\n", " ") + ". " for w, r in zip(seeds, rows)]

    def value_contexts(self) -> list[str]:
        keep = self.s.value_contexts
        return self.contexts if keep is None else self.contexts[:1 + keep]

    def rows(self, batch: list[Request], contexts: list[str]) -> tuple[list[list[int]], list[int]]:
        out, at = [], []
        for r in batch:
            for c in contexts:
                ids, pos = self.words.lookup(r, c)
                out.append(ids)
                at.append(pos)
        return out, at

    def activations(self, batch: list[Request], layer: int, kind: str, contexts: list[str],
                    keep: bool = False) -> np.ndarray:
        """Mean over contexts of an activation at each subject's last token: [requests, width].

        With `keep` the contexts stay apart: [requests * contexts, width], the contexts of a request together.
        """
        rows, at = self.rows(batch, contexts)
        got = self.rt.gather(rows, [layer], kind, at)[layer].astype(np.float64)
        return got if keep else got.reshape(len(batch), len(contexts), -1).mean(axis=1)

    def keys(self, batch: list[Request], layer: int) -> tuple[np.ndarray, int]:
        """Keys as columns, and how many columns each request owns."""
        each = self.s.spread == "each"
        return self.activations(batch, layer, "mlp", self.contexts, keep=each).T, (len(self.contexts) if each else 1)

    def pristine(self, batch: list[Request], layer: int) -> np.ndarray:
        """MLP outputs of the unedited model, which anchor how large a value may grow."""
        self.rt.reset()
        try:
            return self.activations(batch, layer, "out", [""])
        finally:
            self.rt.set_tensors(self.stack.tensors())

    def values(self, batch: list[Request], layer: int) -> tuple[np.ndarray, np.ndarray]:
        """The MLP output at `layer` each request needs, by short Adam descent on a vector added there."""
        start = self.activations(batch, layer, "out", [""])
        anchor = np.linalg.norm(self.pristine(batch, layer), axis=1) * self.s.anchor
        out, loss = np.empty_like(start), np.empty(len(batch))
        for first, last in self.groups(batch):
            out[first:last], loss[first:last] = self.descend(batch[first:last], layer, start[first:last],
                                                             anchor[first:last])
        return out, loss

    def groups(self, batch: list[Request]):
        """Runs of requests whose padded rows fit the runtime's budget, since gradients cost memory per row token."""
        budget = getattr(self.rt, "row_budget", 1 << 30)
        first, rows, width = 0, 0, MIN_ROW
        for i, r in enumerate(batch):
            target = len(self.words.target(r.text(), r.target))
            lengths = [len(self.words.lookup(r, c)[0]) + target for c in self.value_contexts()] + [len(self.words.ids(r.subject)) + 3]
            if i > first and (rows + len(lengths)) * max(width, *lengths) > budget:
                yield first, i
                first, rows, width = i, 0, MIN_ROW
            rows, width = rows + len(lengths), max(width, *lengths)
        yield first, len(batch)

    def descend(self, batch: list[Request], layer: int, start: np.ndarray, anchor: np.ndarray):
        s, words = self.s, self.words
        rows, owner, pos, items, refs = [], [], [], [], []
        for i, r in enumerate(batch):
            target = words.target(r.text(), r.target)
            contexts = self.value_contexts()
            for c in contexts:
                ids, at = words.lookup(r, c)
                for j, tok in enumerate(target):
                    items.append((len(rows), len(ids) - 1 + j, tok, 1 / (len(target) * len(contexts))))
                rows.append(ids + target[:-1])
                owner.append(i)
                pos.append(at)
            ids, at = words.lookup(Request(r.subject, "{} is a", r.target))
            items.append((len(rows), at, -1, s.kl))
            rows.append(ids)
            owner.append(i)
            pos.append(at)
            refs.append(self.rt.logits_at(ids, np.array([at + 1]))[0])
        refs = np.array(refs, dtype=np.float32)
        refs -= refs.max(axis=1, keepdims=True)
        refs -= np.log(np.exp(refs).sum(axis=1, keepdims=True))
        # MLX attention takes a vector kernel for short sequences, and that kernel has no gradient
        width = max(MIN_ROW, *map(len, rows))
        batch_in = DeltaBatch(np.array([r + [0] * (width - len(r)) for r in rows]), np.array(owner), np.array(pos),
                              *(np.array(c) for c in zip(*items)), refs)
        scale = s.lr * np.linalg.norm(start, axis=1, keepdims=True) / np.sqrt(start.shape[1])
        reach = s.clamp * np.linalg.norm(start, axis=1, keepdims=True)
        norm2 = np.linalg.norm(start, axis=1, keepdims=True) ** 2
        delta, state = np.zeros_like(start), {}
        total = np.zeros(len(batch))
        for step in range(1, s.steps + 1):
            nll, grad = self.rt.delta_loss(layer, delta, batch_in)
            total = nll + s.decay * (delta**2).sum(axis=1) / norm2[:, 0]
            if step == s.steps or (total < 0.05).all():
                break
            grad = (grad + 2 * s.decay * delta / norm2) * (total >= 0.05)[:, None]
            delta = delta - adam(grad, state, 1.0, step) * scale
            size = np.linalg.norm(delta, axis=1, keepdims=True)
            delta = delta * np.minimum(1.0, reach / np.maximum(size, 1e-12))
        z = start + delta
        size = np.linalg.norm(z, axis=1, keepdims=True)
        z = z * np.minimum(1.0, anchor[:, None] / np.maximum(size, 1e-12))
        return z, total

    def write(self, batch: list[Request]) -> dict:
        """Add one batch of facts to the stack, layer by layer, and return what it took."""
        s = self.s
        self.prefixes()
        last = self.layers[-1]
        self.say(f"solving the value at layer {last}")
        z, loss = self.values(batch, last)
        passes = self.layers + [last] * s.polish
        gaps = []
        for i, l in enumerate(passes):
            self.say(f"writing layer {l}, pass {i + 1} of {len(passes)}")
            keys, copies = self.keys(batch, l)
            now = self.activations(batch, last, "out", [""])
            resid = np.repeat((z - now).T, copies, axis=1) / (len(passes) - i if i < len(self.layers) else 1)
            gaps.append(float(np.linalg.norm(z - now) / np.sqrt(len(batch))))
            past = self.past.get(l) if s.solver != "memit" else None
            left, right = solve_update(keys, resid, self.cov[l], past, s.lam, s.ridge, self.basis.get(l))
            self.stack.add(l, left, right)
            self.rt.set_tensors(self.stack.tensors())
        if s.solver != "memit":
            for l in self.layers:
                keys, _ = self.keys(batch, l)
                self.past[l] += keys @ keys.T
        after = self.activations(batch, last, "out", [""])
        self.written += len(batch)
        return {"gap_before": gaps[0], "gap_after": float(np.linalg.norm(z - after) / np.sqrt(len(batch))),
                "value_loss": float(loss.mean())}


def pick_layers(effect: np.ndarray, width: int) -> list[int]:
    """The run of `width` layers whose restored MLP outputs recover the most."""
    best = int(np.argmax(effect))
    return list(range(best, best + width))


def next_token(rt, ids: list[int], hooks: Hooks | None = None) -> np.ndarray:
    z = rt.logits_at(ids + [0], np.array([len(ids)]), hooks)[0].astype(np.float64)
    p = np.exp(z - z.max())
    return p / p.sum()


@dataclass
class Trace:
    effect: np.ndarray  # recovery per first layer of a restored window
    width: int
    used: int
    layers: list[int]


def trace(rt, words: Words, requests: list[Request], width: int, limit: int = 24, noise: float = 3.0,
          seed: int = 0, draws: int = 1) -> Trace:
    """Causal tracing (Meng et al. 2022) for the MLP at a subject's last token.

    Hide the subject with noise, then restore the clean MLP outputs of `width` layers at that one
    position. The share of the answer that comes back says where the fact is read from. Hybrid
    stacks have no fixed answer: most layers mix tokens through a recurrent state, so the useful
    window depends on where the few full attention layers sit.
    """
    n_layers = len(rt.layers())
    sigma = noise * rt.embedding_scale()
    rng = np.random.default_rng(seed)
    starts = range(n_layers - width + 1)
    gain, used = np.zeros(len(starts)), 0
    for r in requests[:limit]:
        ids, last = words.lookup(r)
        first, _ = words.span(r)
        at = np.arange(first, last + 1)
        pure = rt.gather([ids], list(range(n_layers)), "out", [last])
        clean = next_token(rt, ids)
        top = int(clean.argmax())
        for _ in range(draws):
            vectors = rng.normal(0, sigma, (len(at), pure[0].shape[1]))
            noisy = next_token(rt, ids, Hooks(noise=(at, vectors)))
            if clean[top] - noisy[top] < 0.1:
                continue
            for k, s in enumerate(starts):
                put = {j: (np.array([last]), pure[j]) for j in range(s, s + width)}
                back = next_token(rt, ids, Hooks(noise=(at, vectors), put=put))[top]
                gain[k] += (back - noisy[top]) / (clean[top] - noisy[top])
            used += 1
    effect = gain / max(used, 1)
    return Trace(effect, width, used, pick_layers(effect, width) if used else [])
