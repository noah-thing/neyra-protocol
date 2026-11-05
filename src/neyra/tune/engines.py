"""Run prompts through the base, the tuned model and quantized files.

The torch engine loads the base once and swaps the tuned tensors in place, so the two share one copy in memory.
Quantized files go through llama-server, which has to be installed; when it is not, the files are skipped and
the report says so.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .prompts import Prompt, encode

Generate = Callable[[list[Prompt]], list[str]]


class EngineError(RuntimeError):
    pass


@dataclass
class Backends:
    """What the verdict needs from a runtime. Tests supply fakes."""

    generate: dict[str, Generate]
    neutral_kl: Callable[[list[str]], list[float]] | None = None
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    has_chat_template: bool = True
    close: Callable[[], None] = lambda: None


def quant_name(path: str) -> str:
    stem = Path(path).stem
    hit = re.search(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16|F32)", stem, re.IGNORECASE)
    return hit[1].upper() if hit else stem


def find_llama_server() -> str | None:
    found = os.environ.get("NEYRA_LLAMA_SERVER") or shutil.which("llama-server")
    if found:
        return found
    for root in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.access(f"{root}/llama-server", os.X_OK):
            return f"{root}/llama-server"
    return None


class LlamaFile:
    """One GGUF file behind llama-server, asked for greedy completions of token ids."""

    def __init__(self, binary: str, gguf: str, tokenizer, max_tokens: int, threads: int | None = None,
                 context: int = 4096):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        cmd = [binary, "-m", gguf, "--port", str(self.port), "--host", "127.0.0.1", "-c", str(context), "-np", "1"]
        if threads:
            cmd += ["-t", str(threads)]
        self.log = Path(gguf).with_suffix(".server.log").open("wb")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise EngineError(f"llama-server exited with {self.proc.returncode} on {gguf}; see {self.log.name}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise EngineError(f"llama-server did not become ready for {gguf} within 120 s")

    def __call__(self, prompts: list[Prompt]) -> list[str]:
        out = []
        for p in prompts:
            body = json.dumps({"prompt": encode(self.tokenizer, p), "n_predict": self.max_tokens, "temperature": 0,
                               "cache_prompt": True, "stream": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", body,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                out.append(json.loads(r.read())["content"])
        return out

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


class LlamaRun:
    """Starts llama-server for one file when asked, answers every prompt, and stops it. Servers do not pile up."""

    def __init__(self, binary: str, gguf: str, tokenizer, max_tokens: int, threads: int | None):
        self.args = (binary, gguf, tokenizer, max_tokens, threads)

    def __call__(self, prompts: list[Prompt]) -> list[str]:
        try:
            srv = LlamaFile(*self.args)
        except OSError as e:
            raise EngineError(f"could not start llama-server: {e}") from e
        try:
            return srv(prompts)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            raise EngineError(f"llama-server failed on {self.args[1]}: {e}") from e
        finally:
            srv.stop()


class TorchEngine:
    def __init__(self, base: str, tuned: str, max_tokens: int, batch: int = 8):
        from ..check import Adapter, peft, targets
        from ..hop import saved_modules
        from ..runtime.torch import TorchRuntime
        from ..store import Checkpoint

        self.rt = TorchRuntime(base)
        self.max_tokens, self.batch = max_tokens, batch
        self.tokenizer = self.rt.tokenizer
        self.state = "base"
        base_ck, tuned_ck = Checkpoint.open(base), Checkpoint.open(tuned)

        def merged() -> dict:
            return self._tuned_tensors(base_ck, tuned_ck, Adapter, peft, targets, saved_modules)

        # Rebuilt on every switch instead of kept: the merged tensors are a third copy of every tuned matrix.
        self.tuned = merged
        self.tuned()
        self.base_logits: list[np.ndarray] | None = None

    def _tuned_tensors(self, base, ck, Adapter, peft, targets, saved_modules) -> dict:
        out: dict[str, tuple[np.ndarray, str]] = {}
        if "adapter_config.json" in ck.files:
            adapter = Adapter(ck)
            for name, tensor in targets(adapter, base).items():
                ref = base.tensors[name]
                out[name] = (peft(ref.dtype, ref.read(), adapter.term(tensor)), ref.dtype)
            if saved_modules(ck):
                raise EngineError("this adapter also stores full trained tensors (modules_to_save); merge it with "
                                  "PEFT and pass the merged folder instead")
            return out
        for name, ref in ck.tensors.items():
            if name not in base.tensors:
                continue
            theirs = ref.read()
            if ref.dtype != base.tensors[name].dtype or not np.array_equal(theirs, base.tensors[name].read()):
                out[name] = (theirs, ref.dtype)
        if not out:
            raise EngineError(f"{ck.spec} has no tensor that differs from the base, so there is nothing to test")
        return out

    def _use(self, state: str) -> None:
        if state != self.state:
            self.rt.set_tensors(self.tuned() if state == "tuned" else {})
            self.state = state

    def generator(self, state: str) -> Generate:
        from ..runtime import Sampling

        def run(prompts: list[Prompt]) -> list[str]:
            self._use(state)
            ids = [encode(self.tokenizer, p) for p in prompts]
            # sort by length so a batch pads little, then restore the order
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
            rolls = self.rt.generate([ids[i] for i in order], self.max_tokens, Sampling(temperature=0), self.batch)
            out = [""] * len(ids)
            for i, r in zip(order, rolls):
                out[i] = r.text
            return out

        return run

    def kl(self, texts: list[str]) -> list[float]:
        """KL(base || tuned) per token on each text, in nats, averaged over its tokens."""
        from ..stats import kl_from_logits

        rows = [list(self.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
        out = []
        for state in ("base", "tuned"):
            self._use(state)
            logits = [self.rt.logits_at(r, np.arange(1, len(r))) for r in rows]
            if state == "base":
                self.base_logits = logits
        for ref, tuned in zip(self.base_logits, logits):
            out.append(float(kl_from_logits(ref, tuned).mean()))
        self.base_logits = None
        return out

    def close(self) -> None:
        self._use("base")


def open_backends(base: str, tuned: str, ggufs: list[str], max_tokens: int, threads: int | None = None) -> Backends:
    engine = TorchEngine(base, tuned, max_tokens)
    b = Backends({"base": engine.generator("base"), "tuned": engine.generator("tuned")}, engine.kl,
                 has_chat_template=bool(getattr(engine.tokenizer, "chat_template", None)))
    if ggufs:
        binary = find_llama_server()
        if binary is None:
            for g in ggufs:
                why = ("llama-server is not installed, so this file was not run. "
                       "Install llama.cpp (brew install llama.cpp) or set NEYRA_LLAMA_SERVER")
                b.skipped.append((quant_name(g), why))
        else:
            used: set[str] = set()
            for g in ggufs:
                label = quant_name(g)
                while label in used:
                    label += "+"
                used.add(label)
                b.generate[label] = LlamaRun(binary, g, engine.tokenizer, max_tokens, threads)
    b.close = engine.close
    return b
