"""Shared pieces for the src_* jobs: range reads, GGUF and safetensors headers, name mapping."""
import glob
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request

import numpy as np

HUB = "https://huggingface.co/%s/resolve/main/%s"
GT = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
      10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
# ggml type id -> (name, block width, bytes per block)
QT = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
      7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
      12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}


def get(url, start, end, tries=8):
    for k in range(tries):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "neyra-src"})
        try:
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) == end - start + 1:
                return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503):
                raise
        except (OSError, TimeoutError):
            time.sleep(1)
        time.sleep(3 * (k + 1))
    raise RuntimeError("range read failed " + url)


def get_big(url, start, end, piece=64 << 20):
    out = bytearray()
    for s in range(start, end + 1, piece):
        out += get(url, s, min(end, s + piece - 1))
    return bytes(out)


class Buf:
    def __init__(self, b):
        self.b, self.p = b, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise EOFError
        r = self.b[self.p:self.p + n]
        self.p += n
        return r

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.u("Q")).decode("utf8", "replace")

    def value(self, t):
        if t in GT:
            return self.u(GT[t][0])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u("I"), self.u("Q")
            if et in GT:
                self.take(GT[et][1] * n)
                return None
            for _ in range(n):
                self.value(et)
            return None
        raise ValueError(t)


def gguf_header(url, first=8 << 20):
    n = first
    while True:
        try:
            b = Buf(get(url, 0, n - 1))
            assert b.take(4) == b"GGUF"
            b.u("I")
            nt, nkv = b.u("Q"), b.u("Q")
            kv = {}
            for _ in range(nkv):
                k = b.string()
                kv[k] = b.value(b.u("I"))
            tensors = {}
            for _ in range(nt):
                name = b.string()
                dims = [b.u("Q") for _ in range(b.u("I"))]
                ty, off = b.u("I"), b.u("Q")
                tensors[name] = {"dims": dims, "type": ty, "off": off}
            al = kv.get("general.alignment", 32)
            return {"url": url, "kv": kv, "tensors": tensors, "data0": (b.p + al - 1) // al * al}
        except EOFError:
            n *= 2
            if n > 128 << 20:
                raise


def gguf_nbytes(t):
    _name, bw, bs = QT[t["type"]]
    n = int(np.prod(t["dims"]))
    return n // bw * bs


def gguf_raw(h, name):
    t = h["tensors"][name]
    s = h["data0"] + t["off"]
    return get_big(h["url"], s, s + gguf_nbytes(t) - 1)


class Safetensors:
    """Read-only view of one safetensors file: a local snapshot when cached on this machine, else range reads."""

    def __init__(self, repo):
        self.repo = repo
        local = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/model.safetensors"))
        self.path = local[0] if local else None
        self.url = HUB % (repo, "model.safetensors")
        if self.path:
            with open(self.path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n))
        else:
            n = struct.unpack("<Q", get(self.url, 0, 7))[0]
            head = json.loads(get(self.url, 8, 8 + n - 1))
        head.pop("__metadata__", None)
        self.base, self.meta = 8 + n, head

    def has(self, name):
        return name in self.meta

    def read(self, name):
        m = self.meta[name]
        s, e = m["data_offsets"]
        if self.path:
            raw = np.memmap(self.path, np.uint8, "r", self.base + s, e - s)
        else:
            raw = np.frombuffer(get_big(self.url, self.base + s, self.base + e - 1), np.uint8)
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[m["dtype"]]
        return raw.view(dt).reshape(m["shape"]), m["dtype"]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


PAIRS = [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
         ("self_attn.o_proj", "attn_output"), ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
         ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"), ("mlp.gate_proj", "ffn_gate"),
         ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down")]


def hf_name(gname):
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gname == "output_norm.weight":
        return "model.norm.weight"
    if gname == "output.weight":
        return "lm_head.weight"
    m = re.match(r"blk\.(\d+)\.(\w+)\.(weight|bias)$", gname)
    if m:
        for hf, g in PAIRS:
            if g == m[2]:
                return f"model.layers.{m[1]}.{hf}.{m[3]}"
    return None
