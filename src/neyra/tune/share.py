"""Opt-in anonymous counts for verdict, lint, train and hop. What is sent is documented in docs/stats.md and
pinned by tests. Nothing here runs unless ENDPOINT is set, the user is at a terminal, no opt-out variable is set,
and the user said yes.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .. import __version__

ENDPOINT = "https://neyra-stats.opencursor.workers.dev/v1/report"
TIMEOUT = 3.0
SCHEMA = 1

FAMILIES = ("qwen", "llama", "gemma", "mistral", "deepseek", "phi", "smollm", "other")
SIZES = ("<1B", "1-3B", "3-8B", "8-30B", ">30B", "unknown")
ROWS = ("<100", "100-999", "1k-9.9k", "10k-99k", "100k+")
FORMATS = ("Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S",
           "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32", "other")
PROBLEMS = ("double_bos", "dropped", "empty", "end_masked", "foreign", "missing_bos", "missing_end", "prefix",
            "rejected", "system", "think", "truncated", "twice", "unlocated", "prompt", "duplicates", "dup_exact",
            "dup_near", "pad_eos", "template_gguf", "tokenizer_gguf", "leakage", "unreadable", "mixed", "other")
PARTS = ("weights", "tokenizer", "template", "bos", "config", "generation")
CAUSES = ("dtype", "extra_tensors", "head", "leftover_adapter", "missing_tensors", "never_applied", "other_base",
          "partly_applied", "rounded_away", "weights_differ", "wrong_scale", "other")
PROBLEM_LIST = {"list": {"max": 12, "key": "id", "item": {"id": list(PROBLEMS), "level": ["error", "warning", "note"]}}}

# Mirrored in stats/src/schema.mjs; a test fails when the two differ.
SPEC = {
    "schema": SCHEMA,
    "common": {"schema": "schema", "kind": "kind", "neyra": "version", "os": ["macos", "linux", "windows", "other"]},
    "kinds": {
        "verdict": {
            "backend": ["torch", "mlx", "other"], "family": list(FAMILIES), "size": list(SIZES),
            "verdict": ["worked", "worked_with_damage", "not_tested", "lost_in_quantization", "prompt_enough",
                        "prompt_better", "did_not_work", "worse_than_base", "cannot_tell"],
            "cases": ["10-29", "30-49", "50-99", "100-199", "200+"], "gain_over_base": "gain",
            "gain_over_prompt": "gain",
            "quants": {"list": {"max": 8, "key": "format", "item": {"format": list(FORMATS), "held": "bool"}}},
        },
        "lint": {"trainer": ["trl", "unsloth", "axolotl", "mlx", "llama-factory"], "family": list(FAMILIES),
                 "size": list(SIZES), "rows": list(ROWS), "blocked": "bool", "problems": PROBLEM_LIST},
        "train": {"family": list(FAMILIES), "size": list(SIZES), "rows": list(ROWS),
                  "outcome": ["trained", "blocked_by_lint"], "loss_improved": ["yes", "no", "unknown"],
                  "revert_exact": "bool", "problems": PROBLEM_LIST},
        "hop": {"family": list(FAMILIES), "size": list(SIZES), "hops": ["2", "3", "4+"], "ok": "bool",
                "first_part": [*PARTS, "other", "none"], "first_cause": [*CAUSES, "none"]},
    },
}
MAX_QUANTS = 8
NOT_QUANTS = ("base", "few-shot", "tuned")
CHOICES = ("ask", "yes", "no")


def config_dir() -> Path:
    if os.environ.get("NEYRA_CONFIG_DIR"):
        return Path(os.environ["NEYRA_CONFIG_DIR"])
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "neyra"


def _name(base: str) -> str:
    return base.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()


def family_of(base: str) -> str:
    for fam in FAMILIES[:-1]:
        if re.search(rf"(?<![a-z]){fam}", _name(base)):
            return fam
    try:
        kind = json.loads((Path(base) / "config.json").read_text()).get("model_type", "")
    except (OSError, ValueError, AttributeError):
        return "other"
    kind = kind.lower() if isinstance(kind, str) else ""
    return next((fam for fam in FAMILIES[:-1] if kind.startswith(fam)), "other")


def size_of(base: str) -> str:
    hit = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])", _name(base))
    if not hit:
        return "unknown"
    b = float(hit[1])
    for limit, label in ((1, "<1B"), (3, "1-3B"), (8, "3-8B"), (30, "8-30B")):
        if b < limit:
            return label
    return ">30B"


def _bucket(n: int, limits: tuple, labels: tuple) -> str:
    return next((lab for lim, lab in zip(limits, labels) if n < lim), labels[-1])


def case_bucket(n: int) -> str:
    return _bucket(n, (30, 50, 100, 200), ("10-29", "30-49", "50-99", "100-199", "200+"))


def rows_bucket(n: int) -> str:
    return _bucket(n, (100, 1000, 10000, 100000), ROWS)


def points(diff: float) -> int:
    return max(-100, min(100, round(diff * 100 / 5) * 5)) + 0


def format_of(label: str) -> str:
    label = label.rstrip("+").upper()
    return label if label in FORMATS else "other"


def os_family() -> str:
    return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system(), "other")


def common(kind: str) -> dict:
    return {"schema": SCHEMA, "kind": kind, "neyra": __version__, "os": os_family()}


def problems_of(findings) -> list[dict]:
    """One entry per problem type at its worst level. Only the type and level are kept."""
    rank = {"note": 0, "warning": 1, "error": 2}
    worst: dict[str, str] = {}
    for f in findings:
        pid = f.id if f.id in PROBLEMS else "other"
        if rank.get(f.level, 0) >= rank.get(worst.get(pid, "note"), 0):
            worst[pid] = f.level if f.level in rank else "note"
    return [{"id": k, "level": worst[k]} for k in sorted(worst)][: PROBLEM_LIST["list"]["max"]]


def build_verdict(r, base: str) -> dict:
    by = {(c.a, c.b): c for c in r.comparisons}
    held: dict[str, bool] = {}
    for arm in r.arms:
        if arm not in NOT_QUANTS:
            ok = all(by[(arm, o)].kind == "better" for o in ("base", "few-shot"))
            fmt = format_of(arm)
            held[fmt] = held.get(fmt, True) and ok
    return {**common("verdict"), "backend": "torch", "family": family_of(base), "size": size_of(base),
            "verdict": r.verdict, "cases": case_bucket(r.n), "gain_over_base": points(by[("tuned", "base")].diff),
            "gain_over_prompt": points(by[("tuned", "few-shot")].diff),
            "quants": [{"format": f, "held": held[f]} for f in sorted(held)][:MAX_QUANTS]}


def build_lint(r, model: str) -> dict:
    return {**common("lint"), "trainer": r.trainer, "family": family_of(model), "size": size_of(model),
            "rows": rows_bucket(r.rows), "blocked": r.errors > 0, "problems": problems_of(r.findings)}


def build_train(model: str, lint_report, result=None) -> dict:
    """result is None when the lint gate stopped the run."""
    improved = "unknown"
    if result is not None and result.loss_before is not None and result.loss_after is not None:
        improved = "yes" if result.loss_after < result.loss_before else "no"
    return {**common("train"), "family": family_of(model), "size": size_of(model),
            "rows": rows_bucket(lint_report.rows), "outcome": "trained" if result is not None else "blocked_by_lint",
            "loss_improved": improved, "revert_exact": bool(result is not None and result.restored),
            "problems": problems_of(lint_report.findings)}


def build_hop(v, first_spec: str) -> dict:
    first = v.first
    return {**common("hop"), "family": family_of(first_spec), "size": size_of(first_spec),
            "hops": "4+" if len(v.hops) > 3 else str(max(len(v.hops), 2)), "ok": first is None,
            "first_part": first.part if first and first.part in PARTS else ("other" if first else "none"),
            "first_cause": ("none" if first is None else first.cause if first.cause in CAUSES else "other")}


def text(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def opted_out() -> bool:
    for var in ("NEYRA_NO_STATS", "DO_NOT_TRACK"):
        if os.environ.get(var, "").strip().lower() not in ("", "0", "false", "no"):
            return True
    return False


def read_choice() -> str:
    try:
        v = json.loads((config_dir() / "stats.json").read_text()).get("choice")
    except (OSError, ValueError, AttributeError):
        return "ask"
    return v if v in CHOICES else "ask"


def write_choice(choice: str) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "stats.json").write_text(json.dumps({"choice": choice}) + "\n")


def last_report() -> str | None:
    try:
        return (config_dir() / "stats-last.json").read_text()
    except OSError:
        return None


def endpoint_ok(url: str) -> bool:
    u = urlparse(url)
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1"))


def post(payload: dict, url: str | None = None) -> bool:
    url = url or ENDPOINT
    if not endpoint_ok(url):
        return False
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "neyra"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001  a failed send must never reach the user
        return False


def send(payload: dict) -> None:
    d = config_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "stats-last.json").write_text(text(payload) + "\n")
    except OSError:
        pass
    post(payload)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def offer(build, json_output: bool = False) -> None:
    """Called once after a command has printed its result. `build` makes the payload; it is not called unless
    a report could be sent. Any failure here is swallowed."""
    try:
        if not ENDPOINT or json_output or opted_out() or os.environ.get("CI") or not interactive():
            return
        choice = read_choice()
        if choice == "no":
            return
        payload = build()
        if choice == "ask":
            print()
            print("Neyra can send an anonymous count of this result, which helps show how often fine-tunes work.")
            print("This is the complete report, with nothing else attached:")
            print(text(payload))
            print(f"It goes to {urlparse(ENDPOINT).hostname}. What is collected: docs/stats.md")
            try:
                answer = input("Send this report, and the same kind of report after later runs? [Y/n] ").strip().lower()
            except EOFError:
                return
            if answer not in ("", "y", "yes"):
                write_choice("no")
                return
            write_choice("yes")
        send(payload)
        if choice == "yes":
            print("Sent an anonymous stats report. `neyra stats show` prints it; `neyra stats off` stops this.")
    except Exception:  # noqa: BLE001  nothing here may break the command that just succeeded
        return
