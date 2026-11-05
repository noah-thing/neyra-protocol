"""Run every command that needs no model against an installed neyra, the way a new user would.

    python -m venv /tmp/v && /tmp/v/bin/pip install dist/neyra-*.whl && /tmp/v/bin/python scripts/smoke_installed.py

It uses the installed `neyra` executable beside the interpreter and nothing from the source tree.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from neyra.demo import build

NEYRA = str(Path(sys.executable).with_name("neyra"))


def run(*args: str, expect: int = 0, cwd: str | None = None) -> str:
    started = time.time()
    done = subprocess.run([NEYRA, *args], check=False, capture_output=True, text=True,
                          cwd=cwd, timeout=120)
    out = done.stdout + done.stderr
    verdict = "ok " if done.returncode == expect else "BAD"
    print(f"{verdict} {time.time() - started:5.2f}s exit {done.returncode} (want {expect})  neyra {' '.join(args)}")
    if done.returncode != expect or "Traceback" in out:
        print(out)
        raise SystemExit(1)
    return out


def main() -> None:
    run("--version")
    run("--help")
    run("doctor")
    out = run("demo")
    assert "THE ADAPTER WAS NEVER APPLIED" in out and "every edited tensor is what peft produces" in out
    run("demo", "edit")
    assert "195/195" in run("conformance")

    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        run("check", "base", "forgot-to-merge", "--adapter", "adapter", expect=1, cwd=tmp)
        run("check", "base", "merged", "--adapter", "adapter", cwd=tmp)
        run("check", "base", "merged", "--json", cwd=tmp, expect=1)
        run("diff", "base", "merged", cwd=tmp)
        run("diff", "base", "merged", "--sample", "2", cwd=tmp)
        run("patch", "base", "merged", "-o", "p.neyra", cwd=tmp)
        run("show", "p.neyra", cwd=tmp)
        run("apply", "base", "p.neyra", "-o", "rebuilt", cwd=tmp)
        run("diff", "rebuilt", "merged", cwd=tmp)
        run("index", "base", "-o", "base.index", cwd=tmp)
        run("keygen", "-o", "k.key", cwd=tmp)
        run("export", "base", "p.neyra", "--format", "peft", "-o", "lora", cwd=tmp)
        run("probe", "base", cwd=tmp, expect=4)
        (Path(tmp) / "facts.json").write_text('[{"subject": "A", "prompt": "{} is", "target": "B"}]')
        run("edit", "base", "facts.json", expect=4, cwd=tmp)
    print("all commands behaved")


if __name__ == "__main__":
    main()
