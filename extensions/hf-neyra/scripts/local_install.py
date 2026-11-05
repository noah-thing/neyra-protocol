"""Run hf's own install and exec code on this extension, with GitHub replaced by local files.

    python scripts/local_install.py WHEELS_DIR

Needs a huggingface_hub that has `hf extensions`, and a folder holding the neyra wheel while neyra is not on
PyPI. HOME is moved to a scratch folder, so the real ~/.local/share/hf is untouched. Only the network is faked:
the virtualenv, pip install, manifest and exec are hf's.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main(wheels: str) -> int:
    with tempfile.TemporaryDirectory() as home:
        os.environ["HOME"] = home
        os.environ["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
        from huggingface_hub.cli import extensions as ext

        real_run = subprocess.run

        def run(cmd, *a, **kw):
            # The archive URL is the only thing swapped: the same source tree, from disk.
            cmd = [str(HERE) if str(c).endswith("/archive/HEAD.zip") else c for c in cmd]
            if "pip" in cmd and "install" in cmd:
                cmd[cmd.index("install") + 1:cmd.index("install") + 1] = ["--find-links", wheels]
            return real_run(cmd, *a, **kw)

        def fetch(method, url, **kw):
            if url.endswith("/hf-neyra"):  # no top level script, so hf takes the Python path
                raise ext.CLIError("404")
            raise ext.CLIError("offline")

        ext.subprocess.run = run
        ext._github_request = fetch
        ext._github_repo_exists = lambda **kw: True
        ext._fetch_latest_commit_sha = lambda **kw: "0" * 40

        manifest = ext._install_extension(owner="noah-thing", repo_name="hf-neyra", short_name="neyra")
        print(f"installed: type={manifest.type} executable={manifest.executable_path}")
        assert manifest.type == "python" and Path(manifest.executable_path).is_file()

        for args in (["--version"], ["demo"], ["conformance"], ["diff", "nope", "nope"]):
            done = real_run([manifest.executable_path, *args], capture_output=True, text=True, check=False)
            print(f"hf neyra {' '.join(args)} -> exit {done.returncode}: {(done.stdout + done.stderr).strip().splitlines()[-1]}")
        # The route a user's `hf neyra ...` takes: unknown top level command, resolved through the manifest.
        code = ext.dispatch_unknown_top_level_extension(["neyra", "conformance"], {"auth", "download"})
        assert code == 0, code
        print("hf dispatch of `hf neyra conformance`: exit", code)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
