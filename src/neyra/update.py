"""Tells people about new releases and installs them.

A command never waits on the network. It reads what an earlier background check saved, and starts a new check
when that is over a day old. The check asks PyPI for the latest version and sends nothing else.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from importlib import metadata
from pathlib import Path

from . import __version__

URL = "https://pypi.org/pypi/neyra/json"
DAY = 24 * 3600
TIMEOUT = 10


def state_file() -> Path:
    base = os.environ.get("NEYRA_CACHE") or Path.home() / ".cache" / "neyra"
    return Path(base) / "update.json"


def parts(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split(".")[:3] if p.isdigit())


def newer(candidate: str, current: str = __version__) -> bool:
    return parts(candidate) > parts(current)


def saved() -> dict:
    try:
        return json.loads(state_file().read_text())
    except (OSError, ValueError):
        return {}


def refresh() -> str | None:
    """Asks PyPI once and remembers the answer."""
    req = urllib.request.Request(URL, headers={"User-Agent": "neyra"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        latest = json.load(r)["info"]["version"]
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checked": time.time(), "latest": latest}))
    return latest


def interactive() -> bool:
    return sys.stderr.isatty()


def quiet(args) -> bool:
    return (getattr(args, "json", False) or not interactive() or bool(os.environ.get("CI"))
            or os.environ.get("NEYRA_NO_UPDATE_CHECK", "") not in ("", "0"))


def check_in_background() -> None:
    subprocess.Popen([sys.executable, "-m", "neyra.update"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def notice(args) -> None:
    if quiet(args) or getattr(args, "fn", None).__name__ == "cmd_update":
        return
    state = saved()
    if time.time() - state.get("checked", 0) > DAY:
        check_in_background()
    latest = state.get("latest")
    if latest and newer(latest):
        print(f"Neyra {latest} is out. Run: neyra update", file=sys.stderr)


def from_checkout() -> bool:
    try:
        direct = metadata.distribution("neyra").read_text("direct_url.json")
        return bool(direct and json.loads(direct).get("dir_info", {}).get("editable"))
    except (metadata.PackageNotFoundError, ValueError):
        return False


def upgrade_command(prefix: str = sys.prefix) -> list[str]:
    """The way Neyra was installed decides how it is upgraded."""
    path = Path(prefix).as_posix()
    if "/uv/tools/" in path and shutil.which("uv"):
        return ["uv", "tool", "upgrade", "neyra"]
    if "/pipx/" in path and shutil.which("pipx"):
        return ["pipx", "upgrade", "neyra"]
    return [sys.executable, "-m", "pip", "install", "--upgrade", "neyra"]


if __name__ == "__main__":
    # A background check has nobody to tell.
    with suppress(Exception):
        refresh()
