"""Install the desktop app from the same feed the app updates itself from.

The app is a separate download, because a wheel cannot run an installer and most people who use the command line
never need it. A file fetched here carries no quarantine flag, so the system opens it without a warning.
"""

from __future__ import annotations

import hashlib
import json
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

FEED = "https://neyra.run/desktop/latest.json"
KEY = "darwin-arm64"
HOSTS = ("https://github.com/noah-thing/neyra-protocol/", "https://neyra.run/")


class Stop(RuntimeError):
    pass


def supported() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def destination() -> Path:
    system = Path("/Applications")
    if system.is_dir() and shutil.os.access(system, shutil.os.W_OK):
        return system / "Neyra.app"
    return Path.home() / "Applications" / "Neyra.app"


def installed_version(app: Path) -> str | None:
    try:
        with (app / "Contents" / "Info.plist").open("rb") as f:
            return plistlib.load(f).get("CFBundleShortVersionString")
    except (OSError, plistlib.InvalidFileException):
        return None


def read_feed(url: str = FEED) -> tuple[str, str, str]:
    with urllib.request.urlopen(url, timeout=30) as r:
        feed = json.loads(r.read())
    build = feed["builds"][KEY]
    if not build["url"].startswith(HOSTS):
        raise Stop(f"the feed points somewhere unexpected: {build['url']}")
    return feed["version"], build["url"], build["sha256"]


def fetch(url: str, target: Path, sha256: str) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as r, target.open("wb") as f:
        while chunk := r.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != sha256:
        raise Stop("the download does not match its checksum, so nothing was installed")


def install(say: Callable[[str], None] = print, feed: str = FEED) -> Path:
    if not supported():
        raise Stop("the desktop app needs an Apple Silicon Mac for now; every neyra command works here without it")
    app = destination()
    version, url, sha256 = read_feed(feed)
    if installed_version(app) == version:
        say(f"Neyra {version} is already installed")
    else:
        say(f"downloading Neyra {version}")
        with tempfile.TemporaryDirectory() as tmp:
            zipped = Path(tmp) / "neyra.zip"
            fetch(url, zipped, sha256)
            subprocess.run(["ditto", "-x", "-k", str(zipped), tmp], check=True)
            app.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(app, ignore_errors=True)
            shutil.move(str(Path(tmp) / "Neyra.app"), str(app))
        say(f"installed to {app}")
    subprocess.run(["open", str(app)], check=False)
    return app
