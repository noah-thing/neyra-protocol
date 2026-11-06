import json
import time
from types import SimpleNamespace

import pytest

from neyra import update


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NEYRA_CACHE", str(tmp_path))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NEYRA_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update.sys.stderr, "isatty", lambda: True)
    started = []
    monkeypatch.setattr(update, "check_in_background", lambda: started.append(True))
    return started


def args(**extra):
    return SimpleNamespace(json=False, fn=lambda: None, **extra)


def save(latest, age=0):
    update.state_file().parent.mkdir(parents=True, exist_ok=True)
    update.state_file().write_text(json.dumps({"checked": time.time() - age, "latest": latest}))


def test_versions_compare_as_numbers():
    assert update.newer("0.10.0", "0.9.9") and not update.newer("0.3.4", "0.3.4")


def test_a_newer_release_gets_one_line(capsys):
    save("99.0.0")
    update.notice(args())
    assert "99.0.0 is out" in capsys.readouterr().err


def test_nothing_is_said_on_the_latest_version(capsys):
    save(update.__version__)
    update.notice(args())
    assert capsys.readouterr().err == ""


def test_a_stale_answer_starts_a_background_check(cache):
    save(update.__version__, age=2 * update.DAY)
    update.notice(args())
    assert cache == [True]


def test_a_fresh_answer_stays_off_the_network(cache):
    save(update.__version__)
    update.notice(args())
    assert cache == []


@pytest.mark.parametrize("env", ["CI", "NEYRA_NO_UPDATE_CHECK"])
def test_automation_hears_nothing(monkeypatch, capsys, cache, env):
    monkeypatch.setenv(env, "1")
    save("99.0.0", age=2 * update.DAY)
    update.notice(args())
    assert capsys.readouterr().err == "" and cache == []


def test_json_output_stays_clean(capsys):
    save("99.0.0")
    update.notice(SimpleNamespace(json=True, fn=lambda: None))
    assert capsys.readouterr().err == ""


def test_upgrade_follows_the_install(monkeypatch):
    monkeypatch.setattr(update.shutil, "which", lambda name: f"/bin/{name}")
    assert update.upgrade_command("/home/a/.local/share/uv/tools/neyra")[:2] == ["uv", "tool"]
    assert update.upgrade_command("/home/a/.local/share/pipx/venvs/neyra")[0] == "pipx"
    assert update.upgrade_command("/home/a/venv")[1:3] == ["-m", "pip"]
