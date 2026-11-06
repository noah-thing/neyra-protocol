import hashlib
import io
import json

import pytest

from neyra import cli, desktop


def feed(url="https://github.com/noah-thing/neyra-protocol/releases/download/x/n.zip", sha="ab"):
    return json.dumps({"version": "1.2.3", "builds": {desktop.KEY: {"url": url, "sha256": sha}}}).encode()


def serve(monkeypatch, body):
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))


def test_refuses_where_there_is_no_build(monkeypatch):
    monkeypatch.setattr(desktop, "supported", lambda: False)
    with pytest.raises(desktop.Stop, match="Apple Silicon"):
        desktop.install()


def test_reads_the_feed(monkeypatch):
    serve(monkeypatch, feed())
    assert desktop.read_feed()[0] == "1.2.3"


def test_refuses_a_feed_pointing_elsewhere(monkeypatch):
    serve(monkeypatch, feed(url="https://example.com/evil.zip"))
    with pytest.raises(desktop.Stop, match="unexpected"):
        desktop.read_feed()


def test_refuses_a_bad_checksum(monkeypatch, tmp_path):
    serve(monkeypatch, b"payload")
    with pytest.raises(desktop.Stop, match="checksum"):
        desktop.fetch("https://neyra.run/x", tmp_path / "x", "00")


def test_keeps_a_good_download(monkeypatch, tmp_path):
    serve(monkeypatch, b"payload")
    desktop.fetch("https://neyra.run/x", tmp_path / "x", hashlib.sha256(b"payload").hexdigest())
    assert (tmp_path / "x").read_bytes() == b"payload"


def test_opens_an_app_already_up_to_date(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(desktop, "supported", lambda: True)
    monkeypatch.setattr(desktop, "destination", lambda: tmp_path / "Neyra.app")
    monkeypatch.setattr(desktop, "installed_version", lambda app: "1.2.3")
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: opened.append(cmd))
    serve(monkeypatch, feed())
    desktop.install(say=lambda m: None)
    assert opened == [["open", str(tmp_path / "Neyra.app")]]


def test_command_reports_a_refusal(monkeypatch, capsys):
    monkeypatch.setattr(desktop, "supported", lambda: False)
    assert cli.main(["app"]) != 0
    assert "Apple Silicon" in capsys.readouterr().err
