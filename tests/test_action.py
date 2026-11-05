"""Run the shell of action.yml as the runner would, with the inputs it gets, against the demo models."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neyra.demo import build

yaml = pytest.importorskip("yaml")
# The action's steps are bash, and the test stands in for the runner with an sh shim.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="runs the action shell with an sh shim")
ROOT = Path(__file__).resolve().parents[1]
ACTION = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))


def step(name):
    return next(s for s in ACTION["runs"]["steps"] if s.get("name") == name)


def run_check(tmp_path, merged, adapter="adapter"):
    build(tmp_path)
    shim = tmp_path / "bin" / "neyra"
    shim.parent.mkdir()
    shim.write_text(f"#!/bin/sh\nexec {sys.executable} -m neyra.cli \"$@\"\n")
    shim.chmod(0o755)
    out = tmp_path / "github_output"
    out.touch()
    (tmp_path / "hollow").mkdir()
    env = {**os.environ, "PATH": f"{shim.parent}:{os.environ['PATH']}", "PYTHONPATH": str(ROOT / "src"),
           "BASE": "base", "MERGED": merged, "ADAPTER": adapter, "ULPS": "1",
           "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "RUNNER_TEMP": str(tmp_path)}
    done = subprocess.run(["bash", "-e", "-c", step("neyra check")["run"]], cwd=tmp_path, env=env,
                          capture_output=True, text=True, check=False)
    return done, out.read_text(), (tmp_path / "summary.md").read_text()


def test_a_good_merge_passes(tmp_path):
    done, outputs, summary = run_check(tmp_path, "merged")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "equivalent=true" in outputs and "every edited tensor is what peft produces" in summary


def test_an_unapplied_adapter_fails_the_job_and_says_why(tmp_path):
    done, outputs, summary = run_check(tmp_path, "forgot-to-merge")
    assert done.returncode == 1
    assert "equivalent=false" in outputs
    assert "::error title=neyra check::the adapter was never applied" in done.stdout
    assert "THE ADAPTER WAS NEVER APPLIED" in summary


def test_a_check_that_cannot_run_is_not_reported_as_a_failed_merge(tmp_path):
    done, outputs, _ = run_check(tmp_path, "hollow")
    assert done.returncode not in (0, 1)
    assert "equivalent" not in outputs and "could not run the check" in done.stdout


def test_inputs_reach_the_shell_through_the_environment_only():
    """A `${{ }}` inside `run:` would let a repository's input text become shell."""
    for s in ACTION["runs"]["steps"]:
        assert "${{" not in s.get("run", "")
