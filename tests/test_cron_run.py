"""Tests for lookout/bin/cron_run.sh, the cron wrapper.

Runs the real script in a throwaway git repo under tmp_path: ``origin`` is a
local bare repo, ``.venv/bin/activate`` is a stub and the script it runs is a
stub that exits with a chosen code. No network, secrets or real venv.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

CRON_RUN = Path(__file__).resolve().parent.parent / "lookout" / "bin" / "cron_run.sh"
SCRIPT_REL = "scripts/stub.py"
STATUS_REL = "logs/cron_scripts_stub.status"
MARKER_REL = "ran.marker"

GIT_ENV: Dict[str, str] = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(cwd: Path, *args: str) -> None:
    """Run git in cwd, failing the test on error.

    :param cwd: directory to run git in
    :param args: git arguments
    """
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={**os.environ, **GIT_ENV},
    )


def _run(repo: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    """Run the wrapper from the repo with no jitter.

    :param repo: working clone to run in
    :param args: wrapper arguments after --sleep 0
    :return: the finished process
    """
    return subprocess.run(
        ["bash", str(CRON_RUN), "--sleep", "0", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, **GIT_ENV},
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clone tracking a local bare origin, with a stub venv and stub script.

    :param tmp_path: pytest temp dir
    :return: the working clone
    """
    bare = tmp_path / "origin.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    _git(tmp_path, "clone", str(bare), str(work))
    _git(work, "checkout", "-b", "main")

    (work / "scripts").mkdir()
    (work / SCRIPT_REL).write_text(
        "import os, sys\n"
        "open(os.environ['MARKER'], 'w').close()\n"
        "sys.exit(int(os.environ['STUB_EXIT']))\n"
    )
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    python = stub_bin / "python"
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python.chmod(0o755)
    (work / ".venv" / "bin").mkdir(parents=True)
    (work / ".venv" / "bin" / "activate").write_text(
        f'export PATH="{stub_bin}:$PATH"\n'
    )
    (work / ".gitignore").write_text(".venv/\nlogs/\nran.marker\n")

    _git(work, "add", "-A")
    _git(work, "commit", "-m", "init")
    _git(work, "push", "-u", "origin", "main")
    return work


def _status_lines(repo: Path) -> List[str]:
    """Lines written to the stub script's status file.

    :param repo: working clone
    :return: status lines, empty when no file exists
    """
    path = repo / STATUS_REL
    return path.read_text().splitlines() if path.exists() else []


def _with_exit(monkeypatch: pytest.MonkeyPatch, repo: Path, code: int) -> None:
    """Point the stub script at its marker file and exit code.

    :param monkeypatch: pytest monkeypatch
    :param repo: working clone
    :param code: exit code the stub script returns
    """
    monkeypatch.setenv("MARKER", str(repo / MARKER_REL))
    monkeypatch.setenv("STUB_EXIT", str(code))


def test_failing_script_exits_with_its_status_and_logs_fail(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing script gives the same exit status and only a FAIL line."""
    _with_exit(monkeypatch, repo, 3)
    result = _run(repo, SCRIPT_REL)
    lines = _status_lines(repo)
    assert result.returncode == 3
    assert len(lines) == 1 and "FAIL" in lines[0]
    assert not any("SUCCESS" in line for line in lines)


def test_passing_script_exits_zero_and_logs_one_success(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passing script gives exit 0 and one SUCCESS line."""
    _with_exit(monkeypatch, repo, 0)
    result = _run(repo, SCRIPT_REL)
    lines = _status_lines(repo)
    assert result.returncode == 0
    assert len(lines) == 1 and lines[0].endswith("SUCCESS")


def test_failed_pull_stops_the_run_before_the_script(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed pull exits non-zero, logs the pull and never runs the script."""
    _with_exit(monkeypatch, repo, 0)
    shutil.rmtree(tmp_path / "origin.git")
    result = _run(repo, SCRIPT_REL)
    lines = _status_lines(repo)
    assert result.returncode != 0
    assert len(lines) == 1 and "FAIL" in lines[0] and "pull" in lines[0]
    assert "pull" in result.stderr
    assert not (repo / MARKER_REL).exists()
