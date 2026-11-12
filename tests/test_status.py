"""
The refresh run's status document (#268): `lookout.snapshot.status`.

`refresh.main` writing it on every run is tested in `tests/test_refresh.py`.
Here: which code a run reports (version, and the checkout's SHA from git,
asked of a throwaway repo in `tmp_path`, never the real checkout), and the
golden status documents the lane server serves at `/status`, under
`tests/golden/status/<scenario>.status.json`. Regenerate them with:

    LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_status.py \
        -p no:cacheprovider
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

import lookout
from lookout.snapshot.status import RunStatus, code_version
from tests.snapshot_helpers import FIXED_CODE, REGENERATE, dry_station
from tests.snapshot_helpers import raining_station, seasoned_station

GOLDEN_DIR = Path(__file__).parent / "golden" / "status"
SCENARIOS = {
    "dry": dry_station,
    "raining": raining_station,
    "seasoned": seasoned_station,
}


def git(root: Path, *args: str) -> str:
    """
    Run git in a throwaway repo, isolated from the user's and the system's
    git config (signing, hooks, templates).

    :param root: The repo.
    :param args: The git arguments.
    :return: Its stdout, stripped.
    """
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestCodeVersion:
    def test_a_git_checkout_reports_its_head(self, tmp_path):
        git(tmp_path, "init", "-q")
        (tmp_path / "VERSION").write_text("2099.1.1\n")
        git(tmp_path, "add", "VERSION")
        git(tmp_path, "commit", "-q", "-m", "one")

        code = code_version(tmp_path)

        assert code.sha == git(tmp_path, "rev-parse", "HEAD")
        assert len(code.sha) == 40

    def test_the_version_is_the_packages_release(self, tmp_path):
        assert code_version(tmp_path).version == lookout.__version__

    def test_a_directory_that_is_not_a_checkout_has_no_sha(self, tmp_path):
        assert code_version(tmp_path).sha is None

    def test_no_git_on_the_path_has_no_sha(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))

        assert code_version(tmp_path).sha is None


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_status_documents_match_the_golden_files(scenario):
    snapshot = SCENARIOS[scenario]()
    document = RunStatus(
        station_mac=snapshot.station.mac,
        code=FIXED_CODE,
        started_at_ms=snapshot.built_at_ms - 40_000,
        ended_at_ms=snapshot.built_at_ms + 2_000,
        outcome="ok",
        built_at_ms=snapshot.built_at_ms,
    ).document()

    path = GOLDEN_DIR / f"{scenario}.status.json"
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if REGENERATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert json.loads(path.read_text()) == {
        "kind": "status",
        "station_mac": snapshot.station.mac,
        "version": "2026.11.1",
        "sha": FIXED_CODE.sha,
        "started_at_ms": snapshot.built_at_ms - 40_000,
        "ended_at_ms": snapshot.built_at_ms + 2_000,
        "outcome": "ok",
        "error": None,
        "built_at_ms": snapshot.built_at_ms,
    }
