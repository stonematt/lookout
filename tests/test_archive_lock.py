"""
The archive-writer lock (#244): `lookout.utils.archive_lock`.

`tests/conftest.py` points `LOOKOUT_LOCK_FILE` at a temp path for every
test, so none of these touch the real lock file.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from lookout.utils import archive_lock
from lookout.utils.archive_lock import ArchiveLockHeld, hold_archive_lock, writer_lock

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_lock_file_is_the_env_var_and_its_directory_is_created(
    archive_lock_file: Path,
):
    assert not archive_lock_file.parent.exists()

    with hold_archive_lock() as held:
        assert held == archive_lock_file
        assert archive_lock_file.exists()


def test_the_default_lock_file_is_under_local_state(monkeypatch, tmp_path):
    monkeypatch.delenv(archive_lock.LOCK_FILE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / ".local" / "state" / "lookout" / "archive-writer.lock"

    assert archive_lock.lock_file() == expected
    with hold_archive_lock():
        assert expected.exists()


def test_a_held_lock_refuses_a_second_writer_naming_the_lock_file(
    archive_lock_file: Path,
):
    with hold_archive_lock():
        with pytest.raises(ArchiveLockHeld) as refused:
            hold_archive_lock()

    assert refused.value.path == archive_lock_file
    assert str(archive_lock_file) in str(refused.value)


def test_a_released_lock_is_free_for_the_next_writer():
    with hold_archive_lock():
        pass
    with hold_archive_lock():
        pass


def test_a_lock_is_released_when_its_holder_raises():
    with pytest.raises(RuntimeError):
        with hold_archive_lock():
            raise RuntimeError("the run failed")

    with hold_archive_lock():
        pass


def test_a_killed_process_leaves_the_lock_free(archive_lock_file: Path):
    code = (
        "import sys\n"
        "from lookout.utils.archive_lock import hold_archive_lock\n"
        "lock = hold_archive_lock()\n"
        "print('held', flush=True)\n"
        "sys.stdin.read()\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ArchiveLockHeld):
            hold_archive_lock()
    finally:
        holder.kill()
        holder.wait()

    with hold_archive_lock():
        pass


def test_a_writer_without_its_save_flag_takes_no_lock(archive_lock_file: Path):
    with hold_archive_lock():
        with writer_lock(False):
            pass
        with pytest.raises(ArchiveLockHeld):
            writer_lock(True)
