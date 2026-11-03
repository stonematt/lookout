"""
archive_lock.py: The archive-writer lock, one per host (#244).

An archive writer is a command that saves the archive, the Event Catalog or
the energy catalog. At most one runs on a host at a time: each takes this
lock at its entry point, before it does any work. The lock is a
non-blocking `flock` on a file, so the OS releases it when the process
exits, however it exits.

The lock file is `~/.local/state/lookout/archive-writer.lock`; the
`LOOKOUT_LOCK_FILE` env var moves it (tests point it under `tmp_path`). A
missing parent directory is created.

Functions:
    lock_file: The lock file's path.
    hold_archive_lock: Take the lock now, or raise `ArchiveLockHeld`.
    writer_lock: The lock for a writer that saves, a no-op for one that
        does not.
"""

import contextlib
import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import ContextManager, Optional, Type, Union

LOCK_FILE_ENV = "LOOKOUT_LOCK_FILE"
DEFAULT_LOCK_FILE = Path(".local") / "state" / "lookout" / "archive-writer.lock"


class ArchiveLockHeld(Exception):
    """
    Another archive writer holds the lock.

    :param path: The lock file.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(f"Another archive writer holds the lock {path}")
        self.path = path


class ArchiveLock:
    """
    A held archive-writer lock; released on `release` or leaving a `with`.

    :param path: The lock file.
    :param fd: The open file descriptor that holds the `flock`.
    """

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd: Optional[int] = fd

    def release(self) -> None:
        """Release the lock; a second call does nothing."""
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> Path:
        return self.path

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()


def lock_file() -> Path:
    """
    The lock file's path: `LOOKOUT_LOCK_FILE`, else the default under `~`.

    :return: The path; read at each call, so a changed env var takes effect.
    """
    override = os.environ.get(LOCK_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_LOCK_FILE


def hold_archive_lock() -> ArchiveLock:
    """
    Take the archive-writer lock now, without waiting.

    :return: The held lock; use it in a `with` so it is released.
    :raises ArchiveLockHeld: If another writer holds it.
    """
    path = lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ArchiveLockHeld(path) from None
    return ArchiveLock(path, fd)


def writer_lock(
    saves: bool,
) -> Union[ArchiveLock, ContextManager[None]]:
    """
    The lock a command takes: held when it saves, none when it does not.

    A read-only command, or a writer run without its save flag, runs while
    another writer holds the lock.

    :param saves: Whether this run saves the archive or a catalog.
    :return: A held lock, or a no-op context when `saves` is False.
    :raises ArchiveLockHeld: If `saves` and another writer holds the lock.
    """
    if saves:
        return hold_archive_lock()
    return contextlib.nullcontext()
