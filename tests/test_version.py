"""Tests for the release version: the root VERSION file and lookout.__version__.

Checks that VERSION holds a calendar version YYYY.MM.N (ADR-0007) and that the
package exposes the same number.
"""

import re
from pathlib import Path

import lookout

VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
CALVER = re.compile(r"^20\d{2}\.([1-9]|1[0-2])\.[1-9]\d*$")


def test_version_file_is_calendar_version() -> None:
    """VERSION holds one YYYY.MM.N line, month and count without leading zeros."""
    text = VERSION_FILE.read_text()
    assert text.endswith("\n")
    assert CALVER.match(text.strip()), text


def test_package_version_matches_version_file() -> None:
    """lookout.__version__ equals the number in VERSION."""
    assert lookout.__version__ == VERSION_FILE.read_text().strip()
