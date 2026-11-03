"""Lookout: the weather station dashboard's Python package.

Offers ``__version__``, the release version read from the root ``VERSION``
file (ADR-0007). Importing the package reads that one file and nothing else.
"""

from pathlib import Path

__version__: str = (
    (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()
)
