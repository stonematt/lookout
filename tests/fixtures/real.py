"""
real.py: Load the frozen real archive slice.

`real_slice.parquet` is two days of the station's own archive (2025-11-17
00:00 to 2025-11-18 21:00 local), every column and dtype as on disk, newest
row first, with the passkey scrubbed. It holds a rain day and a 94-minute
upload gap. `make_real_fixtures.py` regenerates it from a local parquet.
"""

from pathlib import Path

import pandas as pd

from tests.fixtures.make_real_fixtures import SLICE_END, SLICE_START

REAL_SLICE_PATH = Path(__file__).parent / "real_slice.parquet"

__all__ = ["REAL_SLICE_PATH", "SLICE_END", "SLICE_START", "load_real_slice"]


def load_real_slice() -> pd.DataFrame:
    """
    Read the frozen slice; each call returns a fresh copy.

    :return: The archive rows, newest first.
    """
    return pd.read_parquet(REAL_SLICE_PATH, engine="pyarrow")
