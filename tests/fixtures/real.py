"""
real.py: Load the frozen real archive slice.

`real_slice.parquet` is two days of the station's own archive (2025-11-17
00:00 to 2025-11-18 21:00 local), every column and dtype as on disk, newest
row first, with the passkey scrubbed. It holds a rain day and a 94-minute
upload gap. `make_real_fixtures.py` regenerates it from a local parquet.
"""

from pathlib import Path

import pandas as pd

REAL_SLICE_PATH = Path(__file__).parent / "real_slice.parquet"
SLICE_START = "2025-11-17T08:00:00Z"  # 2025-11-17 00:00 local
SLICE_END = "2025-11-19T05:00:00Z"  # 2025-11-18 21:00 local
AMBIENT_END = "2025-11-19T08:00:00Z"  # end of the recorded history
FAKE_PASSKEY = "0" * 32


def load_real_slice() -> pd.DataFrame:
    """
    Read the frozen slice; each call returns a fresh copy.

    :return: The archive rows, newest first.
    """
    return pd.read_parquet(REAL_SLICE_PATH, engine="pyarrow")
