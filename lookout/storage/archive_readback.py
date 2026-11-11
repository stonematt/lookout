"""
archive_readback.py: Check a stored archive against the one a repair started
from, after the repair saved it (#170).

The read-back is raw: `storj.read_df_from_s3`, the stored bytes as stored.
Never read it back through `lookout.cli.diagnostics.load_station` or
`storj.load_archive`: they normalize, which sorts `dateutc` ascending and
drops duplicates, so they cannot show the stored order or a duplicate. That
misled the #95 check. The save guard (#190) does not cover a repair save
either, since a repair may drop or blank readings on purpose.

Functions:
- `ReadbackFailed`: a repair whose read-back failed a check.
- `Check`: one check's verdict and report line.
- `read_back`: the stored archive, raw.
- `content_hash`, `outside`: the outside-the-windows comparison.
- `check_readback`: the order, count and outside-the-windows checks.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from lookout.core import archive_merge
from lookout.storage import storj


class ReadbackFailed(RuntimeError):
    """A repair saved, and the read-back of the saved archive failed a check."""


@dataclass
class Check:
    """
    One read-back check.

    :param name: What it checks.
    :param ok: Whether it passed.
    :param detail: The numbers behind the verdict.
    """

    name: str
    ok: bool
    detail: str

    def line(self, rollback_key: Optional[str] = None) -> str:
        """
        The check's report line.

        :param rollback_key: The backup to roll back to; named on failure.
        :return: One line.
        """
        if self.ok:
            return f"Read-back check passed: {self.name} ({self.detail})"
        line = f"Read-back check FAILED: {self.name} ({self.detail})"
        if rollback_key:
            line += f"; roll back from {rollback_key}"
        return line


def read_back(bucket: str, mac: str) -> pd.DataFrame:
    """
    The stored archive, raw: in stored order, with duplicates, unnormalized.

    Do not replace this with `load_station` or `load_archive`; see the module
    docstring.

    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :return: The stored rows as stored.
    :raises storj.StorageReadError: If the read failed, or the archive is
        missing.
    """
    key = f"{mac}.{storj.HISTORY_FILE_TYPE}"
    stored = storj.read_df_from_s3(bucket, key, file_type=storj.HISTORY_FILE_TYPE)
    if stored is None:
        raise storj.StorageReadError(f"s3://{bucket}/{key} is missing after the save")
    return stored


def content_hash(df: pd.DataFrame, columns: Sequence[str]) -> int:
    """
    A hash of the rows' content, whatever their order.

    :param df: The rows.
    :param columns: The columns to hash.
    :return: One integer for the rows' values, sorted by `dateutc`.
    """
    rows = df.sort_values("dateutc", kind="stable")[list(columns)].reset_index(
        drop=True
    )
    for col in rows.columns:
        if isinstance(rows[col].dtype, pd.CategoricalDtype):
            rows[col] = rows[col].astype(object)
    hashes = pd.util.hash_pandas_object(rows, index=False)
    return int(hashes.sum() % (1 << 63)) ^ len(rows)


def outside(df: pd.DataFrame, bounds: Sequence[Tuple[int, int]]) -> pd.DataFrame:
    """
    The rows outside every window.

    :param df: The rows.
    :param bounds: Each window's `dateutc` start (inclusive) and end (exclusive).
    :return: The rows in no window.
    """
    keep = pd.Series(True, index=df.index)
    for start, end in bounds:
        keep &= ~((df["dateutc"] >= start) & (df["dateutc"] < end))
    return df[keep]


def check_readback(
    stored: pd.DataFrame,
    before: pd.DataFrame,
    expected_rows: int,
    bounds: Sequence[Tuple[int, int]],
) -> List[Check]:
    """
    Check a raw read-back of the saved archive.

    :param stored: The raw read-back (`read_back`).
    :param before: The archive the repair loaded.
    :param expected_rows: The row count the repair predicted.
    :param bounds: Each repair window's `dateutc` start and end (exclusive).
    :return: The row count, outside-the-windows, and order checks.
    """
    checks = [
        Check(
            "row count",
            len(stored) == expected_rows,
            f"{len(stored)} stored, {expected_rows} predicted",
        )
    ]
    columns = [
        c
        for c in before.columns
        if c in stored.columns and c not in archive_merge.DERIVED_COLUMNS
    ]
    kept_before, kept_after = outside(before, bounds), outside(stored, bounds)
    same = len(kept_before) == len(kept_after) and content_hash(
        kept_before, columns
    ) == content_hash(kept_after, columns)
    checks.append(
        Check(
            "rows outside the windows unchanged",
            same,
            f"{len(kept_after)} stored, {len(kept_before)} before",
        )
    )
    dates = stored["dateutc"]
    decreasing = bool(dates.is_monotonic_decreasing and dates.is_unique)
    detail = f"{int(dates.duplicated().sum())} duplicates"
    if not dates.is_monotonic_decreasing:
        detail += ", not newest first"
    checks.append(Check("dateutc strictly decreasing", decreasing, detail))
    return checks
