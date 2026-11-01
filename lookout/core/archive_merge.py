"""
archive_merge.py: how fetched readings join the archive.

The one place that decides who wins when Ambient's readings meet stored ones
(GLOSSARY: **archive merge**). Ambient is the source of truth: for each value
Ambient sends, Ambient's value wins; where Ambient sends nothing (a NaN, or a
column its frame lacks), the stored value stays. Readings are matched on
`dateutc`, and a value, not a whole row, is the unit of precedence.

Every merge reports what it changed (`MergeReport`): readings added, values
filled (stored had none) and values updated (Ambient's differs), plus the
columns only one side has. Comparison is exact after `dateutc.normalize`
(no float tolerance); a value missing on both sides is unchanged.

Pure frame logic: no storage access, no Streamlit. A deliberate correction
is a replace, never a merge.

Functions:
- merge: Merge Ambient's readings into the stored archive; returns `Merged`.

Classes:
- MergeReport: What one merge changed; `describe()` is its log line.
- Merged: `(archive, report)`, the outcome of `merge`.

Example:
    archive, report = archive_merge.merge(stored, fetched)
    logger.info(f"Archive merge: {report.describe()}")
    if report.changed:
        ...  # save `archive`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import pandas as pd

from lookout.utils.dateutc import normalize


@dataclass(frozen=True)
class MergeReport:
    """
    What one archive merge changed.

    :param added: Readings only Ambient had, now in the archive.
    :param filled: Values the stored reading lacked (NaN) that Ambient sent.
    :param updated: Values both sides held, where Ambient's differs.
    :param stored_only: Columns the stored frame has and Ambient's lacks,
        in stored order; their values are kept. Columns are compared only
        when both sides hold readings.
    :param ambient_only: Columns Ambient's frame has and the stored one
        lacks, in Ambient's order; kept, so a new sensor is never dropped.
    """

    added: int = 0
    filled: int = 0
    updated: int = 0
    stored_only: Tuple[str, ...] = ()
    ambient_only: Tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """True when the merge added, filled or updated anything."""
        return bool(self.added or self.filled or self.updated)

    def describe(self) -> str:
        """
        The report as one log line.

        :return: E.g. "3 added, 0 filled, 24 updated; stored only: hour;
            Ambient only: none".
        """
        return (
            f"{self.added} added, {self.filled} filled, {self.updated} updated; "
            f"stored only: {', '.join(self.stored_only) or 'none'}; "
            f"Ambient only: {', '.join(self.ambient_only) or 'none'}"
        )


class Merged(NamedTuple):
    """
    The outcome of `merge`.

    :param archive: The merged archive, newest `dateutc` first, index reset.
    :param report: What the merge changed.
    """

    archive: pd.DataFrame
    report: MergeReport


def merge(stored: pd.DataFrame, ambient: pd.DataFrame) -> Merged:
    """
    Merge Ambient's readings into the stored archive, Ambient winning.

    Both frames go through `dateutc.normalize` first. A reading only Ambient
    has is added. For a reading both have, value by value: Ambient's non-NaN
    value wins; Ambient's NaN, or a column Ambient's frame lacks, keeps the
    stored value. No valid stored reading is dropped; only rows `normalize`
    rejects (no valid `dateutc`, a repeated one) are left out, as
    `storj.load_archive` already does.

    :param stored: The stored archive (or the frame standing in for it).
    :param ambient: Readings fetched from Ambient, raw.
    :return: `Merged(archive, report)`. The archive's columns are the stored
        columns in their order, then the Ambient-only ones.
    """
    stored = normalize(stored)
    ambient = normalize(ambient)
    if ambient.empty:
        return Merged(_newest_first(stored), MergeReport())
    if stored.empty:
        return Merged(_newest_first(ambient), MergeReport(added=len(ambient)))

    stored_at = stored.set_index("dateutc")
    ambient_at = ambient.set_index("dateutc")
    shared = stored_at.index.intersection(ambient_at.index)
    common = [c for c in stored_at.columns if c in ambient_at.columns]
    ambient_only = tuple(c for c in ambient_at.columns if c not in stored_at.columns)

    both = stored_at.loc[shared].copy()
    filled = updated = 0
    for col in common:
        old, new = both[col], ambient_at.loc[shared, col]
        sent = new.notna()
        had = old.notna()
        filled += int((sent & ~had).sum())
        updated += int((sent & had & _differs(old, new)).sum())
        both[col] = new.where(sent, old)
    for col in ambient_only:
        both[col] = ambient_at.loc[shared, col]

    pieces = [
        p
        for p in (stored_at.drop(index=shared), both, ambient_at.drop(index=shared))
        if not p.empty
    ]
    columns = list(stored.columns) + list(ambient_only)
    merged = pd.concat(pieces, sort=False).reset_index().reindex(columns=columns)
    report = MergeReport(
        added=len(ambient_at) - len(shared),
        filled=filled,
        updated=updated,
        stored_only=tuple(c for c in stored_at.columns if c not in ambient_at.columns),
        ambient_only=ambient_only,
    )
    return Merged(_newest_first(merged), report)


def _differs(old: pd.Series, new: pd.Series) -> pd.Series:
    """
    Where two aligned columns hold different values, exactly.

    :param old: The stored values.
    :param new: Ambient's values, same index.
    :return: A boolean Series; True also where the values cannot be
        compared (e.g. a tz-naive and a tz-aware time), which differ.
    """
    try:
        return old.ne(new)
    except TypeError:
        return pd.Series(True, index=old.index)


def _newest_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort readings newest `dateutc` first, the stored order.

    :param df: Readings with a `dateutc` column, or an empty frame.
    :return: `df` sorted (stable) with its index reset; an empty `df` as is.
    """
    if df.empty:
        return df
    return df.sort_values("dateutc", ascending=False, kind="stable").reset_index(
        drop=True
    )
