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

The same comparison guards every archive save (`storj.save_archive`):
`loss` says what a new archive would lose against the stored one, readings
dropped and values blanked, whoever built it. No routine write may lose
either (GLOSSARY: **archive**); only an explicit repair may.

Pure frame logic: no storage access, no Streamlit. A deliberate correction
is a replace, never a merge.

Functions:
- merge: Merge Ambient's readings into the stored archive; returns `Merged`.
- loss: What saving a new archive over the stored one would lose; `Loss`.
- replace: Swap the stored readings in a window for corrected ones; `Replaced`.

Classes:
- MergeReport: What one merge changed; `describe()` is its log line.
- Merged: `(archive, report)`, the outcome of `merge`.
- Loss: Stored readings dropped and stored values blanked by a save.
- ReplaceReport: What one replace swapped out and put in.
- Replaced: `(archive, report)`, the outcome of `replace`.

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


@dataclass(frozen=True)
class Loss:
    """
    What a new archive would lose against the stored one.

    :param dropped: Stored readings (`dateutc`) absent from the new archive.
    :param blanked: Stored values (non-NaN) that are NaN in the new archive,
        or whose column the new archive lacks, over the readings both hold.
    :param blanked_columns: The columns those values are in, stored order.
    """

    dropped: int = 0
    blanked: int = 0
    blanked_columns: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """True when the new archive drops a reading or blanks a value."""
        return bool(self.dropped or self.blanked)

    def describe(self) -> str:
        """
        The loss as one phrase for an error or log line.

        :return: E.g. "drop 1 stored reading and blank 86 stored values
            (hour)".
        """
        readings = "reading" if self.dropped == 1 else "readings"
        values = "value" if self.blanked == 1 else "values"
        columns = ", ".join(self.blanked_columns) or "none"
        return (
            f"drop {self.dropped} stored {readings} and blank {self.blanked} "
            f"stored {values} ({columns})"
        )


def loss(stored: pd.DataFrame, new: pd.DataFrame) -> Loss:
    """
    What saving `new` over the stored archive would lose.

    Both frames go through `dateutc.normalize` first, so a stored duplicate
    or invalid row is not a reading `new` could drop. Readings are matched
    on `dateutc`. A changed value is not a loss (that is an update); only a
    stored value with nothing in its place is. A value missing on both
    sides is not lost.

    :param stored: The stored archive, as read.
    :param new: The archive about to be saved, any order.
    :return: The `Loss`; falsy when nothing stored would be lost.
    """
    stored = normalize(stored)
    if stored.empty:
        return Loss()
    new = normalize(new)
    stored_at = stored.set_index("dateutc")
    new_at = new.set_index("dateutc") if not new.empty else pd.DataFrame()
    kept = stored_at.index.intersection(new_at.index)
    old = stored_at.loc[kept]
    # A column `new` lacks reindexes to NaN: every stored value in it is lost.
    now = new_at.reindex(index=kept, columns=old.columns)
    per_column = (old.notna() & now.isna()).sum()
    lost = per_column[per_column > 0]
    return Loss(
        dropped=len(stored_at) - len(kept),
        blanked=int(lost.sum()),
        blanked_columns=tuple(lost.index),
    )


@dataclass(frozen=True)
class ReplaceReport:
    """
    What one replace changed.

    :param replaced: Stored readings in the window that were swapped out.
    :param written: Replacement readings now in the window.
    """

    replaced: int = 0
    written: int = 0

    def describe(self) -> str:
        """
        The report as one log line.

        :return: E.g. "replaced 120 readings with 118".
        """
        readings = "reading" if self.replaced == 1 else "readings"
        return f"replaced {self.replaced} {readings} with {self.written}"


class Replaced(NamedTuple):
    """
    The outcome of `replace`.

    :param archive: The archive with the window replaced, newest first.
    :param report: What the replace swapped.
    """

    archive: pd.DataFrame
    report: ReplaceReport


def replace(
    stored: pd.DataFrame, start_ms: int, end_ms: int, replacement: pd.DataFrame
) -> Replaced:
    """
    Replace the stored readings in a window with corrected ones.

    A deliberate correction, never a merge: every stored reading with
    `start_ms <= dateutc < end_ms` goes, whatever the replacement holds, and
    the replacement's readings in the window take their place whole.
    Readings outside the window stay exactly as stored, and replacement
    readings outside it are ignored. The save guard (`loss`) sees the
    difference; only an explicit repair may pass it.

    :param stored: The stored archive.
    :param start_ms: The window's start, epoch ms, inclusive.
    :param end_ms: The window's end, epoch ms, exclusive.
    :param replacement: The corrected readings.
    :return: `Replaced(archive, report)`. The archive's columns are the
        stored columns in their order, then the replacement-only ones.
    """
    stored = normalize(stored)
    if replacement.empty:
        replacement = stored.iloc[0:0]
    else:
        replacement = normalize(replacement)
    inside = (stored["dateutc"] >= start_ms) & (stored["dateutc"] < end_ms)
    new = replacement[
        (replacement["dateutc"] >= start_ms) & (replacement["dateutc"] < end_ms)
    ]
    columns = list(stored.columns) + [
        c for c in replacement.columns if c not in stored.columns
    ]
    pieces = [p for p in (stored[~inside], new) if not p.empty]
    if not pieces:
        return Replaced(stored, ReplaceReport(replaced=int(inside.sum())))
    merged = pd.concat(pieces, sort=False).reindex(columns=columns)
    report = ReplaceReport(replaced=int(inside.sum()), written=len(new))
    return Replaced(_newest_first(merged), report)


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
