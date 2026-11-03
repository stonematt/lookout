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
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

# The type rule `pd.concat` uses, so a merged column keeps the type the old
# whole-frame concat gave it (pandas < 3 is pinned in requirements.txt).
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na, find_common_type

from lookout.utils.dateutc import normalize

Dtype = Union[np.dtype, ExtensionDtype]


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

    Both frames go through `dateutc.normalize` first; a stored archive as
    `storj.load_archive` returns it is already normal and is not copied. A
    reading only Ambient has is added. For a reading both have, value by
    value: Ambient's non-NaN value wins; Ambient's NaN, or a column
    Ambient's frame lacks, keeps the stored value. No valid stored reading
    is dropped; only rows `normalize` rejects (no valid `dateutc`, a
    repeated one) are left out, as `storj.load_archive` already does.

    Only the shared readings and the added ones are worked on: Ambient's
    values are written into the shared readings, the added readings are
    appended, and the archive is built once, column by column, in stored
    order. `stored` is never changed. Each column takes the type pandas
    gives the stored, shared and added values together, as a concat of them
    would.

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

    # Both are ascending and unique on `dateutc`, so a sorted search finds
    # each fetched reading's place in the stored archive.
    stored_ms = stored["dateutc"].to_numpy()
    ambient_ms = ambient["dateutc"].to_numpy()
    place = np.searchsorted(stored_ms, ambient_ms)
    is_shared = place < len(stored_ms)
    is_shared[is_shared] = stored_ms[place[is_shared]] == ambient_ms[is_shared]
    rows = place[is_shared]
    sent_rows = ambient[is_shared].reset_index(drop=True)
    added = ambient[~is_shared].reset_index(drop=True)

    common = [c for c in stored.columns if c in ambient.columns]
    ambient_only = tuple(c for c in ambient.columns if c not in stored.columns)
    both = stored.iloc[rows].reset_index(drop=True)
    filled = updated = 0
    for col in common:
        if col == "dateutc":
            continue
        old, new = both[col], sent_rows[col]
        sent = new.notna()
        had = old.notna()
        filled += int((sent & ~had).sum())
        updated += int((sent & had & _differs(old, new)).sum())
        both[col] = new.where(sent, old)
    for col in ambient_only:
        both[col] = sent_rows[col]

    kept = np.ones(len(stored), dtype=bool)
    kept[rows] = False
    pieces = _Pieces(stored, np.flatnonzero(kept), rows, both, added)
    merged = pieces.build(ambient_only)
    report = MergeReport(
        added=len(added),
        filled=filled,
        updated=updated,
        stored_only=tuple(c for c in stored.columns if c not in ambient.columns),
        ambient_only=ambient_only,
    )
    return Merged(merged, report)


class _Pieces(NamedTuple):
    """
    The three pieces a merged archive is built from.

    The stored readings Ambient did not send (`kept`) stay in `stored`; the
    shared readings (`both`, at `rows` in `stored`) and the added ones are
    small frames of their own.

    :param stored: The stored archive, normalized; read, never changed.
    :param kept: Positions in `stored` of the readings Ambient did not send.
    :param rows: Positions in `stored` of the shared readings.
    :param both: The shared readings, Ambient's values written in.
    :param added: The readings only Ambient has.
    """

    stored: pd.DataFrame
    kept: np.ndarray
    rows: np.ndarray
    both: pd.DataFrame
    added: pd.DataFrame

    def build(self, ambient_only: Tuple[str, ...]) -> pd.DataFrame:
        """
        The merged archive, newest first, from one copy of the stored one.

        The stored archive is taken once into stored order, an added
        reading holding a stand-in row; then each column's shared and added
        values are written into that copy. A column whose merged type is not
        its stored type, or that only Ambient has, is built whole instead.

        :param ambient_only: Ambient's columns `stored` lacks, in order.
        :return: The archive; stored columns, then `ambient_only`.
        """
        keys = np.concatenate(
            [
                self.stored["dateutc"].to_numpy()[self.kept],
                self.both["dateutc"].to_numpy(),
                self.added["dateutc"].to_numpy(),
            ]
        )
        order = np.argsort(keys, kind="stable")[::-1]
        # Where each reading of the pieces, end to end, lands in the archive.
        lands = np.empty_like(order)
        lands[order] = np.arange(len(order))
        first_sent = len(self.kept)
        sent_at = lands[first_sent:]
        source = np.concatenate(
            [self.kept, self.rows, np.zeros(len(self.added), dtype=np.intp)]
        )
        merged = self.stored.take(source[order])
        merged.index = pd.RangeIndex(len(merged))
        for i, col in enumerate(self.stored.columns):
            dtype = self.dtype(col)
            if dtype == self.stored[col].dtype:
                merged.iloc[sent_at, i] = self.sent(col, dtype)
            else:
                merged[col] = self.column(col, dtype).take(order)
        for col in ambient_only:
            merged[col] = self.column(col, self.dtype(col)).take(order)
        return merged

    def parts(self, col: str) -> List[Tuple[int, Optional[pd.Series]]]:
        """
        A column's part in each piece, with the piece's size.

        :param col: A column of one piece or more.
        :return: `(size, part)` for the kept, shared and added pieces; the
            part is None where the piece lacks the column.
        """
        stored = self.stored.get(col)
        return [
            (len(self.kept), None if stored is None else stored.iloc[self.kept]),
            (len(self.both), self.both.get(col)),
            (len(self.added), self.added.get(col)),
        ]

    def dtype(self, col: str) -> Dtype:
        """
        The column's merged type, from its parts in the non-empty pieces.

        :param col: A column of one piece or more.
        :return: The type, as a concat of the pieces gives it.
        """
        return _merged_dtype([part for size, part in self.parts(col) if size], col)

    def sent(self, col: str, dtype: Dtype) -> ExtensionArray:
        """
        The column's shared values, then its added ones, in `dtype`.

        :param col: A column of one piece or more.
        :param dtype: The column's merged type.
        :return: The values; NA where Ambient's frame lacks the column.
        """
        return pd.concat(
            [_as(part, size, dtype) for size, part in self.parts(col)[1:]],
            ignore_index=True,
        ).array

    def column(self, col: str, dtype: Dtype) -> ExtensionArray:
        """
        The whole column: kept, shared and added values, end to end.

        :param col: A column of one piece or more.
        :param dtype: The column's merged type.
        :return: The values piece after piece; NA where a piece lacks it.
        """
        return pd.concat(
            [_as(part, size, dtype) for size, part in self.parts(col)],
            ignore_index=True,
        ).array


def _merged_dtype(parts: List[Optional[pd.Series]], col: str) -> Dtype:
    """
    The type a column takes when its parts are put together.

    The rule `pd.concat` follows for a column across frames: one type when
    every part has it; else the common type, widened to hold NA when a part
    lacks the column (an int column becomes float).

    :param parts: The column in each non-empty part; None where a part
        lacks it.
    :param col: The column's name, for the error.
    :return: The merged column's dtype.
    """
    held = [p.dtype for p in parts if p is not None]
    if not held:
        raise ValueError(f"No part holds column {col!r}")
    dtype = find_common_type(held)
    if len(held) < len(parts):
        dtype = ensure_dtype_can_hold_na(dtype)
    return dtype


def _as(part: Optional[pd.Series], size: int, dtype: Dtype) -> pd.Series:
    """
    A piece's column in the merged type.

    :param part: The column, or None when the piece lacks it.
    :param size: The piece's length in readings.
    :param dtype: The merged type.
    :return: `part` cast to `dtype` (itself when it already is), or `size`
        NAs of `dtype` when `part` is None.
    """
    if part is None:
        return pd.Series(index=pd.RangeIndex(size), dtype=dtype)
    return part.astype(dtype, copy=False)


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
    if "dateutc" not in stored:
        stored = pd.DataFrame({"dateutc": pd.Series(dtype="int64")})
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
        return Replaced(
            _newest_first(stored), ReplaceReport(replaced=int(inside.sum()))
        )
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
    return df.sort_values("dateutc", ascending=False, kind="stable", ignore_index=True)
