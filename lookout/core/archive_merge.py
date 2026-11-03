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

The archive holds only what Ambient sends (#191). `DERIVED_COLUMNS` names
the columns Lookout once derived and saved; every archive save strips them
(`strip_derived`), and `loss` does not count them, so leaving them out is
never a blank. Any column not on the list is raw and kept, so a sensor
Ambient adds later is never dropped.

Pure frame logic: no storage access, no Streamlit. A deliberate correction
is a replace, never a merge.

Functions:
- merge: Merge Ambient's readings into the stored archive; returns `Merged`.
- loss: What saving a new archive over the stored one would lose; `Loss`.
- replace: Swap the stored readings in a window for corrected ones; `Replaced`.
- strip_derived: The archive without its `DERIVED_COLUMNS`.

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
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

# The type rule `pd.concat` uses, so a merged column keeps the type a
# whole-frame concat gives it. A private pandas API: `pandas<3` is pinned in
# requirements.txt, and the equivalence tests (TestMergeMatchesTheOracle in
# tests/test_archive_merge.py) catch a pandas upgrade that changes the rule.
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na, find_common_type

from lookout.utils.dateutc import normalize

Dtype = Union[np.dtype, ExtensionDtype]

# Columns Lookout derived that old UI code saved into the archive, never
# Ambient's (#191): the hour of a reading and the wind rose's direction and
# speed bins. The archive's first data, Ambient's 2022 history (a3b1246),
# has none of them; it does carry `passkey`, `time` and `loc`, which no
# Lookout code ever set, so those are Ambient's and stay.
DERIVED_COLUMNS = ("hour", "direction_bin", "value_bin")


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

    Only the shared readings and the added ones are worked on: the stored
    archive is taken once into stored order with room for the added
    readings, and Ambient's values and the added readings are written into
    that copy. `stored` is never changed. Each column takes the type a
    concat of the stored, shared and added readings would give it, so the
    archive, its column order and its dtypes are what a whole-frame merge
    gives.

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

    # Both are ascending and unique on `dateutc`, so a sorted search gives
    # each fetched reading's place in the stored archive; it is shared when
    # the stored reading there has the same `dateutc` (a place past the end
    # is never shared).
    stored_ms = stored["dateutc"].to_numpy()
    ambient_ms = ambient["dateutc"].to_numpy()
    place = np.searchsorted(stored_ms, ambient_ms)
    is_shared = place < len(stored_ms)
    is_shared[is_shared] = stored_ms[place[is_shared]] == ambient_ms[is_shared]
    shared = place[is_shared]
    ambient_shared = ambient[is_shared].reset_index(drop=True)
    added = ambient[~is_shared].reset_index(drop=True)

    common = [c for c in stored.columns if c in ambient.columns]
    ambient_only = tuple(c for c in ambient.columns if c not in stored.columns)
    both = stored.iloc[shared].reset_index(drop=True)
    filled = updated = 0
    for col in common:
        if col == "dateutc":
            continue
        old, new = both[col], ambient_shared[col]
        sent = new.notna()
        had = old.notna()
        filled += int((sent & ~had).sum())
        updated += int((sent & had & _differs(old, new)).sum())
        both[col] = new.where(sent, old)
    for col in ambient_only:
        both[col] = ambient_shared[col]

    kept = np.ones(len(stored), dtype=bool)
    kept[shared] = False
    merged = _Pieces(stored, np.flatnonzero(kept), shared, both, added).build(
        ambient_only
    )
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

    The stored readings Ambient did not send stay in `stored`, at `kept`;
    the shared readings (`both`, at `shared` in `stored`) and the added
    ones are small frames of their own. In that order, end to end, they are
    the merged archive's readings before the sort into stored order.

    :param stored: The stored archive, normalized; read, never changed.
    :param kept: Positions in `stored` of the readings Ambient did not send.
    :param shared: Positions in `stored` of the shared readings.
    :param both: The shared readings, Ambient's values written in.
    :param added: The readings only Ambient has.
    """

    stored: pd.DataFrame
    kept: np.ndarray
    shared: np.ndarray
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
        stored_ms = self.stored["dateutc"].to_numpy()
        keys = np.concatenate(
            [
                stored_ms[self.kept],
                stored_ms[self.shared],
                self.added["dateutc"].to_numpy(),
            ]
        )
        order = np.argsort(keys, kind="stable")[::-1]
        # The archive position of each reading of the pieces, end to end.
        position = np.empty_like(order)
        position[order] = np.arange(len(order))
        from_ambient = len(self.kept)
        written_at = position[from_ambient:]
        source = np.concatenate(
            [self.kept, self.shared, np.zeros(len(self.added), dtype=np.intp)]
        )
        merged = self.stored.take(source[order])
        merged.index = pd.RangeIndex(len(merged))
        for i, col in enumerate(self.stored.columns):
            dtype = self.merged_dtype(col)
            if dtype == self.stored[col].dtype:
                merged.iloc[written_at, i] = self.values(col, dtype, whole=False)
            else:
                merged[col] = self.values(col, dtype, whole=True).take(order)
        for col in ambient_only:
            dtype = self.merged_dtype(col)
            merged[col] = self.values(col, dtype, whole=True).take(order)
        return merged

    def merged_dtype(self, col: str) -> Dtype:
        """
        The column's type in the merged archive.

        The rule `pd.concat` follows for a column across the non-empty
        pieces: one type when every piece has it; else the common type,
        widened to hold NA when a piece lacks the column (an int column
        becomes float).

        :param col: A column of `stored`, of Ambient's frame, or both.
        :return: The merged column's dtype.
        """
        sides = [
            (len(self.kept), self.stored.get(col)),
            (len(self.both), self.both.get(col)),
            (len(self.added), self.added.get(col)),
        ]
        present = [part for size, part in sides if size]
        held = [part.dtype for part in present if part is not None]
        dtype = find_common_type(held)
        if len(held) < len(present):
            dtype = ensure_dtype_can_hold_na(dtype)
        return dtype

    def values(self, col: str, dtype: Dtype, *, whole: bool) -> ExtensionArray:
        """
        A column's values piece after piece, in its merged type.

        :param col: A column of `stored`, of Ambient's frame, or both.
        :param dtype: The column's merged type.
        :param whole: True for the kept, shared and added values; False for
            the shared and added ones only, the values Ambient's fetch puts
            in place.
        :return: The values; NA where a piece lacks the column.
        """
        stored = self.stored.get(col)
        pieces = [
            (len(self.both), self.both.get(col)),
            (len(self.added), self.added.get(col)),
        ]
        if whole:
            kept = None if stored is None else stored.iloc[self.kept]
            pieces.insert(0, (len(self.kept), kept))
        return pd.concat(
            [_cast_piece(part, size, dtype) for size, part in pieces],
            ignore_index=True,
        ).array


def _cast_piece(part: Optional[pd.Series], size: int, dtype: Dtype) -> pd.Series:
    """
    One piece's column in the merged type.

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
    sides is not lost. `DERIVED_COLUMNS` are not compared: the save strips
    them, so a stored derived value is never a loss.

    Only `dateutc` keys and one column at a time are worked on, never a copy
    of either frame: `new`'s readings are the rows `normalize` would keep
    (`_rows_kept`), a stored reading absent from them is dropped, and each
    stored column is compared on the readings both hold.

    :param stored: The stored archive, as read.
    :param new: The archive about to be saved, any order.
    :return: The `Loss`; falsy when nothing stored would be lost.
    """
    stored = normalize(stored)
    if stored.empty:
        return Loss()
    stored_ms = stored["dateutc"].to_numpy()
    new_ms, new_rows = _rows_kept(new)
    # Both key arrays are ascending and unique, so a sorted search finds each
    # stored reading's place among the new ones; it is kept when the new
    # reading there has the same `dateutc` (a place past the end never is).
    place = np.searchsorted(new_ms, stored_ms)
    is_kept = place < len(new_ms)
    is_kept[is_kept] = new_ms[place[is_kept]] == stored_ms[is_kept]
    kept = np.flatnonzero(is_kept)
    at_new = new_rows[place[kept]]

    blanked = {}
    for col in stored.columns:
        if col == "dateutc" or col in DERIVED_COLUMNS:
            continue
        had = stored[col].notna().to_numpy()[kept]
        if col in new.columns:
            # A column `new` lacks blanks every stored value in it.
            had &= new[col].isna().to_numpy()[at_new]
        count = int(had.sum())
        if count:
            blanked[col] = count
    return Loss(
        dropped=len(stored) - len(kept),
        blanked=sum(blanked.values()),
        blanked_columns=tuple(blanked),
    )


def _rows_kept(new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    The readings of `new` that `normalize(new)` keeps, without copying `new`.

    `normalize` runs on `new`'s `dateutc` beside each row's position, so the
    same rows survive (invalid and repeated ones are left out, the same
    repeat kept) and come back in its ascending order.

    :param new: An archive frame, any order and index.
    :return: `(dateutc, rows)`: the kept readings' int64 ms keys, ascending
        and unique, and their positions in `new`.
    """
    if new.empty or "dateutc" not in new.columns:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.intp)
    keys = pd.DataFrame({"dateutc": new["dateutc"].array, "row": np.arange(len(new))})
    kept = normalize(keys)
    if kept.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.intp)
    return kept["dateutc"].to_numpy(), kept["row"].to_numpy()


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


def strip_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    The archive without the columns Lookout derives (`DERIVED_COLUMNS`).

    A shallow copy, so stripping costs no copy of the archive's values: the
    frame returned shares them with `df`, and `df` keeps its columns.

    :param df: An archive frame.
    :return: `df` itself when it holds none of them, else a frame without them.
    """
    derived = [col for col in DERIVED_COLUMNS if col in df.columns]
    if not derived:
        return df
    stripped = df.copy(deep=False)
    for col in derived:
        del stripped[col]
    return stripped


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
