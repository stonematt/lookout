"""
archive_merge_oracle.py: the archive merge as it was before #283, for tests.

A private test oracle (spec #234, user story 30): the equivalence tests in
`tests/test_archive_merge.py` run it and `lookout.core.archive_merge.merge`
on the same frames and assert equal archives and equal counts. It is the
merge's previous implementation, copied unchanged apart from names, so
equivalence is tested against the real previous behaviour, not a rewrite.
The save guard lever (#284) deletes it with the other oracles.

Functions:
- merge_before_overlap: The pre-#283 `archive_merge.merge`.
"""

import pandas as pd

from lookout.core.archive_merge import Merged, MergeReport
from lookout.utils.dateutc import normalize


def merge_before_overlap(stored: pd.DataFrame, ambient: pd.DataFrame) -> Merged:
    """
    Merge Ambient's readings into the stored archive, as before #283.

    :param stored: The stored archive (or the frame standing in for it).
    :param ambient: Readings fetched from Ambient, raw.
    :return: `Merged(archive, report)`, the archive newest first.
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
    :return: A boolean Series; True where the values cannot be compared.
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
