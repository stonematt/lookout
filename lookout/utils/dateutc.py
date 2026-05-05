"""
Canonical dateutc utilities.

Single source of truth for what counts as a valid `dateutc` value
(int64 epoch ms UTC, >= post-2020 floor) and for converting between
`dateutc` and tz-aware datetimes. Every consumer (catchup CLI,
controller's combine_df, UI date logic) goes through this module.

See:
    - CONTEXT.md  -- domain glossary (dateutc, normalize, post-2020 floor)
    - docs/adr/0001-dateutc-post-2020-floor.md
    - docs/adr/0002-station-local-time-as-analytical-frame.md

Pure transforms only. Importing this module performs no I/O, has no
Streamlit/config side effects.
"""

from __future__ import annotations

import datetime as _dt
from typing import Union

import pandas as pd

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

POST_2020_MS: int = 1577836800000
"""Minimum legal `dateutc`: 2020-01-01 UTC in epoch ms.

Rows below the floor are corrupt (small positive ints from parquet
truncation, partial writes, API hiccups) and are dropped at normalize
time. See ADR-0001.
"""

DEFAULT_TZ: str = "America/Los_Angeles"
"""IANA tz of the single lookout post today.

Per ADR-0002 the station-local zone is the analytical frame for all
day-granularity reasoning. Persisting per-row tz is deferred work.
"""


def normalize(
    df: pd.DataFrame, *, min_ms: int = POST_2020_MS
) -> pd.DataFrame:
    """
    Validate and cast a DataFrame's `dateutc` column to canonical form.

    Operations (in order):
        1. Coerce `dateutc` to int64 epoch ms UTC. Accepts numeric or
           datetime64-with-tz input.
        2. Drop rows whose `dateutc` is null, <= 0, or < ``min_ms``.
        3. De-duplicate on `dateutc` (keep last).
        4. Sort ascending.

    Empty input or input lacking a `dateutc` column returns an empty
    DataFrame without raising.

    Logs a single warning per invocation when rows are dropped,
    including the input dtype and a 3-row sample of the original
    column.

    :param df: DataFrame containing a `dateutc` column.
    :param min_ms: Floor (epoch ms) below which rows are dropped.
        Defaults to ``POST_2020_MS``. Pass ``min_ms=0`` to disable
        the floor (test affordance only -- production callers take
        the default).
    :return: New DataFrame, dateutc as int64 ms, ascending, deduped.
    """
    if df is None or df.empty or "dateutc" not in df.columns:
        return pd.DataFrame()

    original = df["dateutc"]
    original_dtype = str(original.dtype)
    sample = original.head(3).tolist()
    before = len(df)

    if pd.api.types.is_datetime64_any_dtype(original):
        as_utc = pd.to_datetime(original, utc=True, errors="coerce")
        ms = as_utc.astype("int64", copy=False) // 1_000_000
        ms = ms.where(as_utc.notna())
    else:
        ms = pd.to_numeric(original, errors="coerce")

    out = df.assign(dateutc=ms).dropna(subset=["dateutc"])
    out["dateutc"] = out["dateutc"].astype("int64")
    out = out[(out["dateutc"] > 0) & (out["dateutc"] >= min_ms)]
    out = (
        out.sort_values("dateutc")
        .drop_duplicates(subset="dateutc", keep="last")
        .reset_index(drop=True)
    )

    dropped = before - len(out)
    if dropped:
        logger.warning(
            "dateutc.normalize: dropped %d of %d row(s) (dtype=%s, "
            "sample=%s, min_ms=%d)",
            dropped,
            before,
            original_dtype,
            sample,
            min_ms,
        )

    return out


def to_utc(series: pd.Series) -> pd.Series:
    """Cast int64 ms `dateutc` series to a tz-aware UTC datetime series."""
    return pd.to_datetime(series, unit="ms", utc=True)


def to_local(series: pd.Series, tz: str = DEFAULT_TZ) -> pd.Series:
    """Cast int64 ms `dateutc` series to a tz-aware station-local datetime series."""
    return to_utc(series).dt.tz_convert(tz)


def from_local_date(
    date: Union[_dt.date, _dt.datetime, pd.Timestamp, str],
    tz: str = DEFAULT_TZ,
) -> int:
    """
    Convert a station-local calendar date to its midnight `dateutc`.

    Used by sliders that hand the user a local-day boundary and need
    the matching int64 ms UTC for filtering.

    :param date: Calendar date in the station-local zone.
    :param tz: Station-local IANA tz. Defaults to ``DEFAULT_TZ``.
    :return: int64 epoch ms UTC for local midnight on ``date``.
    """
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return int(ts.tz_convert("UTC").value // 1_000_000)
