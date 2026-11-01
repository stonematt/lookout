"""Unit tests for lookout.utils.dateutc."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from lookout.utils.dateutc import (
    DEFAULT_TZ,
    INTERVAL_MS,
    POST_2020_MS,
    day_span,
    from_local_date,
    from_local_date_ts,
    local_time,
    normalize,
    to_local,
    to_utc,
)


def _ms(year: int, month: int = 1, day: int = 1) -> int:
    return int(pd.Timestamp(year=year, month=month, day=day, tz="UTC").value // 10**6)


# ---------- normalize ----------


def test_normalize_happy_path_unchanged_content_sorted_ascending():
    rows = [
        {"dateutc": _ms(2024, 6, 1), "v": "a"},
        {"dateutc": _ms(2025, 1, 1), "v": "b"},
        {"dateutc": _ms(2023, 12, 31), "v": "c"},
    ]
    df = pd.DataFrame(rows)

    out = normalize(df)

    assert list(out["dateutc"]) == sorted(r["dateutc"] for r in rows)
    assert out["dateutc"].dtype == "int64"
    assert set(out["v"]) == {"a", "b", "c"}


def test_normalize_drops_nan_zero_negative_and_below_floor():
    df = pd.DataFrame(
        {
            "dateutc": [
                None,
                0,
                -1,
                POST_2020_MS - 1,
                POST_2020_MS,
                _ms(2025, 1, 1),
            ],
            "v": list("abcdef"),
        }
    )

    out = normalize(df)

    assert list(out["dateutc"]) == [POST_2020_MS, _ms(2025, 1, 1)]
    assert list(out["v"]) == ["e", "f"]


def test_normalize_min_ms_override_disables_floor():
    df = pd.DataFrame({"dateutc": [1, 28_800_000, POST_2020_MS]})

    out = normalize(df, min_ms=0)

    assert list(out["dateutc"]) == [1, 28_800_000, POST_2020_MS]


def test_normalize_accepts_datetime64_utc_input():
    ts = [
        pd.Timestamp("2024-06-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    ]
    df = pd.DataFrame({"dateutc": ts})

    out = normalize(df)

    assert out["dateutc"].dtype == "int64"
    assert list(out["dateutc"]) == [_ms(2024, 6, 1), _ms(2025, 1, 1)]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "UTC"])
def test_normalize_accepts_datetime64_at_any_unit(unit, tz):
    """Parquet and pandas 3 hand back `datetime64[us]`; it must still be ms."""
    stamps = pd.Series(pd.to_datetime(["2024-06-01", "2025-01-01"]).as_unit(unit))
    if tz:
        stamps = stamps.dt.tz_localize(tz)
    df = pd.DataFrame({"dateutc": stamps})

    out = normalize(df)

    assert out["dateutc"].dtype == "int64"
    assert list(out["dateutc"]) == [_ms(2024, 6, 1), _ms(2025, 1, 1)]


def test_normalize_empty_dataframe_returns_empty():
    out = normalize(pd.DataFrame())
    assert out.empty


def test_normalize_missing_dateutc_column_returns_empty():
    out = normalize(pd.DataFrame({"other": [1, 2]}))
    assert out.empty


def test_normalize_dedup_keeps_last():
    df = pd.DataFrame(
        {
            "dateutc": [_ms(2025, 1, 1), _ms(2025, 1, 1), _ms(2025, 1, 2)],
            "v": ["first", "second", "third"],
        }
    )

    out = normalize(df)

    assert list(out["dateutc"]) == [_ms(2025, 1, 1), _ms(2025, 1, 2)]
    assert list(out["v"]) == ["second", "third"]


def test_normalize_dedup_only_does_not_warn(caplog):
    df = pd.DataFrame({"dateutc": [_ms(2025, 1, 1)] * 3 + [_ms(2025, 1, 2)] * 2})

    with caplog.at_level("WARNING", logger="lookout.utils.dateutc"):
        normalize(df)

    assert not any(
        "dateutc.normalize" in r.message for r in caplog.records
    ), "dedup-only call must not emit an invalid-rows warning"


def test_normalize_invalid_drops_emit_warning(caplog):
    df = pd.DataFrame({"dateutc": [None, 0, _ms(2025, 1, 1)]})

    with caplog.at_level("WARNING", logger="lookout.utils.dateutc"):
        normalize(df)

    matching = [r for r in caplog.records if "dateutc.normalize" in r.message]
    assert len(matching) == 1
    assert "as invalid" in matching[0].message


# ---------- casts ----------


def test_to_utc_and_to_local_roundtrip_identity_on_ms_vector():
    ms = pd.Series([_ms(2025, 1, 1), _ms(2025, 6, 15), _ms(2026, 5, 4)])

    utc = to_utc(ms)
    local = to_local(ms, tz="America/Los_Angeles")

    assert list(utc.astype("int64") // 10**6) == list(ms)
    assert list(local.dt.tz_convert("UTC").astype("int64") // 10**6) == list(ms)
    assert str(local.dt.tz) == "America/Los_Angeles"


def test_from_local_date_returns_pacific_midnight_for_string_date():
    got = from_local_date("2026-05-04", tz="America/Los_Angeles")
    expected = int(
        pd.Timestamp("2026-05-04", tz="America/Los_Angeles").tz_convert("UTC").value
        // 10**6
    )
    assert got == expected


def test_from_local_date_accepts_date_object_default_tz():
    got = from_local_date(dt.date(2026, 5, 4))
    expected = int(
        pd.Timestamp("2026-05-04", tz=DEFAULT_TZ).tz_convert("UTC").value // 10**6
    )
    assert got == expected


def test_from_local_date_ts_returns_utc_timestamp():
    ts = from_local_date_ts("2026-05-04", tz="America/Los_Angeles")
    assert isinstance(ts, pd.Timestamp)
    assert str(ts.tz) == "UTC"
    assert ts == pd.Timestamp("2026-05-04", tz="America/Los_Angeles").tz_convert("UTC")


def test_from_local_date_is_local_midnight_across_a_dst_change():
    # 2026-03-08 starts at 00:00 PST (08:00 UTC); the next day at 00:00 PDT.
    assert from_local_date(dt.date(2026, 3, 8), "America/Los_Angeles") == (
        1772956800000
    )
    assert from_local_date(dt.date(2026, 3, 9), "America/Los_Angeles") == (
        1773039600000
    )


def test_local_time_is_the_reading_in_the_station_zone():
    # 2026-01-01T00:00Z
    assert local_time(1767225600000, "America/Los_Angeles") == pd.Timestamp(
        "2025-12-31 16:00", tz="America/Los_Angeles"
    )
    assert local_time(1767225600000, "Asia/Tokyo") == pd.Timestamp(
        "2026-01-01 09:00", tz="Asia/Tokyo"
    )


def test_local_time_defaults_to_the_station_zone():
    assert str(local_time(1767225600000).tz) == DEFAULT_TZ


def test_day_span_is_every_day_inclusive():
    assert day_span(dt.date(2026, 2, 27), dt.date(2026, 3, 2)) == [
        dt.date(2026, 2, 27),
        dt.date(2026, 2, 28),
        dt.date(2026, 3, 1),
        dt.date(2026, 3, 2),
    ]
    assert day_span(dt.date(2026, 3, 1), dt.date(2026, 3, 1)) == [dt.date(2026, 3, 1)]
    assert day_span(dt.date(2026, 3, 2), dt.date(2026, 3, 1)) == []


def test_interval_is_five_minutes():
    assert INTERVAL_MS == 300_000


# ---------- April 2026 regression ----------


def test_normalize_drops_april_2026_corruption_shapes():
    """
    Regression: small positive ints (parquet truncation) survived a naive
    `> 0` guard but mapped to 1969-12-31 PT after tz conversion. Floor
    must drop them; valid rows must survive sorted ascending.
    """
    valid_a = _ms(2025, 1, 1)
    valid_b = _ms(2026, 5, 4)
    df = pd.DataFrame(
        {
            "dateutc": [
                1,
                28_800_000,
                1_776_665,
                1_776_725_051,
                valid_b,
                valid_a,
            ]
        }
    )

    out = normalize(df)

    assert list(out["dateutc"]) == [valid_a, valid_b]
