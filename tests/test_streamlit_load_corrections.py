"""
The Streamlit load path applies rain corrections once (#95).

The archive in storage is raw. `lookout.core.data_processing.load_or_update_data`
corrects it once on the first load, and corrects only the rows each later
update fetches, so session rows are never corrected twice. Driven on a fake
session state (a plain dict) against the moto bucket and the fake Ambient API,
as `tests/test_load_path_golden.py` does. Delete with the Streamlit app.
"""

from typing import Dict

import pandas as pd
import pytest
import streamlit as st

from lookout.core import archive_merge
from lookout.core import data_processing as lo_dp
from lookout.core import rain_corrections
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import FAKE_MAC, SCENARIO, RainSpell, build_archive
from tests.fixtures.storage import BUCKET, flaky_storj
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    CatchupStation,
    catchup_station,
    ms,
    reporting,
)

CARRY = 0.48  # SCENARIO's carryover onto Friday 2026-01-09, local
THURSDAY = ms("2026-01-08T20:00:00Z")  # 12:00 local, before the carryover
FRIDAY = ms("2026-01-09T20:00:00Z")  # 12:00 local, archived
SATURDAY = ms("2026-01-10T14:00:00Z")  # 06:00 local, fetched by the update
ARCHIVED_UNTIL = ms("2026-01-10T12:00:00Z")


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Dict:
    """A fake Streamlit session state."""
    state: Dict = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


@pytest.fixture
def weather() -> pd.DataFrame:
    """SCENARIO's week, with 0.24" of Friday's own rain after the carryover."""
    return build_archive(
        SCENARIO.start,
        SCENARIO.end,
        rain=(*SCENARIO.rain, RainSpell("2026-01-09T18:00:00Z", 2, 0.12)),
        gaps=SCENARIO.gaps,
    )


@pytest.fixture
def station(catchup_station, weather) -> CatchupStation:  # noqa: F811
    """Raw archive to Saturday 04:00 local; Ambient reports the whole week."""
    rain_corrections.save_catalog(
        {"corrections": rain_corrections.detect_carryovers(weather)}, FAKE_MAC
    )
    return catchup_station(
        archive=weather[weather["dateutc"] < ARCHIVED_UNTIL],
        history=to_ambient_records(weather),
        devices=reporting(weather),
        now=SCENARIO.end,
    )


def load(state: Dict, device: Dict) -> None:
    """One script run's data step, as `streamlit_app.py` performs it."""
    state["device"] = device
    lo_dp.load_or_update_data(
        device=device,
        bucket=BUCKET,
        file_type="parquet",
        auto_update=True,
        short_minutes=6,
        long_minutes=3 * 24 * 60,
    )


def device(station: CatchupStation, dateutc: int) -> Dict:
    """The station's `/devices` entry, reporting a reading at `dateutc`."""
    (entry,) = reporting(station.archive)
    entry["lastData"] = {**entry["lastData"], "dateutc": dateutc}
    return entry


def counters(state: Dict, row: int) -> pd.Series:
    """One session row's rain counters."""
    history = state["history_df"].set_index("dateutc")
    return history.loc[row, ["dailyrainin", "weeklyrainin", "monthlyrainin"]]


def test_the_first_load_corrects_the_raw_archive_once(station, session, weather):
    raw = weather.set_index("dateutc")

    load(session, device(station, int(station.archive["dateutc"].max())))

    assert station.replay.calls == []  # nothing fetched: the load alone
    friday = counters(session, FRIDAY)
    assert friday["dailyrainin"] == pytest.approx(
        raw.loc[FRIDAY, "dailyrainin"] - CARRY
    )
    assert friday["weeklyrainin"] == pytest.approx(
        raw.loc[FRIDAY, "weeklyrainin"] - CARRY
    )
    thursday = counters(session, THURSDAY)
    assert thursday["weeklyrainin"] == pytest.approx(raw.loc[THURSDAY, "weeklyrainin"])


def test_an_update_corrects_only_the_rows_it_fetched(station, session, weather):
    raw = weather.set_index("dateutc")
    load(session, device(station, int(station.archive["dateutc"].max())))

    load(session, device(station, int(weather["dateutc"].max())))

    assert session["history_df"]["dateutc"].max() == weather["dateutc"].max()
    assert session["history_df"]["dateutc"].is_unique
    for row in (FRIDAY, SATURDAY):
        got = counters(session, row)
        for col in ("weeklyrainin", "monthlyrainin"):
            assert got[col] == pytest.approx(raw.loc[row, col] - CARRY)
    # The archive in storage is untouched by the session: still raw.
    stored = station.stored().set_index("dateutc")
    assert stored.loc[FRIDAY, "weeklyrainin"] == pytest.approx(
        raw.loc[FRIDAY, "weeklyrainin"]
    )


# ---------- the Diagnostics view's gap fill (#95) ----------


PLAIN_GAP = SCENARIO.gaps[1]  # Saturday 10:00 to 14:00 local


def test_a_diagnostics_fill_saves_raw_and_shows_corrected(
    catchup_station, session, weather  # noqa: F811
):
    from lookout.ui import diagnostics

    rain_corrections.save_catalog(
        {"corrections": rain_corrections.detect_carryovers(weather)}, FAKE_MAC
    )
    truth = build_archive(
        SCENARIO.start,
        SCENARIO.end,
        rain=(*SCENARIO.rain, RainSpell("2026-01-09T18:00:00Z", 2, 0.12)),
        gaps=SCENARIO.gaps[:1],
    )
    station = catchup_station(
        archive=weather,
        history=to_ambient_records(truth),
        devices=reporting(weather),
        now=SCENARIO.end,
    )
    load(session, device(station, int(weather["dateutc"].max())))
    raw = truth.set_index("dateutc")
    filled_row = ms(PLAIN_GAP.start) + 3600 * 1000

    outcome = diagnostics.fill_gap(
        pd.Timestamp(PLAIN_GAP.start), pd.Timestamp(PLAIN_GAP.end)
    )
    assert outcome.status == "filled"
    assert diagnostics.save_filled_archive() is True

    stored = station.stored().merge(truth, on="dateutc", suffixes=("", "_raw"))
    assert len(stored) == len(truth)
    for col in ("dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"):
        assert (stored[col] - stored[col + "_raw"]).abs().max() < 1e-9
    for row in (FRIDAY, filled_row):
        got = counters(session, row)
        assert got["weeklyrainin"] == pytest.approx(
            raw.loc[row, "weeklyrainin"] - CARRY
        )


def test_a_diagnostics_save_goes_through_the_archive_merge(
    catchup_station, session, weather  # noqa: F811
):
    """Stored values the session's raw archive lacks survive the save."""
    from lookout.ui import diagnostics

    truth = build_archive(
        SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps[:1]
    )
    station = catchup_station(
        archive=weather,
        history=to_ambient_records(truth),
        devices=reporting(weather),
        now=SCENARIO.end,
    )
    load(session, device(station, int(weather["dateutc"].max())))
    diagnostics.fill_gap(pd.Timestamp(PLAIN_GAP.start), pd.Timestamp(PLAIN_GAP.end))
    stored_before = station.stored().set_index("dateutc")
    slot = int(weather["dateutc"].max())
    raw = session[diagnostics.RAW_HISTORY_KEY]
    assert (raw["dateutc"] == slot).any()
    raw.loc[raw["dateutc"] == slot, "tempf"] = float("nan")
    assert pd.notna(stored_before.loc[slot, "tempf"])

    assert diagnostics.save_filled_archive() is True

    saved = station.stored().set_index("dateutc")
    assert saved.loc[slot, "tempf"] == stored_before.loc[slot, "tempf"]


def test_a_diagnostics_save_with_nothing_filled_writes_nothing(
    station, session  # noqa: F811
):
    from lookout.ui import diagnostics

    load(session, device(station, int(station.archive["dateutc"].max())))
    keys = station.keys()

    assert diagnostics.save_filled_archive() is False
    assert station.keys() == keys


def test_a_diagnostics_save_keeps_rows_catchup_stored_since_the_fill(
    catchup_station, session, weather  # noqa: F811
):
    """The fills merge into the archive as stored at save time."""
    from lookout.storage import storj
    from lookout.ui import diagnostics

    truth = build_archive(
        SCENARIO.start,
        SCENARIO.end,
        rain=(*SCENARIO.rain, RainSpell("2026-01-09T18:00:00Z", 2, 0.12)),
        gaps=SCENARIO.gaps[:1],
    )
    older = weather[weather["dateutc"] < ARCHIVED_UNTIL + 6 * 3600 * 1000]
    station = catchup_station(
        archive=older,
        history=to_ambient_records(truth),
        devices=reporting(older),
        now=SCENARIO.end,
    )
    load(session, device(station, int(older["dateutc"].max())))
    diagnostics.fill_gap(pd.Timestamp(PLAIN_GAP.start), pd.Timestamp(PLAIN_GAP.end))
    # Catchup saves the rest of the week before the Diagnostics save.
    storj.save_df_to_s3(weather, BUCKET, f"{FAKE_MAC}.parquet")

    assert diagnostics.save_filled_archive() is True

    stored = station.stored()
    assert set(stored["dateutc"]) == set(truth["dateutc"])


# ---------- an oldest-first archive (#133) ----------


UP = "↑"
TREND_START, TREND_END = "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"


@pytest.fixture
def oldest_first(catchup_station) -> CatchupStation:  # noqa: F811
    """
    Two dry days stored oldest first; only the newest reading moved.

    Temperature and pressure hold at 50.0F and 30.0" except the newest row,
    60.0F and 30.3", so read from it both trends rise; read from the oldest
    row they are steady.
    """
    archive = build_archive(TREND_START, TREND_END, newest_first=False)
    archive["tempf"] = 50.0
    archive["baromrelin"] = 30.0
    newest = archive["dateutc"].idxmax()
    archive.loc[newest, ["tempf", "baromrelin"]] = [60.0, 30.3]
    return catchup_station(
        archive=archive, history=[], devices=reporting(archive), now=TREND_END
    )


def test_the_first_load_reads_trends_from_the_newest_row(oldest_first, session):
    load(session, device(oldest_first, int(oldest_first.archive["dateutc"].max())))

    assert (session["temp_trend"], session["barom_trend"]) == (UP, UP)


def test_a_live_update_reads_trends_from_the_newest_row(oldest_first, session):
    newest = int(oldest_first.archive["dateutc"].max())
    load(session, device(oldest_first, newest))
    session["temp_trend"] = session["barom_trend"] = None

    # The device reports half an hour on; Ambient has nothing new to send.
    load(session, device(oldest_first, newest + 30 * 60 * 1000))

    assert oldest_first.replay.calls  # the update did ask Ambient
    assert (session["temp_trend"], session["barom_trend"]) == (UP, UP)


def test_a_diagnostics_save_that_would_drop_a_stored_reading_is_refused(
    catchup_station, session, weather, monkeypatch  # noqa: F811
):
    """#190: the Diagnostics save is guarded by the archive it read back."""
    from lookout.storage import storj
    from lookout.ui import diagnostics

    truth = build_archive(
        SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps[:1]
    )
    station = catchup_station(
        archive=weather,
        history=to_ambient_records(truth),
        devices=reporting(weather),
        now=SCENARIO.end,
    )
    load(session, device(station, int(weather["dateutc"].max())))
    diagnostics.fill_gap(pd.Timestamp(PLAIN_GAP.start), pd.Timestamp(PLAIN_GAP.end))
    before = station.stored()
    monkeypatch.setattr(
        diagnostics.archive_merge,
        "merge",
        lambda stored, raw: archive_merge.Merged(
            stored.iloc[1:], archive_merge.MergeReport()
        ),
    )

    with pytest.raises(storj.ArchiveLossError):
        diagnostics.save_filled_archive()

    pd.testing.assert_frame_equal(station.stored(), before)


def test_a_failed_archive_read_stops_the_diagnostics_save(
    catchup_station, session, weather, monkeypatch  # noqa: F811
):
    """#160: the save reads the stored archive strictly, before any backup."""
    from lookout.storage import storj
    from lookout.ui import diagnostics

    truth = build_archive(
        SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps[:1]
    )
    station = catchup_station(
        archive=weather,
        history=to_ambient_records(truth),
        devices=reporting(weather),
        now=SCENARIO.end,
    )
    load(session, device(station, int(weather["dateutc"].max())))
    diagnostics.fill_gap(pd.Timestamp(PLAIN_GAP.start), pd.Timestamp(PLAIN_GAP.end))
    keys = station.keys()
    flaky_storj(monkeypatch, station.s3, read=f"{FAKE_MAC}.parquet")

    with pytest.raises(storj.StorageReadError):
        diagnostics.save_filled_archive()

    assert station.keys() == keys
