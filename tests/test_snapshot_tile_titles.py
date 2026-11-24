"""
The ten-day temperature tile's title (#665, spec #647).

GLOSSARY tile title: a title names what its number counts where the old one
named a window. The ten-day tile's number is the last ten days' highs
against their usual highs, so its title reads "Highs over usual", "Highs
under usual" or "Highs at usual"; a window with a gap reads "Gap in record",
never a level reading. With nothing to compare yet the tile keeps its fixed
name, "Last 10 days". The snapshot carries the text in
`tile_titles.ten_days`, an addition in version 4.

Through `build_snapshot`; archives are synthetic five-minute stations from
`tests.fixtures.archive` (the same day-shaped temperature every day), local
days Pacific, with the last ten days shifted to run warm or cool.
"""

from typing import Sequence

import pandas as pd

from lookout.snapshot import Snapshot
from tests.fixtures.archive import Gap, build_archive
from tests.snapshot_helpers import build

START = "2024-12-01T08:00:00Z"  # one earlier year around the ten days
NOW = "2026-01-07T07:50:00Z"  # 23:50 local Tue Jan 6: today's high is in
TEN_DAYS_FROM = "2025-12-28T08:00:00Z"  # local midnight starting Dec 28


def snapshot_of(shift_f: float = 0.0, gaps: Sequence[Gap] = ()) -> Snapshot:
    """
    The station at `NOW`, its last ten days `shift_f` degrees off usual.

    :param shift_f: Added to every reading from Dec 28 on.
    :param gaps: Spans with no readings.
    :return: The snapshot.
    """
    archive = build_archive(START, "2026-01-07T07:45:00Z", gaps=gaps)
    recent = archive["dateutc"] >= pd.Timestamp(TEN_DAYS_FROM).value // 1_000_000
    archive.loc[recent, "tempf"] = (archive.loc[recent, "tempf"] + shift_f).round(1)
    return build(archive, None, NOW)


class TestTheTitleNamesTheNumber:
    def test_a_warm_ten_days_is_highs_over_usual(self):
        snapshot = snapshot_of(shift_f=5.0)

        assert snapshot.temperature_context.ten_day_highs.delta_f == 5
        assert snapshot.tile_titles.ten_days == "Highs over usual"

    def test_a_cool_ten_days_is_highs_under_usual(self):
        snapshot = snapshot_of(shift_f=-4.0)

        assert snapshot.temperature_context.ten_day_highs.delta_f == -4
        assert snapshot.tile_titles.ten_days == "Highs under usual"

    def test_a_level_ten_days_is_highs_at_usual(self):
        snapshot = snapshot_of()

        assert snapshot.temperature_context.ten_day_highs.delta_f == 0
        assert snapshot.tile_titles.ten_days == "Highs at usual"


class TestAGapIsNeverLevel:
    def test_a_missing_day_reads_gap_in_record(self):
        # Jan 2 local has no readings; the other nine still pair, level.
        gone = Gap("2026-01-02T08:00:00Z", "2026-01-03T08:00:00Z")

        snapshot = snapshot_of(gaps=(gone,))

        assert snapshot.temperature_context.ten_day_highs.days == 9
        assert snapshot.temperature_context.ten_day_highs.delta_f == 0
        assert snapshot.tile_titles.ten_days == "Gap in record"

    def test_a_gap_reads_so_when_over_usual_too(self):
        gone = Gap("2026-01-02T08:00:00Z", "2026-01-03T08:00:00Z")

        snapshot = snapshot_of(shift_f=5.0, gaps=(gone,))

        assert snapshot.tile_titles.ten_days == "Gap in record"

    def test_nothing_to_compare_keeps_the_fixed_name(self):
        # A station two weeks old: nothing to compare, the fixed name.
        archive = build_archive("2025-12-31T08:00:00Z", "2026-01-07T07:45:00Z")

        snapshot = build(archive, None, NOW)

        assert snapshot.temperature_context.ten_day_highs is None
        assert snapshot.tile_titles.ten_days == "Last 10 days"


def test_the_dashboard_document_carries_the_title():
    document = snapshot_of(shift_f=5.0).dashboard()

    assert document["tile_titles"]["ten_days"] == "Highs over usual"
    assert document["schema_version"] == 4
