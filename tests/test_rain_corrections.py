"""
Unit tests for rain corrections CLI.

Tests rain carryover detection and catalog management CLI.
"""

import unittest.mock as mock
import pytest
import pandas as pd

from lookout.cli.rain_corrections_cli import (
    get_station,
    load_archive_s3,
    handle_detect,
    handle_generate,
    handle_show,
)
from lookout.core import rain_corrections
from lookout.core.rain_corrections import apply_corrections, detect_carryovers
from lookout.storage.storj import StorageReadError
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    Gap,
    build_archive,
    scenario_archive,
)
from tests.fixtures.real import load_real_slice
from tests.fixtures.storage import flaky_storj

CORRECTIONS_KEY = f"{FAKE_MAC}.rain_corrections.json"


class TestDeviceDiscovery:
    """Test device auto-discovery functionality."""

    @mock.patch("lookout.cli.rain_corrections_cli.ambient_client.get_devices")
    def test_single_device_discovery(self, mock_get_devices):
        """Test discovery when one device available."""
        mock_devices = [
            {"macAddress": "98:CD:AC:22:0D:E5", "info": {"name": "Test Station"}}
        ]
        mock_get_devices.return_value = mock_devices

        device, mac, name = get_station()
        assert mac == "98:CD:AC:22:0D:E5"
        assert name == "Test Station"
        assert device == mock_devices[0]

    @mock.patch("lookout.cli.rain_corrections_cli.ambient_client.get_devices")
    def test_multiple_devices_uses_the_pinned_station(
        self, mock_get_devices, monkeypatch
    ):
        """The pinned STATION_MAC is used, even when it is listed second."""
        monkeypatch.setenv("STATION_MAC", "12:34:56:78:9A:BC")
        mock_devices = [
            {"macAddress": "98:CD:AC:22:0D:E5", "info": {"name": "Other Station"}},
            {"macAddress": "12:34:56:78:9A:BC", "info": {"name": "Pinned Station"}},
        ]
        mock_get_devices.return_value = mock_devices

        device, mac, name = get_station()
        assert mac == "12:34:56:78:9A:BC"
        assert name == "Pinned Station"
        assert device == mock_devices[1]

    @mock.patch("builtins.print")
    @mock.patch("lookout.cli.rain_corrections_cli.ambient_client.get_devices")
    def test_no_devices_exits_with_the_reason(self, mock_get_devices, mock_print):
        """No device list: exit 1, saying why the station was not found."""
        mock_get_devices.return_value = []

        with pytest.raises(SystemExit) as exited:
            get_station()

        assert exited.value.code == 1
        assert "Ambient returned no devices" in str(mock_print.call_args)


class TestArchiveLoading:
    """Test S3 archive loading."""

    @mock.patch("lookout.storage.storj.get_df_from_s3")
    def test_successful_load(self, mock_get_df_from_s3):
        """Test successful archive loading from S3."""
        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_get_df_from_s3.return_value = mock_df

        from lookout.cli.rain_corrections_cli import load_archive_s3

        result = load_archive_s3("98:CD:AC:22:0D:E5", "lookout")
        assert not result.empty
        mock_get_df_from_s3.assert_called_once_with(
            "lookout", "98:CD:AC:22:0D:E5.parquet", "parquet"
        )

    @mock.patch("lookout.storage.storj.get_df_from_s3")
    def test_load_error(self, mock_get_df_from_s3):
        """Test handling of S3 loading errors."""
        mock_get_df_from_s3.side_effect = Exception("S3 failed")

        from lookout.cli.rain_corrections_cli import load_archive_s3

        result = load_archive_s3("98:CD:AC:22:0D:E5", "lookout")
        assert result.empty


class TestCliCommands:
    """Test CLI command handlers."""

    @mock.patch("builtins.print")
    @mock.patch("lookout.cli.rain_corrections_cli.ambient_client.get_devices")
    def test_detect_no_device(self, mock_get_devices, mock_print, monkeypatch):
        """detect exits 1 with the reason when the pinned station is missing."""
        monkeypatch.setenv("STATION_MAC", "12:34:56:78:9A:BC")
        mock_get_devices.return_value = [
            {"macAddress": "98:CD:AC:22:0D:E5", "info": {"name": "Other"}}
        ]

        with pytest.raises(SystemExit) as exited:
            handle_detect(mock.Mock())

        assert exited.value.code == 1
        assert "Station 12:34:56:78:9A:BC is not on this Ambient account" in str(
            mock_print.call_args
        )

    @mock.patch("builtins.print")
    @mock.patch("lookout.cli.rain_corrections_cli.load_archive_s3")
    @mock.patch("lookout.cli.rain_corrections_cli.get_station")
    def test_detect_empty_archive(
        self, mock_get_station, mock_load_archive_s3, mock_print
    ):
        """Test detect command with empty archive."""
        mock_get_station.return_value = (
            {"macAddress": "98:CD:AC:22:0D:E5"},
            "98:CD:AC:22:0D:E5",
            "Test Station",
        )
        mock_load_archive_s3.return_value = pd.DataFrame()

        args = mock.Mock()
        handle_detect(args)

        calls = mock_print.call_args_list
        assert "❌ Archive is empty or not found." in str(calls)

    @mock.patch("builtins.print")
    @mock.patch("lookout.core.rain_corrections.detect_carryovers")
    @mock.patch("lookout.cli.rain_corrections_cli.load_archive_s3")
    @mock.patch("lookout.cli.rain_corrections_cli.get_station")
    def test_detect_no_carryovers(
        self,
        mock_get_station,
        mock_load_archive_s3,
        mock_detect_carryovers,
        mock_print,
    ):
        """Test detect command when no carryovers found."""
        mock_get_station.return_value = (
            {"macAddress": "98:CD:AC:22:0D:E5"},
            "98:CD:AC:22:0D:E5",
            "Test Station",
        )
        mock_load_archive_s3.return_value = pd.DataFrame({"test": [1]})
        mock_detect_carryovers.return_value = []

        args = mock.Mock()
        handle_detect(args)

        calls = mock_print.call_args_list
        assert "✅ No carryovers detected." in str(calls)

    @mock.patch("builtins.print")
    @mock.patch("lookout.core.rain_corrections.detect_carryovers")
    @mock.patch("lookout.cli.rain_corrections_cli.load_archive_s3")
    @mock.patch("lookout.cli.rain_corrections_cli.get_station")
    def test_detect_with_carryovers(
        self,
        mock_get_station,
        mock_load_archive_s3,
        mock_detect_carryovers,
        mock_print,
    ):
        """Test detect command with carryovers found."""
        mock_get_station.return_value = (
            {"macAddress": "98:CD:AC:22:0D:E5"},
            "98:CD:AC:22:0D:E5",
            "Test Station",
        )
        mock_load_archive_s3.return_value = pd.DataFrame({"test": [1]})

        # Mock the known carryover data
        mock_carryovers = [
            {
                "affected_date": "2024-12-16",
                "carryover_amount": 0.063,
                "gap_start": "2024-12-15 18:25:00",
                "gap_end": "2024-12-16 00:17:00",
            }
        ]
        mock_detect_carryovers.return_value = mock_carryovers

        args = mock.Mock()
        handle_detect(args)

        calls = mock_print.call_args_list
        assert "🚨 Found 1 carryover(s):" in str(calls)
        assert '2024-12-16: 0.063"' in str(calls)

    @mock.patch("builtins.print")
    @mock.patch("lookout.core.rain_corrections.save_catalog")
    @mock.patch("lookout.core.rain_corrections.detect_carryovers")
    @mock.patch("lookout.cli.rain_corrections_cli.load_archive_s3")
    @mock.patch("lookout.cli.rain_corrections_cli.get_station")
    def test_generate_dry_run(
        self,
        mock_get_station,
        mock_load_archive_s3,
        mock_detect_carryovers,
        mock_save_catalog,
        mock_print,
    ):
        """Test generate command with dry-run flag."""
        mock_get_station.return_value = (
            {"macAddress": "98:CD:AC:22:0D:E5"},
            "98:CD:AC:22:0D:E5",
            "Test Station",
        )
        mock_load_archive_s3.return_value = pd.DataFrame({"test": [1]})
        mock_detect_carryovers.return_value = [
            {
                "affected_date": "2024-12-16",
                "carryover_amount": 0.063,
                "gap_start": "2024-12-15 18:25:00",
                "gap_end": "2024-12-16 00:17:00",
            }
        ]
        mock_save_catalog.return_value = True

        args = mock.Mock()
        args.dry_run = True
        args.bucket = "lookout"

        handle_generate(args)

        calls = mock_print.call_args_list
        assert "🔒 --dry-run: Catalog not saved." in str(calls)
        mock_save_catalog.assert_not_called()

    @mock.patch("builtins.print")
    @mock.patch("lookout.core.rain_corrections.save_catalog")
    @mock.patch("lookout.core.rain_corrections.detect_carryovers")
    @mock.patch("lookout.cli.rain_corrections_cli.load_archive_s3")
    @mock.patch("lookout.cli.rain_corrections_cli.get_station")
    def test_generate_save_success(
        self,
        mock_get_station,
        mock_load_archive_s3,
        mock_detect_carryovers,
        mock_save_catalog,
        mock_print,
    ):
        """Test generate command saving to S3 successfully."""
        mock_get_station.return_value = (
            {"macAddress": "98:CD:AC:22:0D:E5"},
            "98:CD:AC:22:0D:E5",
            "Test Station",
        )
        mock_load_archive_s3.return_value = pd.DataFrame({"test": [1]})
        mock_detect_carryovers.return_value = [
            {
                "affected_date": "2024-12-16",
                "carryover_amount": 0.063,
                "gap_start": "2024-12-15 18:25:00",
                "gap_end": "2024-12-16 00:17:00",
            }
        ]
        mock_save_catalog.return_value = True

        args = mock.Mock()
        args.dry_run = False
        args.bucket = "lookout"

        handle_generate(args)

        calls = mock_print.call_args_list
        assert "💾 Saved catalog with 1 corrections to S3." in str(calls)
        mock_save_catalog.assert_called_once()

    @mock.patch("builtins.print")
    @mock.patch("lookout.cli.rain_corrections_cli.ambient_client.get_devices")
    def test_show_no_device(self, mock_get_devices, mock_print, monkeypatch):
        """show exits 1 with the reason when the pinned station is missing."""
        monkeypatch.setenv("STATION_MAC", "12:34:56:78:9A:BC")
        mock_get_devices.return_value = [
            {"macAddress": "98:CD:AC:22:0D:E5", "info": {"name": "Other"}}
        ]

        with pytest.raises(SystemExit) as exited:
            handle_show(mock.Mock())

        assert exited.value.code == 1
        assert "Station 12:34:56:78:9A:BC is not on this Ambient account" in str(
            mock_print.call_args
        )

    @staticmethod
    def run_show(capsys) -> str:
        """Run `show` for FAKE_MAC against the fake bucket; what it printed."""
        args = mock.Mock()
        args.bucket = "lookout"
        station = ({"macAddress": FAKE_MAC}, FAKE_MAC, "Test Station")
        with mock.patch(
            "lookout.cli.rain_corrections_cli.get_station", return_value=station
        ):
            handle_show(args)
        return capsys.readouterr().out

    def test_show_no_catalog(self, fake_s3, capsys):
        """show says so when no corrections catalog is stored."""
        assert "📭 No corrections catalog found." in self.run_show(capsys)

    def test_show_with_catalog(self, fake_s3, capsys):
        """show prints the stored catalog."""
        rain_corrections.save_catalog(
            {
                "version": "1.0",
                "updated_at": "2025-12-20T20:00:00Z",
                "corrections": [
                    {
                        "affected_date": "2024-12-16",
                        "carryover_amount": 0.063,
                        "gap_start": "2024-12-15 18:25:00",
                        "gap_end": "2024-12-16 00:17:00",
                    }
                ],
            },
            FAKE_MAC,
        )

        out = self.run_show(capsys)

        assert "Version: 1.0" in out
        assert "Corrections: 1" in out
        assert '2024-12-16: 0.063"' in out

    def test_show_exits_1_when_the_catalog_read_fails(
        self, fake_s3, capsys, monkeypatch
    ):
        """A failed read is an error, never "version None, 0 corrections"."""
        rain_corrections.save_catalog({"version": "1.0", "corrections": []}, FAKE_MAC)
        flaky_storj(monkeypatch, fake_s3, read=CORRECTIONS_KEY)

        with pytest.raises(SystemExit) as exited:
            self.run_show(capsys)

        assert exited.value.code == 1
        out = capsys.readouterr().out
        assert "storj read timed out" in out
        assert "Version" not in out and "Corrections" not in out


class TestApplyCorrections:
    """Test rain carryover corrections application."""

    @mock.patch("lookout.core.rain_corrections.read_catalog")
    def test_no_catalog_returns_original_df(self, mock_read_catalog):
        """Test returns original DataFrame when no catalog exists."""
        mock_read_catalog.return_value = None

        original_df = pd.DataFrame({"dailyrainin": [1.0, 2.0]})
        result = apply_corrections(original_df, "98:CD:AC:22:0D:E5")

        assert result.equals(original_df)

    @mock.patch("lookout.core.rain_corrections.read_catalog")
    def test_single_correction_applied_correctly(self, mock_read_catalog):
        """Test correct subtraction amounts and period boundaries."""
        mock_catalog = {
            "corrections": [{"affected_date": "2024-12-16", "carryover_amount": 0.063}]
        }
        mock_read_catalog.return_value = mock_catalog

        df = pd.DataFrame(
            {
                "date": ["2024-12-16 00:00:00", "2024-12-17 00:00:00"],
                "dailyrainin": [0.1, 0.05],
                "weeklyrainin": [1.0, 0.5],
                "monthlyrainin": [10.0, 5.0],
                "yearlyrainin": [50.0, 25.0],
            }
        )

        result = apply_corrections(df, "98:CD:AC:22:0D:E5")

        # Verify daily correction on affected date only
        assert result.loc[0, "dailyrainin"] == pytest.approx(
            0.037, abs=1e-10
        )  # 0.1 - 0.063
        assert result.loc[1, "dailyrainin"] == 0.05  # unchanged

    @mock.patch("lookout.core.rain_corrections.read_catalog")
    def test_negative_values_clipped_to_zero(self, mock_read_catalog):
        """Test clipping prevents negative rain values."""
        mock_catalog = {
            "corrections": [
                {
                    "affected_date": "2024-12-16",
                    "carryover_amount": 1.0,  # Large correction
                }
            ]
        }
        mock_read_catalog.return_value = mock_catalog

        df = pd.DataFrame(
            {
                "date": ["2024-12-16 00:00:00"],
                "dailyrainin": [0.5],
                "weeklyrainin": [0.5],
                "monthlyrainin": [0.5],
                "yearlyrainin": [0.5],
            }
        )

        result = apply_corrections(df, "98:CD:AC:22:0D:E5")

        # All values should be clipped to 0, not negative
        assert result.loc[0, "dailyrainin"] == 0.0
        assert result.loc[0, "weeklyrainin"] == 0.0
        assert result.loc[0, "monthlyrainin"] == 0.0
        assert result.loc[0, "yearlyrainin"] == 0.0


COUNTERS = ["dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"]
TZ = "America/Los_Angeles"


def carryover_free_scenario() -> pd.DataFrame:
    """The SCENARIO week as the station would have reported it with no carryover."""
    gaps = (Gap(SCENARIO.gaps[0].start, SCENARIO.gaps[0].end), SCENARIO.gaps[1])
    return build_archive(SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=gaps)


def counter_frame(rows: list) -> pd.DataFrame:
    """
    Build an archive-shaped frame from `(local time, value)` rows, newest first.

    Each rain counter holds `value`, so a correction shows as the difference.
    """
    times = pd.DatetimeIndex([t for t, _ in rows]).tz_localize(TZ)
    df = pd.DataFrame({"date": times, "dateutc": times.tz_convert("UTC").asi8 // 10**6})
    for col in COUNTERS:
        df[col] = [float(v) for _, v in rows]
    return df.sort_values("dateutc", ascending=False).reset_index(drop=True)


def stored(catalog_rows: list) -> dict:
    """A corrections catalog with one entry per `(affected_date, amount)`."""
    return {
        "corrections": [
            {"affected_date": d, "carryover_amount": a} for d, a in catalog_rows
        ]
    }


class TestDetectCarryovers:
    """`detect_carryovers`: outages across midnight that left the daily counter up."""

    def test_finds_the_scenario_carryover(self):
        found = detect_carryovers(scenario_archive())

        assert len(found) == 1
        assert found[0]["affected_date"] == "2026-01-09"
        assert found[0]["source_date"] == "2026-01-08"
        assert found[0]["carryover_amount"] == pytest.approx(0.48)
        assert pd.Timestamp(found[0]["gap_start"]) == pd.Timestamp(
            "2026-01-09T06:55:00Z"
        )
        assert pd.Timestamp(found[0]["gap_end"]) == pd.Timestamp("2026-01-09T09:00:00Z")

    def test_order_of_the_input_does_not_matter_and_it_is_not_mutated(self):
        newest_first = scenario_archive()
        oldest_first = scenario_archive(newest_first=False)
        before = newest_first.copy()

        assert detect_carryovers(oldest_first) == detect_carryovers(newest_first)
        pd.testing.assert_frame_equal(newest_first, before)

    def test_midnight_outage_where_the_counter_reset_is_not_a_carryover(self):
        reset_ok = carryover_free_scenario()

        assert detect_carryovers(reset_ok) == []

    def test_outage_within_a_day_is_not_a_carryover(self):
        no_gap_at_midnight = build_archive(
            SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=(SCENARIO.gaps[1],)
        )

        assert detect_carryovers(no_gap_at_midnight) == []

    def test_real_station_slice_has_none(self):
        assert detect_carryovers(load_real_slice()) == []

    def test_empty_or_without_dailyrainin_finds_nothing(self):
        assert detect_carryovers(pd.DataFrame()) == []
        assert detect_carryovers(pd.DataFrame({"dateutc": [1], "date": [1]})) == []

    @pytest.mark.parametrize(
        "prev, first, found",
        [
            (0.30, 0.30, True),  # unchanged across the outage
            (0.30, 0.34, True),  # inside the 0.05 tolerance
            (0.30, 0.36, False),  # outside it
            (0.30, 0.0, False),  # it reset
            (0.01, 0.01, False),  # trace amounts are ignored
            (0.02, 0.02, True),  # just over the floor
        ],
    )
    def test_matches_within_tolerance_above_the_floor(self, prev, first, found):
        df = counter_frame([("2026-01-08 23:55", prev), ("2026-01-09 02:00", first)])

        assert bool(detect_carryovers(df)) is found

    def test_a_first_record_inside_the_midnight_window_is_never_flagged(self):
        on_time = counter_frame([("2026-01-08 23:55", 0.3), ("2026-01-09 00:05", 0.3)])
        just_late = counter_frame(
            [("2026-01-08 23:55", 0.3), ("2026-01-09 00:06", 0.3)]
        )

        assert detect_carryovers(on_time) == []
        assert len(detect_carryovers(just_late)) == 1


class TestApplyCorrectionsAgainstScenario:
    """`apply_corrections` against the SCENARIO carryover, via the Storj catalog."""

    def test_detected_corrections_restore_the_counters_the_station_should_report(
        self, fake_s3
    ):
        carried = scenario_archive()
        rain_corrections.save_catalog(
            {"corrections": detect_carryovers(carried)}, FAKE_MAC
        )

        corrected = apply_corrections(carried, FAKE_MAC)

        pd.testing.assert_frame_equal(corrected, carryover_free_scenario())

    def test_input_is_untouched_and_row_order_is_kept(self, fake_s3):
        carried = scenario_archive()
        before = carried.copy()
        rain_corrections.save_catalog(
            {"corrections": detect_carryovers(carried)}, FAKE_MAC
        )

        corrected = apply_corrections(carried, FAKE_MAC)

        pd.testing.assert_frame_equal(carried, before)
        assert list(corrected["dateutc"]) == list(carried["dateutc"])

    def test_no_catalog_in_storage_returns_the_archive_as_is(self, fake_s3):
        carried = scenario_archive()

        assert apply_corrections(carried, FAKE_MAC) is carried

    def test_a_failed_catalog_read_raises_rather_than_return_uncorrected_rain(
        self, fake_s3, monkeypatch
    ):
        carried = scenario_archive()
        rain_corrections.save_catalog(
            {"corrections": detect_carryovers(carried)}, FAKE_MAC
        )
        flaky_storj(monkeypatch, fake_s3, read=CORRECTIONS_KEY)

        with pytest.raises(StorageReadError):
            apply_corrections(carried, FAKE_MAC)

    def test_empty_corrections_list_returns_the_archive_as_is(self, fake_s3):
        carried = scenario_archive()
        rain_corrections.save_catalog({"corrections": []}, FAKE_MAC)

        assert apply_corrections(carried, FAKE_MAC) is carried

    def test_applying_twice_subtracts_twice(self, fake_s3):
        """
        Held finding: corrections are not idempotent.

        Nothing marks a row as corrected, so a second pass over a corrected
        archive takes the carryover off again.
        """
        carried = scenario_archive()
        rain_corrections.save_catalog(
            {"corrections": detect_carryovers(carried)}, FAKE_MAC
        )

        once = apply_corrections(carried, FAKE_MAC)
        twice = apply_corrections(once, FAKE_MAC)

        friday_noon = carried["dateutc"] == 1767988800000  # 2026-01-09T20:00Z
        assert once.loc[friday_noon, "weeklyrainin"].item() == pytest.approx(1.32)
        assert twice.loc[friday_noon, "weeklyrainin"].item() == pytest.approx(0.84)


class TestCorrectionPeriods:
    """`apply_corrections` ends each correction where its counter next resets."""

    @staticmethod
    def corrected(rows, catalog, mock_catalog) -> pd.DataFrame:
        mock_catalog.return_value = catalog
        return apply_corrections(counter_frame(rows), "mac").sort_values("date")

    @staticmethod
    def drop_by_date(result: pd.DataFrame, col: str) -> dict:
        """How far each local calendar day's rows moved from 5.0."""
        days = result["date"].dt.strftime("%Y-%m-%d")
        return {d: round(5.0 - v, 9) for d, v in zip(days, result[col])}

    ROWS = [
        ("2024-12-13 12:00", 5),  # Fri before
        ("2024-12-14 12:00", 5),  # Sat before
        ("2024-12-15 12:00", 5),  # Sun, week reset
        ("2024-12-16 12:00", 5),  # Mon: affected
        ("2024-12-17 12:00", 5),
        ("2024-12-21 12:00", 5),  # Sat: last day of the week
        ("2024-12-22 12:00", 5),  # Sun: next week
        ("2024-12-31 12:00", 5),  # last day of the month and the year
        ("2025-01-01 12:00", 5),
    ]

    @mock.patch("lookout.core.rain_corrections.read_catalog")
    def test_each_counter_is_corrected_until_it_resets(self, read_catalog):
        result = self.corrected(self.ROWS, stored([("2024-12-16", 0.5)]), read_catalog)

        daily = self.drop_by_date(result, "dailyrainin")
        weekly = self.drop_by_date(result, "weeklyrainin")
        monthly = self.drop_by_date(result, "monthlyrainin")
        yearly = self.drop_by_date(result, "yearlyrainin")

        assert {d for d, v in daily.items() if v} == {"2024-12-16"}
        assert {d for d, v in weekly.items() if v} == {
            "2024-12-16",
            "2024-12-17",
            "2024-12-21",
        }
        assert {d for d, v in monthly.items() if v} == {
            "2024-12-16",
            "2024-12-17",
            "2024-12-21",
            "2024-12-22",
            "2024-12-31",
        }
        # The year ends on the 31st, so here yearly moves with monthly.
        assert {d for d, v in yearly.items() if v} == {
            d for d, v in monthly.items() if v
        }
        assert set(daily.values()) <= {0.0, 0.5}

    @mock.patch("lookout.core.rain_corrections.read_catalog")
    def test_a_saturday_carryover_also_takes_the_following_week(self, read_catalog):
        """
        Held finding: a Saturday carryover over-corrects the next week.

        The week is closed with `Week(weekday=5)`, which moves a Saturday on to
        the next Saturday, so Sunday to Saturday after it lose the carryover
        from `weeklyrainin` although the counter reset on Sunday.
        """
        rows = [
            ("2024-12-21 12:00", 5),  # Sat: affected
            ("2024-12-22 12:00", 5),  # Sun: reset
            ("2024-12-28 12:00", 5),  # next Sat
            ("2024-12-29 12:00", 5),  # Sun after that
        ]
        result = self.corrected(rows, stored([("2024-12-21", 0.5)]), read_catalog)

        weekly = self.drop_by_date(result, "weeklyrainin")

        assert weekly == {
            "2024-12-21": 0.5,
            "2024-12-22": 0.5,
            "2024-12-28": 0.5,
            "2024-12-29": 0.0,
        }

    @mock.patch("lookout.core.rain_corrections.read_catalog")
    def test_corrections_add_up_and_other_columns_are_untouched(self, read_catalog):
        rows = [("2024-12-16 12:00", 5), ("2024-12-17 12:00", 5)]
        df = counter_frame(rows)
        df["tempf"] = 60.0
        read_catalog.return_value = stored([("2024-12-16", 0.5), ("2024-12-17", 1.0)])

        result = apply_corrections(df, "mac").sort_values("date")

        assert result["weeklyrainin"].tolist() == [4.5, 3.5]
        assert result["dailyrainin"].tolist() == [4.5, 4.0]
        assert (result["tempf"] == 60.0).all()
        assert df["weeklyrainin"].tolist() == [5.0, 5.0]


class TestCatalogStorage:
    """The corrections catalog's place in storage."""

    def test_catalog_round_trips_under_the_mac_keyed_name(self, fake_s3):
        catalog = stored([("2026-01-09", 0.48)])

        assert rain_corrections.read_catalog(FAKE_MAC) is None
        assert rain_corrections.save_catalog(catalog, FAKE_MAC)

        assert rain_corrections.read_catalog(FAKE_MAC) == catalog
        keys = [o["Key"] for o in fake_s3.list_objects_v2(Bucket="lookout")["Contents"]]
        assert keys == [f"{FAKE_MAC}.rain_corrections.json"]

    def test_missing_catalog_reads_as_none(self, fake_s3):
        assert rain_corrections.read_catalog(FAKE_MAC) is None

    def test_a_failed_read_raises(self, fake_s3, monkeypatch):
        rain_corrections.save_catalog({"corrections": []}, FAKE_MAC)
        flaky_storj(monkeypatch, fake_s3, read=CORRECTIONS_KEY)

        with pytest.raises(StorageReadError):
            rain_corrections.read_catalog(FAKE_MAC)
