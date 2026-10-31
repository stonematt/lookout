"""
Healthchecks heartbeats: slugs, freshness and the ping sender.

The endpoint is the conftest `healthchecks` fake; nothing reaches the web.
"""

from types import SimpleNamespace

import pytest

from lookout.secrets_loader import MissingSecretError
from lookout.utils import healthchecks as hc

NOW = 1_700_000_000_000
MINUTE = 60_000


class TestSlugs:
    def test_the_station_slug_is_the_slugified_name(self):
        assert (
            hc.station_slug("Matt's Station #2", "AA:BB") == "station-matt-s-station-2"
        )

    def test_an_unnamed_station_falls_back_to_its_mac(self):
        assert hc.station_slug(None, "AA:BB:CC") == "station-aa-bb-cc"
        assert hc.station_slug("!!!", "AA:BB:CC") == "station-aa-bb-cc"


class TestFreshness:
    @pytest.mark.parametrize(
        "age_minutes, fresh",
        [(0, True), (14, True), (15, False), (90, False)],
    )
    def test_a_reading_is_fresh_under_fifteen_minutes(self, age_minutes, fresh):
        assert hc.is_fresh(NOW - age_minutes * MINUTE, NOW) is fresh

    def test_no_reading_is_not_fresh(self):
        assert hc.is_fresh(None, NOW) is False


class TestHeartbeat:
    def test_pings_use_the_key_from_get_secret(self, healthchecks, stub_secrets):
        hc.Heartbeat.from_secrets().start()

        assert healthchecks.urls == [
            f"https://hc-ping.com/{stub_secrets['HC_PING_KEY']}/lookout-refresh/start"
        ]

    def test_the_exit_status_is_the_suffix(self, healthchecks):
        beat = hc.Heartbeat.from_secrets()
        beat.finish(0)
        beat.finish(1)

        assert healthchecks.pings == ["lookout-refresh/0", "lookout-refresh/1"]

    def test_a_missing_key_skips_pinging_with_a_warning(
        self, healthchecks, monkeypatch, caplog
    ):
        def missing(name):
            raise MissingSecretError(name)

        monkeypatch.setattr(hc, "get_secret", missing)
        beat = hc.Heartbeat.from_secrets()

        assert beat.start() is False
        assert beat.finish(0) is False
        assert healthchecks.pings == []

    def test_a_down_endpoint_is_swallowed(self, healthchecks):
        healthchecks.fail = True

        assert hc.Heartbeat("key").start() is False

    def test_a_bad_status_is_swallowed(self, monkeypatch):
        def refuse():
            raise RuntimeError("502")

        monkeypatch.setattr(
            hc,
            "requests",
            SimpleNamespace(
                get=lambda *a, **k: SimpleNamespace(raise_for_status=refuse)
            ),
        )

        assert hc.Heartbeat("key").start() is False

    def test_a_stale_reading_sends_no_station_ping(self, healthchecks):
        beat = hc.Heartbeat("key")

        assert beat.station("Home", "AA", NOW - 20 * MINUTE, NOW) is False
        assert healthchecks.pings == []

    def test_a_fresh_reading_pings_the_station_slug(self, healthchecks):
        assert hc.Heartbeat("key").station("Home", "AA", NOW - MINUTE, NOW) is True
        assert healthchecks.pings == ["station-home"]
