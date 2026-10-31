"""
healthchecks.py: Healthchecks.io heartbeats for the refresh run.

Two checks, both auto-created on first ping from the project's ping key plus
a slug (`https://hc-ping.com/<ping key>/<slug>?create=1`):

- the cron check (`CRON_SLUG`): `/start` when a run begins, then the exit
  status when it ends. A non-zero status fails the check at once.
- the station check (`station-<slugified station name>`): pinged only while
  the live reading is under `STATION_FRESH_MINUTES` old, so a station that
  stops reporting stops pinging and the check goes late.

Period and grace are set on the Healthchecks side, not in the ping (cron: 5
min period, 10 min grace; station: 5 min period, 30 min grace).

A heartbeat never fails the run: no ping key skips pinging with one warning,
and a ping that errors or times out is logged and dropped.
"""

import re
from typing import Callable, Optional

import requests

from lookout.secrets_loader import get_secret
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="refresh.log")

PING_BASE = "https://hc-ping.com"
CRON_SLUG = "lookout-refresh"
STATION_FRESH_MINUTES = 15
PING_TIMEOUT_SECONDS = 10


def slugify(text: str) -> str:
    """
    Reduce text to a Healthchecks slug: lowercase letters, digits, hyphens.

    :param text: Any text, e.g. a station name.
    :return: The slug; empty when the text has no letters or digits.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def station_slug(name: Optional[str], mac: str) -> str:
    """
    The station check's slug, `station-<slugified name>`.

    :param name: The station's Ambient name, or None when unnamed.
    :param mac: Its MAC address, used when the name gives no slug.
    :return: The slug.
    """
    return f"station-{slugify(name or '') or slugify(mac)}"


def is_fresh(
    observed_at_ms: Optional[int],
    now_ms: int,
    max_age_minutes: int = STATION_FRESH_MINUTES,
) -> bool:
    """
    Whether a reading is recent enough to count as the station reporting.

    :param observed_at_ms: The reading's time, epoch ms UTC; None if unknown.
    :param now_ms: The current time, epoch ms UTC.
    :param max_age_minutes: The oldest age that still counts.
    :return: True when the reading is under `max_age_minutes` old.
    """
    if observed_at_ms is None:
        return False
    return now_ms - observed_at_ms < max_age_minutes * 60_000


class Heartbeat:
    """
    Sends pings to one Healthchecks project; every method swallows failure.

    :param ping_key: The project's ping key; None or empty disables pinging.
    :param get: The HTTP GET to use; `requests.get` by default.
    """

    def __init__(
        self,
        ping_key: Optional[str],
        get: Optional[Callable[..., object]] = None,
    ) -> None:
        self.ping_key = ping_key or None
        self._get = get or requests.get
        if self.ping_key is None:
            logger.warning("HC_PING_KEY is not set; Healthchecks pings are skipped")

    @classmethod
    def from_secrets(cls) -> "Heartbeat":
        """
        Build from the `HC_PING_KEY` secret; disabled when it is missing.

        :return: The heartbeat sender.
        """
        try:
            return cls(get_secret("HC_PING_KEY"))
        except Exception as e:  # missing key or unreadable secrets file
            logger.warning(f"HC_PING_KEY unavailable: {type(e).__name__}")
            return cls(None)

    def ping(self, slug: str, suffix: str = "") -> bool:
        """
        Ping one check, creating it on first use.

        :param slug: The check's slug.
        :param suffix: "" for success, "start", "fail" or an exit status.
        :return: True when Healthchecks accepted the ping.
        """
        if self.ping_key is None:
            return False
        path = f"{slug}/{suffix}" if suffix else slug
        url = f"{PING_BASE}/{self.ping_key}/{path}"
        try:
            response = self._get(
                url, params={"create": 1}, timeout=PING_TIMEOUT_SECONDS
            )
            response.raise_for_status()
        except Exception as e:
            # Never log the URL: it holds the ping key.
            logger.warning(f"Healthchecks ping {path} failed: {type(e).__name__}")
            return False
        return True

    def start(self) -> bool:
        """Cron check: a run has begun."""
        return self.ping(CRON_SLUG, "start")

    def finish(self, status: int) -> bool:
        """
        Cron check: the run ended; non-zero fails the check at once.

        :param status: The run's exit status.
        """
        return self.ping(CRON_SLUG, str(status))

    def station(
        self,
        name: Optional[str],
        mac: str,
        observed_at_ms: Optional[int],
        now_ms: int,
    ) -> bool:
        """
        Station check: ping only while the live reading is fresh.

        :param name: The station's Ambient name, or None.
        :param mac: Its MAC address.
        :param observed_at_ms: The live reading's time, epoch ms UTC.
        :param now_ms: The current time, epoch ms UTC.
        :return: True when a ping was sent and accepted.
        """
        if not is_fresh(observed_at_ms, now_ms):
            logger.info("Live reading is stale; station heartbeat skipped")
            return False
        return self.ping(station_slug(name, mac))
