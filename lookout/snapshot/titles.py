"""
titles.py: The title text of tiles whose title follows a mode or data.

GLOSSARY tile title: where a tile's number depends on a mode or counts a
window, its title names what it counts, and Python decides the text. The
site draws it on the tile, its zoom heading and the sources footer. The
wet-or-dry tile follows the wet mode (#658); the ten-day temperature tile
follows its number, the highs against usual (#665).

Functions:
- tile_titles(mode, temperature): The `TileTitles` field.
- ten_days_title(temperature): The ten-day temperature tile's title.
"""

from lookout.snapshot.model import TemperatureContext, TileTitles, WetMode

WET_OR_DRY = {"wet": "Wet hours", "dry": "Dry spell"}
"""The wet-or-dry tile's title in each mode."""

TEN_DAYS = "Last 10 days"
"""The ten-day tile's fixed name, while there is nothing to compare."""

TEN_DAYS_GAP = "Gap in record"
"""The ten-day tile's title when a day the record covers is missing."""


def ten_days_title(temperature: TemperatureContext) -> str:
    """
    The ten-day temperature tile's title, named by its number.

    A day is a gap when the record covers it (its status is shown) and it
    has no low or high, the same days the tile's chart hatches. A gap wins
    over the highs' side of usual, so a missing day never reads as level.

    :param temperature: The snapshot's temperature context.
    :return: "Highs over usual", "Highs under usual", "Highs at usual",
        "Gap in record", or "Last 10 days" with no highs against usual.
    """
    highs = temperature.ten_day_highs
    if highs is None:
        return TEN_DAYS
    if any(
        day.status.shown and (day.low is None or day.high is None)
        for day in temperature.ten_days
    ):
        return TEN_DAYS_GAP
    if highs.delta_f > 0:
        return "Highs over usual"
    if highs.delta_f < 0:
        return "Highs under usual"
    return "Highs at usual"


def tile_titles(mode: WetMode, temperature: TemperatureContext) -> TileTitles:
    """
    Each mode- or data-led tile's title.

    :param mode: Today's wet mode.
    :param temperature: The snapshot's temperature context.
    :return: The titles.
    """
    return TileTitles(
        wet_or_dry=WET_OR_DRY[mode.mode],
        ten_days=ten_days_title(temperature),
    )
