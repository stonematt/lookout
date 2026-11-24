"""
titles.py: The title text of tiles whose title follows a mode or data.

GLOSSARY tile title: where a tile's number depends on a mode or counts a
window, its title names what it counts, and Python decides the text. The
site draws it on the tile, its zoom heading and the sources footer. The
wet-or-dry tile follows the wet mode (#658); the ten-day temperature tile
follows its number, the highs against usual (#665).

Functions:
- tile_titles(mode, highs): The `TileTitles` field.
- ten_days_title(highs): The ten-day temperature tile's title.
"""

from typing import Optional

from lookout.snapshot.model import TenDayHighs, TileTitles, WetMode

WET_OR_DRY = {"wet": "Wet hours", "dry": "Dry spell"}
"""The wet-or-dry tile's title in each mode."""

TEN_DAYS = "Last 10 days"
"""The ten-day tile's fixed name, while there is nothing to compare."""

TEN_DAYS_GAP = "Gap in record"
"""The ten-day tile's title when a day the record covers is missing."""

HIGHS_OVER = "Highs over usual"
HIGHS_UNDER = "Highs under usual"
HIGHS_AT = "Highs at usual"
"""The ten-day tile's title by the highs' side of usual. The site's words
for a document without the title (`highsWords` in
`web/src/overview/mosaic-tiles.ts`) follow the same sign."""


def ten_days_title(highs: Optional[TenDayHighs]) -> str:
    """
    The ten-day temperature tile's title, named by its number.

    A gap wins over the highs' side of usual, so a missing day never reads
    as level.

    :param highs: The ten-day highs against usual, or None with nothing to
        compare.
    :return: "Highs over usual", "Highs under usual", "Highs at usual",
        "Gap in record", or "Last 10 days" with nothing to compare.
    """
    if highs is None:
        return TEN_DAYS
    if highs.gap:
        return TEN_DAYS_GAP
    if highs.delta_f > 0:
        return HIGHS_OVER
    if highs.delta_f < 0:
        return HIGHS_UNDER
    return HIGHS_AT


def tile_titles(mode: WetMode, highs: Optional[TenDayHighs]) -> TileTitles:
    """
    Each mode- or data-led tile's title.

    :param mode: Today's wet mode.
    :param highs: The ten-day highs against usual
        (`TemperatureContext.ten_day_highs`).
    :return: The titles.
    """
    return TileTitles(
        wet_or_dry=WET_OR_DRY[mode.mode],
        ten_days=ten_days_title(highs),
    )
