"""
titles.py: The title text of tiles whose title follows a mode or data (#658).

GLOSSARY tile title: where a tile's number depends on a mode or counts a
window, its title names what it counts, and Python decides the text. The
site draws it on the tile, its zoom heading and the sources footer.

Functions:
- tile_titles(mode): The `TileTitles` field.
"""

from lookout.snapshot.model import TileTitles, WetMode

WET_OR_DRY = {"wet": "Wet hours", "dry": "Dry spell"}
"""The wet-or-dry tile's title in each mode."""


def tile_titles(mode: WetMode) -> TileTitles:
    """
    Each mode- or data-led tile's title.

    :param mode: Today's wet mode.
    :return: The titles.
    """
    return TileTitles(wet_or_dry=WET_OR_DRY[mode.mode])
