"""
event_category.py: A rain event's category, from its total and peak rate.

Every event in the Event Catalog is one of four categories, smallest first:
sprinkle, shower, rain and storm. The rain tab's event list filters on it,
and sprinkles are hidden there by default: a single bucket tip makes an
event, and on a wet-winter station a third of the catalog is sprinkles.

A sprinkle is judged on its total alone, since one tip reads as a rate of
0.05 to 0.15 in/hr between 5-minute readings. Above that, an event takes the
larger of its total's category and its peak rate's, so a short cloudburst
counts as a storm. The cutoffs were set against the Salem station's catalog
(259 events, water years 2023 to 2027): storms come 5 or 6 a year.

Functions:
- event_category(total_in, peak_rate_in_hr): The category.
"""

from typing import Optional

SPRINKLE = "sprinkle"
SHOWER = "shower"
RAIN = "rain"
STORM = "storm"

CATEGORIES = (SPRINKLE, SHOWER, RAIN, STORM)
"""The categories, smallest first."""

SPRINKLE_BELOW_IN = 0.05
"""An event with less than this total is a sprinkle, whatever its rate."""

RAIN_FROM_IN = 0.5
"""A total of at least this is rain."""

STORM_FROM_IN = 2.0
"""A total of at least this is a storm."""

RAIN_FROM_IN_HR = 0.5
"""A peak rate of at least this makes a shower rain."""

STORM_FROM_IN_HR = 1.0
"""A peak rate of at least this makes a shower or rain a storm."""


def event_category(total_in: float, peak_rate_in_hr: Optional[float]) -> str:
    """
    One event's category.

    :param total_in: Its total rain, in inches (the live total while live).
    :param peak_rate_in_hr: Its highest rate between readings; None reads as
        no rate, so the total decides.
    :return: "sprinkle", "shower", "rain" or "storm".
    """
    if total_in < SPRINKLE_BELOW_IN:
        return SPRINKLE
    rate = peak_rate_in_hr or 0.0
    if total_in >= STORM_FROM_IN or rate >= STORM_FROM_IN_HR:
        return STORM
    if total_in >= RAIN_FROM_IN or rate >= RAIN_FROM_IN_HR:
        return RAIN
    return SHOWER
