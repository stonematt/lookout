"""
widget.py: The widget view, which of the widget's views shows now.

A Lookout widget shows one view at a time: the current value against the
station's own history. Python chooses the view so every widget (Scriptable
now, a WidgetKit app later) shows the same one. The rules are checked in
order, and the first that holds is the view:

1. **rain_event**: an event is live and more than `RAIN_EVENT_IN` has fallen
   in it (`ActiveRainEvent.total_in`, the live `eventrainin`).
2. **dry_spell**: the station-local date is in a summer month
   (`SUMMER_MONTHS`) and it has been dry for at least `DRY_SPELL_DAYS`: the
   hero lead's calm rule value, the days since `lastRain`.
3. **temperature_ranges**, otherwise: now against today, 7, 30 and 365 days.

Functions:
- widget_view(active, lead, clock): The `WidgetView` for one snapshot.
"""

from typing import Optional

from lookout.snapshot.clock import StationClock
from lookout.snapshot.model import ActiveRainEvent, HeroLead, WidgetView

RAIN_EVENT_IN = 0.04
"""The rain event view needs more than this much rain in the live event."""

SUMMER_MONTHS = (6, 7, 8, 9)
"""Station-local months in which the dry spell view can show, June to
September (the station's dry season; northern hemisphere)."""

DRY_SPELL_DAYS = 3.0
"""The dry spell view needs at least this many days since it last rained."""


def _days_dry(lead: HeroLead) -> Optional[float]:
    """
    The days since it last rained: the calm rule's value.

    :param lead: The hero lead, which carries every rule's value.
    :return: The days, or None when there is no `lastRain` reading.
    """
    calm = next((rule for rule in lead.rules if rule.kind == "calm"), None)
    return None if calm is None else calm.value


def widget_view(
    active: Optional[ActiveRainEvent], lead: HeroLead, clock: StationClock
) -> WidgetView:
    """
    Choose the widget's view.

    :param active: The active rain event, or None.
    :param lead: The hero lead, for the days since it last rained.
    :param clock: The build's station clock, for the station-local month.
    :return: The view name.
    """
    if active is not None and active.total_in > RAIN_EVENT_IN:
        return "rain_event"
    days_dry = _days_dry(lead)
    if (
        clock.today.month in SUMMER_MONTHS
        and days_dry is not None
        and days_dry >= DRY_SPELL_DAYS
    ):
        return "dry_spell"
    return "temperature_ranges"
