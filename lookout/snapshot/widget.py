"""
widget.py: The widget view, which of the widget's views shows now.

A Lookout widget shows one view at a time: the current value against the
station's own history. Python chooses the view so every widget (Scriptable
now, a WidgetKit app later) shows the same one. The rules are checked in
order, and the first that holds is the view:

1. **rain_event**: an event is live and more than `RAIN_EVENT_IN` has fallen
   in it (`ActiveRainEvent.total_in`, the live `eventrainin`).
2. **dry_spell**: the current dry spell is longer than the usual dry spell,
   in any month. Both are the snapshot's `dry_spell` field: `days_since_rain`
   is the days since `lastRain`, `usual_days` the median completed dry
   spell over the last 365 days. Without a usual dry spell (an invalid window) it cannot
   hold.
3. **temperature_ranges**, otherwise: now against today, 7, 30 and 365 days.

Functions:
- widget_view(active, dry): The `WidgetView` for one snapshot.
"""

from typing import Optional

from lookout.snapshot.model import ActiveRainEvent, DrySpellNow, WidgetView

RAIN_EVENT_IN = 0.04
"""The rain event view needs more than this much rain in the live event."""


def _longer_than_usual(dry: DrySpellNow) -> bool:
    """
    Whether the current dry spell is longer than the usual dry spell.

    :param dry: The dry spell field.
    :return: True only when both values are known and the spell is longer.
    """
    if dry.days_since_rain is None or dry.usual_days is None:
        return False
    return dry.days_since_rain > dry.usual_days


def widget_view(active: Optional[ActiveRainEvent], dry: DrySpellNow) -> WidgetView:
    """
    Choose the widget's view.

    :param active: The active rain event, or None.
    :param dry: The dry spell field (days since rain, the usual dry spell).
    :return: The view name.
    """
    if active is not None and active.total_in > RAIN_EVENT_IN:
        return "rain_event"
    if _longer_than_usual(dry):
        return "dry_spell"
    return "temperature_ranges"
