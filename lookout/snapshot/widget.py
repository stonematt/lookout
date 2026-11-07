"""
widget.py: The widget view, which of the widget's views shows now.

A Lookout widget shows one view at a time: the current value against the
station's own history. Python chooses the view so every widget (Scriptable
now, a WidgetKit app later) shows the same one. The rules are checked in
order, and the first that holds is the view:

1. **rain_event**: an event is live and more than `RAIN_EVENT_IN` has fallen
   in it (`ActiveRainEvent.total_in`, the live `eventrainin`).
2. **dry_spell**: the current dry spell is longer than the usual dry spell,
   in any month. Both are the hero lead's calm rule: its value is the days
   since `lastRain`, its reference the median completed dry spell over the
   last 365 days. Without a usual dry spell (an invalid window) it cannot
   hold.
3. **temperature_ranges**, otherwise: now against today, 7, 30 and 365 days.

Functions:
- widget_view(active, lead): The `WidgetView` for one snapshot.
"""

from typing import Optional

from lookout.snapshot.model import ActiveRainEvent, HeroLead, LeadRule, WidgetView

RAIN_EVENT_IN = 0.04
"""The rain event view needs more than this much rain in the live event."""


def _longer_than_usual(calm: Optional[LeadRule]) -> bool:
    """
    Whether the current dry spell is longer than the usual dry spell.

    :param calm: The calm lead rule, or None.
    :return: True only when both values are known and the spell is longer.
    """
    if calm is None or calm.value is None or calm.reference is None:
        return False
    return calm.value > calm.reference


def widget_view(active: Optional[ActiveRainEvent], lead: HeroLead) -> WidgetView:
    """
    Choose the widget's view.

    :param active: The active rain event, or None.
    :param lead: The hero lead, whose calm rule holds the dry spell.
    :return: The view name.
    """
    if active is not None and active.total_in > RAIN_EVENT_IN:
        return "rain_event"
    calm = next((rule for rule in lead.rules if rule.kind == "calm"), None)
    if _longer_than_usual(calm):
        return "dry_spell"
    return "temperature_ranges"
