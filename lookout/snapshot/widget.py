"""
widget.py: The widget view, which of the widget's views shows now.

A Lookout widget shows one view at a time: the current value against the
station's own history. Python chooses the view so every widget (Scriptable
now, a WidgetKit app later) shows the same one. It has no rules of its own:
it is the feature (`lookout.snapshot.feature`, ADR-0009) mapped to the three
views a widget can draw:

- feature **rain** -> **rain_event**;
- feature **dry_spell** -> **dry_spell**;
- anything else (**unusual**, **quiet**) -> **temperature_ranges**, now
  against today, 7, 30 and 365 days. A widget cannot draw wind or pressure,
  so an unusual reading of any kind maps here.

Functions:
- widget_view(feature): The `WidgetView` for one snapshot.
"""

from typing import Dict

from lookout.snapshot.model import Feature, FeatureKind, WidgetView

VIEWS: Dict[FeatureKind, WidgetView] = {
    "rain": "rain_event",
    "dry_spell": "dry_spell",
}
"""The feature kinds a widget has a view for; every other kind is
`temperature_ranges`."""


def widget_view(feature: Feature) -> WidgetView:
    """
    The widget's view: the feature, mapped.

    :param feature: The snapshot's feature.
    :return: The view name.
    """
    return VIEWS.get(feature.kind, "temperature_ranges")
