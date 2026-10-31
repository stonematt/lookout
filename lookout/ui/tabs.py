"""
tabs.py: The dashboard's tab registry.

Which tabs the Streamlit app shows, in order, and the UI module that renders
each. Kept out of `streamlit_app.py` so it can be imported without running the
app (which fetches live data at import).

Functions:
- tab_names(is_dev): Tab names to show, with Diagnostics and Playground in dev.

Constants:
- TAB_MODULES: Tab name -> UI module exposing `render()`, in display order.
- DEV_ONLY_TABS: Tabs shown only in dev mode.
"""

from types import ModuleType
from typing import Dict, List

from lookout.ui import diagnostics, overview, playground, rain, rain_events, solar

# Tab name -> UI module exposing render(), in display order.
TAB_MODULES: Dict[str, ModuleType] = {
    "Overview": overview,
    "Rain": rain,
    "Rain Events": rain_events,
    "Solar": solar,
    "Diagnostics": diagnostics,
    "Playground": playground,
}

DEV_ONLY_TABS = frozenset({"Diagnostics", "Playground"})


def tab_names(is_dev: bool) -> List[str]:
    """
    Return the tab names to show, in display order.

    :param is_dev: True in dev mode, which adds the diagnostic tabs.
    :return: List of tab names, each a key of TAB_MODULES.
    """
    return [name for name in TAB_MODULES if is_dev or name not in DEV_ONLY_TABS]
