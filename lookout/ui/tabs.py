"""
tabs.py: The dashboard's tab registry.

Which tabs the Streamlit app shows, in order, and the UI module that renders
each. Kept out of `streamlit_app.py` so it can be imported without running the
app (which fetches live data at import).

Functions:
- tab_names(is_dev): Tab names to show, with Diagnostics and Playground in dev.

Constants:
- TAB_MODULES: Tab name -> UI module exposing `render()`.
"""

from types import ModuleType
from typing import Dict, List

from lookout.ui import diagnostics, overview, playground, rain, rain_events, solar

PROD_TABS = ["Overview", "Rain", "Rain Events", "Solar"]
DEV_ONLY_TABS = ["Diagnostics", "Playground"]

TAB_MODULES: Dict[str, ModuleType] = {
    "Overview": overview,
    "Rain": rain,
    "Rain Events": rain_events,
    "Solar": solar,
    "Diagnostics": diagnostics,
    "Playground": playground,
}


def tab_names(is_dev: bool) -> List[str]:
    """
    Return the tab names to show, in display order.

    :param is_dev: True in dev mode, which adds the diagnostic tabs.
    :return: List of tab names, each a key of TAB_MODULES.
    """
    return PROD_TABS + DEV_ONLY_TABS if is_dev else list(PROD_TABS)
