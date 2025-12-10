"""
Solar Energy Visualizations
Plotly-based charts and heatmaps for solar production data.
"""

import pandas as pd
import plotly.graph_objects as go


# Color constants for all solar visualizations
SOLAR_COLORSCALE = [
    [0.0, "#FFF9E6"],  # Dawn/dusk - pale yellow
    [0.3, "#FFE680"],  # Morning - light yellow
    [0.6, "#FFB732"],  # Midday - golden orange
    [1.0, "#FF8C00"],  # Peak sun - deep orange
]

SOLAR_BAR_COLOR = "#FFB732"  # Golden orange for bar charts


def create_month_day_heatmap(periods_df: pd.DataFrame) -> go.Figure:
    """
    Create month/day heatmap showing daily energy production.

    TODO: Implement in Phase 2 - Epic 2.1
    """
    raise NotImplementedError("Phase 2 - Epic 2.1 in progress")


def create_day_column_chart(periods_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """
    Create hourly column chart for a specific day.

    TODO: Implement in Phase 2 - Epic 2.2
    """
    raise NotImplementedError("Phase 2 - Epic 2.2 in progress")


def create_day_15min_heatmap(periods_df: pd.DataFrame) -> go.Figure:
    """
    Create day/15min heatmap showing granular production patterns.

    TODO: Implement in Phase 2 - Epic 2.3
    """
    raise NotImplementedError("Phase 2 - Epic 2.3 in progress")


def create_15min_bar_chart(periods_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """
    Create 15-minute bar chart for a specific day.

    TODO: Implement in Phase 2 - Epic 2.4
    """
    raise NotImplementedError("Phase 2 - Epic 2.4 in progress")
