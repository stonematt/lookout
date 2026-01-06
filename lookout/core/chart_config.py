"""
chart_config.py

Reusable Plotly configuration helpers to eliminate code duplication in chart creation.

Provides standardized layout, axis, annotation, and styling configurations
for common chart patterns used throughout the Lookout application.
"""

from typing import Optional, Dict, Any
import plotly.graph_objects as go


def get_default_margins(compact: bool = False) -> Dict[str, int]:
    """
    Get standard margin configurations for charts.

    :param compact: If True, returns reduced margins for overview displays
    :return: Dictionary with margin settings
    """
    if compact:
        return dict(l=30, r=20, t=30, b=40)
    else:
        return dict(l=50, r=20, t=40, b=40)


def get_standard_colors() -> Dict[str, str]:
    """
    Get standard color palette used across charts.

    :return: Dictionary with color definitions
    """
    return {
        "muted_line": "#1f77b4",
        "muted_marker": "#aec7e8",
        "muted_current": "#2ca02c",
        "rainfall_line": "#4682B4",
        "rainfall_fill": "rgba(70, 130, 180, 0.3)",
        "rate_low": "#90EE90",
        "rate_medium": "#FFD700",
        "rate_high": "#FF6347",
        "gap_fill": "#B0B0B0",
        "solar_line": "#FFA500",
        "solar_fill": "rgba(255, 165, 0, 0.3)",
        "solar_high": "#FF8C00",
        "solar_medium": "#FFA500",
        "solar_low": "#FFD700",
    }


def get_solar_colors() -> Dict[str, str]:
    """
    Get solar-specific color palette for solar visualizations.

    :return: Dictionary with solar color definitions
    """
    return {
        "solar_bar": "#FFB732",  # Golden orange for bar charts
        "solar_line": "#FFA500",
        "solar_fill": "rgba(255, 165, 0, 0.3)",
        "solar_high": "#FF8C00",
        "solar_medium": "#FFA500",
        "solar_low": "#FFD700",
    }


def get_solar_colorscale() -> list:
    """
    Get solar radiation colorscale for heatmaps.

    :return: Plotly colorscale list for solar radiation visualization
    """
    return [
        [0.0, "#FFF9E6"],  # Dawn/dusk - pale yellow
        [0.3, "#FFE680"],  # Morning - light yellow
        [0.6, "#FFB732"],  # Midday - golden orange
        [1.0, "#FF8C00"],  # Peak sun - deep orange
    ]


def apply_time_series_layout(
    fig: go.Figure,
    height: int = 450,
    showlegend: bool = False,
    title: Optional[str] = None,
    compact: bool = False,
    hovermode: str = "x unified",
) -> go.Figure:
    """
    Apply standard time series chart layout configuration.

    :param fig: Plotly figure to configure
    :param height: Chart height in pixels
    :param showlegend: Whether to show legend
    :param title: Chart title (optional)
    :param compact: Use compact margins if True
    :param hovermode: Hover mode setting
    :return: Configured figure
    """
    layout_config = {
        "height": height,
        "margin": get_default_margins(compact),
        "showlegend": showlegend,
        "hovermode": hovermode,
        "template": "plotly_white",
    }

    if title:
        layout_config["title"] = title

    fig.update_layout(**layout_config)
    return fig


def apply_standard_axes(
    fig: go.Figure,
    xaxis_title: str = "",
    yaxis_title: str = "",
    showgrid_x: bool = False,
    showgrid_y: bool = True,
    rangemode_y: str = "tozero",
    type_x: Optional[str] = None,
    type_y: Optional[str] = None,
    autorange_y: Optional[str] = None,
) -> go.Figure:
    """
    Apply standard axis configuration to charts.

    :param fig: Plotly figure to configure
    :param xaxis_title: X-axis title
    :param yaxis_title: Y-axis title
    :param showgrid_x: Whether to show x-axis grid
    :param showgrid_y: Whether to show y-axis grid
    :param rangemode_y: Y-axis range mode
    :param type_x: X-axis type (e.g., "category")
    :param type_y: Y-axis type (e.g., "category")
    :param autorange_y: Y-axis autorange setting
    :return: Configured figure
    """
    xaxis_config = {
        "title": xaxis_title,
        "showgrid": showgrid_x,
        "gridcolor": "lightgray",
    }

    yaxis_config = {
        "title": yaxis_title,
        "showgrid": showgrid_y,
        "gridcolor": "lightgray",
        "rangemode": rangemode_y,
    }

    if type_x:
        xaxis_config["type"] = type_x
    if type_y:
        yaxis_config["type"] = type_y
    if autorange_y:
        yaxis_config["autorange"] = autorange_y

    fig.update_xaxes(**xaxis_config)
    fig.update_yaxes(**yaxis_config)
    return fig


def create_standard_annotation(
    text: str,
    position: str = "top_right",
    xref: str = "paper",
    yref: str = "paper",
    showarrow: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standard annotation with common positioning.

    :param text: Annotation text
    :param position: Position preset ("top_right", "top_left", "bottom_right")
    :param xref: X reference ("paper" or "data")
    :param yref: Y reference ("paper" or "data")
    :param showarrow: Whether to show arrow
    :param kwargs: Additional annotation parameters
    :return: Annotation configuration dictionary
    """
    positions = {
        "top_right": dict(x=0.98, y=0.95, xanchor="right", yanchor="top"),
        "top_left": dict(x=0.02, y=0.95, xanchor="left", yanchor="top"),
        "bottom_right": dict(x=0.98, y=0.05, xanchor="right", yanchor="bottom"),
    }

    pos_config = positions.get(position, positions["top_right"])

    annotation = {
        "text": text,
        "xref": xref,
        "yref": yref,
        "showarrow": showarrow,
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": "gray",
        "borderwidth": 1,
        "borderpad": 4,
        **pos_config,
        **kwargs,
    }

    return annotation


def apply_heatmap_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 600,
    compact: bool = False,
    showlegend: Optional[bool] = None,
    colorbar_title: str = "Rain (in)",
    **kwargs
) -> go.Figure:
    """
    Apply standard heatmap layout configuration.

    :param fig: Plotly figure to configure
    :param title: Chart title
    :param xaxis_title: X-axis title
    :param yaxis_title: Y-axis title
    :param height: Chart height in pixels
    :param compact: Use compact styling if True
    :param showlegend: Legend visibility override
    :param colorbar_title: Colorbar title
    :param kwargs: Additional layout parameters
    :return: Configured figure
    """
    margin = get_default_margins(compact)

    layout_config = {
        "title": title,
        "height": height,
        "margin": margin,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        **kwargs,
    }

    if showlegend is not None:
        layout_config["showlegend"] = showlegend

    fig.update_layout(**layout_config)

    # Configure axes
    apply_standard_axes(
        fig=fig,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showgrid_x=True,
        showgrid_y=True,
        type_x="category",
        type_y="category",
        autorange_y="reversed",
    )

    # Configure colorbar
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=colorbar_title,
            tickmode="linear",
            tick0=0,
            dtick=0.1 if not compact else None,
        )
    )

    return fig


def get_rainfall_colorscale() -> list:
    """
    Get standard rainfall colorscale for heatmaps.

    :return: Plotly colorscale list
    """
    return [
        [0.0, "white"],  # 0 maps to white
        [0.001, "#f7fbff"],  # Very light blue
        [0.25, "#deebf7"],  # Light blue
        [0.5, "#9ecae1"],  # Medium blue
        [0.75, "#4292c6"],  # Darker blue
        [1.0, "#08519c"],  # Darkest blue at max
    ]


def apply_violin_layout(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 500,
    yaxis_title: str = "",
    showlegend: bool = True,
) -> go.Figure:
    """
    Apply standard violin/box plot layout configuration.

    :param fig: Plotly figure to configure
    :param title: Chart title (optional)
    :param height: Chart height in pixels
    :param yaxis_title: Y-axis title
    :param showlegend: Whether to show legend
    :return: Configured figure
    """
    layout_config = {
        "height": height,
        "yaxis_title": yaxis_title,
        "xaxis_title": "",
        "showlegend": showlegend,
        "template": "plotly_white",
        "margin": get_default_margins(),
        "hovermode": "x unified",
    }

    if title:
        layout_config["title"] = title

    fig.update_layout(**layout_config)

    # Standard violin axes
    fig.update_yaxes(rangemode="tozero", showgrid=True, gridcolor="lightgray")
    fig.update_xaxes(showgrid=False)

    return fig
