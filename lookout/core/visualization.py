"""
visualizations.py
Collection of functions to create charts and visualizations for streamlit
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from colour import Color

from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import BYTES_TO_MB, force_garbage_collection

logger = app_logger(__name__)

gauge_defaults = {
    "default": {
        "start_color": "#33CCFF",
        "end_color": "#FF3300",
        "steps": 100,
        "min_val": 0,
        "max_val": 120,
        "title": "Temperature (°F)",
    },
    "temps": {
        "start_color": "#33CCFF",
        "end_color": "#FF3300",
        "steps": 100,
        "min_val": 0,
        "max_val": 120,
        "title": "Temperature (°F)",
    },
    "wind": {
        "start_color": "#CCEFFF",
        "end_color": "#6666FF",
        "steps": 5,
        "min_val": 0,
        "max_val": 100,
        "title": "Wind Speed (mph)",
    },
    "pressure": {
        "start_color": "#FFCCCC",
        "end_color": "#FF6666",
        "steps": 8,
        "min_val": 950,
        "max_val": 1050,
        "title": "Barometric Pressure (hPa)",
    },
    "rain": {
        "start_color": "#ADD8E6",  # Light blue, representing light rain
        "end_color": "#00008B",  # Dark blue, representing heavy rainfall
        "steps": 100,
        "min_val": 0,
        "max_val": 30,
        "title": "Rainfall (inches)",
    },
    "rain_rate": {
        "start_color": "#ADD8E6",  # Light blue, representing light rain
        "end_color": "#00008B",  # Dark blue, representing heavy rainfall
        "steps": 100,
        "min_val": 0,
        "max_val": 10,
        "title": "Rainfall (inches)",
    },
}


def generate_gradient_steps(start_color, end_color, steps):
    """
    Generate gradient steps for a Plotly gauge chart.

    :param start_color: Hex string for the start color (e.g., "#FF0000" for red).
    :param end_color: Hex string for the end color (e.g., "#00FF00" for green).
    :param steps: Number of gradient steps to generate.
    :return: A list of colors in hex format representing the gradient steps.
    """
    # Create Color objects for start and end
    start = Color(start_color)
    end = Color(end_color)

    # Generate colors in the gradient
    gradient_colors = list(start.range_to(end, steps))

    # Convert colors to hex format
    hex_colors = [color.get_hex() for color in gradient_colors]

    return hex_colors


def create_windrose_chart(
    grouped_data,
    value_labels,
    title="Windrose",
    sector_size=30,
    color_palette="default",
):
    """
    Generates a windrose chart using Plotly, with customizable sectors and color palettes.
    Includes enhanced hover text with category, direction, and percentage details.

    :param grouped_data: pd.DataFrame - DataFrame with percentage data.
    :param value_labels: list - Bin labels for the values.
    :param title: str - Title for the chart. Defaults to "Windrose".
    :param sector_size: int - Size of directional sectors (degrees). Defaults to 30°.
    :param color_palette: str - Key for the color palette in gauge_defaults.
    :return: go.Figure - Plotly figure object.
    """
    # Generate gradient colors
    chart_config = gauge_defaults.get(color_palette, gauge_defaults["default"])
    colors = generate_gradient_steps(
        start_color=chart_config["start_color"],
        end_color=chart_config["end_color"],
        steps=len(value_labels),
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add data to the windrose chart
    for i, value_label in enumerate(value_labels):
        bin_data = grouped_data[grouped_data["value_bin"] == value_label]

        # Create the barpolar trace with enhanced hover text
        fig.add_trace(
            go.Barpolar(
                r=bin_data["percentage"],
                theta=[int(label.split("-")[0]) for label in bin_data["direction_bin"]],
                width=np.full(len(bin_data), sector_size),  # Width of the bars
                name=value_label,  # Legend entry
                marker_color=colors[i],
                marker_line=dict(color="black", width=1.5),
                hoverinfo="text",  # Custom hover info
                text=[  # Detailed hover text
                    f"<b>Category:</b> {value_label}<br>"
                    f"<b>Direction:</b> {label}<br>"
                    f"<b>Percentage:</b> {percentage:.2f}%"
                    for label, percentage in zip(
                        bin_data["direction_bin"], bin_data["percentage"]
                    )
                ],
            )
        )

    # Customize chart layout
    fig.update_layout(
        title=dict(text=title),
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=np.arange(0, 360, sector_size),
                ticktext=[f"{int(val)}°" for val in np.arange(0, 360, sector_size)],
                rotation=90,  # North is up
                direction="clockwise",  # Angles increase clockwise
            ),
            radialaxis=dict(
                visible=True,
                ticksuffix="%",  # Append "%" to radial labels
            ),
        ),
        legend=dict(
            title="Categories",
            orientation="h",
            yanchor="top",
            y=-0.15,  # Move legend below chart
            xanchor="center",
            x=0.5,  # Center the legend horizontally
        ),
    )

    return fig


def create_gauge_chart(
    value, metric_type, title="Gauge", gauge_defaults=gauge_defaults, chart_height=450
):
    """
    Create a Plotly gauge chart with gradient steps and a specified chart height.

    :param value: The value to display on the gauge.
    :param metric_type: The type of metric, which determines the gauge defaults.
    :param title: Title for the gauge chart.
    :param gauge_defaults: Default settings and preferences by metric type.
    :param chart_height: The height of the chart in pixels.
    :return: Plotly figure object for the gauge chart.
    """
    defaults = gauge_defaults[metric_type]
    gradient_colors = generate_gradient_steps(
        defaults["start_color"], defaults["end_color"], defaults["steps"]
    )
    step_values = np.linspace(
        defaults["min_val"], defaults["max_val"], len(gradient_colors) + 1
    )
    gauge_steps = [
        {"range": [step_values[i], step_values[i + 1]], "color": gradient_colors[i]}
        for i in range(len(gradient_colors))
    ]

    # Create the gauge figure
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            gauge={
                "axis": {"range": [defaults["min_val"], defaults["max_val"]]},
                "bar": {"color": "black"},
                "steps": gauge_steps,
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )

    # Update the layout to set the chart height
    fig.update_layout(height=chart_height)

    return fig


def display_current_data(data):
    """
    Display current device data in the Streamlit app.

    :param data: Dictionary with device's last known data.
    """
    st.subheader("Current Data")
    for key, value in data.items():
        st.text(f"{key}: {value}")


def display_heatmap(df, metric, interval="15T"):
    """
    Generate and display a heatmap for a given metric over specified intervals.

    :param df: DataFrame with the data.
    :param metric: Metric to visualize.
    :param interval: Interval for aggregation, default 15 minutes.
    """
    df["date"] = pd.to_datetime(df["date"])
    df["interval"] = df["date"].dt.floor(interval).dt.strftime("%H:%M")
    fig = px.density_heatmap(df, x="date", y="interval", z=metric, histfunc="avg")
    st.plotly_chart(fig, width="stretch")


def display_line_chart(df, metrics):
    """
    Display a line chart for selected metrics from the data.

    :param df: DataFrame containing the data.
    :param metrics: List of metric names to include in the chart.
    """
    fig = px.line(df, x="date", y=metrics, title="Historical Data")
    st.plotly_chart(fig, width="stretch")


def draw_horizontal_bars(data_dict, label="Value", xaxis_range=None):
    """
    Draws a horizontal bar plot with markers for a range (min-max) and a current value.

    :param data_dict: dict - Data in the format:
                          {
                              "category": {"min": value, "max": value, "current": value},
                              ...
                          }
    :param label: str - Label for the x-axis and chart title.
    :param xaxis_range: tuple or None - Custom x-axis range (min, max). If None, defaults
                       to ±10% of the lowest min and highest max in data.
    """
    # Muted color palette
    muted_palette = {
        "line": "#1f77b4",  # Muted blue for the range line
        "marker": "#aec7e8",  # Light blue for end markers
        "current": "#2ca02c",  # Muted green for the current value
    }

    # Convert dictionary to a list of dictionaries for easier processing
    categories = list(data_dict.keys())
    plots = []

    # Prepare data for plotting
    for category in categories:
        values = data_dict[category]
        if values["min"] is not None and values["max"] is not None:
            plots.append(
                {
                    "Category": category,
                    "Min": values["min"],
                    "Max": values["max"],
                    "Current": values["current"],
                }
            )

    # Determine x-axis range if not provided
    if xaxis_range is None:
        min_values = [p["Min"] for p in plots]
        max_values = [p["Max"] for p in plots]
        min_range = min(min_values) - 0.1 * abs(min(min_values))
        max_range = max(max_values) + 0.1 * abs(max(max_values))
        xaxis_range = (min_range, max_range)

    # Create Plotly figure
    fig = go.Figure()

    # Add horizontal bars and current markers for each category
    for plot in plots:
        # Add the horizontal bar for the range
        fig.add_trace(
            go.Scatter(
                x=[plot["Min"], plot["Max"]],
                y=[plot["Category"], plot["Category"]],
                mode="lines+markers",
                line=dict(color=muted_palette["line"], width=8),
                marker=dict(color=muted_palette["marker"], size=12),
                name=plot["Category"],
                text=[  # Tooltip for the line
                    f"min: {plot['Min']}<br>max: {plot['Max']}<br>current: {plot['Current']}",
                    f"min: {plot['Min']}<br>max: {plot['Max']}<br>current: {plot['Current']}",
                ],
                hoverinfo="text",
            )
        )
        # Add a marker for the current value
        if plot["Current"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=[plot["Current"]],
                    y=[plot["Category"]],
                    mode="markers",
                    marker=dict(
                        symbol="diamond-tall",
                        line=dict(width=2, color="DarkSlateGrey"),
                        color=muted_palette["current"],
                        size=14,
                    ),
                    name=f"{plot['Category']} Current",
                    text=[  # Tooltip for the marker
                        f"current: {plot['Current']}",
                    ],
                    hoverinfo="text",
                )
            )

    # Customize layout
    fig.update_layout(
        title=f"{label} Overview",
        xaxis=dict(
            range=xaxis_range,
            showgrid=True,
        ),
        yaxis=dict(title="", showgrid=False),
        showlegend=False,
        font=dict(family="Arial", size=18),  # Use a clean font
        # plot_bgcolor="#f9f9f9",  # Light background for consistency with Streamlit
    )

    # Render the plot in Streamlit
    st.plotly_chart(fig)


def better_heatmap_table(df, metric, aggfunc="max", interval=1800):
    """
    Create a pivot table of aggregate values for a given metric, with the row index
    as a time stamp for every `interval`-second interval and the column index as
    the unique dates in the "date" column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with a "date" column and a column with the desired `metric`.
    metric : str
        The name of the column in `df` containing the desired metric.
    aggfunc : str or function
        The aggregation function to use when computing the pivot table. Can be a string
        of a built-in function (e.g., "mean", "sum", "count"), or a custom function.
    interval : int
        The number of seconds for each interval. For example, `interval=15` would
        create an interval of 15 seconds.

    Returns
    -------
    pandas.DataFrame
        A pivot table where the row index is a time stamp for every `interval`-second
        interval, and the column index is the unique dates in the "date" column.
        The values are the aggregate value of the `metric` column for each interval
        and each date.
    """

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date
    df["interval"] = df["date"].dt.floor(f"{interval}s").dt.strftime("%H:%M:%S")
    table = df.pivot_table(
        index=["interval"],
        columns=["day"],
        values=metric,
        aggfunc=aggfunc,
    )

    return table


def heatmap_chart(heatmap_table):
    fig = px.imshow(heatmap_table, x=heatmap_table.columns, y=heatmap_table.index)
    st.plotly_chart(fig)


def make_column_gauges(gauge_list, chart_height=300):
    """
    Take a list of metrics and produce a row of gauges, with min, median, and max values displayed below each gauge.

    :param gauge_list: list of dicts with metrics, titles to render as gauges, and their types.
    :param chart_height: height of the charts in the row
    """
    # Create columns for gauges
    cols = st.columns(len(gauge_list))
    last_data = st.session_state.get("last_data", {})
    history_df = st.session_state.get("history_df")

    for i, gauge in enumerate(gauge_list):
        metric = gauge["metric"]
        title = gauge["title"]
        metric_type = gauge["metric_type"]

        # Retrieve the last value for the metric
        value = last_data.get(metric, 0)

        # Calculate min, median, max for the current metric from history_df
        min_val = median_val = max_val = 0  # Default fallback

        if isinstance(history_df, pd.DataFrame) and metric in history_df.columns:
            min_val = history_df[metric].min()
            median_val = history_df[metric].median()
            max_val = history_df[metric].max()

        # Create the gauge chart for the current metric
        gauge_fig = create_gauge_chart(
            value=value, metric_type=metric_type, title=title, chart_height=chart_height
        )

        # Plot the gauge in the respective column, fitting it to the column width
        with cols[i]:
            st.plotly_chart(gauge_fig, width="stretch")

            # Use markdown to display min, median, and max values below the gauge with less vertical space
            stats_md = f"""<small>
            <b>Min:</b> {min_val:.2f} <br>
            <b>Median:</b> {median_val:.2f} <br>
            <b>Max:</b> {max_val:.2f}
            </small>"""
            st.markdown(stats_md, unsafe_allow_html=True)


def display_data_coverage_heatmap(df, metric="tempf", interval_minutes=60):
    """
    Show data density heatmap based on datapoint counts per interval.

    :param df: Filtered DataFrame with a datetime column "date".
    :param metric: A valid metric column (used for presence, not value).
    :param interval_minutes: Interval granularity in minutes.
    """
    interval_seconds = interval_minutes * 60
    table = better_heatmap_table(
        df, metric=metric, aggfunc="count", interval=interval_seconds
    )
    heatmap_chart(table)


def display_hourly_coverage_heatmap(df):
    """
    Render a grid-style heatmap showing the number of samples per hour per day.
    Missing days/hours are zero-filled to preserve visual continuity.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.date

    # Count samples per (day, hour)
    grouped = df.groupby(["day", "hour"]).size().reset_index(name="samples")

    # Create full (day, hour) index to fill gaps
    full_days = pd.date_range(df["day"].min(), df["day"].max(), freq="D").date
    full_hours = list(range(24))
    full_index = pd.MultiIndex.from_product(
        [full_days, full_hours], names=["day", "hour"]
    )

    full_df = (
        grouped.set_index(["day", "hour"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    pivot = full_df.pivot(index="day", columns="hour", values="samples")

    hover_text = [
        [
            f"Hour: {hour}<br>Date: {date}<br>Samples: {pivot.at[date, hour]}"
            for hour in pivot.columns
        ]
        for date in pivot.index
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(h) for h in pivot.columns],
            y=pivot.index.astype(str),
            text=hover_text,
            hoverinfo="text",
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
            zsmooth=False,  # No smoothing — preserves grid
        )
    )

    fig.update_traces(xgap=2, ygap=2)  # Visual cell spacing

    fig.update_layout(
        title="Hourly Coverage",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            dtick=1,
            type="category",
            showgrid=True,
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="Date",
            type="category",
            autorange="reversed",
            showgrid=True,
            gridcolor="lightgrey",
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, width="stretch")


def create_rainfall_violin_plot(
    window: str,
    violin_data: dict,
    unit: str = "in",
    title: Optional[str] = None,
) -> None:
    """
    Create single violin plot showing rainfall distribution for specified window.

    :param window: Window size (e.g., "7d", "30d")
    :param violin_data: Data from prepare_violin_plot_data()
    :param unit: Unit for display (e.g., "in")
    :param title: Chart title (auto-generated if None)
    """
    if window not in violin_data or len(violin_data[window]["values"]) == 0:
        st.warning(f"No historical data available for {window} period.")
        return

    data = violin_data[window]
    values = data["values"]
    current = data["current"]
    percentile = data["percentile"]

    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            y=values,
            name=f"Historical {window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(56, 128, 191, 0.6)",
            line_color="rgba(56, 128, 191, 1.0)",
            x0=f"{window} Periods",
        )
    )

    if not np.isnan(current):
        fig.add_trace(
            go.Scatter(
                x=[f"{window} Periods"],
                y=[current],
                mode="markers",
                marker=dict(
                    symbol="diamond-tall",
                    size=16,
                    color="red",
                    line=dict(width=2, color="darkred"),
                ),
                name=f"Current ({current:.2f}{unit})",
                text=[
                    (
                        f"Current: {current:.2f}{unit}<br>Percentile: {percentile:.1f}th"
                        if not np.isnan(percentile)
                        else f"Current: {current:.2f}{unit}"
                    )
                ],
                hoverinfo="text",
            )
        )

    chart_title = title or f"Rainfall Distribution: {window} Rolling Periods"
    fig.update_layout(
        title=chart_title,
        yaxis_title=f"Rainfall ({unit})",
        xaxis_title="",
        showlegend=True,
        height=500,
        template="plotly_white",
    )

    st.plotly_chart(fig, width="stretch")

    if not np.isnan(percentile):
        if percentile >= 90:
            status = "🔴 **Extremely wet**"
        elif percentile >= 75:
            status = "🟡 **Above normal**"
        elif percentile >= 25:
            status = "🟢 **Normal range**"
        else:
            status = "🔵 **Below normal**"

        st.markdown(
            f"{status} - Current {window} total ({current:.2f}{unit}) ranks at **{percentile:.1f}th percentile** of {len(values):,} historical periods."
        )


def create_dual_violin_plot(
    left_window: str,
    right_window: str,
    violin_data: dict,
    unit: str = "in",
    title: Optional[str] = None,
) -> None:
    """
    Create dual violin plot comparing two different rainfall windows side-by-side.

    :param left_window: Left window size (e.g., "1d")
    :param right_window: Right window size (e.g., "7d")
    :param violin_data: Data from prepare_violin_plot_data()
    :param unit: Unit for display (e.g., "in")
    :param title: Chart title (auto-generated if None)
    """
    from plotly.subplots import make_subplots

    missing_data = []
    for window in [left_window, right_window]:
        if window not in violin_data or len(violin_data[window]["values"]) == 0:  # type: ignore
            missing_data.append(window)

    if missing_data:
        st.warning(f"No historical data available for: {', '.join(missing_data)}")
        return

    left_data = violin_data[left_window]
    right_data = violin_data[right_window]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{left_window} Periods", f"{right_window} Periods"],
        shared_yaxes=True,
    )

    colors = {
        left_window: "rgba(56, 128, 191, 0.6)",
        right_window: "rgba(191, 128, 56, 0.6)",
    }
    line_colors = {
        left_window: "rgba(56, 128, 191, 1.0)",
        right_window: "rgba(191, 128, 56, 1.0)",
    }

    fig.add_trace(
        go.Violin(
            y=left_data["values"],
            name=f"Historical {left_window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[left_window],
            line_color=line_colors[left_window],
            x0=f"{left_window}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Violin(
            y=right_data["values"],
            name=f"Historical {right_window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[right_window],
            line_color=line_colors[right_window],
            x0=f"{right_window}",
        ),
        row=1,
        col=2,
    )

    for i, (window, data) in enumerate(
        [(left_window, left_data), (right_window, right_data)], 1
    ):
        current = data["current"]
        percentile = data["percentile"]

        if not np.isnan(current):
            fig.add_trace(
                go.Scatter(
                    x=[window],
                    y=[current],
                    mode="markers",
                    marker=dict(
                        symbol="diamond-tall",
                        size=16,
                        color="red",
                        line=dict(width=2, color="darkred"),
                    ),
                    name=f"Current {window} ({current:.2f}{unit})",
                    text=[
                        (
                            f"Current: {current:.2f}{unit}<br>Percentile: {percentile:.1f}th"
                            if not np.isnan(percentile)
                            else f"Current: {current:.2f}{unit}"
                        )
                    ],
                    hoverinfo="text",
                    showlegend=True,
                ),
                row=1,
                col=i,
            )

    chart_title = (
        title or f"Rainfall Distribution Comparison: {left_window} vs {right_window}"
    )
    fig.update_layout(
        title=chart_title,
        height=600,
        template="plotly_white",
    )

    fig.update_yaxes(title_text=f"Rainfall ({unit})", row=1, col=1)

    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        left_current = left_data["current"]
        left_percentile = left_data["percentile"]
        if not np.isnan(left_percentile):
            st.markdown(
                f"**{left_window} Period**: {left_current:.2f}{unit} ({left_percentile:.1f}th percentile)"
            )

    with col2:
        right_current = right_data["current"]
        right_percentile = right_data["percentile"]
        if not np.isnan(right_percentile):
            st.markdown(
                f"**{right_window} Period**: {right_current:.2f}{unit} ({right_percentile:.1f}th percentile)"
            )


def create_event_accumulation_chart(
    event_data: pd.DataFrame, event_info: dict
) -> go.Figure:
    """
    Create area chart showing cumulative rainfall for a rain event.

    :param event_data: DataFrame with dateutc and eventrainin columns (sorted by time)
    :param event_info: Dict with total_rainfall, duration_minutes, start_time, end_time
    :return: Plotly figure
    """
    df = event_data.copy()
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["time_pst"] = df["timestamp"].dt.tz_convert("America/Los_Angeles")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time_pst"],
            y=df["eventrainin"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#4682B4", width=2),
            fillcolor="rgba(70, 130, 180, 0.3)",
            hovertemplate='%{x|%b %d %I:%M %p}<br>%{y:.3f}"<extra></extra>',
            name="Rainfall",
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="",
        yaxis_title="Rainfall (in)",
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")

    fig.add_annotation(
        text=f"Total: {event_info['total_rainfall']:.3f}\"",
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.95,
        xanchor="right",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
    )

    return fig


def create_event_rate_chart(event_data: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing rainfall intensity with time-aware rate calculation.

    Rates calculated from actual time intervals. Data gaps (>10 min) are filled with
    synthetic 5-min interval bars showing average rate over the gap period, colored
    gray to distinguish from instantaneous measurements.

    :param event_data: DataFrame with dateutc and dailyrainin columns (sorted by time)
    :return: Plotly figure
    """
    df = event_data.copy()
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["time_pst"] = df["timestamp"].dt.tz_convert("America/Los_Angeles")

    df["time_diff_min"] = df["timestamp"].diff().dt.total_seconds() / 60
    df.loc[df.index[0], "time_diff_min"] = 5

    df["interval_rain"] = df["dailyrainin"].diff().clip(lower=0)
    df.loc[df.index[0], "interval_rain"] = 0

    df["rate"] = df["interval_rain"] / (df["time_diff_min"] / 60)

    times = []
    rates = []
    colors = []
    customdata = []

    for idx in df.index:
        row = df.loc[idx]
        time_gap = row["time_diff_min"]
        interval_rain = row["interval_rain"]
        rate = row["rate"]

        if time_gap > 10:
            num_intervals = max(1, int(time_gap / 5))
            avg_rate = interval_rain / (time_gap / 60)

            prev_idx = df.index[df.index.get_loc(idx) - 1]
            prev_time = df.loc[prev_idx, "time_pst"]
            curr_time = row["time_pst"]

            for i in range(num_intervals):
                synthetic_time = prev_time + pd.Timedelta(minutes=5 * (i + 1))
                if synthetic_time <= curr_time:
                    times.append(synthetic_time)
                    rates.append(avg_rate)
                    colors.append("#B0B0B0")
                    customdata.append(f"({int(time_gap)}min avg)")
        else:
            times.append(row["time_pst"])
            rates.append(rate)

            if rate < 0.1:
                colors.append("#90EE90")
            elif rate < 0.3:
                colors.append("#FFD700")
            else:
                colors.append("#FF6347")
            customdata.append("")

    hover_template = (
        "%{x|%b %d %I:%M %p}<br>" "%{y:.3f} in/hr<br>" "%{customdata}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=times,
            y=rates,
            marker_color=colors,
            hovertemplate=hover_template,
            customdata=customdata,
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis_title="",
        yaxis_title="Rate (in/hr)",
        showlegend=False,
        bargap=0,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")

    return fig


def create_rainfall_summary_violin(
    daily_rain_df: pd.DataFrame,
    current_values: dict,
    rolling_context_df: pd.DataFrame,
    end_date: pd.Timestamp,
    windows: list = None,
    title: str = None,
) -> go.Figure:
    """
    Create box plot showing current rainfall vs historical distributions.

    Shows specified windows with box plots (via violin with hidden shape) and
    current values as colored diamond markers.

    :param daily_rain_df: DataFrame with daily rainfall totals
    :param current_values: Dict with today, yesterday, 7d, 30d, 90d, 365d values
    :param rolling_context_df: DataFrame from compute_rolling_rain_context
    :param end_date: Current date for analysis
    :param windows: List of window keys to display (default: all 6)
    :return: Plotly figure
    """
    import numpy as np

    if windows is None:
        windows = ["Today", "Yesterday", "7d", "30d", "90d", "365d"]

    fig = go.Figure()

    # Convert dates without copying the entire DataFrame
    dates = pd.to_datetime(daily_rain_df["date"])
    all_single_days = daily_rain_df[dates.dt.year != end_date.year]["rainfall"].values

    annotations_data = []

    for window in windows:
        if window in ["Today", "Yesterday"]:
            category = window
            current_val = current_values.get(window.lower(), 0)
            distribution = all_single_days
        else:
            category = window
            window_days = int(window.rstrip("d"))

            window_row = rolling_context_df[
                rolling_context_df["window_days"] == window_days
            ]

            if len(window_row) == 0:
                continue

            row = window_row.iloc[0]
            current_val = current_values.get(window, row.get("total", 0))

            # Convert date column to datetime and set as index for proper year comparison
            df_temp = daily_rain_df.copy()
            df_temp["date"] = pd.to_datetime(df_temp["date"])
            s = df_temp.set_index("date")["rainfall"].sort_index()
            historical_data = s[s.index.year != end_date.year]

            if len(historical_data) < window_days:
                continue

            all_periods = (
                historical_data.rolling(window=window_days).sum().dropna().values
            )
            distribution = all_periods[np.isfinite(all_periods)]

        if len(distribution) > 0:
            q25, q75 = np.percentile(distribution, [25, 75])

            fig.add_trace(
                go.Box(
                    y=distribution,
                    name=category,
                    boxpoints="outliers",
                    marker=dict(color="lightblue", size=3),
                    line=dict(color="steelblue"),
                    fillcolor="lightblue",
                    showlegend=False,
                )
            )

            percentile = (
                (distribution < current_val).sum() / len(distribution) * 100
                if len(distribution) > 0
                else 50
            )

            marker_color = (
                "red"
                if current_val > q75
                else "green" if current_val < q25 else "orange"
            )

            fig.add_trace(
                go.Scatter(
                    x=[category],
                    y=[current_val],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color=marker_color,
                        line=dict(width=2, color="black"),
                    ),
                    showlegend=False,
                    hovertemplate=f'{current_val:.2f}" ({percentile:.0f}th percentile)<extra></extra>',
                )
            )

            annotations_data.append((category, current_val, percentile))

    fig.update_layout(
        height=400,
        yaxis_title="Rainfall (inches)",
        yaxis=dict(rangemode="tozero", showgrid=True, gridcolor="lightgray"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=50, r=20, t=30, b=100),
        hovermode="x unified",
        title=title,
    )

    for cat, val, pct in annotations_data:
        fig.add_annotation(
            x=cat,
            y=0,
            yshift=-40,
            text=f'{val:.2f}"<br>{pct:.0f}th',
            showarrow=False,
            yref="paper",
            yanchor="top",
            font=dict(size=10),
        )

    # Memory cleanup for large DataFrames created during visualization
    try:
        del pivot, full_data, indexed
        force_garbage_collection()
    except:
        pass

    return fig


def prepare_rain_accumulation_heatmap_data(
    archive_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    timezone: str = "America/Los_Angeles",
    num_days: Optional[int] = None,
    include_gaps: bool = False,
    row_mode: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare rainfall accumulation data for heatmap with simplified aggregation.

    All rainfall increments are included to ensure accurate totals.
    Each dailyrainin.diff() represents real rain that fell, captured
    at the time of the reading regardless of data gaps.

    Row modes: 'day', 'week', 'month', 'year_month', 'auto'
    - day: Daily rows with Hour of Day columns
    - week: Weekly rows with Day of Week columns
    - month: Monthly rows with Day of Month columns
    - year_month: YY-MM rows with Day of Month columns
    - auto: Automatically select based on period length

    :param archive_df: Archive with dateutc and dailyrainin columns
    :param start_date: Filter start date (timezone-aware or naive UTC)
    :param end_date: Filter end date (timezone-aware or naive UTC)
    :param timezone: Timezone for hour bucketing
    :param num_days: Number of days in period (helps determine auto mode)
    :param include_gaps: Deprecated parameter (has no effect, all data included)
    :param row_mode: Row aggregation mode ('day', 'week', 'month', 'year_month', 'auto')
    :return: DataFrame with (date, hour, accumulation) columns
    """
    if archive_df.empty or "dailyrainin" not in archive_df.columns:
        logger.warning("No dailyrainin data available for accumulation heatmap")
        return pd.DataFrame(columns=["date", "hour", "accumulation"])

    df = archive_df.copy()

    # Convert to datetime and timezone
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(timezone)

    # Filter by date range if specified
    if start_date:
        if start_date.tz is None:
            start_date = start_date.tz_localize("UTC")
        df = df[df["timestamp"] >= start_date]

    if end_date:
        if end_date.tz is None:
            end_date = end_date.tz_localize("UTC")
        df = df[df["timestamp"] <= end_date]

    if df.empty:
        logger.warning("No data in specified date range")
        return pd.DataFrame(columns=["date", "hour", "accumulation"])

    # Sort by timestamp ascending (archive may be DESC)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calculate interval accumulation
    df["time_diff_min"] = df["timestamp"].diff().dt.total_seconds() / 60
    df["interval_rain"] = df["dailyrainin"].diff().clip(lower=0)

    # Handle first row
    first_idx = df.index[0]
    df.loc[first_idx, "time_diff_min"] = 5
    df.loc[first_idx, "interval_rain"] = 0

    # Extract date and hour from local time
    df["date"] = df["timestamp_local"].dt.date
    df["hour"] = df["timestamp_local"].dt.hour

    # NOTE: Gap filtering removed - all accumulation data is included
    # to ensure accurate totals and proper hourly distribution

    # Aggregate by (date, hour)
    hourly_accum = df.groupby(["date", "hour"])["interval_rain"].sum().reset_index()
    hourly_accum.columns = ["date", "hour", "accumulation"]

    # Determine aggregation mode
    if row_mode is None or row_mode == "auto":
        if num_days and num_days > 730:  # 2 years
            row_mode = "year_month"
        elif num_days and num_days > 180:
            row_mode = "week"
        else:
            row_mode = "day"

    # Add timestamp for date operations
    hourly_accum["date_ts"] = pd.to_datetime(hourly_accum["date"])

    # Apply aggregation based on row mode (column type is determined by row type)
    if row_mode == "month":
        logger.info(f"Aggregating by month/day-of-month")

        # Add month and day columns
        hourly_accum["month"] = hourly_accum["date_ts"].dt.month
        hourly_accum["day_of_month"] = hourly_accum["date_ts"].dt.day

        # Aggregate by (month, day_of_month)
        monthly_accum = (
            hourly_accum.groupby(["month", "day_of_month"])["accumulation"]
            .sum()
            .reset_index()
        )
        monthly_accum.columns = ["date", "hour", "accumulation"]  # Reuse column names

        logger.info(
            f"Prepared monthly/day heatmap data: {len(monthly_accum)} cells "
            f"across all months"
        )

        return monthly_accum

    elif row_mode == "year_month":
        logger.info(f"Aggregating by year-month/day-of-month")

        # Add year-month and day columns
        hourly_accum["year_month"] = (
            hourly_accum["date_ts"].dt.to_period("M").dt.strftime("%Y-%m")
        )
        hourly_accum["day_of_month"] = hourly_accum["date_ts"].dt.day

        # Aggregate by (year_month, day_of_month)
        year_month_accum = (
            hourly_accum.groupby(["year_month", "day_of_month"])["accumulation"]
            .sum()
            .reset_index()
        )
        year_month_accum.columns = [
            "date",
            "hour",
            "accumulation",
        ]  # Reuse column names

        logger.info(
            f"Prepared year-month/day heatmap data: {len(year_month_accum)} cells "
            f"from {year_month_accum['date'].min()} to {year_month_accum['date'].max()}"
        )

        return year_month_accum

    elif row_mode == "week":
        logger.info(f"Aggregating by week/day-of-week")

        # Add week and day-of-week columns
        hourly_accum["week_start"] = (
            hourly_accum["date_ts"].dt.to_period("W").dt.start_time.dt.date
        )
        hourly_accum["day_of_week"] = hourly_accum["date_ts"].dt.dayofweek  # 0=Monday

        # Aggregate by (week, day_of_week)
        weekly_accum = (
            hourly_accum.groupby(["week_start", "day_of_week"])["accumulation"]
            .sum()
            .reset_index()
        )
        weekly_accum.columns = ["date", "hour", "accumulation"]  # Reuse column names

        logger.info(
            f"Prepared weekly heatmap data: {len(weekly_accum)} weekly cells "
            f"from {weekly_accum['date'].min()} to {weekly_accum['date'].max()}"
        )

        return weekly_accum

    logger.info(
        f"Prepared accumulation heatmap data: {len(hourly_accum)} hourly cells "
        f"from {hourly_accum['date'].min()} to {hourly_accum['date'].max()}"
    )

    return hourly_accum


def create_rain_accumulation_heatmap(
    accumulation_df: pd.DataFrame,
    height: int = 600,
    max_accumulation: Optional[float] = None,
    num_days: Optional[int] = None,
    row_mode: Optional[str] = None,
    compact: bool = False,
) -> go.Figure:
    """
    Create heatmap showing rainfall accumulation with simplified grid options.

    Row modes: 'day', 'week', 'month', 'year_month', 'auto'
    - day: Daily rows with Hour of Day columns
    - week: Weekly rows with Day of Week columns
    - month: Monthly rows with Day of Month columns
    - year_month: YY-MM rows with Day of Month columns
    - auto: Automatically select based on period length

    Auto behavior:
    - ≤180 days: daily × hourly
    - 180-730 days: weekly × day-of-week
    - >730 days: year_month × day-of-month

    :param accumulation_df: DataFrame with (date, hour, accumulation)
    :param height: Chart height in pixels (auto-calculated if using default)
    :param max_accumulation: Cap color scale (auto-calculated at 90th percentile if None)
    :param num_days: Number of days in period (helps determine auto mode)
    :param row_mode: Row aggregation mode ('day', 'week', 'month', 'year_month', 'auto')
    :param compact: If True, removes legend and axis labels for overview display
    :return: Plotly figure
    """
    if accumulation_df.empty:
        logger.warning("Empty accumulation data for heatmap")
        return go.Figure()

    # Determine display mode
    if row_mode is None or row_mode == "auto":
        if num_days and num_days > 730:  # 2 years
            row_mode = "year_month"
        elif num_days and num_days > 180:
            row_mode = "week"
        else:
            row_mode = "day"

    # Setup grid based on row mode (column type is determined by row type)
    if row_mode == "month":
        # Month/Day grid: 12 rows x 31 columns
        all_months = list(range(1, 13))  # 1-12
        all_days = list(range(1, 32))  # 1-31

        full_index = pd.MultiIndex.from_product(
            [all_months, all_days], names=["date", "hour"]
        )
        x_labels = [str(d) for d in all_days]
        x_title = "Day of Month"
        y_title = "Month"
        chart_title = "Monthly Rainfall Patterns"
        height = 400  # Fixed height for 12 rows
        grid_gap = 0

    elif row_mode == "year_month":
        # YY-MM/Day grid: variable rows x 31 columns
        # Get unique year-month values from data
        year_months = sorted(accumulation_df["date"].unique())
        all_days = list(range(1, 32))  # 1-31

        full_index = pd.MultiIndex.from_product(
            [year_months, all_days], names=["date", "hour"]
        )
        x_labels = [str(d) for d in all_days]
        x_title = "Day of Month"
        y_title = "Year-Month"
        chart_title = "Monthly Timeline Rainfall Patterns"
        height = min(800, max(400, len(year_months) * 25))  # Dynamic height
        grid_gap = 0

    elif row_mode == "week":
        # Weekly mode: rows are weeks, columns are days of week
        all_weeks = pd.date_range(
            accumulation_df["date"].min(), accumulation_df["date"].max(), freq="W-MON"
        ).date
        all_days = list(range(7))  # 0=Monday to 6=Sunday

        full_index = pd.MultiIndex.from_product(
            [all_weeks, all_days], names=["date", "hour"]
        )
        x_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        x_title = "Day of Week"
        y_title = "Week Starting"
        chart_title = "Weekly Rainfall Patterns"
        height = min(800, max(400, len(all_weeks) * 15))  # Dynamic height
        grid_gap = 0

    else:  # day
        # Default hourly mode: rows are dates, columns are hours
        all_dates = pd.date_range(
            accumulation_df["date"].min(), accumulation_df["date"].max(), freq="D"
        ).date
        all_hours = list(range(24))

        full_index = pd.MultiIndex.from_product(
            [all_dates, all_hours], names=["date", "hour"]
        )
        x_labels = [f"{h:02d}:00" for h in range(24)]
        x_title = "Hour of Day"
        y_title = "Date"
        chart_title = "Hourly Rainfall Accumulation"
        height = min(1200, max(600, len(all_dates) * 10))  # Dynamic height
        grid_gap = 1  # Small gaps for daily view

    # Reindex and fill missing with NaN
    indexed = accumulation_df.set_index(["date", "hour"])
    full_data = indexed.reindex(full_index, fill_value=np.nan).reset_index()

    # Pivot: rows=dates/weeks/months, columns=hours/days
    pivot = full_data.pivot(index="date", columns="hour", values="accumulation")

    # Auto-scale colorbar at 90th percentile of non-zero values if not specified
    if max_accumulation is None:
        valid_values = pivot.values[~np.isnan(pivot.values)]
        non_zero_values = valid_values[valid_values > 0]

        if len(non_zero_values) > 0:
            max_accumulation = float(np.percentile(non_zero_values, 90))
            max_accumulation = max(max_accumulation, 0.05)  # Minimum 0.05"
        else:
            max_accumulation = 0.05

    # Create heatmap
    hover_template = (
        "<b>%{y}</b><br>"
        f"{x_title}: %{{x}}<br>"
        'Accumulation: %{z:.3f}"'
        "<extra></extra>"
    )

    # Custom colorscale: white at 0, then blue gradient for positive values
    colorscale = [
        [0.0, "white"],  # 0 maps to white
        [0.001, "#f7fbff"],  # Very light blue
        [0.25, "#deebf7"],  # Light blue
        [0.5, "#9ecae1"],  # Medium blue
        [0.75, "#4292c6"],  # Darker blue
        [1.0, "#08519c"],  # Darkest blue at 90th percentile
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=pivot.index.astype(str),
            colorscale=colorscale,
            zmin=0,
            zmax=max_accumulation,
            colorbar=dict(title="Rain (in)"),
            hovertemplate=hover_template,
            zsmooth=False,
            hoverongaps=False,
        )
    )

    fig.update_traces(xgap=grid_gap, ygap=grid_gap)

    # Apply compact styling if requested
    if compact:
        margin = dict(l=30, r=20, t=30, b=40)
        showlegend = False
        xaxis_showticklabels = False
        colorbar_title = ""
        colorbar_dtick = 0.1
    else:
        margin = dict(l=80, r=20, t=60, b=60)
        showlegend = None  # Use default
        xaxis_showticklabels = None  # Use default
        colorbar_title = "Rain (in)"
        colorbar_dtick = None  # Use default

    fig.update_layout(
        title=chart_title,
        xaxis=dict(
            title=x_title,
            tickmode="linear",
            dtick=(
                1 if row_mode in ["week", "month", "year_month"] else 2
            ),  # Show all for aggregated views, every 2 for hourly
            type="category",
            showgrid=True,
            gridcolor="lightgrey",
            showticklabels=xaxis_showticklabels,
        ),
        yaxis=dict(
            title=y_title,
            type="category",
            autorange="reversed",  # Most recent at top
            showgrid=True,
            gridcolor="lightgrey",
        ),
        height=height,
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=showlegend,
        coloraxis_colorbar=dict(
            title=colorbar_title,
            tickmode="linear",
            tick0=0,
            dtick=colorbar_dtick,
        ),
    )

    return fig


def create_year_over_year_accumulation_chart(
    yoy_data: pd.DataFrame, start_day: int = 1, end_day: int = 365
) -> go.Figure:
    """
    Create year-over-year cumulative rainfall line chart.

    Displays multiple lines, one for each year, showing cumulative rainfall
    progression through the specified day range. Enables visual comparison of
    rainfall patterns across different years for specific time periods.

    :param yoy_data: DataFrame with day_of_year, year, cumulative_rainfall columns.
    :param start_day: Start day of year displayed (for axis labeling).
    :param end_day: End day of year displayed (for axis labeling).
    :return: Plotly figure with line chart.
    """
    if yoy_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No year-over-year data available",
            height=400,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return fig

    fig = go.Figure()

    # Color palette for different years
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Get unique years and sort them
    years = sorted(yoy_data["year"].unique())

    # Add a line for each year
    for i, year in enumerate(years):
        year_data = yoy_data[yoy_data["year"] == year].copy()
        year_data = year_data.sort_values("day_of_year")

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=year_data["day_of_year"],
                y=year_data["cumulative_rainfall"],
                mode="lines",
                line=dict(color=color, width=2),
                name=str(year),
                hovertemplate=(
                    "Day %{x}<br>"
                    "Year: " + str(year) + "<br>"
                    'Cumulative: %{y:.2f}"<extra></extra>'
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title="Year-over-Year Rainfall Accumulation",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis_title="Day of Year",
        yaxis_title="Cumulative Rainfall (inches)",
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        range=[start_day, end_day],
        tickmode="array",
        tickvals=[1, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
        ticktext=[
            "Jan 1",
            "Feb 1",
            "Mar 1",
            "Apr 1",
            "May 1",
            "Jun 1",
            "Jul 1",
            "Aug 1",
            "Sep 1",
            "Oct 1",
            "Nov 1",
            "Dec 1",
            "Dec 31",
        ],
    )

    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")

    return fig


def _create_event_headline(current_event):
    """
    Create formatted headline for rain event display.

    :param current_event: Event dictionary from catalog
    :return: Formatted headline string
    """
    start_time = pd.to_datetime(current_event["start_time"], utc=True)
    end_time = pd.to_datetime(current_event["end_time"], utc=True)

    # Convert to Pacific time
    start_pst = start_time.tz_convert("America/Los_Angeles")
    end_pst = end_time.tz_convert("America/Los_Angeles")

    # Format date strings
    start_str = start_pst.strftime("%b %-d")
    if end_pst.date() != start_pst.date():
        end_str = end_pst.strftime("%-d, %Y")
    else:
        end_str = end_pst.strftime("%-I:%M %p").lower().lstrip("0")

    # Duration formatting
    duration_h = current_event["duration_minutes"] / 60
    if duration_h >= 48:
        duration_str = f"{duration_h/24:.1f}d"
    else:
        duration_str = f"{duration_h:.1f}h"

    # Extract values
    total_rain = current_event["total_rainfall"]
    peak_rate = current_event["max_hourly_rate"]

    # Create headline without quality and flags
    headline = f'Rain Event: {start_str}-{end_str} • {duration_str} • {total_rain:.3f}" • {peak_rate:.3f} in/hr'

    return headline


def create_event_detail_charts(history_df, current_event, event_key="event"):
    """
    Create both accumulation and rate charts for rain event detail.

    :param history_df: Full weather history DataFrame
    :param current_event: Current event dictionary from catalog
    :param event_key: Key prefix for chart uniqueness
    :return: Tuple of (accumulation_fig, rate_fig)
    """
    # Extract event data from history
    history_df = history_df.copy()
    history_df["timestamp"] = pd.to_datetime(history_df["dateutc"], unit="ms", utc=True)

    start_time = pd.to_datetime(current_event["start_time"], utc=True)
    end_time = pd.to_datetime(current_event["end_time"], utc=True)

    mask = (history_df["timestamp"] >= start_time) & (
        history_df["timestamp"] <= end_time
    )
    event_data = history_df[mask].sort_values("timestamp").copy()

    if len(event_data) == 0:
        return None, None, None

    # Create event info for accumulation chart
    event_info = {
        "total_rainfall": current_event["total_rainfall"],
        "duration_minutes": current_event["duration_minutes"],
        "start_time": start_time,
        "end_time": end_time,
    }

    # Create charts using existing functions
    acc_fig = create_event_accumulation_chart(event_data, event_info)
    rate_fig = create_event_rate_chart(event_data)

    # Create headline for overview display
    headline = _create_event_headline(current_event)

    return acc_fig, rate_fig, headline
