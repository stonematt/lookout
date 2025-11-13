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

    daily_rain_df_copy = daily_rain_df.copy()
    daily_rain_df_copy["date"] = pd.to_datetime(daily_rain_df_copy["date"])

    all_single_days = daily_rain_df_copy[
        daily_rain_df_copy["date"].dt.year != end_date.year
    ]["rainfall"].values

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

            s = daily_rain_df_copy.set_index("date")["rainfall"].sort_index()
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

    return fig


def prepare_rain_accumulation_heatmap_data(
    archive_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    timezone: str = "America/Los_Angeles",
    include_gaps: bool = False,
) -> pd.DataFrame:
    """
    Prepare hourly rainfall accumulation data for heatmap.

    All rainfall increments are included to ensure accurate totals.
    Each dailyrainin.diff() represents real rain that fell, captured
    at the time of the reading regardless of data gaps.

    :param archive_df: Archive with dateutc and dailyrainin columns
    :param start_date: Filter start date (timezone-aware or naive UTC)
    :param end_date: Filter end date (timezone-aware or naive UTC)
    :param timezone: Timezone for hour bucketing
    :param include_gaps: Deprecated parameter (has no effect, all data included)
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

    logger.info(
        f"Prepared accumulation heatmap data: {len(hourly_accum)} hourly cells "
        f"from {hourly_accum['date'].min()} to {hourly_accum['date'].max()}"
    )

    return hourly_accum


def create_rain_accumulation_heatmap(
    accumulation_df: pd.DataFrame,
    height: int = 600,
    max_accumulation: Optional[float] = None,
) -> go.Figure:
    """
    Create heatmap showing hourly rainfall accumulation.

    :param accumulation_df: DataFrame with (date, hour, accumulation)
    :param height: Chart height in pixels
    :param max_accumulation: Cap color scale (auto if None)
    :return: Plotly figure
    """
    if accumulation_df.empty:
        logger.warning("Empty accumulation data for heatmap")
        return go.Figure()

    # Create full grid (all dates, all hours)
    all_dates = pd.date_range(
        accumulation_df["date"].min(), accumulation_df["date"].max(), freq="D"
    ).date
    all_hours = list(range(24))

    full_index = pd.MultiIndex.from_product(
        [all_dates, all_hours], names=["date", "hour"]
    )

    # Reindex and fill missing with NaN
    indexed = accumulation_df.set_index(["date", "hour"])
    full_data = indexed.reindex(full_index, fill_value=np.nan).reset_index()

    # Pivot: rows=dates, columns=hours
    pivot = full_data.pivot(index="date", columns="hour", values="accumulation")

    # Auto-scale colorbar if not specified
    if max_accumulation is None:
        # Use 95th percentile to avoid outliers washing out scale
        valid_values = pivot.values[~np.isnan(pivot.values)]
        if len(valid_values) > 0:
            max_accumulation = float(np.percentile(valid_values, 95))
            max_accumulation = max(max_accumulation, 0.05)  # Minimum 0.05"
        else:
            max_accumulation = 0.05

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=pivot.index.astype(str),
            colorscale="Blues",
            zmin=0,
            zmax=max_accumulation,
            colorbar=dict(title="Rain (in)"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Hour: %{x}<br>"
                'Accumulation: %{z:.3f}"'
                "<extra></extra>"
            ),
            zsmooth=False,
            hoverongaps=False,
        )
    )

    fig.update_traces(xgap=1, ygap=1)

    fig.update_layout(
        title="Hourly Rainfall Accumulation",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            dtick=2,  # Show every 2 hours
            type="category",
            showgrid=True,
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="Date",
            type="category",
            autorange="reversed",  # Most recent at top
            showgrid=True,
            gridcolor="lightgrey",
        ),
        height=height,
        margin=dict(l=80, r=20, t=60, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig
