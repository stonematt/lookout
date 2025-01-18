""" visualizations.py"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from colour import Color

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
    st.plotly_chart(fig, use_container_width=True)


def display_line_chart(df, metrics):
    """
    Display a line chart for selected metrics from the data.

    :param df: DataFrame containing the data.
    :param metrics: List of metric names to include in the chart.
    """
    fig = px.line(df, x="date", y=metrics, title="Historical Data")
    st.plotly_chart(fig, use_container_width=True)


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
