"""
overview.py rendering for overview tab
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lookout.core.data_processing as lo_dp
import lookout.core.visualization as lo_viz
from lookout import config as cfg
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def render():
    history_df = st.session_state["history_df"]
    last_data = st.session_state["last_data"]

    st.header("Overview")
    # Access the updated history_df and max_dateutc

    # Active Event Headline (under page title)
    render_active_event_headline()

    # Present the dashboard ########################

    row1 = st.columns([1, 1])  # Two equal columns

    with row1[0]:  # LEFT: Current Conditions
        # Temperature bars (existing)
        temp_bars = lo_dp.get_history_min_max(history_df, "date", "tempf", "temp")
        lo_viz.draw_horizontal_bars(temp_bars, label="Temperature (°F)")

        st.markdown("---")  # Visual divider

        # Wind rose (existing, moved down)
        # Parameters for the polar chart
        # value_col = "windspeedmph"
        # direction_col = "winddir"
        # Define valid column pairs for the polar chart
        valid_pairs = [
            ("windspeedmph", "winddir", "Wind Speed"),
            ("windspdmph_avg10m", "winddir_avg10m", "10m Average Wind"),
            ("windgustmph", "winddir", "Wind Gust"),
            ("maxdailygust", "winddir_avg10m", "Max Daily Gust"),
        ]

        # Initialize session state for the selected pair
        if "selected_pair" not in st.session_state:
            st.session_state["selected_pair"] = valid_pairs[
                0
            ]  # Default to the first pair

        # Unpack the selected pair from session state
        value_col, direction_col, wind_description = st.session_state["selected_pair"]

        # Use the wrapper function to prepare data
        grouped_data, value_labels, direction_labels = lo_dp.prepare_polar_chart_data(
            history_df,
            value_col,
            direction_col,
        )

        # Create and display the chart
        fig = lo_viz.create_windrose_chart(
            grouped_data, value_labels, color_palette="wind", title=value_col
        )

        st.plotly_chart(fig, width="stretch")

        # Dropdown for selecting the pair (below the chart)
        selected_pair = st.selectbox(
            "Wind Metric:",
            valid_pairs,
            format_func=lambda pair: pair[2],
            index=valid_pairs.index(
                st.session_state["selected_pair"]
            ),  # Set current session value
        )

        # Update session state if the user changes the dropdown value
        if selected_pair != st.session_state["selected_pair"]:
            st.session_state["selected_pair"] = selected_pair
            st.rerun()

    with row1[1]:  # RIGHT: Rainfall Summary (NEW)
        render_rainfall_summary_widget()

    # rain_bars = lo_dp.get_history_min_max(history_df, data_column= , )

    # Display the header
    st.subheader("Current")

    if last_data:
        lo_viz.make_column_gauges(cfg.TEMP_GAUGES)
        # make_column_gauges(rain_guages)

    st.subheader("Temps Plots")
    # Let the user select multiple metrics for comparison
    metric_titles = [metric["title"] for metric in cfg.BOX_PLOT_METRICS]
    selected_titles = st.multiselect(
        "Select metrics for the box plot:", metric_titles, default=metric_titles[0]
    )

    # Find the selected metrics based on the titles
    selected_metrics = [
        metric for metric in cfg.BOX_PLOT_METRICS if metric["title"] in selected_titles
    ]

    # User selects a box width
    box_width_option = st.selectbox(
        "Select box width:", ["hour", "day", "week", "month"]
    )

    if selected_metrics and "date" in history_df.columns:
        # Convert 'date' column to datetime if it's not already
        history_df["date"] = pd.to_datetime(history_df["date"])
        group_column = ""

        # Group by the selected box width option
        if box_width_option == "hour":
            history_df["hour"] = history_df["date"].dt.hour
            group_column = "hour"
        elif box_width_option == "day":
            history_df["day"] = history_df["date"].dt.dayofyear
            group_column = "day"
        elif box_width_option == "week":
            history_df["week"] = history_df["date"].dt.isocalendar().week
            group_column = "week"
        elif box_width_option == "month":
            history_df["month"] = history_df["date"].dt.month
            group_column = "month"

        # Create and render the box plot for each selected metric
        fig = go.Figure()
        for metric in selected_metrics:
            # Filter the DataFrame for the selected metric
            df_filtered = history_df[["date", group_column, metric["metric"]]].dropna()

            # Create a box plot for the current metric
            fig.add_trace(
                go.Box(
                    x=df_filtered[group_column],
                    y=df_filtered[metric["metric"]],
                    name=metric["title"],
                )
            )

        # Update plot layout
        fig.update_layout(
            title=f"Comparison of Selected Metrics by {box_width_option.capitalize()}",
            xaxis_title=box_width_option.capitalize(),
            yaxis_title="Value",
        )

        st.plotly_chart(fig)


def render_active_event_headline():
    """Render active rain event headline under page title."""
    import json
    
    # Active Event Detection
    active_event_headline = None
    
    try:
        if "device" in st.session_state and "history_df" in st.session_state:
            device = st.session_state["device"]
            device_mac = device["macAddress"]
            file_type = "parquet"
            
            from lookout.core.rain_events import RainEventCatalog
            catalog = RainEventCatalog(device_mac, file_type)
            
            # Try to get events from session or storage
            events_df = None
            if "rain_events_catalog" in st.session_state:
                events_df = st.session_state["rain_events_catalog"]
            elif catalog.catalog_exists():
                events_df = catalog.load_catalog()
            
            if events_df is not None and not events_df.empty:
                # Check for ongoing events
                ongoing_events = []
                for _, event in events_df.iterrows():
                    is_ongoing = False
                    if "ongoing" in event and event["ongoing"]:
                        is_ongoing = True
                    elif "flags" in event and event["flags"]:
                        flags = event["flags"]
                        if isinstance(flags, str):
                            flags = json.loads(flags)
                        is_ongoing = flags.get("ongoing", False)
                    
                    if is_ongoing:
                        ongoing_events.append(event)
                
                if ongoing_events:
                    # Get the most recent ongoing event
                    latest_event = max(ongoing_events, key=lambda x: x["start_time"])
                    
                    # Calculate duration and format
                    duration_h = latest_event["duration_minutes"] / 60
                    total_rain = latest_event["total_rainfall"]
                    
                    # Get last rain time from current data
                    last_rain_time = "unknown"
                    if "last_data" in st.session_state:
                        last_data = st.session_state["last_data"]
                        try:
                            last_rain = pd.to_datetime(last_data["lastRain"])
                            current_time = pd.to_datetime(last_data["dateutc"], unit="ms", utc=True)
                            time_since = current_time - last_rain
                            
                            if time_since.total_seconds() < 3600:  # Less than 1 hour
                                last_rain_time = f"{time_since.total_seconds()/60:.0f}min ago"
                            else:
                                hours = time_since.total_seconds() / 3600
                                last_rain_time = f"{hours:.1f}h ago"
                        except Exception:
                            last_rain_time = "unknown"
                    
                    # Format duration string
                    if duration_h >= 24:
                        duration_str = f"{duration_h/24:.1f}d"
                    else:
                        duration_str = f"{duration_h:.1f}h"
                    
                    active_event_headline = f"🌧️ {duration_str} event: {total_rain:.2f}\" (last rain {last_rain_time})"
    
    except ImportError:
        # Rain events not available
        pass
    except Exception as e:
        logger.warning(f"Error detecting active rain events: {e}")
    
    # Display active event headline if present
    if active_event_headline:
        st.markdown(f"**{active_event_headline}**")


def render_rainfall_summary_widget():
    """Render rainfall summary widget with placeholder for future components."""
    # Placeholder for remaining components
    st.info("🌧️ Today/Yesterday chart and 30-day heatmap coming next...")
