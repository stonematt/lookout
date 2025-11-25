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

    # Access the updated history_df and max_dateutc

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
            # st.rerun()

    with row1[1]:  # RIGHT: Rainfall Summary (NEW)
        render_rainfall_summary_widget()

    # rain_bars = lo_dp.get_history_min_max(history_df, data_column= , )

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
        "Select box width:", ["week", "hour", "day", "month"]
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


def render_rainfall_summary_widget():
    """Render rainfall summary widget with today/yesterday chart and placeholder for heatmap."""
    import lookout.core.rainfall_analysis as rain_analysis

    try:
        if "history_df" not in st.session_state:
            st.info("🌧️ No weather data available")
            return

        df = st.session_state["history_df"]

        # Extract daily rainfall data
        with st.spinner("Processing rainfall data..."):
            daily_rain_df = rain_analysis.extract_daily_rainfall(df)

        if daily_rain_df.empty:
            st.info("🌧️ No rainfall data available")
            return

        # Get today and yesterday values
        end_date = pd.to_datetime(daily_rain_df["date"]).max()
        yesterday_date = end_date - pd.Timedelta(days=1)

        today_rain = 0.0
        yesterday_rain = 0.0

        # Get today's rainfall from current data if available
        if "last_data" in st.session_state:
            today_rain = st.session_state["last_data"].get("dailyrainin", 0) or 0

        # Get yesterday's rainfall from daily data
        yesterday_rain = (
            daily_rain_df[pd.to_datetime(daily_rain_df["date"]) == yesterday_date][
                "rainfall"
            ].sum()
            if len(daily_rain_df) > 0
            else 0.0
        )

        # Get historical distributions (excluding current year for context)
        daily_rain_df["date_dt"] = pd.to_datetime(daily_rain_df["date"])
        historical_today = daily_rain_df[
            daily_rain_df["date_dt"].dt.year != end_date.year
        ]["rainfall"].tolist()

        # For yesterday, we need historical data for same day-of-year
        yesterday_doy = yesterday_date.dayofyear
        historical_yesterday = daily_rain_df[
            (daily_rain_df["date_dt"].dt.year != end_date.year)
            & (daily_rain_df["date_dt"].dt.dayofyear == yesterday_doy)
        ]["rainfall"].tolist()

        # Calculate rolling context for the violin plot (same as rain tab)
        context_df = None
        current_values = {}

        if len(daily_rain_df) > 0:
            context_df = rain_analysis.compute_rolling_rain_context(
                daily_rain_df=daily_rain_df,
                windows=(1, 7, 30, 90, 365),
                normals_years=None,
                end_date=end_date,
            )

            # Get current values for the violin plot
            current_values = {
                "today": today_rain,
                "yesterday": yesterday_rain,
                "7d": (
                    context_df[context_df["window_days"] == 7]["total"].iloc[0]
                    if len(context_df[context_df["window_days"] == 7]) > 0
                    else 0
                ),
                "30d": (
                    context_df[context_df["window_days"] == 30]["total"].iloc[0]
                    if len(context_df[context_df["window_days"] == 30]) > 0
                    else 0
                ),
                "90d": (
                    context_df[context_df["window_days"] == 90]["total"].iloc[0]
                    if len(context_df[context_df["window_days"] == 90]) > 0
                    else 0
                ),
                "365d": (
                    context_df[context_df["window_days"] == 365]["total"].iloc[0]
                    if len(context_df[context_df["window_days"] == 365]) > 0
                    else 0
                ),
            }

        # Use the same violin plot as rain tab (today/yesterday only)
        # Hide chart completely if both days have no rainfall
        if today_rain == 0.0 and yesterday_rain == 0.0:
            # Silent - no chart or message when no rainfall
            pass
        elif context_df is not None and not context_df.empty:
            chart = lo_viz.create_rainfall_summary_violin(
                daily_rain_df=daily_rain_df,
                current_values=current_values,
                rolling_context_df=context_df,
                end_date=end_date,
                windows=["Today", "Yesterday"],
                title="Recent Rainfall",
            )

            st.plotly_chart(chart, width="stretch", key="today_yesterday_violin")

        # 30-day compact heatmap
        st.markdown("**Last 30 Days:**")

        # Get date range for last 30 days
        df_timestamps = pd.to_datetime(
            df["dateutc"], unit="ms", utc=True
        ).dt.tz_convert("America/Los_Angeles")
        max_date = df_timestamps.max().date()
        start_date = max_date - pd.Timedelta(days=29)  # 30 days inclusive

        # Prepare heatmap data using existing function
        start_ts = (
            pd.Timestamp(start_date)
            .tz_localize("America/Los_Angeles")
            .tz_convert("UTC")
        )
        end_ts = (
            (pd.Timestamp(max_date) + pd.Timedelta(days=1))
            .tz_localize("America/Los_Angeles")
            .tz_convert("UTC")
        )

        accumulation_df = lo_viz.prepare_rain_accumulation_heatmap_data(
            archive_df=df,
            start_date=start_ts,
            end_date=end_ts,
            timezone="America/Los_Angeles",
            num_days=30,
            row_mode="auto",  # Let it choose best mode for 30 days
        )

        # Render compact heatmap
        if not accumulation_df.empty:
            fig = lo_viz.create_rain_accumulation_heatmap(
                accumulation_df=accumulation_df,
                num_days=30,
                row_mode="auto",
                max_accumulation=None,  # Let function auto-scale
                height=300,  # Compact height
                compact=True,  # Remove legend and axis labels
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="compact_30day_heatmap_v2",
                config={"displayModeBar": False},
            )

            # Summary stats
            total_period = accumulation_df["accumulation"].sum()
            max_cell = accumulation_df["accumulation"].max()
            st.caption(f'📈 Total: {total_period:.2f}" • Peak daily: {max_cell:.2f}"')
        else:
            st.info("🌧️ No rainfall data in last 30 days")

    except Exception as e:
        logger.error(f"Error rendering rainfall summary widget: {e}")
        st.info("🌧️ Rainfall summary temporarily unavailable")
