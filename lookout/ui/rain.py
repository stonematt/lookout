"""
Precipitation analysis UI for Lookout weather station dashboard.

This module provides the Streamlit presentation layer for rainfall analysis,
including caching wrappers and table rendering. Core data processing is handled
by lookout.core.rainfall_analysis and visualizations by lookout.core.visualization.
"""

import pandas as pd
import streamlit as st

import lookout.core.rainfall_analysis as rain_analysis
import lookout.core.visualization as lo_viz
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


@st.cache_data(show_spinner=False)
def _cached_rolling_context(
    daily_rain_df: pd.DataFrame, windows, normals_years, end_date, version: str = "v2"
):
    return rain_analysis.compute_rolling_rain_context(
        daily_rain_df, windows, normals_years, end_date
    )


def render_rolling_rain_context_table(stats_df: pd.DataFrame, unit: str = "in") -> None:
    """
    Render rolling rainfall context as a compact table.

    :param stats_df: DataFrame with rolling window statistics
    :param unit: Unit for display (e.g., "in")
    """
    if stats_df.empty:
        st.info("No rolling-window rainfall stats available.")
        return

    view = stats_df.assign(
        Window=stats_df["window_days"].astype(str) + "d",
        PeriodBegins=stats_df["period_start"].dt.strftime("%Y-%m-%d"),
        Total=stats_df["total"].map(lambda x: f"{x:g} {unit}"),
        Normal=stats_df["normal"].map(
            lambda x: f"{x:g} {unit}" if pd.notna(x) else "—"
        ),
        Anomaly=stats_df["anomaly_pct"].map(
            lambda x: f"{x:+.0f} %" if pd.notna(x) else "—"
        ),
        Rank=stats_df.apply(
            lambda r: (
                f"{int(r['rank'])} / {int(r['n_periods'])}"
                if pd.notna(r["rank"])
                else "—"
            ),
            axis=1,
        ),
        Percentile=stats_df["percentile"].map(
            lambda x: f"{x:.0f}th" if pd.notna(x) else "—"
        ),
    )[["Window", "PeriodBegins", "Total", "Normal", "Anomaly", "Rank", "Percentile"]]

    try:
        st.dataframe(view, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(view.set_index("Window"), width="stretch")


@st.cache_data(show_spinner=False)
def _cached_violin_data(
    daily_rain_df: pd.DataFrame, windows, normals_years, end_date, version: str = "v1"
):
    return rain_analysis.prepare_violin_plot_data(
        daily_rain_df, windows, normals_years, end_date
    )


@st.cache_data(show_spinner=False)
def _cached_accumulation_data(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    version: str = "v2",
):
    """
    Cache wrapper for accumulation heatmap data preparation.

    :param df: Archive DataFrame with dateutc and dailyrainin
    :param start_date: Start date for range (date object)
    :param end_date: End date for range (date object)
    :param version: Cache version for invalidation
    :return: Prepared accumulation DataFrame
    """
    # Convert dates to timestamps
    start_ts = (
        pd.Timestamp(start_date).tz_localize("America/Los_Angeles").tz_convert("UTC")
    )
    end_ts = (
        (pd.Timestamp(end_date) + pd.Timedelta(days=1))
        .tz_localize("America/Los_Angeles")
        .tz_convert("UTC")
    )

    num_days = (end_date - start_date).days + 1

    return lo_viz.prepare_rain_accumulation_heatmap_data(
        archive_df=df,
        start_date=start_ts,
        end_date=end_ts,
        timezone="America/Los_Angeles",
        num_days=num_days,
        row_mode=None,  # Will be set in UI
    )


def render():
    """Render the precipitation analysis tab."""
    st.header("Precipitation Analysis")
    st.write("Comprehensive rainfall data analysis and visualization")

    if "history_df" not in st.session_state:
        st.error("No weather data available in session")
        return

    df = st.session_state["history_df"]

    if "last_data" in st.session_state:
        latest = st.session_state["last_data"]
        try:
            last_rain = pd.to_datetime(latest["lastRain"])
            current_time = pd.to_datetime(latest["dateutc"], unit="ms", utc=True)
            dry_period = current_time - last_rain
            current_dry_days = dry_period.days
            current_dry_hours = (dry_period.total_seconds() % (24 * 3600)) / 3600

            if current_dry_days > 0:
                time_since_rain = f"{current_dry_days}d {current_dry_hours:.1f}h"
            else:
                time_since_rain = f"{current_dry_hours:.1f}h"
        except Exception:
            current_dry_days = 0
            time_since_rain = "0h"
    else:
        latest = df.iloc[-1].to_dict()
        current_dry_days = 0
        time_since_rain = "0h"

    with st.spinner("Processing daily rainfall data..."):
        daily_rain_df = rain_analysis.extract_daily_rainfall(df)

    with st.spinner("Calculating rainfall statistics..."):
        stats = rain_analysis.calculate_rainfall_statistics(df, daily_rain_df)
        if "last_data" in st.session_state:
            stats.update(
                {
                    "current_ytd": latest.get("yearlyrainin", 0),
                    "current_monthly": latest.get("monthlyrainin", 0),
                    "current_weekly": latest.get("weeklyrainin", 0),
                    "current_daily": latest.get("dailyrainin", 0),
                    "last_rain": latest.get("lastRain", "Unknown"),
                    "current_dry_days": current_dry_days,
                    "time_since_rain": time_since_rain,
                }
            )

    yesterday_date = pd.to_datetime(daily_rain_df["date"]).max() - pd.Timedelta(days=1)
    yesterday_rain = (
        daily_rain_df[pd.to_datetime(daily_rain_df["date"]) == yesterday_date][
            "rainfall"
        ].sum()
        if len(daily_rain_df) > 0
        else 0.0
    )

    # Calculate rolling context for relative periods
    if len(daily_rain_df) > 0:
        end_date = pd.to_datetime(daily_rain_df["date"]).max()
        context_df = _cached_rolling_context(
            daily_rain_df=daily_rain_df,
            windows=(1, 7, 30, 90, 365),
            normals_years=None,
            end_date=end_date,
            version="v2",
        )

        # Get relative period values
        today_val = stats["current_daily"]
        yesterday_val = yesterday_rain
        last_7d_val = (
            context_df[context_df["window_days"] == 7]["total"].iloc[0]
            if len(context_df[context_df["window_days"] == 7]) > 0
            else 0
        )
        last_30d_val = (
            context_df[context_df["window_days"] == 30]["total"].iloc[0]
            if len(context_df[context_df["window_days"] == 30]) > 0
            else 0
        )
        last_90d_val = (
            context_df[context_df["window_days"] == 90]["total"].iloc[0]
            if len(context_df[context_df["window_days"] == 90]) > 0
            else 0
        )
        last_365d_val = (
            context_df[context_df["window_days"] == 365]["total"].iloc[0]
            if len(context_df[context_df["window_days"] == 365]) > 0
            else 0
        )
    else:
        today_val = stats["current_daily"]
        yesterday_val = yesterday_rain
        last_7d_val = last_30d_val = last_90d_val = last_365d_val = 0

    st.markdown(
        f"**Today:** {today_val:.2f}\" • "
        f'**Yesterday:** {yesterday_val:.2f}" • '
        f"**Last 7d:** {last_7d_val:.2f}\" • "
        f"**Last 30d:** {last_30d_val:.2f}\" • "
        f"**Last 90d:** {last_90d_val:.2f}\" • "
        f"**Last 365d:** {last_365d_val:.2f}\""
    )

    # Use values already calculated for overview
    current_values = {
        "today": today_val,
        "yesterday": yesterday_val,
        "7d": last_7d_val,
        "30d": last_30d_val,
        "90d": last_90d_val,
        "365d": last_365d_val,
    }

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_daily = lo_viz.create_rainfall_summary_violin(
            daily_rain_df=daily_rain_df,
            current_values=current_values,
            rolling_context_df=context_df,
            end_date=end_date,
            windows=["Today", "Yesterday"],
        )
        st.plotly_chart(fig_daily, width="stretch", key="daily_viz")

    with col2:
        fig_rolling = lo_viz.create_rainfall_summary_violin(
            daily_rain_df=daily_rain_df,
            current_values=current_values,
            rolling_context_df=context_df,
            end_date=end_date,
            windows=["7d", "30d", "90d", "365d"],
        )
        st.plotly_chart(fig_rolling, width="stretch", key="rolling_viz")

    rain_percentage = (
        (stats["total_rain_days"] / stats["total_days"] * 100)
        if stats["total_days"] > 0
        else 0
    )

    try:
        last_rain_dt = pd.to_datetime(stats["last_rain"])
        last_rain_local = last_rain_dt.tz_convert("America/Los_Angeles")
        formatted_date = last_rain_local.strftime("%m/%d/%Y %H:%M")
    except Exception:
        formatted_date = str(stats.get("last_rain", "Unknown"))

    st.caption(
        f"📊 Historical: Avg annual {stats['avg_annual']:.1f}\" • "
        f"{stats['total_days']:,} days • {stats['total_rain_days']:,} rain days ({rain_percentage:.0f}%)"
    )
    st.caption(
        f"🌧️ Recent: Max daily {stats['max_daily_this_year']:.2f}\" • "
        f"Last rain {formatted_date} • Dry {stats.get('time_since_rain', '0h')}"
    )

    with st.expander("History Windows"):
        st.subheader("Rolling Historical Context (1d / 7d / 30d / 90d)")

        if len(daily_rain_df) > 0:
            end_date = pd.to_datetime(daily_rain_df["date"]).max()
            context_df = _cached_rolling_context(
                daily_rain_df=daily_rain_df,
                windows=(1, 7, 30, 90),
                normals_years=None,
                end_date=end_date,
                version="v2",
            )
            render_rolling_rain_context_table(context_df, unit="in")
        else:
            st.info("No daily totals to compute rolling context.")

    st.divider()

    st.subheader("Historical Rainfall Distribution")
    st.write(
        "Compare current rainfall periods against the full distribution of all historical periods"
    )

    if len(daily_rain_df) > 0:
        end_date = pd.to_datetime(daily_rain_df["date"]).max()
        violin_data = _cached_violin_data(
            daily_rain_df=daily_rain_df,
            windows=(1, 7, 30, 90),
            normals_years=None,
            end_date=end_date,
            version="v1",
        )

        available_windows = [
            w
            for w in ["1d", "7d", "30d", "90d"]
            if w in violin_data and len(violin_data[w]["values"]) > 0  # type: ignore
        ]

        if available_windows:
            viz_mode = st.radio(
                "Visualization mode:",
                ["Single Window", "Compare Two Windows"],
                horizontal=True,
                help="Choose single window analysis or side-by-side comparison",
            )

            if viz_mode == "Single Window":
                selected_window = st.selectbox(
                    "Select time window for distribution analysis:",
                    available_windows,
                    index=min(1, len(available_windows) - 1),
                    help="Choose the rolling period length to analyze",
                )

                lo_viz.create_rainfall_violin_plot(
                    window=selected_window, violin_data=violin_data, unit="in"
                )

            else:
                col1, col2 = st.columns(2)

                with col1:
                    left_window = st.selectbox(
                        "Left window:",
                        available_windows,
                        index=0,
                        help="Choose the left period for comparison",
                    )

                with col2:
                    right_window = st.selectbox(
                        "Right window:",
                        available_windows,
                        index=min(1, len(available_windows) - 1),
                        help="Choose the right period for comparison",
                    )

                if left_window != right_window:
                    lo_viz.create_dual_violin_plot(
                        left_window=left_window,
                        right_window=right_window,
                        violin_data=violin_data,
                        unit="in",
                    )
                else:
                    st.info("Please select different windows for comparison.")
        else:
            st.warning("Insufficient historical data for distribution analysis.")
    else:
        st.info("No daily rainfall data available for distribution analysis.")

    st.divider()

    st.subheader("Year-over-Year Accumulation")
    st.info(
        "Coming next: Line chart showing cumulative rainfall by day of year, with separate lines for each year"
    )

    st.subheader("Rain Accumulation Heatmap")

    # Get available date range from data
    df_timestamps = pd.to_datetime(df["dateutc"], unit="ms", utc=True).dt.tz_convert(
        "America/Los_Angeles"
    )
    min_date = df_timestamps.min().date()
    max_date = df_timestamps.max().date()

    # Default to last 90 days
    default_start = max(min_date, max_date - pd.Timedelta(days=90))

    st.write("**Date Range:**")
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="MMM DD, YYYY",
        label_visibility="collapsed",
    )

    start_date, end_date = date_range
    num_days = (end_date - start_date).days + 1

    # Grid selection control
    row_mode = st.selectbox(
        "Grid type:",
        options=["auto", "day", "week", "month", "year_month"],
        format_func=lambda x: {
            "auto": "Auto (based on period)",
            "day": "Daily × Hourly",
            "week": "Weekly × Day-of-week",
            "month": "Monthly × Day-of-month",
            "year_month": "Timeline × Day-of-month",
        }[x],
        index=0,
        help="Choose grid type (column aggregation is automatic)",
    )

    # Display mode info
    mode_descriptions = {
        "day": "Daily rows × Hourly columns",
        "week": "Weekly rows × Day-of-week columns",
        "month": "Monthly rows × Day-of-month columns",
        "year_month": "Timeline rows × Day-of-month columns",
    }

    actual_row_mode = (
        row_mode
        if row_mode != "auto"
        else ("year_month" if num_days > 730 else "week" if num_days > 180 else "day")
    )
    st.caption(f"📅 {num_days} days selected • {mode_descriptions[actual_row_mode]}")

    # Data preparation with caching
    with st.spinner("Preparing accumulation data..."):
        # Re-aggregate data if needed for selected mode
        if row_mode != "auto":
            start_ts = (
                pd.Timestamp(start_date)
                .tz_localize("America/Los_Angeles")
                .tz_convert("UTC")
            )
            end_ts = (
                (pd.Timestamp(end_date) + pd.Timedelta(days=1))
                .tz_localize("America/Los_Angeles")
                .tz_convert("UTC")
            )

            accumulation_df = lo_viz.prepare_rain_accumulation_heatmap_data(
                archive_df=df,
                start_date=start_ts,
                end_date=end_ts,
                timezone="America/Los_Angeles",
                num_days=num_days,
                row_mode=row_mode,
            )
        else:
            # Use cached data for auto mode
            accumulation_df = _cached_accumulation_data(
                df=df,
                start_date=start_date,
                end_date=end_date,
                version="v3",  # New version for aggregation modes
            )

    # Render heatmap
    if not accumulation_df.empty:
        fig = lo_viz.create_rain_accumulation_heatmap(
            accumulation_df=accumulation_df, num_days=num_days, row_mode=row_mode
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Summary statistics
        max_cell = accumulation_df["accumulation"].max()
        total_period = accumulation_df["accumulation"].sum()

        # Get actual mode being used
        actual_row_mode = (
            row_mode
            if row_mode != "auto"
            else (
                "year_month" if num_days > 730 else "week" if num_days > 180 else "day"
            )
        )

        if actual_row_mode == "month":
            st.caption(
                f'Peak monthly/day cell: {max_cell:.3f}" • '
                f'Total in period: {total_period:.2f}"'
            )
        elif actual_row_mode == "year_month":
            st.caption(
                f'Peak timeline/day cell: {max_cell:.3f}" • '
                f'Total in period: {total_period:.2f}"'
            )
        elif actual_row_mode == "week":
            st.caption(
                f'Peak weekly cell: {max_cell:.3f}" • '
                f'Total in period: {total_period:.2f}"'
            )
        else:  # day
            max_row = accumulation_df.loc[accumulation_df["accumulation"].idxmax()]
            st.caption(
                f"Peak hourly: {max_cell:.3f}\" on {max_row['date']} at {max_row['hour']:02d}:00 • "
                f'Total in period: {total_period:.2f}"'
            )
    else:
        st.info("No rainfall data in selected period")

    st.subheader("Dry Spell & Event Analysis")
    st.info(
        "Coming next: Analysis of dry periods, rain events, and precipitation patterns"
    )

    with st.expander("Data Health Check"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            date_range = (df["date"].max() - df["date"].min()).days
            st.metric("Days of Data", date_range)
        with col3:
            rain_fields = [col for col in df.columns if "rain" in col.lower()]
            st.metric("Rain Fields", len(rain_fields))

        if st.checkbox("Show daily rainfall sample"):
            st.write("**Recent daily totals:**")
            st.dataframe(daily_rain_df.tail(10))
