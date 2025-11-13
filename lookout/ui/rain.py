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
        PeriodEnd=stats_df["period_end"].dt.strftime("%Y-%m-%d"),
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
    )[["Window", "PeriodEnd", "Total", "Normal", "Anomaly", "Rank", "Percentile"]]

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

    st.markdown(
        f"**YTD:** {stats['current_ytd']:.2f}\" • "
        f"**Month:** {stats['current_monthly']:.2f}\" • "
        f"**Week:** {stats['current_weekly']:.2f}\" • "
        f'**Yesterday:** {yesterday_rain:.2f}" • '
        f"**Today:** {stats['current_daily']:.2f}\""
    )

    if len(daily_rain_df) > 0:
        end_date = pd.to_datetime(daily_rain_df["date"]).max()
        context_df = _cached_rolling_context(
            daily_rain_df=daily_rain_df,
            windows=(1, 7, 30, 90, 365),
            normals_years=None,
            end_date=end_date,
            version="v2",
        )

        current_values = {
            "today": stats["current_daily"],
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

        fig = lo_viz.create_rainfall_summary_boxplot(
            daily_rain_df=daily_rain_df,
            current_values=current_values,
            rolling_context_df=context_df,
            end_date=end_date,
        )
        st.plotly_chart(fig, use_container_width=True)

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

    st.divider()

    st.subheader(
        "Rolling Historical Context vs All N-day Periods (1d / 7d / 30d / 90d)"
    )

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

    st.subheader("Rain Intensity Heatmap")
    st.info(
        "Coming next: Heatmap of hourly rain rates with configurable time granularity (daily avg, max, etc.)"
    )

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
