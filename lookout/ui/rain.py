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
        st.dataframe(view, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(view.set_index("Window"), use_container_width=True)


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

    st.subheader("Rainfall Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Year to Date", f"{stats['current_ytd']:.2f}\"")
    with col2:
        st.metric("This Month", f"{stats['current_monthly']:.2f}\"")
    with col3:
        st.metric("This Week", f"{stats['current_weekly']:.2f}\"")
    with col4:
        st.metric("Yesterday", f"{stats['current_yesterday']:.2f}\"")
    with col5:
        st.metric("Today", f"{stats['current_daily']:.2f}\"")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Historical Context:**")
        st.write(f"• Average Annual: {stats['avg_annual']:.1f}\"")
        st.write(f"• Total Data Days: {stats['total_days']:,}")
        st.write(f"• Days with Rain: {stats['total_rain_days']:,}")
        rain_percentage = (
            (stats["total_rain_days"] / stats["total_days"] * 100)
            if stats["total_days"] > 0
            else 0
        )
        st.write(f"• Rain Frequency: {rain_percentage:.1f}%")

    with col2:
        st.write("**Recent Activity:**")
        st.write(f"• Max Daily Rain: {stats['max_daily_this_year']:.2f}\"")
        if "time_since_rain" in stats:
            st.write(f"• Time Since Rain: {stats['time_since_rain']}")
        else:
            st.write(f"• Days Since Rain: {stats['current_dry_days']}")

        try:
            last_rain_dt = pd.to_datetime(stats["last_rain"])
            last_rain_local = last_rain_dt.tz_convert("America/Los_Angeles")
            formatted_date = last_rain_local.strftime("%m/%d/%Y %H:%M")
            st.write(f"• Last Rain: {formatted_date}")
        except Exception:
            st.write(f"• Last Rain: {stats['last_rain']}")

    st.divider()

    st.subheader("Daily Rainfall Chart")

    if len(daily_rain_df) > 0:
        rain_accumulations = rain_analysis.calculate_rainfall_accumulations(
            daily_rain_df, df
        )

        if rain_accumulations:
            lo_viz.draw_horizontal_bars(
                rain_accumulations, label="Rainfall Accumulation (inches)"
            )
        else:
            st.error("Could not calculate rainfall accumulations")
    else:
        st.error("No daily rainfall data available")

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

    st.subheader("Rain Event Catalog")
    st.write("Browse and analyze individual rain events detected from historical data")

    try:
        from lookout.core.rain_events import RainEventCatalog

        if "device" in st.session_state:
            device = st.session_state["device"]
            device_mac = device["macAddress"]
            file_type = "parquet"

            catalog = RainEventCatalog(device_mac, file_type)

            events_df = None

            if "rain_events_catalog" in st.session_state:
                events_df = st.session_state["rain_events_catalog"]
                catalog_source = "session"
                logger.info(
                    f"Using catalog from session state: {len(events_df)} events"
                )

            elif catalog.catalog_exists():
                with st.spinner("Loading event catalog from storage..."):
                    logger.info(f"Loading catalog from Storj: {catalog.catalog_path}")
                    stored_catalog = catalog.load_catalog()
                    if not stored_catalog.empty:
                        logger.info(
                            f"Loaded {len(stored_catalog)} events from storage, checking for updates..."
                        )
                        events_df = catalog.update_catalog_with_new_data(
                            df, stored_catalog
                        )
                        st.session_state["rain_events_catalog"] = events_df
                        catalog_source = "storage"
                        logger.info(
                            f"Catalog updated and cached in session: {len(events_df)} events"
                        )
                    else:
                        logger.warning("Loaded catalog from storage is empty")
                        catalog_source = None

            else:
                with st.spinner(
                    "Generating event catalog from historical data... This may take a minute."
                ):
                    logger.info(
                        "No catalog found in session or storage, generating fresh catalog..."
                    )
                    events_df = catalog.detect_and_catalog_events(df, auto_save=False)
                    st.session_state["rain_events_catalog"] = events_df
                    catalog_source = "generated"
                    logger.info(
                        f"Fresh catalog generated and cached: {len(events_df)} events"
                    )

            if events_df is not None and not events_df.empty:
                zero_rate_count = (events_df["max_hourly_rate"] == 0).sum()
                logger.debug(
                    f"Displaying catalog: {len(events_df)} events, {zero_rate_count} with zero max_rate"
                )
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
                with btn_col1:
                    if st.button("💾 Save to Storage"):
                        if catalog.save_catalog(events_df):
                            st.success("✅ Catalog saved to Storj!")
                        else:
                            st.error("❌ Failed to save catalog")

                with btn_col2:
                    if st.button("🔄 Regenerate"):
                        logger.info("User requested catalog regeneration")

                        if "rain_events_catalog" in st.session_state:
                            old_count = len(st.session_state["rain_events_catalog"])
                            del st.session_state["rain_events_catalog"]
                            logger.info(
                                f"Cleared old catalog from session: {old_count} events"
                            )

                        with st.spinner("Regenerating event catalog from archive..."):
                            new_events = catalog.detect_and_catalog_events(
                                df, auto_save=False
                            )
                            st.session_state["rain_events_catalog"] = new_events
                            events_df = new_events
                            logger.info(
                                f"Catalog regenerated: {len(new_events)} events cached in session state"
                            )

                        st.cache_data.clear()
                        logger.info("Cleared streamlit data cache")

                        st.success(
                            f"✅ Regenerated {len(new_events)} events! Data updated in current view."
                        )

                st.divider()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", f"{len(events_df)}")
                with col2:
                    total_rain = events_df["total_rainfall"].sum()
                    st.metric("Total Rainfall", f'{total_rain:.1f}"')
                with col3:
                    avg_duration = events_df["duration_minutes"].mean()
                    st.metric("Avg Duration", f"{avg_duration/60:.1f}h")
                with col4:
                    excellent_pct = (
                        events_df["quality_rating"] == "excellent"
                    ).mean() * 100
                    st.metric("Data Quality", f"{excellent_pct:.0f}% excellent")

                st.write("**Select a rain event to analyze:**")

                events_df["event_label"] = events_df.apply(
                    lambda row: f"{pd.to_datetime(row['start_time']).tz_convert('America/Los_Angeles').strftime('%Y-%m-%d %H:%M')} - "
                    f"{row['total_rainfall']:.2f}\" in {row['duration_minutes']/60:.1f}h "
                    f"({row['quality_rating']})",
                    axis=1,
                )

                events_df = events_df.sort_values("start_time", ascending=False)

                if len(events_df) > 0:
                    selected_event_idx = st.selectbox(
                        "Choose event:",
                        range(len(events_df)),
                        format_func=lambda x: events_df.iloc[x]["event_label"],
                        help="Events sorted by most recent first",
                    )

                    selected_event = events_df.iloc[selected_event_idx]

                    st.write("**Event Details:**")

                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        start_time = pd.to_datetime(
                            selected_event["start_time"]
                        ).tz_convert("America/Los_Angeles")
                        end_time = pd.to_datetime(
                            selected_event["end_time"]
                        ).tz_convert("America/Los_Angeles")
                        st.write(
                            f"• **Start**: {start_time.strftime('%Y-%m-%d %H:%M %Z')}"
                        )
                        st.write(f"• **End**: {end_time.strftime('%Y-%m-%d %H:%M %Z')}")
                        st.write(
                            f"• **Duration**: {selected_event['duration_minutes']/60:.1f} hours"
                        )
                        st.write(
                            f"• **Total Rainfall**: {selected_event['total_rainfall']:.3f} inches"
                        )

                    with detail_col2:
                        st.write(
                            f"• **Data Quality**: {selected_event['quality_rating'].title()}"
                        )
                        st.write(
                            f"• **Completeness**: {selected_event['data_completeness']*100:.1f}%"
                        )
                        st.write(
                            f"• **Max Gap**: {selected_event['max_gap_minutes']:.1f} minutes"
                        )
                        st.write(
                            f"• **Max Rate**: {selected_event['max_hourly_rate']:.3f} in/hr"
                        )

                    if selected_event.get("flags"):
                        flags = selected_event["flags"]
                        if isinstance(flags, str):
                            import json

                            flags = json.loads(flags)

                        flag_indicators = []
                        if flags.get("ongoing"):
                            flag_indicators.append("🔄 Ongoing")
                        if flags.get("interrupted"):
                            flag_indicators.append("⚠️ Interrupted")
                        if flags.get("has_gaps"):
                            flag_indicators.append("🕳️ Has gaps")
                        if flags.get("low_completeness"):
                            flag_indicators.append("📉 Low completeness")

                        if flag_indicators:
                            st.write(f"**Flags**: {' '.join(flag_indicators)}")

                    st.info(
                        "🚧 Event data visualization coming next - this will show the intensity-duration curve for the selected event"
                    )
            else:
                st.info("No events found in catalog")
        else:
            st.error("Device not found in session state. Please refresh the page.")

    except ImportError as e:
        st.error(f"Event catalog feature not available: {e}")
    except Exception as e:
        st.error(f"Error loading event catalog: {e}")

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
