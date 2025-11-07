"""
Rain Event Catalog UI for Lookout weather station dashboard.

This module provides event browsing, selection, and management interface
for rain events detected from historical weather data.
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lookout.core.rain_events import RainEventCatalog
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def create_event_histogram(events_df: pd.DataFrame, selected_range: tuple) -> go.Figure:
    """
    Create histogram showing event count over time with selected range highlighted.

    :param events_df: DataFrame with event data
    :param selected_range: Tuple of (start_date, end_date) for highlighting
    :return: Plotly figure
    """
    events_by_week = (
        events_df.set_index("start_time").resample("W").size().reset_index(name="count")
    )

    fig = go.Figure()

    start_date, end_date = selected_range
    start_ts = (
        pd.Timestamp(start_date).tz_localize("America/Los_Angeles").tz_convert("UTC")
    )
    end_ts = (
        (pd.Timestamp(end_date) + pd.Timedelta(days=1))
        .tz_localize("America/Los_Angeles")
        .tz_convert("UTC")
    )

    def get_bar_color(date):
        date_ts = (
            pd.Timestamp(date).tz_localize("UTC")
            if pd.Timestamp(date).tz is None
            else pd.Timestamp(date)
        )
        return "steelblue" if start_ts <= date_ts < end_ts else "lightgray"

    colors = [get_bar_color(date) for date in events_by_week["start_time"]]

    fig.add_trace(
        go.Bar(
            x=events_by_week["start_time"],
            y=events_by_week["count"],
            marker_color=colors,
            hovertemplate="<b>Week of %{x|%Y-%m-%d}</b><br>Events: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Events by Week",
        xaxis_title="",
        yaxis_title="Event Count",
        height=200,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


def render_event_visualization(selected_event: pd.Series, archive_df: pd.DataFrame):
    """
    Render dense, minimalistic visualization for selected rain event.

    :param selected_event: Event row from catalog DataFrame
    :param archive_df: Full weather archive (unsorted)
    """
    import json

    from lookout.core.visualization import (
        create_event_accumulation_chart,
        create_event_rate_chart,
    )

    archive_df = archive_df.copy()
    archive_df["timestamp"] = pd.to_datetime(archive_df["dateutc"], unit="ms", utc=True)
    start_time = pd.to_datetime(selected_event["start_time"], utc=True)
    end_time = pd.to_datetime(selected_event["end_time"], utc=True)

    mask = (archive_df["timestamp"] >= start_time) & (
        archive_df["timestamp"] <= end_time
    )
    event_data = archive_df[mask].sort_values("timestamp").copy()

    if len(event_data) == 0:
        st.error(f"No data found for event time range {start_time} to {end_time}")
        return

    logger.debug(f"Extracted {len(event_data)} records for event")

    start_pst = pd.to_datetime(selected_event["start_time"]).tz_convert(
        "America/Los_Angeles"
    )
    end_pst = pd.to_datetime(selected_event["end_time"]).tz_convert(
        "America/Los_Angeles"
    )

    start_str = start_pst.strftime("%b %-d")
    if end_pst.date() != start_pst.date():
        end_str = end_pst.strftime("%-d, %Y")
    else:
        end_str = end_pst.strftime("%-I:%M %p").lower().lstrip("0")

    duration_h = selected_event["duration_minutes"] / 60
    total_rain = selected_event["total_rainfall"]
    peak_rate = selected_event["max_hourly_rate"]
    quality = selected_event["quality_rating"].title()

    flag_str = ""
    if selected_event.get("flags"):
        flags = (
            selected_event["flags"]
            if isinstance(selected_event["flags"], dict)
            else json.loads(selected_event["flags"])
        )
        if flags.get("ongoing"):
            flag_str = " • 🔄"
        elif flags.get("interrupted"):
            flag_str = " • ⚠️"

    if duration_h >= 48:
        duration_str = f"{duration_h/24:.1f}d"
    else:
        duration_str = f"{duration_h:.1f}h"

    header = f'{start_str}-{end_str} • {duration_str} • {total_rain:.3f}" • {peak_rate:.3f} in/hr • {quality}{flag_str}'
    st.markdown(f"**{header}**")

    event_info = {
        "total_rainfall": total_rain,
        "duration_minutes": selected_event["duration_minutes"],
        "start_time": start_time,
        "end_time": end_time,
    }

    acc_fig = create_event_accumulation_chart(event_data, event_info)
    st.plotly_chart(acc_fig, use_container_width=True, key=f"acc_{selected_event.name}")

    rate_fig = create_event_rate_chart(event_data)
    st.plotly_chart(
        rate_fig, use_container_width=True, key=f"rate_{selected_event.name}"
    )

    completeness = selected_event["data_completeness"] * 100
    gap = selected_event["max_gap_minutes"]
    gap_str = "No gaps" if gap == 0 else f"{gap:.0f}min gap"

    with st.expander("📊 Data quality", expanded=False):
        st.caption(f"{quality} • {completeness:.0f}% • {gap_str}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Completeness", f"{completeness:.1f}%")
            st.metric("Max Gap", f"{gap:.1f} min")
        with col2:
            st.metric("Quality", quality)
            st.metric(
                "Data Points",
                int(selected_event.get("data_point_count", len(event_data))),
            )


def render():
    """Render the rain events catalog tab."""
    st.header("Rain Event Catalog")
    st.write("Browse and analyze individual rain events detected from historical data")

    if "history_df" not in st.session_state:
        st.error("No weather data available in session")
        return

    if "device" not in st.session_state:
        st.error("Device not found in session state. Please refresh the page.")
        return

    df = st.session_state["history_df"]
    device = st.session_state["device"]
    device_mac = device["macAddress"]
    file_type = "parquet"

    try:
        catalog = RainEventCatalog(device_mac, file_type)

        events_df = None

        if "rain_events_catalog" in st.session_state:
            events_df = st.session_state["rain_events_catalog"]
            catalog_source = "session"
            logger.info(f"Using catalog from session state: {len(events_df)} events")

        elif catalog.catalog_exists():
            with st.spinner("Loading event catalog from storage..."):
                logger.info(f"Loading catalog from Storj: {catalog.catalog_path}")
                stored_catalog = catalog.load_catalog()
                if not stored_catalog.empty:
                    logger.info(
                        f"Loaded {len(stored_catalog)} events from storage, checking for updates..."
                    )
                    events_df = catalog.update_catalog_with_new_data(df, stored_catalog)
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

            min_date = events_df["start_time"].min().date()
            max_date = events_df["start_time"].max().date()

            st.write("**Date Range:**")

            date_range = st.slider(
                "Select date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="MMM DD, YYYY",
                label_visibility="collapsed",
            )

            histogram_fig = create_event_histogram(events_df, date_range)
            st.plotly_chart(
                histogram_fig, use_container_width=True, key="date_histogram"
            )

            with st.expander("Advanced Filter"):
                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    st.write("**Minimum Rainfall:**")
                    max_rainfall = float(events_df["total_rainfall"].max())
                    min_rainfall_threshold = st.slider(
                        "Minimum rainfall (inches)",
                        min_value=0.0,
                        max_value=max_rainfall,
                        value=0.0,
                        step=0.01,
                        format="%.2f",
                        label_visibility="collapsed",
                    )
                    st.caption(f"≥ {min_rainfall_threshold:.2f} inches")

                with filter_col2:
                    st.write("**Data Quality:**")
                    quality_options = ["excellent", "good", "fair", "poor"]
                    selected_quality = st.multiselect(
                        "Select quality ratings",
                        options=quality_options,
                        default=quality_options,
                        label_visibility="collapsed",
                    )

            filtered_events_df = events_df.copy()

            if len(date_range) == 2:
                start_date, end_date = date_range
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
                filtered_events_df = filtered_events_df[
                    (filtered_events_df["start_time"] >= start_ts)
                    & (filtered_events_df["start_time"] < end_ts)
                ]

            if selected_quality:
                filtered_events_df = filtered_events_df[
                    filtered_events_df["quality_rating"].isin(selected_quality)
                ]

            filtered_events_df = filtered_events_df[
                filtered_events_df["total_rainfall"] >= min_rainfall_threshold
            ]

            logger.info(
                f"Filtered events: {len(filtered_events_df)} of {len(events_df)} "
                f"(date: {date_range}, quality: {selected_quality}, min_rain: {min_rainfall_threshold})"
            )

            st.divider()

            if len(filtered_events_df) < len(events_df):
                st.info(
                    f"📊 Showing {len(filtered_events_df)} of {len(events_df)} events "
                    f"({len(events_df) - len(filtered_events_df)} filtered out)"
                )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", f"{len(filtered_events_df)}")
            with col2:
                total_rain = filtered_events_df["total_rainfall"].sum()
                st.metric("Total Rainfall", f'{total_rain:.1f}"')
            with col3:
                if len(filtered_events_df) > 0:
                    med_rainfall = filtered_events_df["total_rainfall"].median()
                    st.metric("Median Rainfall", f'{med_rainfall:.2f}"')
                else:
                    st.metric("Median Rainfall", "—")
            with col4:
                if len(filtered_events_df) > 0:
                    avg_duration = filtered_events_df["duration_minutes"].mean()
                    st.metric("Avg Duration", f"{avg_duration/60:.1f}h")
                else:
                    st.metric("Avg Duration", "—")

            if len(filtered_events_df) > 0:
                st.write("**Select a rain event to analyze:**")

                display_df = filtered_events_df.copy()
                display_df = display_df.sort_values("start_time", ascending=False)
                display_df["Date"] = (
                    display_df["start_time"]
                    .dt.tz_convert("America/Los_Angeles")
                    .dt.strftime("%Y-%m-%d %H:%M")
                )
                display_df["Duration (h)"] = (
                    display_df["duration_minutes"] / 60
                ).round(1)
                display_df["Rainfall (in)"] = display_df["total_rainfall"].round(2)
                display_df["Max Rate (in/hr)"] = display_df["max_hourly_rate"].round(3)
                display_df["Quality"] = display_df["quality_rating"].str.title()

                event_selection = st.dataframe(
                    display_df[
                        [
                            "Date",
                            "Duration (h)",
                            "Rainfall (in)",
                            "Max Rate (in/hr)",
                            "Quality",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    height=400,
                )

                if event_selection["selection"]["rows"]:
                    selected_idx = event_selection["selection"]["rows"][0]
                    selected_event = display_df.iloc[selected_idx]
                else:
                    selected_event = None

                if selected_event is not None:
                    render_event_visualization(selected_event, df)
            else:
                st.warning(
                    "No events match the selected filters. Try adjusting your criteria."
                )

            st.divider()

            st.write("**Catalog Management:**")
            btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 2])
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
        else:
            st.info("No events found in catalog")

    except ImportError as e:
        st.error(f"Event catalog feature not available: {e}")
    except Exception as e:
        st.error(f"Error loading event catalog: {e}")
