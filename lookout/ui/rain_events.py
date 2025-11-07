"""
Rain Event Catalog UI for Lookout weather station dashboard.

This module provides event browsing, selection, and management interface
for rain events detected from historical weather data.
"""

import pandas as pd
import streamlit as st

from lookout.core.rain_events import RainEventCatalog
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


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
                    end_time = pd.to_datetime(selected_event["end_time"]).tz_convert(
                        "America/Los_Angeles"
                    )
                    st.write(f"• **Start**: {start_time.strftime('%Y-%m-%d %H:%M %Z')}")
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
