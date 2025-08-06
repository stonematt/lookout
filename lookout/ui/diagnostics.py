import pandas as pd
import streamlit as st

import lookout.core.data_processing as lo_dp
from lookout.core.data_processing import detect_gaps, get_human_readable_duration


def render():
    """
    Render the Diagnostics tab for archive data review.
    """

    if "history_df" not in st.session_state:
        st.warning("No archive data loaded.")
        return

    history_df = st.session_state["history_df"]
    last_data = st.session_state["last_data"]

    st.header("Archive Diagnostics")
    history_max_dateutc = st.session_state["history_max_dateutc"]
    device_last_dateutc = last_data.get("dateutc")

    # Inline archive status (already in sidebar too)
    history_age_h = lo_dp.get_human_readable_duration(
        device_last_dateutc, history_max_dateutc
    )

    st.caption(
        f"📅 Archive current to: {history_df.date.max()} -- 🕐 Archive lag: {history_age_h}"
    )

    # --- 1. Status Summary ---
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    total_records = len(history_df)
    col1.metric("Total Records", f"{total_records:,}")

    # Gap analysis
    gaps_df = detect_gaps(history_df)  # default is 10min gaps
    total_gaps = len(gaps_df)
    col2.metric("Total Gaps", f"{total_gaps:,}")
    if not gaps_df.empty:
        longest_gap_min = gaps_df["duration_minutes"].max()
        # Convert minutes to ms for compatibility with the utility
        dummy_now = 0
        dummy_then = -longest_gap_min * 60 * 1000
        human_gap = get_human_readable_duration(dummy_now, dummy_then)

        col3.metric("Longest Gap", human_gap)
    else:
        col3.metric("Longest Gap", "—")

    with st.expander("View Gap Details"):
        if gaps_df.empty:
            st.success("No significant gaps found.")
        else:
            st.write(f"Detected {len(gaps_df)} gaps over 10 minutes.")
            st.dataframe(
                gaps_df.style.format(
                    {
                        "start": lambda x: x.strftime("%Y-%m-%d %H:%M"),
                        "end": lambda x: x.strftime("%Y-%m-%d %H:%M"),
                        "duration_minutes": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

    st.markdown("---")

    # --- 2. Gap Heatmap Placeholder ---
    st.subheader("Data Coverage")
    st.info("Heatmap showing data presence by hour/day goes here.")
    st.empty()

    # --- 3. Recent Device Status ---
    st.subheader("Device vs Archive Check")
    st.json({"device_last_dateutc": "—", "archive_max_dateutc": "—"})

    # --- 4. Anomaly Check Placeholder ---
    st.subheader("Anomaly Scan")
    st.warning("Sensor range and duplicate checks will appear here.")

    # --- 5. Raw Gap View Placeholder ---
    with st.expander("Raw Gap List (Coming Soon)"):
        st.dataframe(pd.DataFrame([], columns=["Start", "End", "Duration"]))

    st.subheader("Last Data Record")
    st.write(last_data)
