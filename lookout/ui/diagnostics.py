import pandas as pd
import streamlit as st


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

    # --- 1. Status Summary ---
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "—")
    col2.metric("Total Gaps", "—")
    col3.metric("Longest Gap", "—")

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
        st.dataframe(pd.DataFrame(columns=["Start", "End", "Duration"]))

    st.subheader("Last Data Record")
    st.write(last_data)
