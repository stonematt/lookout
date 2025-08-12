import pandas as pd
import streamlit as st

import lookout.core.data_processing as lo_dp
from lookout.core.data_processing import detect_gaps, get_human_readable_duration
from lookout.core.visualization import (
    display_data_coverage_heatmap,
    display_hourly_coverage_heatmap,
)


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

    # --- Status Summary ---
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)

    total_records = len(history_df)
    col1.metric("Total Records", f"{total_records:,}")

    # --- Gap Analysis ---
    st.subheader("Gap Analysis")

    # Threshold slider
    threshold = st.slider(
        "Minimum gap size (minutes)",
        min_value=5,
        max_value=2000,
        value=30,
        step=15,
    )

    # Time window selector
    col1, col2 = st.columns(2)
    with col1:
        window_value = st.number_input("Time range value", min_value=1, value=12)
    with col2:
        window_unit = st.selectbox("Time range unit", ["days", "weeks"], index=1)

    # Compute filter window
    window_kwargs = {window_unit: window_value}
    cutoff = pd.Timestamp.now(tz=history_df["date"].dt.tz) - pd.Timedelta(
        **window_kwargs
    )
    filtered_df = history_df[history_df["date"] >= cutoff]

    # Detect gaps with threshold
    gaps_df = detect_gaps(filtered_df, threshold_minutes=threshold)

    col1, col2 = st.columns(2)

    total_gaps = len(gaps_df)
    col1.metric("Gaps", f"{total_gaps}")
    if not gaps_df.empty:
        longest_gap_min = gaps_df["duration_minutes"].max()
        # Convert minutes to ms for compatibility with the utility
        dummy_now = 0
        dummy_then = -longest_gap_min * 60 * 1000
        human_gap = get_human_readable_duration(dummy_now, dummy_then)

        col2.metric("Longest Gap", human_gap)
    else:
        col2.metric("Longest Gap", "—")

    # Gap Summary + Table
    with st.expander("View Gap Details"):
        if gaps_df.empty:
            st.success(
                f"No gaps over {threshold} min in the last {window_value} {window_unit}."
            )
        else:
            st.write(f"Detected {len(gaps_df)} gaps > {threshold} min.")
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

    # --- 2. Gap Heatmap Placeholder ---
    #
    # --- Data Coverage Heatmap ---
    st.subheader("Data Coverage")

    display_hourly_coverage_heatmap(df=filtered_df.copy())

    # # Filtered data reused from Gap Analysis: `filtered_df`
    # coverage_df = filtered_df.copy()
    # coverage_df["date"] = pd.to_datetime(coverage_df["dateutc"], unit="ms")
    #
    # # Resample into daily/hourly intervals
    # coverage_df["interval"] = coverage_df["date"].dt.floor(f"{interval_minutes}min")
    # coverage_df["day"] = coverage_df["date"].dt.date
    #
    # heatmap_data = (
    #     coverage_df.groupby(["day", "interval"]).size().reset_index(name="count")
    # )
    #
    # # Pivot to matrix format
    # pivot = heatmap_data.pivot_table(
    #     index="day",
    #     columns=heatmap_data["interval"].dt.strftime("%H:%M"),
    #     values="count",
    #     aggfunc="sum",
    #     fill_value=0,
    # )
    #
    # # Plot heatmap
    # fig = px.imshow(
    #     pivot,
    #     labels=dict(x="Hour", y="Day", color="Samples"),
    #     aspect="auto",
    #     color_continuous_scale="Blues",
    # )
    # fig.update_layout(
    #     xaxis_title="Hour of Day",
    #     yaxis_title="Date",
    #     margin=dict(t=30, b=30),
    # )
    #
    # st.plotly_chart(fig, use_container_width=True)
    # st.subheader("Data Coverage")
    # st.info("Heatmap showing data presence by hour/day goes here.")
    # st.empty()

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
