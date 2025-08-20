import pandas as pd
import streamlit as st

import lookout.core.data_processing as lo_dp
from lookout.api.awn_controller import fill_archive_gap
from lookout.core.data_processing import detect_gaps, get_human_readable_duration
from lookout.core.visualization import display_hourly_coverage_heatmap
from lookout.storage.storj import backup_and_save_history


def render():
    """
    Render the Diagnostics tab for archive data review.
    """

    if "history_df" not in st.session_state:
        st.warning("No archive data loaded.")
        return

    device = st.session_state["device"]
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

            # Format gaps for display
            styled_gaps = gaps_df.copy()
            styled_gaps["start_str"] = styled_gaps["start"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            styled_gaps["end_str"] = styled_gaps["end"].dt.strftime("%Y-%m-%d %H:%M")
            styled_gaps["duration_minutes"] = styled_gaps["duration_minutes"].round(1)

            # Show as read-only table
            styled_display_df = (
                styled_gaps[["start_str", "end_str", "duration_minutes"]]
                .copy()
                .rename(
                    columns={  # explicitly cast to Dict[str, str] to satisfy Pyright
                        "start_str": "Gap Start",
                        "end_str": "Gap End",
                        "duration_minutes": "Minutes",
                    }  # type: Dict[str, str]
                )
            )

            st.dataframe(styled_display_df, use_container_width=True)

            # Gap selector
            selected_idx = st.selectbox(
                "Select a gap to fill",
                options=styled_gaps.index,
                format_func=lambda i: (
                    f"{styled_gaps.loc[i, 'start_str']} → {styled_gaps.loc[i, 'end_str']} "
                    f"({styled_gaps.loc[i, 'duration_minutes']:.1f} min)"
                ),
            )

            col1, col2, _ = st.columns([1, 1, 4])  # 2 narrow columns, 1 spacer
            with col1:
                fill_clicked = st.button("Fill Selected Gap")
            with col2:
                save_clicked = st.button("💾 Save Archive")

            if fill_clicked:
                row = gaps_df.loc[selected_idx]
                start_ts = pd.to_datetime(row["start"])
                end_ts = pd.to_datetime(row["end"])
                st.info(f"Filling gap: {start_ts} to {end_ts}")

                updated_df = fill_archive_gap(
                    st.session_state["device"],
                    st.session_state["history_df"],
                    start_ts,
                    end_ts,
                )
                st.session_state["history_df"] = updated_df
                st.success("Gap filled and archive updated.")
                st.rerun()

            if save_clicked:
                backup_and_save_history(
                    df=st.session_state["history_df"],
                    device=st.session_state["device"],
                )
                st.success("Archive saved to Storj.")
    # --- 2. Gap Heatmap Placeholder ---
    #
    # --- Data Coverage Heatmap ---
    st.subheader("Data Coverage")

    display_hourly_coverage_heatmap(df=filtered_df.copy())

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
