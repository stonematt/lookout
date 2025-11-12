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

    # 1. Archive Status Summary
    history_age_h = lo_dp.get_human_readable_duration(
        device_last_dateutc, history_max_dateutc
    )
    st.caption(
        f"📅 Archive current to: {history_df.date.max()} -- 🕐 Archive lag: {history_age_h}"
    )

    # 2. Gap Analysis
    st.subheader("Gap Analysis")

    threshold = st.slider(
        "Minimum gap size (minutes)", min_value=5, max_value=2000, value=30, step=15
    )

    col1, col2 = st.columns(2)
    with col1:
        window_value = st.number_input("Time range value", min_value=1, value=12)
    with col2:
        window_unit = st.selectbox("Time range unit", ["days", "weeks"], index=1)

    window_kwargs = {window_unit: window_value}
    cutoff = pd.Timestamp.now(tz=history_df["date"].dt.tz) - pd.Timedelta(
        **window_kwargs
    )
    filtered_df = history_df[history_df["date"] >= cutoff]
    gaps_df = detect_gaps(filtered_df, threshold_minutes=threshold)

    skipped_gaps = st.session_state.get("skipped_gaps", [])
    skipped_set = set(
        (pd.to_datetime(g["start"], utc=True), pd.to_datetime(g["end"], utc=True))
        for g in skipped_gaps
    )

    styled_gaps = gaps_df.copy()
    styled_gaps["start_str"] = styled_gaps["start"].dt.strftime("%Y-%m-%d %H:%M")
    styled_gaps["end_str"] = styled_gaps["end"].dt.strftime("%Y-%m-%d %H:%M")
    styled_gaps["duration_minutes"] = styled_gaps["duration_minutes"].round(1)
    styled_gaps["processed"] = styled_gaps.apply(
        lambda row: (row["start"], row["end"]) in skipped_set, axis=1
    )

    col1, col2 = st.columns(2)
    col1.metric("Gaps", f"{len(gaps_df)}")
    if not gaps_df.empty:
        human_gap = get_human_readable_duration(
            0, -gaps_df["duration_minutes"].max() * 60 * 1000
        )
        col2.metric("Longest Gap", human_gap)
    else:
        col2.metric("Longest Gap", "—")

    with st.expander("View Gap Details"):
        if gaps_df.empty:
            st.success(
                f"No gaps over {threshold} min in the last {window_value} {window_unit}."
            )
        else:
            st.write(f"Detected {len(gaps_df)} gaps > {threshold} min.")

            styled_display_df = styled_gaps[
                ["start_str", "end_str", "duration_minutes"]
            ].copy()
            styled_display_df.rename(
                columns={
                    "start_str": "Gap Start",
                    "end_str": "Gap End",
                    "duration_minutes": "Minutes",
                },
                inplace=True,
            )
            st.dataframe(styled_display_df, width='stretch')

            available_gaps = styled_gaps[~styled_gaps["processed"]]
            if not available_gaps.empty:
                selected_idx = st.selectbox(
                    "Select a gap to fill",
                    options=available_gaps.index,
                    format_func=lambda i: (
                        f"{available_gaps.loc[i, 'start_str']} → {available_gaps.loc[i, 'end_str']} "
                        f"({available_gaps.loc[i, 'duration_minutes']:.1f} min)"
                    ),
                )
            else:
                selected_idx = None

            col1, col2, _ = st.columns([1, 1, 4])
            with col1:
                fill_clicked = st.button("Fill Selected Gap")
            with col2:
                save_clicked = st.button("💾 Save Archive")

            if fill_clicked and selected_idx is not None:
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

    # 3. Data Coverage Heatmap
    st.subheader("Data Coverage")
    display_hourly_coverage_heatmap(df=filtered_df.copy())

    # 4. Device vs Archive
    st.subheader("Device vs Archive Check")
    st.json({"device_last_dateutc": "—", "archive_max_dateutc": "—"})

    # 5. Anomaly Check
    st.subheader("Anomaly Scan")
    st.warning("Sensor range and duplicate checks will appear here.")

    # 6. Raw Gap Placeholder
    with st.expander("Raw Gap List (Coming Soon)"):
        st.dataframe(pd.DataFrame([], columns=["Start", "End", "Duration"]))

    # 7. Skipped Gap Review
    skipped_gaps = st.session_state.get("skipped_gaps", [])
    if skipped_gaps:
        st.error("🚫 Skipped Gaps (No Data):")
        st.json(skipped_gaps)

    # 8. Final Data Row
    st.subheader("Last Data Record")
    st.write(last_data)
