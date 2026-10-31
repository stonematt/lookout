import gc
import inspect
import sys

import pandas as pd
import streamlit as st

import lookout.core.data_processing as lo_dp
import lookout.ui.rain as rain_module
from lookout.config import DEFAULT_BUCKET
from lookout.api.awn_controller import (
    GapFill,
    combine_full_history,
    extend_corrected,
    fill_archive_gap,
    load_archive_for_device,
)
from lookout.core.data_processing import detect_gaps, get_human_readable_duration
from lookout.core.visualization import display_hourly_coverage_heatmap
from lookout.storage.storj import backup_and_save_history
from lookout.utils.log_util import app_logger
from lookout.core.energy_catalog_func import (
    detect_and_calculate_periods,
    save_energy_catalog,
)
from lookout.utils.memory_utils import (
    BYTES_TO_MB,
    MEMORY_UNAVAILABLE,
    force_garbage_collection,
    get_df_memory_usage,
    get_memory_usage,
    get_object_counts,
    get_object_memory_usage,
    log_memory_usage,
)

logger = app_logger(__name__)

# The raw archive the gap fills extend; what "Save Archive" writes (#95).
RAW_HISTORY_KEY = "raw_history_df"
BUCKET = DEFAULT_BUCKET


def fill_gap(start: pd.Timestamp, end: pd.Timestamp) -> GapFill:
    """
    Fill one gap in the raw archive and show it in the session history.

    The session's `history_df` has its rain corrected for display, so it is
    never what gets saved. The fill extends the raw archive instead (loaded
    from storage on the first fill, then kept in the session), and only the
    rows it added are corrected into `history_df`.

    :param start: The gap's start, UTC.
    :param end: The gap's end, UTC.
    :return: The fill's outcome.
    """
    device = st.session_state["device"]
    raw = st.session_state.get(RAW_HISTORY_KEY)
    if raw is None:
        raw = load_archive_for_device(device, BUCKET, "parquet")
    outcome = fill_archive_gap(device, raw, start, end)
    if outcome.status == "filled":
        st.session_state[RAW_HISTORY_KEY] = outcome.history_df
        st.session_state["history_df"] = extend_corrected(
            st.session_state["history_df"],
            outcome.history_df,
            device["macAddress"],
            BUCKET,
        )
    return outcome


def save_filled_archive() -> bool:
    """
    Back up the stored archive and save it with this session's fills added.

    The fills are merged into the archive as stored now, not as loaded at
    the first fill, so rows catchup saved in between are kept. All raw.

    :return: True when saved; False when no gap was filled this session, so
        there is nothing new to save.
    """
    raw = st.session_state.get(RAW_HISTORY_KEY)
    if raw is None:
        return False
    device = st.session_state["device"]
    stored = load_archive_for_device(device, BUCKET, "parquet")
    backup_and_save_history(
        df=combine_full_history(stored, raw), device=device, bucket=BUCKET
    )
    return True


def analyze_cache_usage():
    """Analyze Streamlit cache usage patterns."""
    try:
        # Get all objects before and after cache operations
        before_objects = len(gc.get_objects())

        # Try to trigger cache usage
        cache_info = {
            "gc_objects_before": before_objects,
            "cache_functions": [],
            "memory_estimate": 0,
        }

        # Check for cached functions

        for name in dir(rain_module):
            obj = getattr(rain_module, name)
            if hasattr(obj, "_is_cache"):
                cache_info["cache_functions"].append(name)

        gc.collect()
        after_objects = len(gc.get_objects())
        cache_info["gc_objects_after"] = after_objects
        cache_info["object_delta"] = after_objects - before_objects

        return cache_info

    except Exception as e:
        return {"error": str(e)}


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
            st.dataframe(styled_display_df, width="stretch")

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

                outcome = fill_gap(start_ts, end_ts)
                if outcome.skipped:
                    # The tab keeps the skipped-gaps record (it is session
                    # state); the core function no longer does.
                    st.session_state.setdefault("skipped_gaps", []).append(
                        {"start": outcome.start, "end": outcome.end}
                    )
                if outcome.status == "filled":
                    st.success("Gap filled and archive updated.")
                elif outcome.skipped:
                    st.warning("No new data for that gap; marked as skipped.")
                else:
                    st.error("Gap refused: inverted, or no archive data before it.")
                # st.rerun()

            if save_clicked:
                if save_filled_archive():
                    st.success("Archive saved to Storj.")
                else:
                    st.info("No gap filled yet; the stored archive is unchanged.")

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

    # 8. Memory Usage Analysis
    st.subheader("Memory Usage Analysis")

    # Get memory stats
    memory_mb = get_memory_usage()
    if memory_mb == MEMORY_UNAVAILABLE:
        st.info("Install psutil (`pip install psutil`) for detailed memory monitoring")
        return

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Process Memory", f"{memory_mb:.1f} MB")
        with col2:
            st.metric("Session Counter", st.session_state.get("session_counter", 0))
        with col3:
            gc.collect()
            st.metric("After GC", f"{get_memory_usage():.1f} MB")

        # Detailed memory analysis
        st.write("**Detailed Memory Analysis:**")

        # Analyze all session state objects
        session_memory = {}
        total_session_memory = 0

        for key, value in st.session_state.items():
            try:
                if hasattr(value, "__sizeof__"):
                    size_mb = get_object_memory_usage(value)
                    session_memory[key] = size_mb
                    total_session_memory += size_mb

                    # Special handling for DataFrames
                    if hasattr(value, "shape"):
                        st.write(f"**{key}**: {size_mb:.1f}MB, shape: {value.shape}")
                    else:
                        st.write(f"**{key}**: {size_mb:.1f}MB")
            except Exception as e:
                st.write(f"**{key}**: Unable to measure ({type(value).__name__})")

        st.write(f"**Total Session State Memory**: {total_session_memory:.1f}MB")

        # DataFrame specific analysis
        st.write("**DataFrame Memory Usage:**")
        df_info = []

        if "history_df" in st.session_state:
            df = st.session_state["history_df"]
            df_size = sys.getsizeof(df) / 1024 / 1024
            # More accurate DataFrame memory usage
            df_memory = get_df_memory_usage(df)
            df_info.append(
                {
                    "DataFrame": "history_df",
                    "Rows": len(df),
                    "Size_MB": f"{df_memory:.1f}",
                    "Columns": len(df.columns),
                }
            )

        if "rain_events_catalog" in st.session_state:
            df = st.session_state["rain_events_catalog"]
            df_memory = get_df_memory_usage(df)
            df_info.append(
                {
                    "DataFrame": "rain_events_catalog",
                    "Rows": len(df),
                    "Size_MB": f"{df_memory:.1f}",
                    "Columns": len(df.columns),
                }
            )

        if df_info:
            st.dataframe(pd.DataFrame(df_info), width="stretch", hide_index=True)

        # Memory gap analysis
        process_memory = memory_mb
        accounted_memory = total_session_memory
        memory_gap = process_memory - accounted_memory

        st.write("**Memory Gap Analysis:**")
        st.write(f"Process Memory: {process_memory:.1f}MB")
        st.write(f"Session State: {accounted_memory:.1f}MB")
        st.write(
            f"Unaccounted: {memory_gap:.1f}MB ({memory_gap/process_memory*100:.1f}%)"
        )

        if memory_gap > 100:
            st.warning(
                f"Large memory gap ({memory_gap:.1f}MB) suggests cache or other objects consuming memory"
            )
        elif memory_gap > 300:
            st.error(
                f"Very large memory gap ({memory_gap:.1f}MB) indicates significant memory leak in cache or other components"
            )

        # JSON export for quick sharing
        st.write("**Memory Analysis JSON:**")

        # Build comprehensive memory dict
        memory_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "process_memory_mb": round(process_memory, 1),
            "session_counter": st.session_state.get("session_counter", 0),
            "session_state": {},
            "dataframes": {},
            "memory_gap": {
                "total_mb": round(memory_gap, 1),
                "percentage": round(memory_gap / process_memory * 100, 1),
            },
        }

        # Add session state details
        for key, value in st.session_state.items():
            try:
                if hasattr(value, "__sizeof__"):
                    size_mb = round(get_object_memory_usage(value), 2)
                    memory_data["session_state"][key] = {
                        "size_mb": size_mb,
                        "type": type(value).__name__,
                    }
                    if hasattr(value, "shape"):
                        memory_data["session_state"][key]["shape"] = str(value.shape)
            except:
                memory_data["session_state"][key] = {
                    "size_mb": "unknown",
                    "type": type(value).__name__,
                }

        # Add DataFrame details
        if "history_df" in st.session_state:
            df = st.session_state["history_df"]
            df_memory = get_df_memory_usage(df)
            memory_data["dataframes"]["history_df"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": round(df_memory, 1),
                "date_range": {
                    "start": (
                        df["date"].min().isoformat() if "date" in df.columns else None
                    ),
                    "end": (
                        df["date"].max().isoformat() if "date" in df.columns else None
                    ),
                },
            }

        if "rain_events_catalog" in st.session_state:
            df = st.session_state["rain_events_catalog"]
            df_memory = get_df_memory_usage(df)
            memory_data["dataframes"]["rain_events_catalog"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": round(df_memory, 1),
            }

        # Add tab memory history
        if "tab_memory_history" in st.session_state:
            memory_data["tab_memory_history"] = st.session_state["tab_memory_history"][
                -10:
            ]  # Last 10 entries

        # Display JSON with copy button
        json_str = str(memory_data).replace("'", '"')
        st.code(json_str, language="json")

        if st.button("📋 Copy Memory JSON"):
            st.write("JSON copied to clipboard (use browser copy)")
            st.json(memory_data)

        # Tab navigation memory tracking
        st.write("**Tab Memory Tracking:**")
        st.info("Visit different tabs and return here to see memory changes")

        if "tab_memory_history" not in st.session_state:
            st.session_state["tab_memory_history"] = []

        current_memory = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "process_memory_mb": round(process_memory, 1),
            "session_counter": st.session_state.get("session_counter", 0),
            "memory_gap_mb": round(memory_gap, 1),
            "gap_percentage": round(memory_gap / process_memory * 100, 1),
        }

        # Add to history
        st.session_state["tab_memory_history"].append(current_memory)

        # Log memory snapshot with GC analysis (DEBUG level only)
        gc_objects = force_garbage_collection()

        logger.debug(
            f"MEMORY_SNAPSHOT: {current_memory['process_memory_mb']:.1f}MB "
            f"(gap: {current_memory['memory_gap_mb']:.1f}MB, "
            f"{current_memory['gap_percentage']:.1f}%), "
            f"session_counter: {current_memory['session_counter']}, "
            f"gc_objects: {gc_objects}"
        )

        # Show last 5 measurements
        recent_history = st.session_state["tab_memory_history"][-5:]
        if len(recent_history) > 1:
            history_df = pd.DataFrame(recent_history)
            st.dataframe(history_df, width="stretch", hide_index=True)

            # Show memory trend
            if len(recent_history) >= 2:
                memory_change = (
                    recent_history[-1]["process_memory_mb"]
                    - recent_history[-2]["process_memory_mb"]
                )
                if memory_change > 50:
                    st.error(
                        f"⚠️ Memory increased by {memory_change:.1f}MB since last check"
                    )
                elif memory_change < -50:
                    st.success(
                        f"✅ Memory decreased by {abs(memory_change):.1f}MB since last check"
                    )

        # Cache stats and management
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Cache"):
                # Selective cache clearing to avoid UI disruption
                try:
                    cleared_count = 0
                    if hasattr(rain_module, "_cached_rolling_context"):
                        rain_module._cached_rolling_context.clear()
                        cleared_count += 1
                    gc.collect()
                    st.success(f"Selective cache cleared ({cleared_count} functions)!")
                except Exception as e:
                    st.warning(f"Cache clearing failed: {e}")

        with col2:
            if st.button("🔄 Force GC"):
                gc.collect()
                st.success("Garbage collection completed!")
                # st.rerun()

        with col3:
            if st.button("🔍 Cache Analysis"):
                # Force cache analysis
                memory_data["cache_analysis"] = analyze_cache_usage()
                st.success("Cache analysis completed!")

        st.write("**Cache Investigation:**")
        try:
            # Try to get cache info

            # Check if we can access cache internals
            cache_info = {
                "cache_data_available": hasattr(st, "cache_data"),
                "cache_resource_available": hasattr(st, "cache_resource"),
                "streamlit_version": (
                    st.__version__ if hasattr(st, "__version__") else "unknown"
                ),
            }

            st.json(cache_info)

            # Try to estimate cache size by calling cached functions
            if "history_df" in st.session_state:
                df = st.session_state["history_df"]
                st.write("**Testing cache memory impact:**")

                # Test rolling context cache
                try:
                    from lookout.ui.rain import _cached_rolling_context

                    with st.spinner("Testing cache..."):
                        result = _cached_rolling_context(
                            df,
                            [7, 30, 90],
                            [2023, 2024],
                            pd.Timestamp.now().date(),
                            "test",
                        )
                        cache_size = get_object_memory_usage(result)
                        st.write(f"Rolling context cache: {cache_size:.1f}MB")
                except Exception as e:
                    st.write(f"Cache test failed: {e}")

        except Exception as e:
            st.write(f"Cache investigation failed: {e}")

        # Memory trend warning
        if memory_mb > 500:  # Warning threshold
            st.warning(
                f"⚠️ High memory usage: {memory_mb:.1f} MB. Consider clearing cache or restarting."
            )
        elif memory_mb > 800:  # Critical threshold
            st.error(
                f"🚨 Critical memory usage: {memory_mb:.1f} MB. Memory leak likely in progress."
            )

    # 9. Solar Energy Catalog Management
    st.subheader("Solar Energy Catalog Management")

    if "energy_catalog" in st.session_state:
        catalog_df = st.session_state["energy_catalog"]
        if not catalog_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Energy Periods", f"{len(catalog_df)}")
            with col2:
                if not catalog_df.empty:
                    date_range = f"{catalog_df['period_start'].min().date()} → {catalog_df['period_start'].max().date()}"
                    st.metric("Date Range", date_range)
            with col3:
                total_energy = catalog_df["energy_kwh"].sum()
                st.metric("Total Energy", f"{total_energy:.1f} kWh")

            # Management controls
            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "🔄 Regenerate Energy Catalog",
                    help="Recalculate all energy periods from scratch",
                ):
                    with st.spinner("Regenerating energy catalog..."):
                        from lookout.core.energy_catalog import EnergyCatalog

                        catalog = EnergyCatalog(
                            st.session_state["device"]["macAddress"]
                        )
                        periods_df = detect_and_calculate_periods(
                            st.session_state["history_df"], auto_save=False
                        )
                        st.session_state["energy_catalog"] = periods_df
                        st.success(f"✓ Regenerated {len(periods_df)} energy periods")
                        st.rerun()

            with col2:
                if st.button(
                    "💾 Save Energy Catalog", help="Save current catalog to storage"
                ):
                    with st.spinner("Saving energy catalog..."):
                        from lookout.core.energy_catalog_func import save_energy_catalog

                        save_energy_catalog(st.session_state["energy_catalog"])
                        st.success("✓ Energy catalog saved to storage")

            # Catalog details
            with st.expander("📊 Catalog Details", expanded=False):
                st.write("**Recent Periods:**")
                st.dataframe(catalog_df.tail(10), width="stretch")
        else:
            st.warning("Energy catalog is empty")
    else:
        st.info("Energy catalog not loaded yet")

    # 10. Final Data Row
    st.subheader("Last Data Record")
    st.write(last_data)
