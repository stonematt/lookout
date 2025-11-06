"""
Precipitation analysis module for Lookout weather station dashboard.

This module provides comprehensive rainfall data analysis including current accumulations,
historical statistics, and data processing functions for precipitation visualizations.
Handles daily rainfall extraction from accumulating fields and dry spell calculations.
"""

from typing import Iterable, List, Optional, Tuple, cast

# --- Rolling rainfall context (1d, 7d, 30d, 90d) ----------------------------
import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

import lookout.core.visualization as lo_viz
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def compute_rolling_rain_context(
    daily_rain_df: pd.DataFrame,
    windows: Iterable[int] = (1, 7, 30, 90),
    normals_years: Optional[Tuple[int, int]] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Rolling-window rainfall totals compared to ALL historical N-day periods.

    Compares current rainfall totals against the full distribution of all possible
    N-day periods in historical data, not just the same calendar dates.

    :param daily_rain_df: DataFrame with daily totals ['date', 'rainfall'].
    :param windows: Window lengths in days.
    :param normals_years: (start_year, end_year); if None, uses all years except current.
    :param end_date: Period end (defaults to latest 'date' in data).
    :return: DataFrame with totals, normal, anomaly %, rank, percentile for each window.
    """
    if daily_rain_df.empty:
        return pd.DataFrame(
            columns=[  # type: ignore
                "window_days",
                "period_start",
                "period_end",
                "total",
                "normal",
                "anomaly_pct",
                "rank",
                "percentile",
                "n_periods",
            ]
        )

    df = daily_rain_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    s = df.set_index("date")["rainfall"].sort_index()

    end_dt = (
        pd.to_datetime(end_date).normalize() if end_date is not None else s.index.max()
    )

    # Filter historical data (exclude current year unless specified)
    if normals_years is None:
        historical_data = s[s.index.year != end_dt.year]  # type: ignore
    else:
        y0, y1 = normals_years
        historical_data = s[
            (s.index.year >= y0) & (s.index.year <= y1) & (s.index.year != end_dt.year)  # type: ignore
        ]

    rows = []
    for w in windows:
        period_end = end_dt
        period_start = end_dt - pd.Timedelta(days=w - 1)  # type: ignore

        # Current period total
        cur = float(s.loc[(s.index >= period_start) & (s.index <= period_end)].sum())

        # Generate all possible N-day rolling totals from historical data
        if len(historical_data) >= w:
            all_rolling_totals = historical_data.rolling(window=w).sum().dropna()  # type: ignore
            normals = all_rolling_totals.values  # type: ignore
            normals = normals[np.isfinite(normals)]  # type: ignore
        else:
            normals = []

        n_periods = len(normals)
        normal_mean = float(np.mean(normals)) if n_periods else np.nan
        anomaly_pct = (
            (100.0 * (cur / normal_mean - 1.0))
            if (n_periods and normal_mean != 0)
            else np.nan
        )

        if n_periods:
            # Rank: 1 = wettest (highest total)
            rank = int(sum(n > cur for n in normals) + 1)
            percentile = 100.0 * (1.0 - rank / (n_periods + 1.0))
        else:
            rank, percentile = np.nan, np.nan

        rows.append(
            {
                "window_days": int(w),
                "period_start": period_start,
                "period_end": period_end,
                "total": round(cur, 3),
                "normal": round(normal_mean, 3) if np.isfinite(normal_mean) else np.nan,
                "anomaly_pct": (
                    round(anomaly_pct, 1) if np.isfinite(anomaly_pct) else np.nan
                ),
                "rank": rank,
                "percentile": (
                    round(percentile, 1) if np.isfinite(percentile) else np.nan
                ),
                "n_periods": n_periods,
            }
        )

    return pd.DataFrame(rows).sort_values("window_days").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _cached_rolling_context(
    daily_rain_df: pd.DataFrame, windows, normals_years, end_date, version: str = "v2"
):
    return compute_rolling_rain_context(daily_rain_df, windows, normals_years, end_date)


def render_rolling_rain_context_table(stats_df: pd.DataFrame, unit: str = "in") -> None:
    """
    Render rolling rainfall context as a compact table.
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
        # older Streamlit compatibility
        st.dataframe(view.set_index("Window"), use_container_width=True)


def prepare_violin_plot_data(
    daily_rain_df: pd.DataFrame,
    windows: Iterable[int] = (1, 7, 30, 90),
    normals_years: Optional[Tuple[int, int]] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Extract all N-day rolling totals and current values for violin plot visualization.

    :param daily_rain_df: DataFrame with daily totals ['date', 'rainfall'].
    :param windows: Window lengths in days.
    :param normals_years: (start_year, end_year); if None, uses all years except current.
    :param end_date: Period end (defaults to latest 'date' in data).
    :return: Dict with window -> {"values": array, "current": float, "percentile": float}
    """
    if daily_rain_df.empty:
        return {}

    df = daily_rain_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    s = df.set_index("date")["rainfall"].sort_index()

    end_dt = (
        pd.to_datetime(end_date).normalize() if end_date is not None else s.index.max()
    )

    # Filter historical data (exclude current year unless specified)
    if normals_years is None:
        historical_data = s[s.index.year != end_dt.year]  # type: ignore
    else:
        y0, y1 = normals_years
        historical_data = s[
            (s.index.year >= y0) & (s.index.year <= y1) & (s.index.year != end_dt.year)  # type: ignore
        ]

    violin_data = {}
    for w in windows:
        period_end = end_dt
        period_start = end_dt - pd.Timedelta(days=w - 1)  # type: ignore

        # Current period total
        current_total = float(
            s.loc[(s.index >= period_start) & (s.index <= period_end)].sum()
        )

        # Generate all possible N-day rolling totals from historical data
        if len(historical_data) >= w:
            all_rolling_totals = historical_data.rolling(window=w).sum().dropna()  # type: ignore
            historical_values = all_rolling_totals.values  # type: ignore
            historical_values = historical_values[np.isfinite(historical_values)]  # type: ignore

            # Calculate percentile of current value
            if len(historical_values) > 0:  # type: ignore
                rank = int(sum(v > current_total for v in historical_values) + 1)  # type: ignore
                percentile = 100.0 * (1.0 - rank / (len(historical_values) + 1.0))
            else:
                percentile = np.nan
        else:
            historical_values = np.array([])
            percentile = np.nan

        violin_data[f"{w}d"] = {
            "values": historical_values,
            "current": current_total,
            "percentile": percentile,
        }

    return violin_data


def create_rainfall_violin_plot(
    window: str,
    violin_data: dict,
    unit: str = "in",
    title: Optional[str] = None,
) -> None:
    """
    Create single violin plot showing rainfall distribution for specified window.

    :param window: Window size (e.g., "7d", "30d")
    :param violin_data: Data from prepare_violin_plot_data()
    :param unit: Unit for display (e.g., "in")
    :param title: Chart title (auto-generated if None)
    """
    import plotly.graph_objects as go

    if window not in violin_data or len(violin_data[window]["values"]) == 0:
        st.warning(f"No historical data available for {window} period.")
        return

    data = violin_data[window]
    values = data["values"]
    current = data["current"]
    percentile = data["percentile"]

    # Create violin plot
    fig = go.Figure()

    # Add violin showing distribution of historical values
    fig.add_trace(
        go.Violin(
            y=values,
            name=f"Historical {window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(56, 128, 191, 0.6)",  # Blue from rain palette
            line_color="rgba(56, 128, 191, 1.0)",
            x0=f"{window} Periods",
        )
    )

    # Add current value as overlay marker
    if not np.isnan(current):
        fig.add_trace(
            go.Scatter(
                x=[f"{window} Periods"],
                y=[current],
                mode="markers",
                marker=dict(
                    symbol="diamond-tall",
                    size=16,
                    color="red",
                    line=dict(width=2, color="darkred"),
                ),
                name=f"Current ({current:.2f}{unit})",
                text=[
                    (
                        f"Current: {current:.2f}{unit}<br>Percentile: {percentile:.1f}th"
                        if not np.isnan(percentile)
                        else f"Current: {current:.2f}{unit}"
                    )
                ],
                hoverinfo="text",
            )
        )

    # Customize layout
    chart_title = title or f"Rainfall Distribution: {window} Rolling Periods"
    fig.update_layout(
        title=chart_title,
        yaxis_title=f"Rainfall ({unit})",
        xaxis_title="",
        showlegend=True,
        height=500,
        template="plotly_white",
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Show summary stats
    if not np.isnan(percentile):
        if percentile >= 90:
            status = "🔴 **Extremely wet**"
        elif percentile >= 75:
            status = "🟡 **Above normal**"
        elif percentile >= 25:
            status = "🟢 **Normal range**"
        else:
            status = "🔵 **Below normal**"

        st.markdown(
            f"{status} - Current {window} total ({current:.2f}{unit}) ranks at **{percentile:.1f}th percentile** of {len(values):,} historical periods."
        )


def create_dual_violin_plot(
    left_window: str,
    right_window: str,
    violin_data: dict,
    unit: str = "in",
    title: Optional[str] = None,
) -> None:
    """
    Create dual violin plot comparing two different rainfall windows side-by-side.

    :param left_window: Left window size (e.g., "1d")
    :param right_window: Right window size (e.g., "7d")
    :param violin_data: Data from prepare_violin_plot_data()
    :param unit: Unit for display (e.g., "in")
    :param title: Chart title (auto-generated if None)
    """
    import plotly.graph_objects as go

    # Check data availability
    missing_data = []
    for window in [left_window, right_window]:
        if window not in violin_data or len(violin_data[window]["values"]) == 0:  # type: ignore
            missing_data.append(window)

    if missing_data:
        st.warning(f"No historical data available for: {', '.join(missing_data)}")
        return

    left_data = violin_data[left_window]
    right_data = violin_data[right_window]

    # Create subplot with shared y-axis
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{left_window} Periods", f"{right_window} Periods"],
        shared_yaxes=True,
    )

    # Colors for distinction
    colors = {
        left_window: "rgba(56, 128, 191, 0.6)",  # Blue
        right_window: "rgba(191, 128, 56, 0.6)",  # Orange
    }
    line_colors = {
        left_window: "rgba(56, 128, 191, 1.0)",
        right_window: "rgba(191, 128, 56, 1.0)",
    }

    # Add left violin
    fig.add_trace(
        go.Violin(
            y=left_data["values"],
            name=f"Historical {left_window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[left_window],
            line_color=line_colors[left_window],
            x0=f"{left_window}",
        ),
        row=1,
        col=1,
    )

    # Add right violin
    fig.add_trace(
        go.Violin(
            y=right_data["values"],
            name=f"Historical {right_window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[right_window],
            line_color=line_colors[right_window],
            x0=f"{right_window}",
        ),
        row=1,
        col=2,
    )

    # Add current value markers
    for i, (window, data) in enumerate(
        [(left_window, left_data), (right_window, right_data)], 1
    ):
        current = data["current"]
        percentile = data["percentile"]

        if not np.isnan(current):
            fig.add_trace(
                go.Scatter(
                    x=[window],
                    y=[current],
                    mode="markers",
                    marker=dict(
                        symbol="diamond-tall",
                        size=16,
                        color="red",
                        line=dict(width=2, color="darkred"),
                    ),
                    name=f"Current {window} ({current:.2f}{unit})",
                    text=[
                        (
                            f"Current: {current:.2f}{unit}<br>Percentile: {percentile:.1f}th"
                            if not np.isnan(percentile)
                            else f"Current: {current:.2f}{unit}"
                        )
                    ],
                    hoverinfo="text",
                    showlegend=True,
                ),
                row=1,
                col=i,
            )

    # Customize layout
    chart_title = (
        title or f"Rainfall Distribution Comparison: {left_window} vs {right_window}"
    )
    fig.update_layout(
        title=chart_title,
        height=600,
        template="plotly_white",
    )

    # Update y-axis
    fig.update_yaxes(title_text=f"Rainfall ({unit})", row=1, col=1)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Show comparison summary
    col1, col2 = st.columns(2)

    with col1:
        left_current = left_data["current"]
        left_percentile = left_data["percentile"]
        if not np.isnan(left_percentile):
            st.markdown(
                f"**{left_window} Period**: {left_current:.2f}{unit} ({left_percentile:.1f}th percentile)"
            )

    with col2:
        right_current = right_data["current"]
        right_percentile = right_data["percentile"]
        if not np.isnan(right_percentile):
            st.markdown(
                f"**{right_window} Period**: {right_current:.2f}{unit} ({right_percentile:.1f}th percentile)"
            )


@st.cache_data(show_spinner=False)
def _cached_violin_data(
    daily_rain_df: pd.DataFrame, windows, normals_years, end_date, version: str = "v1"
):
    return prepare_violin_plot_data(daily_rain_df, windows, normals_years, end_date)


def extract_daily_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract daily rainfall totals from dailyrainin field.
    :param df: pd.DataFrame - Weather data with 'dateutc' and 'dailyrainin' columns.
    :return: pd.DataFrame - With 'date' and 'rainfall' columns containing daily totals.
    """
    df_local = df.copy()

    # Convert to local time and extract dates
    df_local["local_datetime"] = pd.to_datetime(
        df_local["dateutc"], unit="ms", utc=True
    ).dt.tz_convert("America/Los_Angeles")
    df_local["local_date"] = df_local["local_datetime"].dt.date

    # Group by date and take max dailyrainin (handles accumulation)
    daily_max = df_local.groupby("local_date")["dailyrainin"].max()

    # Calculate daily totals (handle midnight resets)
    daily_totals = daily_max.diff().fillna(daily_max.iloc[0])
    daily_totals = daily_totals.clip(lower=0)

    # Handle edge case where field doesn't reset
    mask = (daily_totals == 0) & (daily_max > 0)  # type: ignore
    daily_totals[mask] = daily_max[mask]

    return pd.DataFrame({"date": daily_totals.index, "rainfall": daily_totals.values})


def calculate_dry_spell_stats(df: pd.DataFrame) -> dict:
    """
    Calculate current dry spell duration using lastRain timestamp.
    :param df: pd.DataFrame - Weather data with 'lastRain' and 'dateutc' columns.
    :return: dict - With 'current_dry_days' and 'current_dry_hours' keys.
    """
    if "lastRain" not in df.columns:
        return {"current_dry_days": 0, "current_dry_hours": 0}

    latest_record = df.iloc[-1]
    last_rain_str = latest_record["lastRain"]
    current_time = pd.to_datetime(latest_record["dateutc"], unit="ms", utc=True)

    try:
        last_rain = pd.to_datetime(last_rain_str)
        dry_period = current_time - last_rain
        return {
            "current_dry_days": dry_period.days,
            "current_dry_hours": dry_period.total_seconds() / 3600,
        }
    except Exception:
        return {"current_dry_days": 0, "current_dry_hours": 0}


def calculate_rainfall_accumulations(
    daily_rain_df: pd.DataFrame, df: pd.DataFrame
) -> dict:
    """
    Calculate rainfall accumulations over different time periods.
    :param daily_rain_df: pd.DataFrame - Daily rainfall data with 'date' and 'rainfall' columns.
    :param df: pd.DataFrame - Raw weather data for current hourly rate.
    :return: dict - Accumulation totals for different periods with current rate.
    """
    if len(daily_rain_df) == 0:
        return {}

    # Convert date to datetime for easier filtering
    df_calc = daily_rain_df.copy()
    df_calc["date"] = pd.to_datetime(df_calc["date"])

    # Get current date (use last date in data)
    current_date = df_calc["date"].max()

    # Get current hourly rain rate (same for all periods)
    current_rate = df["hourlyrainin"].iloc[-1] if len(df) > 0 else 0

    # Define periods
    periods = {
        "today": 1,
        "last 7d": 7,
        "last 30d": 30,
        "last 90d": 90,
        "last 365d": 365,
    }

    results = {}

    for period_name, days in periods.items():
        # Calculate start date
        start_date = current_date - pd.Timedelta(days=days - 1)

        # Filter data for period
        period_data = df_calc[
            (df_calc["date"] >= start_date) & (df_calc["date"] <= current_date)
        ]

        # Sum rainfall for period
        total_rainfall = period_data["rainfall"].sum()

        # Structure like get_history_min_max
        results[period_name] = {
            "min": 0,  # Rainfall can't be negative
            "max": total_rainfall,
            "current": current_rate,  # Current hourly rate for all periods
        }

    return results


def calculate_rainfall_statistics(
    df: pd.DataFrame, daily_rain_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Calculate comprehensive rainfall statistics and metrics.
    :param df: pd.DataFrame - Weather data with rain accumulation fields.
    :param daily_rain_df: pd.DataFrame - Pre-computed daily rainfall data (optional).
    :return: dict - Current accumulations, historical averages, and dry spell info.
    """
    daily_rain = (
        daily_rain_df if daily_rain_df is not None else extract_daily_rainfall(df)
    )
    dry_stats = calculate_dry_spell_stats(df)

    # Current accumulations from latest record
    latest = df.iloc[-1]

    # Historical analysis
    rain_days = (daily_rain["rainfall"] > 0).sum()  # type: ignore
    max_daily = daily_rain["rainfall"].max()
    total_days = len(daily_rain)
    avg_annual = (
        daily_rain["rainfall"].sum() / (total_days / 365.25) if total_days > 365 else 0
    )

    # Calculate yesterday's rainfall
    yesterday_rainfall = 0.0
    if len(daily_rain) >= 2:
        # Sort by date and get second-to-last entry (yesterday)
        daily_sorted = daily_rain.sort_values("date")
        yesterday_rainfall = daily_sorted.iloc[-2]["rainfall"]

    return {
        "current_ytd": latest.get("yearlyrainin", 0),
        "current_monthly": latest.get("monthlyrainin", 0),
        "current_weekly": latest.get("weeklyrainin", 0),
        "current_daily": latest.get("dailyrainin", 0),
        "current_yesterday": yesterday_rainfall,
        "avg_annual": avg_annual,
        "total_rain_days": rain_days,
        "max_daily_this_year": max_daily,
        "current_dry_days": dry_stats["current_dry_days"],
        "last_rain": latest.get("lastRain", "Unknown"),
        "total_days": total_days,
    }


def render():
    """Render the precipitation analysis tab."""
    st.header("Precipitation Analysis")
    st.write("Comprehensive rainfall data analysis and visualization")

    # Get data from session state
    if "history_df" not in st.session_state:
        st.error("No weather data available in session")
        return

    df = st.session_state["history_df"]

    # Use most recent data from last_data if available, otherwise latest record
    if "last_data" in st.session_state:
        latest = st.session_state["last_data"]
        # Calculate dry spell from current data
        try:
            last_rain = pd.to_datetime(latest["lastRain"])
            current_time = pd.to_datetime(latest["dateutc"], unit="ms", utc=True)
            dry_period = current_time - last_rain
            current_dry_days = dry_period.days
            current_dry_hours = (dry_period.total_seconds() % (24 * 3600)) / 3600

            # Format time since rain
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

    # Extract daily rainfall data once for all calculations
    with st.spinner("Processing daily rainfall data..."):
        daily_rain_df = extract_daily_rainfall(df)

    # Calculate statistics
    with st.spinner("Calculating rainfall statistics..."):
        stats = calculate_rainfall_statistics(df, daily_rain_df)
        # Override current values with last_data if available
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

    # Summary Section
    st.subheader("Rainfall Summary")

    # Current accumulations
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

    # Historical context
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

        # Format last rain date properly
        try:
            last_rain_dt = pd.to_datetime(stats["last_rain"])
            # Convert to local timezone for display
            last_rain_local = last_rain_dt.tz_convert("America/Los_Angeles")
            formatted_date = last_rain_local.strftime("%m/%d/%Y %H:%M")
            st.write(f"• Last Rain: {formatted_date}")
        except Exception:
            st.write(f"• Last Rain: {stats['last_rain']}")

    st.divider()

    # Daily Rainfall Chart
    st.subheader("Daily Rainfall Chart")

    # Use the pre-computed daily rainfall data for accumulations
    if len(daily_rain_df) > 0:
        rain_accumulations = calculate_rainfall_accumulations(daily_rain_df, df)

        if rain_accumulations:
            # Data is already in the correct format for draw_horizontal_bars
            lo_viz.draw_horizontal_bars(
                rain_accumulations, label="Rainfall Accumulation (inches)"
            )
        else:
            st.error("Could not calculate rainfall accumulations")
    else:
        st.error("No daily rainfall data available")

    st.divider()

    # --- Rolling Historical Context ---------------------------------------------

    st.subheader(
        "Rolling Historical Context vs All N-day Periods (1d / 7d / 30d / 90d)"
    )

    # Use the pre-computed daily rainfall data for rolling context
    if len(daily_rain_df) > 0:
        end_date = pd.to_datetime(daily_rain_df["date"]).max()
        context_df = _cached_rolling_context(
            daily_rain_df=daily_rain_df,
            windows=(1, 7, 30, 90),
            normals_years=None,  # or fixed range, e.g., (2015, 2024)
            end_date=end_date,
            version="v2",
        )
        render_rolling_rain_context_table(context_df, unit="in")
    else:
        st.info("No daily totals to compute rolling context.")

    st.divider()

    # --- Rainfall Distribution Analysis -------------------------------------

    st.subheader("Historical Rainfall Distribution")
    st.write(
        "Compare current rainfall periods against the full distribution of all historical periods"
    )

    if len(daily_rain_df) > 0:
        # Prepare violin plot data (cached for performance)
        end_date = pd.to_datetime(daily_rain_df["date"]).max()
        violin_data = _cached_violin_data(
            daily_rain_df=daily_rain_df,
            windows=(1, 7, 30, 90),
            normals_years=None,
            end_date=end_date,
            version="v1",
        )

        # Window selection
        available_windows = [
            w
            for w in ["1d", "7d", "30d", "90d"]
            if w in violin_data and len(violin_data[w]["values"]) > 0  # type: ignore
        ]

        if available_windows:
            # Visualization mode selection
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
                    index=min(
                        1, len(available_windows) - 1
                    ),  # Default to 7d if available
                    help="Choose the rolling period length to analyze",
                )

                # Create violin plot for selected window
                create_rainfall_violin_plot(
                    window=selected_window, violin_data=violin_data, unit="in"
                )

            else:  # Compare Two Windows
                col1, col2 = st.columns(2)

                with col1:
                    left_window = st.selectbox(
                        "Left window:",
                        available_windows,
                        index=0,  # Default to first available
                        help="Choose the left period for comparison",
                    )

                with col2:
                    right_window = st.selectbox(
                        "Right window:",
                        available_windows,
                        index=min(
                            1, len(available_windows) - 1
                        ),  # Default to second available
                        help="Choose the right period for comparison",
                    )

                if left_window != right_window:
                    # Create dual violin plot
                    create_dual_violin_plot(
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

    # --- Rain Event Catalog ------------------------------------------------

    st.subheader("Rain Event Catalog")
    st.write("Browse and analyze individual rain events detected from historical data")

    try:
        from lookout.core.rain_events import RainEventCatalog

        # Get device info from session state (set in streamlit_app.py)
        if "device" in st.session_state:
            device = st.session_state["device"]
            device_mac = device["macAddress"]
            file_type = "parquet"

            catalog = RainEventCatalog(device_mac, file_type)

            # Session-first workflow: check session state, then storage, then generate
            events_df = None

            if "rain_events_catalog" in st.session_state:
                # Use cached catalog from session
                events_df = st.session_state["rain_events_catalog"]
                catalog_source = "session"
                logger.info(
                    f"Using catalog from session state: {len(events_df)} events"
                )

            elif catalog.catalog_exists():
                # Load from storage and update with new data
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
                # Auto-generate catalog on first visit (don't save)
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

            # Display catalog if we have events
            if events_df is not None and not events_df.empty:
                # Debug: Check if we're showing updated data
                zero_rate_count = (events_df["max_hourly_rate"] == 0).sum()
                logger.debug(
                    f"Displaying catalog: {len(events_df)} events, {zero_rate_count} with zero max_rate"
                )
                # Save and regenerate buttons
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

                        # Clear existing catalog from session
                        if "rain_events_catalog" in st.session_state:
                            old_count = len(st.session_state["rain_events_catalog"])
                            del st.session_state["rain_events_catalog"]
                            logger.info(
                                f"Cleared old catalog from session: {old_count} events"
                            )

                        # Generate fresh catalog
                        with st.spinner("Regenerating event catalog from archive..."):
                            new_events = catalog.detect_and_catalog_events(
                                df, auto_save=False
                            )
                            st.session_state["rain_events_catalog"] = new_events
                            events_df = new_events  # Update current UI reference
                            logger.info(
                                f"Catalog regenerated: {len(new_events)} events cached in session state"
                            )

                        # Clear any cached data that might hold old event info
                        st.cache_data.clear()
                        logger.info("Cleared streamlit data cache")

                        st.success(
                            f"✅ Regenerated {len(new_events)} events! Data updated in current view."
                        )

                st.divider()

                # Event catalog summary
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

                # Event selection and details
                st.write("**Select a rain event to analyze:**")

                # Create event options with meaningful labels (convert to Pacific time)
                events_df["event_label"] = events_df.apply(
                    lambda row: f"{pd.to_datetime(row['start_time']).tz_convert('America/Los_Angeles').strftime('%Y-%m-%d %H:%M')} - "
                    f"{row['total_rainfall']:.2f}\" in {row['duration_minutes']/60:.1f}h "
                    f"({row['quality_rating']})",
                    axis=1,
                )

                # Sort by start time (most recent first)
                events_df = events_df.sort_values("start_time", ascending=False)

                # Event selection
                if len(events_df) > 0:
                    selected_event_idx = st.selectbox(
                        "Choose event:",
                        range(len(events_df)),
                        format_func=lambda x: events_df.iloc[x]["event_label"],
                        help="Events sorted by most recent first",
                    )

                    selected_event = events_df.iloc[selected_event_idx]

                    # Display event details
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

                    # Quality flags
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

                    # Future: Add event data visualization here
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

    # Placeholder sections for upcoming visualizations
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

    # Data health check (condensed)
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

        # Show sample daily rainfall data
        if st.checkbox("Show daily rainfall sample"):
            st.write("**Recent daily totals:**")
            st.dataframe(daily_rain_df.tail(10))
