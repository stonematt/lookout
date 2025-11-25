"""
Rainfall data processing and statistical analysis.

This module provides core rainfall data processing functions including daily
rainfall extraction, accumulation calculations, rolling window analysis, and
statistical metrics. All functions are pure data processing with no UI dependencies.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def extract_daily_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract daily rainfall totals from dailyrainin field.

    The dailyrainin field is a rolling 24-hour accumulation that resets to zero
    at midnight local time. Therefore, max(dailyrainin) for each calendar day
    represents the total rainfall for that day.

    :param df: Weather data with 'dateutc' and 'dailyrainin' columns.
    :return: DataFrame with 'date' and 'rainfall' columns containing daily totals.
    """
    df_local = df.copy()

    df_local["local_datetime"] = pd.to_datetime(
        df_local["dateutc"], unit="ms", utc=True
    ).dt.tz_convert("America/Los_Angeles")
    df_local["local_date"] = df_local["local_datetime"].dt.date

    # Since dailyrainin resets at midnight, max value for each day IS the daily total
    daily_max = df_local.groupby("local_date")["dailyrainin"].max()

    return pd.DataFrame({"date": daily_max.index, "rainfall": daily_max.values})


def calculate_dry_spell_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate current dry spell duration using lastRain timestamp.

    :param df: Weather data with 'lastRain' and 'dateutc' columns.
    :return: Dictionary with 'current_dry_days' and 'current_dry_hours' keys.
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
) -> Dict:
    """
    Calculate rainfall accumulations over different time periods.

    :param daily_rain_df: Daily rainfall data with 'date' and 'rainfall' columns.
    :param df: Raw weather data for current hourly rate.
    :return: Accumulation totals for different periods with current rate.
    """
    if len(daily_rain_df) == 0:
        return {}

    df_calc = daily_rain_df.copy()
    df_calc["date"] = pd.to_datetime(df_calc["date"])

    current_date = df_calc["date"].max()

    current_rate = df["hourlyrainin"].iloc[-1] if len(df) > 0 else 0

    periods = {
        "today": 1,
        "last 7d": 7,
        "last 30d": 30,
        "last 90d": 90,
        "last 365d": 365,
    }

    results = {}

    for period_name, days in periods.items():
        start_date = current_date - pd.Timedelta(days=days - 1)

        period_data = df_calc[
            (df_calc["date"] >= start_date) & (df_calc["date"] <= current_date)
        ]

        total_rainfall = period_data["rainfall"].sum()

        results[period_name] = {
            "min": 0,
            "max": total_rainfall,
            "current": current_rate,
        }

    return results


def calculate_rainfall_statistics(
    df: pd.DataFrame, daily_rain_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Calculate comprehensive rainfall statistics and metrics.

    :param df: Weather data with rain accumulation fields.
    :param daily_rain_df: Pre-computed daily rainfall data (optional).
    :return: Current accumulations, historical averages, and dry spell info.
    """
    daily_rain = (
        daily_rain_df if daily_rain_df is not None else extract_daily_rainfall(df)
    )
    dry_stats = calculate_dry_spell_stats(df)

    latest = df.iloc[-1]

    rain_days = (daily_rain["rainfall"] > 0).sum()  # type: ignore
    max_daily = daily_rain["rainfall"].max()
    total_days = len(daily_rain)
    avg_annual = (
        daily_rain["rainfall"].sum() / (total_days / 365.25) if total_days > 365 else 0
    )

    yesterday_rainfall = 0.0
    if len(daily_rain) >= 2:
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


def compute_rolling_rain_context(
    daily_rain_df: pd.DataFrame, windows, normals_years, end_date
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

    if normals_years is None:
        historical_data = s[s.index.year != end_dt.year]  # type: ignore
    else:
        y0, y1 = normals_years
        historical_data = s[
            (s.index.year >= y0)
            & (s.index.year <= y1)
            & (s.index.year != end_dt.year)  # type: ignore
        ]

    df = daily_rain_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    s = df.set_index("date")["rainfall"].sort_index()

    end_dt = (
        pd.to_datetime(end_date).normalize() if end_date is not None else s.index.max()
    )

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

        cur = float(s.loc[(s.index >= period_start) & (s.index <= period_end)].sum())

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


def prepare_violin_plot_data(
    daily_rain_df: pd.DataFrame, windows, normals_years, end_date
) -> Dict:
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

    if normals_years is None:
        historical_data = s[s.index.year != end_dt.year]  # type: ignore
    else:
        y0, y1 = normals_years
        historical_data = s[
            (s.index.year >= y0)
            & (s.index.year <= y1)
            & (s.index.year != end_dt.year)  # type: ignore
        ]

    violin_data = {}
    for w in windows:
        period_end = end_dt
        period_start = end_dt - pd.Timedelta(days=w - 1)  # type: ignore

        current_total = float(
            s.loc[(s.index >= period_start) & (s.index <= period_end)].sum()
        )

        if len(historical_data) >= w:
            all_rolling_totals = historical_data.rolling(window=w).sum().dropna()  # type: ignore
            historical_values = all_rolling_totals.values  # type: ignore
            historical_values = historical_values[np.isfinite(historical_values)]  # type: ignore

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


def prepare_year_over_year_accumulation(
    daily_rain_df: pd.DataFrame, start_day: int = 1, end_day: int = 365
) -> pd.DataFrame:
    """
    Prepare year-over-year cumulative rainfall data for visualization.

    For each year in the dataset, calculates the running cumulative total
    of rainfall within the specified day range (start_day to end_day). This enables
    comparison of rainfall accumulation patterns across different years for specific
    time periods.

    :param daily_rain_df: DataFrame with 'date' and 'rainfall' columns.
    :param start_day: Start day of year to include (1-365). Defaults to 1.
    :param end_day: End day of year to include (1-365). Defaults to 365.
    :return: DataFrame with columns: day_of_year, year, cumulative_rainfall.
    """
    if daily_rain_df.empty:
        return pd.DataFrame(columns=["day_of_year", "year", "cumulative_rainfall"])

    # Prepare data with proper date handling
    df = daily_rain_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    # Filter to requested day range
    df_filtered = df[
        (df["day_of_year"] >= start_day) & (df["day_of_year"] <= end_day)
    ].copy()

    # Calculate cumulative rainfall by year
    result_rows = []
    unique_years = df_filtered["year"].unique()

    for year_val in sorted(unique_years):
        year_data = df_filtered[df_filtered["year"] == year_val].copy()
        year_data = year_data.sort_values("day_of_year")
        year_data["cumulative_rainfall"] = year_data["rainfall"].cumsum()

        # Add to result
        for _, row in year_data.iterrows():
            result_rows.append(
                {
                    "day_of_year": int(row["day_of_year"]),
                    "year": int(row["year"]),
                    "cumulative_rainfall": float(row["cumulative_rainfall"]),
                }
            )

    return pd.DataFrame(result_rows)
