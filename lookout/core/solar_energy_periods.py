"""
Solar Energy Period Calculations
Converts raw solar radiation measurements into 15-minute energy periods.
"""

import numpy as np
import pandas as pd
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

PERIOD = pd.Timedelta(minutes=15)

# Longest interval between two rows that is integrated: one period. Across a
# longer gap the radiation between the rows is not known, so nothing is
# credited for it.
MAX_INTERVAL = PERIOD

STATION_TZ = "America/Los_Angeles"


def calculate_15min_energy_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate energy production for 15-minute periods using trapezoidal integration.

    Every pair of consecutive rows is integrated. A pair that crosses a period
    boundary is split at the boundary and each part is credited to the period
    it lies in, so over unbroken rows the periods add up to the trapezoid of
    the whole curve.
    A pair more than one period apart is a gap and credits nothing; so does a
    pair with missing radiation.

    :param df: DataFrame with datetime and solarradiation columns
               - datetime: TZ-aware datetime column in US/Pacific (already converted)
               - solarradiation: Solar radiation in W/m²
    :return: DataFrame with period_start, period_end, energy_kwh columns, one row
             per period that holds a reading
             - period_start: TZ-aware datetime in US/Pacific (start of 15min period)
             - period_end: TZ-aware datetime in US/Pacific (end of 15min period)
             - energy_kwh: Energy accumulated in kWh (can be 0.0 for nighttime periods)
    """
    # Input validation
    required_cols = ["datetime", "solarradiation"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        logger.info("No solar data provided for period calculation")
        return pd.DataFrame(
            {
                "period_start": pd.Series(dtype=f"datetime64[ns, {STATION_TZ}]"),
                "period_end": pd.Series(dtype=f"datetime64[ns, {STATION_TZ}]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )

    # Sort by datetime ascending (archive is reverse-sorted)
    df_sorted = df.sort_values("datetime", kind="stable")

    logger.debug(
        f"Processing {len(df_sorted)} solar radiation records for 15-minute periods"
    )

    # Periods aligned to clock times, floored in UTC to avoid DST issues
    times = df_sorted["datetime"].dt.tz_convert("UTC").reset_index(drop=True)
    watts = df_sorted["solarradiation"].astype(float).reset_index(drop=True)
    energy_wh = _credit_row_pairs(times, watts)

    # Every period holding a row is reported, with 0.0 when nothing is credited
    periods = pd.DatetimeIndex(times.dt.floor(PERIOD).unique())
    energy_wh = energy_wh.reindex(periods, fill_value=0.0).fillna(0.0)

    # Convert Wh/m² to kWh/m²
    period_start = energy_wh.index.tz_convert(STATION_TZ)
    result_df = pd.DataFrame(
        {
            "period_start": period_start,
            "period_end": period_start + PERIOD,
            "energy_kwh": (energy_wh / 1000).values,
        }
    )
    result_df = result_df.sort_values("period_start").reset_index(drop=True)

    logger.info(f"Generated {len(result_df)} 15-minute solar energy periods")
    return result_df


def _credit_row_pairs(times: pd.Series, watts: pd.Series) -> pd.Series:
    """
    Integrate each pair of consecutive rows and credit it to its periods.

    The trapezoid rule, (y0 + y1) / 2 * dx, over each pair. A pair that crosses
    a period boundary is split there, at the radiation read off the straight
    line between its two rows. Cutting a trapezoid along its own line keeps
    its area, so the credited energy adds up to the trapezoid of the whole
    curve. NaN radiation credits nothing, as the sum skips it.

    :param times: Row times in UTC, ascending, index 0..n-1
    :param watts: Solar radiation in W/m² at each row, same index
    :return: Wh/m² credited to each UTC period start that begins or ends a pair
    """
    hour = np.timedelta64(1, "h")
    start, end = times.iloc[:-1].values, times.iloc[1:].values
    start_wm2, end_wm2 = watts.iloc[:-1].values, watts.iloc[1:].values
    start_period = times.iloc[:-1].dt.floor(PERIOD).values
    end_period = times.iloc[1:].dt.floor(PERIOD).values

    span_h = (end - start) / hour
    counted = (span_h > 0) & (span_h <= MAX_INTERVAL / pd.Timedelta(hours=1))
    # A pair of at most one period crosses at most one boundary: the start of
    # its later row's period. A later row on the boundary does not cross it.
    crosses = counted & (end_period > start) & (end_period < end)

    first_part_h = np.where(crosses, (end_period - start) / hour, span_h)
    second_part_h = span_h - first_part_h
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary_wm2 = np.where(
            crosses,
            start_wm2 + (end_wm2 - start_wm2) * first_part_h / span_h,
            end_wm2,
        )
    first_wh = np.where(counted, (start_wm2 + boundary_wm2) / 2 * first_part_h, 0.0)
    second_wh = np.where(crosses, (boundary_wm2 + end_wm2) / 2 * second_part_h, 0.0)

    credited = pd.concat(
        [
            pd.Series(first_wh, index=pd.DatetimeIndex(start_period, tz="UTC")),
            pd.Series(second_wh, index=pd.DatetimeIndex(end_period, tz="UTC")),
        ]
    )
    return credited.groupby(level=0).sum()


def recompute_from_period(
    rows: pd.DataFrame, period_start: pd.Timestamp
) -> pd.DataFrame:
    """
    Energy periods from `period_start` on, as a full calculation would give.

    The interval from the last row before `period_start` into it is part of
    that period's energy, so that row is integrated too; the period it falls
    in is left out of the result. A catalog update replaces its stored periods
    from `period_start` on with these.

    :param rows: DataFrame with datetime and solarradiation columns
    :param period_start: TZ-aware start of the first period to recompute
    :return: Periods as from `calculate_15min_energy_periods`, starting at or
             after `period_start`; empty when no row is that late
    """
    from_start = rows["datetime"] >= period_start
    if not from_start.any():
        return calculate_15min_energy_periods(rows.iloc[0:0])
    earlier_times = rows.loc[~from_start, "datetime"]
    if not earlier_times.empty:
        from_start |= rows["datetime"] == earlier_times.max()
    periods = calculate_15min_energy_periods(rows[from_start])
    return periods[periods["period_start"] >= period_start].reset_index(drop=True)


def aggregate_to_daily(periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to daily totals.

    :param periods_df: DataFrame from calculate_15min_energy_periods with columns [period_start, period_end, energy_kwh]
    :return: DataFrame with columns [date, daily_kwh]
    """
    if periods_df.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "daily_kwh": pd.Series(dtype=float),
            }
        )

    # Extract date from period_start (already TZ-aware US/Pacific)
    periods_df = periods_df.copy()
    periods_df["date"] = periods_df["period_start"].dt.date

    # Group by date and sum energy_kwh, keeping all dates including zeros
    daily_df = periods_df.groupby("date", as_index=False)["energy_kwh"].sum()
    daily_df.columns = ["date", "daily_kwh"]

    # Sort by date
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    return daily_df


def aggregate_to_hourly(periods_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to hourly for a specific date.

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :param date: String in format 'YYYY-MM-DD' (US/Pacific timezone)
    :return: DataFrame with columns [hour, hourly_kwh]
    """
    if periods_df.empty:
        return pd.DataFrame(
            {"hour": pd.Series(dtype=int), "hourly_kwh": pd.Series(dtype=float)}
        )

    try:
        target_date = pd.to_datetime(date).date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Expected 'YYYY-MM-DD'")

    # Filter to specified date
    date_filter = periods_df["period_start"].dt.date == target_date
    filtered_df = periods_df[date_filter].copy()

    if filtered_df.empty:
        # Return all hours with zero energy
        hourly_dict = {hour: 0.0 for hour in range(24)}
    else:
        # Extract hour from period_start
        filtered_df["hour"] = filtered_df["period_start"].dt.hour

        # Group by hour and sum energy_kwh
        hourly_dict = filtered_df.groupby("hour")["energy_kwh"].sum().to_dict()

        # Ensure all hours 0-23 are present (including zeros)
        for hour in range(24):
            if hour not in hourly_dict:
                hourly_dict[hour] = 0.0

    # Create result DataFrame
    result_df = pd.DataFrame(
        {"hour": list(hourly_dict.keys()), "hourly_kwh": list(hourly_dict.values())}
    )

    # Sort by hour
    result_df = result_df.sort_values("hour").reset_index(drop=True)

    return result_df
