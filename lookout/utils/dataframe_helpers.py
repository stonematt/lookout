# lookout/utils/dataframe_helpers.py
import pandas as pd


def max_dateutc_ms(df: pd.DataFrame) -> int:
    """
    Return the maximum value in 'dateutc' column as milliseconds since epoch.
    Handles both pandas Timestamps and datetimes.

    :param df: DataFrame with 'dateutc' column
    :return: int milliseconds since epoch
    """
    if "dateutc" not in df.columns or df.empty:
        return 0
    max_ts = pd.to_datetime(df["dateutc"].max(), utc=True)
    return int(max_ts.timestamp() * 1000)
