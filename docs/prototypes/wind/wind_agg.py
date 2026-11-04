"""Aggregate a Lookout archive parquet into the wind prototype's data.

Prints JSON: roses for the last 7 days and each season, hourly profiles,
the last 24 hours of readings, the last reading and winter gust percentiles.
Usage: ``wind_agg.py <archive.parquet> > wind.json``.
"""

import json
import sys

import numpy as np
import pandas as pd


df = pd.read_parquet(
    sys.argv[1],
    columns=[
        "dateutc",
        "winddir",
        "windspeedmph",
        "windgustmph",
        "winddir_avg10m",
        "windspdmph_avg10m",
    ],
)
df = df.dropna(subset=["windspeedmph", "windgustmph"]).sort_values("dateutc")
t = pd.to_datetime(df.dateutc, unit="ms", utc=True).dt.tz_convert("America/Los_Angeles")
df["t"] = t
df["month"] = t.dt.month
df["hour"] = t.dt.hour
SB = [0.1, 2, 5, 10, 1e9]  # speed bins upper edges; 0 = calm


def rose(d, col):
    out = {"n": int(len(d))}
    calm = d[col] < 0.1
    out["calm"] = round(float(calm.mean()), 4)
    m = d[~calm]
    sec = (((m.winddir + 11.25) % 360) // 22.5).astype(int)
    b = np.digitize(m[col], [2, 5, 10])
    grid = np.zeros((16, 4))
    for s, k in zip(sec, b):
        grid[s, k] += 1
    out["grid"] = (grid / max(len(d), 1)).round(5).tolist()
    return out


end = df.t.max()
recent = df[df.t > end - pd.Timedelta(days=7)]
season_of = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
df["season"] = df.month.map(season_of)
res = {
    "station_end": end.isoformat(),
    "start": df.t.min().isoformat(),
    "recent_days": 7,
    "recent": {
        "speed": rose(recent, "windspeedmph"),
        "gust": rose(recent, "windgustmph"),
    },
    "seasons": {
        s: {
            "speed": rose(g, "windspeedmph"),
            "gust": rose(g, "windgustmph"),
            "years": sorted(g.t.dt.year.unique().tolist()),
        }
        for s, g in df.groupby("season")
    },
}


def hourly(d):
    g = d.groupby("hour")
    return {
        "speed_med": g.windspeedmph.median().round(2).tolist(),
        "speed_p90": g.windspeedmph.quantile(0.9).round(2).tolist(),
        "gust_med": g.windgustmph.median().round(2).tolist(),
        "gust_p90": g.windgustmph.quantile(0.9).round(2).tolist(),
    }


res["hourly"] = {
    "recent": hourly(recent),
    "winter": hourly(df[df.season == "winter"]),
    "summer": hourly(df[df.season == "summer"]),
}
day = df[df.t > end - pd.Timedelta(hours=24)]
res["day"] = [
    [
        int(r.dateutc),
        float(r.windspeedmph),
        float(r.windgustmph),
        None if pd.isna(r.winddir) else int(r.winddir),
    ]
    for r in day.itertuples()
]
last = df.iloc[-1]
res["now"] = {
    "speed": float(last.windspeedmph),
    "gust": float(last.windgustmph),
    "dir": int(last.winddir),
    "dir_avg10m": None if pd.isna(last.winddir_avg10m) else int(last.winddir_avg10m),
    "speed_avg10m": (
        None if pd.isna(last.windspdmph_avg10m) else float(last.windspdmph_avg10m)
    ),
    "max_gust_today": float(df[df.t.dt.date == end.date()].windgustmph.max()),
}
# percentiles for "is this unusual" on the gust: seasonal distribution of daily max gust
dm = df[df.season == "winter"].groupby(df.t.dt.date).windgustmph.max()
res["winter_daily_max_gust_pct"] = {
    p: round(float(dm.quantile(p / 100)), 1) for p in (10, 25, 50, 75, 90, 99)
}
ratio = df.windgustmph / df.windspeedmph.where(df.windspeedmph >= 1)
res["gust_ratio_median"] = round(float(ratio.median()), 2)
print(json.dumps(res, separators=(",", ":")))
