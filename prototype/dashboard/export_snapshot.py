"""
Throwaway: export a JSON snapshot of REAL station data for the static HTML
dashboard prototype. Read-only against Storj; never writes catalogs back.

Run from repo root:
    conda run -n lookout-new env PYTHONPATH=. python prototype/dashboard/export_snapshot.py
Credentials come from .streamlit/secrets.toml (via st.secrets). Output contains
no keys, MAC address, or lat/lon.
"""

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "data.json"
TZ = "America/Los_Angeles"
BUCKET = "lookout"
NOTES: list = []

import streamlit as st  # noqa: E402

import lookout.core.data_processing as lo_dp  # noqa: E402
import lookout.core.rainfall_analysis as ra  # noqa: E402
import lookout.core.rain_viz as rain_viz  # noqa: E402
import lookout.core.solar_data_transformer as sdt  # noqa: E402
from lookout import config as cfg  # noqa: E402
from lookout.core.rain_events import RainEventCatalog  # noqa: E402
from lookout.core.energy_catalog_func import (  # noqa: E402
    load_energy_catalog,
    update_energy_catalog,
)
from lookout.core.solar_energy_periods import aggregate_to_daily  # noqa: E402
from lookout.core.solar_tiles import calculate_tile_metrics  # noqa: E402
from lookout.storage import storj as sj  # noqa: E402
from lookout.utils.trend_utils import (  # noqa: E402
    calculate_barometer_trend,
    calculate_temperature_trend,
)

MAC = cfg.DEFAULT_MAC_ADDRESS


def clean(o):
    """Recursively make JSON-safe (NaN -> None, numpy -> py)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 4)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def section(name):
    def deco(fn):
        try:
            return fn()
        except Exception as e:  # keep going; record gap
            traceback.print_exc()
            NOTES.append(f"{name}: unavailable ({type(e).__name__}: {e})")
            return None

    return deco


def r(s, n=2):
    return [None if pd.isna(v) else round(float(v), n) for v in s]


# ------------------------------------------------------------------ load
t0 = time.time()
hist = sj.get_df_from_s3(BUCKET, f"{MAC}.parquet", file_type="parquet")
assert not hist.empty, "archive load failed"
hist = hist.sort_values("dateutc", ascending=False).reset_index(drop=True)  # app order
hist["date"] = pd.to_datetime(hist["date"]).dt.tz_convert(TZ)
latest = hist.iloc[0]
print(f"archive: {len(hist)} rows {hist.date.min()} -> {hist.date.max()}")
asc = hist.sort_values("dateutc").reset_index(drop=True)

# live last_data + station name from Ambient (what the app does); fall back to archive
station_name = "Weather Station"
last_data = latest.to_dict()
live = False
try:
    import lookout.api.ambient_client as ac

    devs = ac.get_devices()
    if devs:
        station_name = devs[0].get("info", {}).get("name", station_name)
        ld = devs[0].get("lastData") or {}
        if ld.get("dateutc", 0) >= latest["dateutc"]:
            last_data = ld
            live = True
except Exception as e:
    NOTES.append(f"live device call failed, used archive latest row ({e})")
last_ts = pd.to_datetime(last_data["dateutc"], unit="ms", utc=True).tz_convert(TZ)

# ------------------------------------------------------------------ current
temp_trend = calculate_temperature_trend(hist, last_data.get("tempf"))
barom_trend = calculate_barometer_trend(hist, last_data.get("baromrelin"))
p3 = asc[asc.dateutc <= last_data["dateutc"] - 3 * 3600_000].tail(1)
baro_3h = (
    None
    if p3.empty
    else round(float(last_data["baromrelin"] - p3.iloc[0]["baromrelin"]), 3)
)
g = last_data.get
current = {
    "timestamp": last_ts.isoformat(),
    "dateutc": int(last_data["dateutc"]),
    "source": "ambient_live" if live else "archive_latest",
    "temp_f": g("tempf"),
    "feels_like_f": g("feelsLike"),
    "humidity": g("humidity"),
    "dew_point_f": g("dewPoint"),
    "wind_mph": g("windspeedmph"),
    "wind_avg10m_mph": g("windspdmph_avg10m"),
    "wind_gust_mph": g("windgustmph"),
    "max_daily_gust_mph": g("maxdailygust"),
    "wind_dir": g("winddir"),
    "pressure_inhg": g("baromrelin"),
    "pressure_trend_3h_inhg": baro_3h,
    "pressure_trend_symbol": barom_trend,
    "temp_trend_symbol": temp_trend,
    "rain_rate_in_hr": g("hourlyrainin"),
    "rain_event_in": g("eventrainin"),
    "rain_today_in": g("dailyrainin"),
    "rain_week_in": g("weeklyrainin"),
    "rain_month_in": g("monthlyrainin"),
    "rain_year_in": g("yearlyrainin"),
    "last_rain": pd.to_datetime(g("lastRain")).isoformat() if g("lastRain") is not None else None,
    "solar_wm2": g("solarradiation"),
    "uv": g("uv"),
    "indoor_temp_f": g("tempinf"),
    "indoor_humidity": g("humidityin"),
    "indoor_feels_like_f": g("feelsLikein"),
    "indoor_dew_point_f": g("dewPointin"),
}

# ------------------------------------------------------------------ recent (7d, 15-min means)
end = asc.dateutc.max()
rec = asc[asc.dateutc >= end - 7 * 86400_000].copy()
rec["bucket"] = (rec.dateutc // (15 * 60_000)) * (15 * 60_000)
agg = rec.groupby("bucket").agg(
    tempf=("tempf", "mean"), feelsLike=("feelsLike", "mean"),
    humidity=("humidity", "mean"), dewPoint=("dewPoint", "mean"),
    windspeedmph=("windspeedmph", "mean"), windgustmph=("windgustmph", "max"),
    winddir=("winddir", "last"), baromrelin=("baromrelin", "mean"),
    solarradiation=("solarradiation", "mean"), uv=("uv", "max"),
    hourlyrainin=("hourlyrainin", "max"), dailyrainin=("dailyrainin", "max"),
    tempinf=("tempinf", "mean"), humidityin=("humidityin", "mean"),
)
recent = {
    "interval_min": 15,
    "t": [int(i) for i in agg.index],
    "temp_f": r(agg.tempf, 1), "feels_like_f": r(agg.feelsLike, 1),
    "humidity": r(agg.humidity, 0), "dew_point_f": r(agg.dewPoint, 1),
    "wind_mph": r(agg.windspeedmph, 1), "wind_gust_mph": r(agg.windgustmph, 1),
    "wind_dir": r(agg.winddir, 0), "pressure_inhg": r(agg.baromrelin, 3),
    "solar_wm2": r(agg.solarradiation, 0), "uv": r(agg.uv, 1),
    "rain_rate_in_hr": r(agg.hourlyrainin, 3), "rain_today_in": r(agg.dailyrainin, 3),
    "indoor_temp_f": r(agg.tempinf, 1), "indoor_humidity": r(agg.humidityin, 0),
}

# ------------------------------------------------------------------ daily
asc["local_date"] = asc.date.dt.date
dgrp = asc.groupby("local_date")
daily_core = pd.DataFrame({
    "temp_min": dgrp.tempf.min(), "temp_max": dgrp.tempf.max(), "temp_mean": dgrp.tempf.mean(),
    "humidity_mean": dgrp.humidity.mean(), "gust_max": dgrp.windgustmph.max(),
    "wind_mean": dgrp.windspeedmph.mean(), "pressure_mean": dgrp.baromrelin.mean(),
    "uv_max": dgrp.uv.max(), "n": dgrp.size(),
})
daily_rain = ra.extract_daily_rainfall(hist)  # repo calc (max dailyrainin per local day)
daily_core["rain_in"] = daily_rain.set_index("date")["rainfall"]

# energy catalog (read-only; same flow as load_or_update_data)
energy = None


@section("energy_catalog")
def _():
    global energy
    ex = load_energy_catalog()
    energy = update_energy_catalog(hist, ex) if not ex.empty else None
    if energy is not None:
        energy["period_start"] = pd.to_datetime(energy["period_start"]).dt.tz_convert(TZ)
    print("energy periods:", None if energy is None else len(energy))


if energy is not None:
    de = aggregate_to_daily(energy).set_index("date")["daily_kwh"]
    daily_core["solar_kwh_m2"] = de
else:
    NOTES.append("solar energy catalog unavailable")
    daily_core["solar_kwh_m2"] = np.nan

d365 = daily_core.sort_index().iloc[-366:]
daily = {
    "date": [str(i) for i in d365.index],
    **{k: r(d365[k], 3 if k in ("rain_in", "solar_kwh_m2") else 1)
       for k in ["temp_min", "temp_max", "temp_mean", "rain_in", "solar_kwh_m2",
                 "gust_max", "wind_mean", "humidity_mean", "pressure_mean", "uv_max"]},
}

# daily_all: whole archive, one row per station-local day, with coverage
dall = daily_core.sort_index()
dall["coverage"] = (dall["n"] / 288.0).clip(upper=1.0)
daily_all = {
    "date": [str(i) for i in dall.index],
    **{k: r(dall[k], 3 if k in ("rain_in", "solar_kwh_m2", "coverage") else 1)
       for k in ["temp_min", "temp_max", "temp_mean", "rain_in", "solar_kwh_m2",
                 "gust_max", "coverage"]},
}

# ------------------------------------------------------------------ overview extras
overview = {}


@section("overview.temp_ranges")
def _():
    # Windows are anchored on the archive's last local day (the archive can lag the live
    # reading by about a day), so "today" is never empty. Adds p25/p75 from 5-minute readings.
    end_ts = asc.date.iloc[-1]
    day0 = end_ts.normalize()
    wins = {
        "today": (day0, end_ts),
        "yesterday": (day0 - pd.Timedelta(days=1), day0),
        "last 7d": (end_ts - pd.Timedelta(days=7), end_ts),
        "last 30d": (end_ts - pd.Timedelta(days=30), end_ts),
        "last 90d": (end_ts - pd.Timedelta(days=90), end_ts),
        "last 365d": (end_ts - pd.Timedelta(days=365), end_ts),
    }
    cur = float(asc.tempf.iloc[-1])
    out = {}
    for label, (a, b) in wins.items():
        x = asc.loc[(asc.date >= a) & (asc.date <= b), "tempf"].dropna()
        out[label] = {
            "min": None if x.empty else round(float(x.min()), 1),
            "max": None if x.empty else round(float(x.max()), 1),
            "p25": None if x.empty else round(float(x.quantile(.25)), 1),
            "p75": None if x.empty else round(float(x.quantile(.75)), 1),
            "n": int(len(x)),
            "start": a.isoformat(), "end": b.isoformat(),
            "current": round(cur, 1),
        }
    out["_anchor"] = {"archive_end": end_ts.isoformat(), "source": "5-minute archive readings"}
    overview["temp_ranges"] = out


@section("overview.windrose")
def _():
    out = {}
    for vcol, dcol, label in [("windspeedmph", "winddir", "wind"), ("windgustmph", "winddir", "gust")]:
        gd, vl, dl = lo_dp.prepare_polar_chart_data(hist.copy(), vcol, dcol)
        out[label] = {
            "value_labels": [str(x) for x in vl], "direction_labels": [str(x) for x in dl],
            "rows": gd.astype(str).values.tolist() if False else
            [[str(a), str(b), float(c)] for a, b, c in
             gd[[gd.columns[0], gd.columns[1], gd.columns[-1]]].itertuples(index=False)],
            "columns": [str(c) for c in gd.columns],
        }
    overview["windrose"] = out


@section("overview.monthly_temp")
def _():
    # replaces the Streamlit box plot (month grouping): quantiles per calendar month
    m = asc.assign(month=asc.date.dt.month).groupby("month").tempf
    overview["temp_by_month"] = {
        "month": list(range(1, 13)),
        "min": r(m.min(), 1), "q1": r(m.quantile(.25), 1), "median": r(m.median(), 1),
        "q3": r(m.quantile(.75), 1), "max": r(m.max(), 1),
    }


@section("overview.rain_last30_by_hour")
def _():
    end_d = pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)
    st_d = end_d - pd.Timedelta(days=30)
    acc = rain_viz.prepare_rain_accumulation_heatmap_data(
        archive_df=hist, start_date=st_d.tz_convert("UTC"), end_date=end_d.tz_convert("UTC"),
        timezone=TZ, num_days=30, row_mode="day")
    acc = acc[acc["accumulation"] > 0]
    overview["rain_last30_hourly"] = {
        "date": [str(d) for d in acc["date"]], "hour": [int(h) for h in acc["hour"]],
        "rain_in": r(acc["accumulation"], 3),
    }


# ------------------------------------------------------------------ rain tab
rain = {}


@section("rain")
def _():
    drd, stats, cur, ctx = ra.prepare_rain_tab_data(hist, last_data)
    rain["current_values"] = cur
    rain["stats"] = {k: (str(v) if not isinstance(v, (int, float, np.number)) else v) for k, v in stats.items()}
    rain["rolling_context"] = ctx.to_dict(orient="records")
    # violin: summarize each window's historical distribution as quantiles (raw arrays are huge)
    end_date = pd.to_datetime(drd["date"]).max()
    vd = ra.prepare_violin_plot_data(drd, (1, 7, 30, 90, 365), None, end_date)
    qs = [0, .05, .1, .25, .5, .75, .9, .95, 1]
    rain["distribution"] = {
        str(w): {"current": v.get("current"), "percentile": v.get("percentile"),
                 "quantile_levels": qs, "quantiles": [float(np.quantile(v["values"], q)) for q in qs],
                 "n": int(len(v["values"]))}
        for w, v in vd.items()
    }
    # YoY cumulative, one array per year
    yoy = ra.prepare_year_over_year_accumulation(drd, 1, 366)
    rain["yoy_cumulative"] = {
        str(y): {"day_of_year": g_.day_of_year.tolist(), "cum_in": r(g_.cumulative_rainfall, 3)}
        for y, g_ in yoy.groupby("year")
    }
    dd = drd.copy()
    dd["date"] = pd.to_datetime(dd["date"])
    rain["daily_all"] = {"date": [d.strftime("%Y-%m-%d") for d in dd.date], "rain_in": r(dd.rainfall, 3)}
    mo = dd.assign(ym=dd.date.dt.strftime("%Y-%m")).groupby("ym").rainfall.sum()
    rain["monthly_totals"] = {"ym": list(mo.index), "rain_in": r(mo, 2)}
    rain["dry_spell"] = ra.calculate_dry_spell_stats(hist)


# ------------------------------------------------------------------ rain events
events_out = None
cat = RainEventCatalog(MAC, "parquet")


@section("rain_events")
def _():
    global events_out
    ev = cat.load_catalog()
    if ev.empty:
        ev = cat.detect_and_catalog_events(hist, auto_save=False)
    else:
        ev = cat.update_catalog_with_new_data(hist, ev)  # in-memory only, as the app does
    ev = ev.sort_values("start_time").reset_index(drop=True)
    print("events:", len(ev), ev.columns.tolist())
    rows = []
    for _, e in ev.iterrows():
        row = {}
        for c in ev.columns:
            v = e[c]
            if isinstance(v, str) and c == "flags":
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            row[c] = v
        rows.append(row)
    events_out = rows
    # detail series (accumulation + rate) for last 40 events
    cut = asc.dateutc.values
    for row in rows[-40:]:
        s = int(pd.Timestamp(row["start_time"]).value // 1_000_000)
        e_ = int(pd.Timestamp(row["end_time"]).value // 1_000_000)
        w = asc[(asc.dateutc >= s) & (asc.dateutc <= e_)]
        if len(w) > 90:
            w = w.iloc[np.linspace(0, len(w) - 1, 90).astype(int)]
        row["series"] = {"t": [int(x) for x in w.dateutc], "accum_in": r(w.eventrainin, 3),
                         "rate_in_hr": r(w.hourlyrainin, 3)}


# ------------------------------------------------------------------ solar
solar = {}


@section("solar")
def _():
    if energy is None:
        raise RuntimeError("no energy catalog")
    tiles = calculate_tile_metrics(energy)
    solar["tiles"] = {k: {kk: vv for kk, vv in v.items()} for k, v in tiles.items()}
    dly = aggregate_to_daily(energy)
    dly["date"] = pd.to_datetime(dly["date"])
    solar["daily_all"] = {"date": [d.strftime("%Y-%m-%d") for d in dly.date], "kwh_m2": r(dly.daily_kwh, 3)}
    ym = dly.assign(y=dly.date.dt.year, m=dly.date.dt.month).groupby(["y", "m"]).daily_kwh.sum().reset_index()
    solar["year_month_kwh"] = ym.rename(columns={"daily_kwh": "kwh_m2"}).round(2).to_dict(orient="records")
    # 15-min periods, last 7 days (Day/15min view drill-down)
    e7 = energy[energy.period_start >= energy.period_start.max() - pd.Timedelta(days=7)]
    solar["periods_15min_7d"] = {"start": [p.isoformat() for p in e7.period_start], "kwh_m2": r(e7.energy_kwh, 4)}
    # Day/15min heatmap, last 120 days, hours 4-21: (date, slot, kwh) sparse
    e120 = energy[energy.period_start >= energy.period_start.max() - pd.Timedelta(days=120)].copy()
    e120["hr"] = e120.period_start.dt.hour + e120.period_start.dt.minute / 60
    e120 = e120[(e120.hr >= 4) & (e120.hr < 21)]
    solar["heatmap_15min_120d"] = {
        "date": [p.strftime("%Y-%m-%d") for p in e120.period_start],
        "hour": r(e120.hr, 2), "kwh_m2": r(e120.energy_kwh, 4)}
    # typical hourly profile per month (mean W/m2 equivalent: kWh/m2 per hour)
    e = energy.copy()
    e["month"] = e.period_start.dt.month
    e["hour"] = e.period_start.dt.hour
    prof = e.groupby(["month", "hour"]).energy_kwh.sum() * 1.0
    ndays = e.groupby("month").period_start.apply(lambda s: s.dt.date.nunique())
    prof = (prof / ndays.reindex(prof.index.get_level_values(0)).values).reset_index(name="kwh_m2_per_day")
    solar["hourly_profile_by_month"] = prof.round(4).to_dict(orient="records")
    # peak times per day (hour of max 15-min period) last 365d + peak W/m2 from archive
    a = asc[asc.dateutc >= end - 365 * 86400_000]
    idx = a.groupby("local_date").solarradiation.idxmax()
    pk = a.loc[idx.values]
    solar["daily_peak"] = {
        "date": [str(d) for d in pk.local_date],
        "peak_wm2": r(pk.solarradiation, 0),
        "peak_time": [t.strftime("%H:%M") for t in pk.date],
        "uv_max": r(a.groupby("local_date").uv.max().reindex(pk.local_date), 1),
    }


# ------------------------------------------------------------------ assemble
snapshot = {
    "meta": {
        "station_name": station_name, "timezone": TZ,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_range": [asc.date.min().isoformat(), asc.date.max().isoformat()],
        "archive_rows": int(len(asc)), "recent_interval_min": 15,
        "units": {"temp": "F", "wind": "mph", "pressure": "inHg", "rain": "in",
                  "rain_rate": "in/hr", "solar_irradiance": "W/m2",
                  "solar_energy": "kWh/m2", "uv": "index"},
        "notes": NOTES,
    },
    "current": current,
    "recent": recent,
    "daily": daily,
    "daily_all": daily_all,
    "overview": overview,
    "rain": rain,
    "rain_events": events_out,
    "solar": solar,
}
OUT.write_text(json.dumps(clean(snapshot), separators=(",", ":")))
print(f"wrote {OUT} {OUT.stat().st_size/1e6:.2f} MB in {time.time()-t0:.0f}s; notes={NOTES}")
