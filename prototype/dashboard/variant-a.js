// Variant A, "Today": one continuous scroll, phone first. Measured data only.
(function () {
  const TZ = "America/Los_Angeles";
  const H = 3600e3, MIN = 60e3, DAY = 24 * H;
  const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];

  const CSS = `
  .va { --col: 680px; font-family: var(--sans); color: var(--ink); background: var(--ground); min-height: 100vh; line-height: 1.4; }
  .va * { box-sizing: border-box; }
  .va-nav { position: sticky; top: 0; z-index: 50; background: color-mix(in srgb, var(--ground) 92%, transparent); backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule); }
  .va-nav-in { max-width: var(--col); margin: 0 auto; padding: 8px 16px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
  .va-nav button { all: unset; cursor: pointer; text-align: center; min-height: 40px; display: flex; align-items: center; justify-content: center; border-radius: 999px; font-size: var(--step--1); font-weight: 600; color: var(--ink-2); letter-spacing: .01em; }
  .va-nav button[aria-current="true"] { background: var(--ink); color: var(--ground); }
  .va-nav button:focus-visible { outline: 2px solid var(--rain-2); outline-offset: 1px; }
  .va-main { max-width: var(--col); margin: 0 auto; padding: 0 16px 112px; }
  .va section { scroll-margin-top: 56px; }
  .va-sec { border-top: 1px solid var(--rule-strong); margin-top: var(--s-7); padding-top: var(--s-5); }
  .va-eyebrow { font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-2); margin: 0 0 var(--s-2); }
  .va h1, .va h2, .va h3 { font-family: var(--serif); font-weight: 600; margin: 0; letter-spacing: -.01em; }
  .va h2 { font-size: 1.5rem; line-height: 1.18; text-wrap: balance; }
  .va h3 { font-size: 1.25rem; line-height: 1.22; text-wrap: balance; }
  .va-sub { margin: var(--s-2) 0 var(--s-4); font-size: .9375rem; color: var(--ink-2); text-wrap: pretty; }
  .va-foot { margin: var(--s-2) 0 0; font-size: .75rem; color: var(--ink-2); text-wrap: pretty; }
  .va-block { margin-top: var(--s-7); }

  .va-station { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding-top: var(--s-5); font-size: var(--step--1); color: var(--ink-2); }
  .va-station b { color: var(--ink); font-weight: 700; letter-spacing: .02em; }
  .va-summary { font-size: 1.875rem; line-height: 1.1; letter-spacing: -.015em; margin: var(--s-3) 0 var(--s-2) !important; text-wrap: balance; }
  .va-lastrain { margin: 0 0 var(--s-5); font-size: .875rem; color: var(--ink-2); }
  .va-hero { display: grid; grid-template-columns: auto 1fr; gap: var(--s-4) var(--s-5); align-items: end; }
  .va-hero { grid-template-columns: auto minmax(0, 1fr); gap: var(--s-4) var(--s-4); }
  .va-temp { font-size: clamp(3.5rem, 16.5vw, 4.5rem); line-height: .86; font-weight: 600; letter-spacing: -.045em; white-space: nowrap; }
  .va-temp sup { font-size: .4em; vertical-align: top; position: relative; top: .1em; margin-left: 2px; font-weight: 500; letter-spacing: 0; }
  .va-temp small { display: block; margin-top: 10px; font-size: .8125rem; letter-spacing: 0; font-weight: 500; color: var(--ink-2); }
  .va-kv { margin: 0; display: grid; grid-template-columns: 1fr; }
  .va-kv div { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; padding: 5px 0; border-top: 1px solid var(--rule); font-size: .875rem; }
  .va-kv div:first-child { border-top: 0; }
  .va-kv dt { color: var(--ink-2); }
  .va-kv dd { margin: 0; font-weight: 600; text-align: right; white-space: nowrap; }
  .va-kv dd i { font-style: normal; font-weight: 400; color: var(--ink-2); }
  @media (max-width: 360px) { .va-hero { grid-template-columns: 1fr; } }

  .va-rib { position: relative; margin-top: var(--s-3); }
  .va-band { position: relative; height: 84px; border-radius: 6px; overflow: hidden; background: var(--night); }
  .va-band svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .va-hatch { position: absolute; top: 0; bottom: 0; }
  .va-tlabel { position: absolute; transform: translate(-50%, -125%); font-size: .75rem; font-weight: 700; color: var(--ink); text-shadow: 0 0 3px var(--surface), 0 0 3px var(--surface), 0 0 6px var(--surface); white-space: nowrap; }
  .va-tlabel.lo { transform: translate(-50%, 30%); }
  .va-nowdot { position: absolute; right: 3px; width: 10px; height: 10px; border-radius: 50%; background: var(--note); box-shadow: 0 0 0 2px var(--surface); transform: translateY(-50%); }
  .va-ticks { position: relative; height: 36px; margin-top: 4px; font-size: .6875rem; color: var(--ink-2); }
  .va-tick { position: absolute; top: 0; transform: translateX(-50%); text-align: center; white-space: nowrap; line-height: 1.25; }
  .va-tick::before { content: ""; display: block; width: 1px; height: 5px; background: var(--rule-strong); margin: 0 auto 2px; }
  .va-tick b { display: block; color: var(--ink); font-weight: 600; }
  .va-nowlab { position: absolute; right: 0; top: 0; text-align: right; font-size: .75rem; font-weight: 700; color: var(--note); line-height: 1.25; }
  .va-key { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; margin-top: var(--s-2); font-size: .75rem; color: var(--ink-2); }
  .va-key > span { display: inline-flex; align-items: center; gap: 5px; }
  .va-sw { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
  .va-ramp { display: inline-flex; gap: 1px; }
  .va-ramp i { width: 12px; height: 10px; display: block; }

  .va-week { margin-top: var(--s-2); }
  .va-wrow, .va-whead { display: grid; grid-template-columns: 48px 62px 30px 1fr 30px; gap: 0 8px; align-items: center; }
  .va-wrow { min-height: 42px; border-top: 1px solid var(--rule); font-size: .875rem; }
  .va-wday { font-weight: 600; line-height: 1.15; }
  .va-wday small { display: block; font-weight: 400; color: var(--ink-2); font-size: .6875rem; }
  .va-wrain { display: flex; align-items: center; gap: 4px; }
  .va-wrain svg { width: 11px; height: 14px; flex: none; }
  .va-wrain.zero { color: var(--ink-3); }
  .va-wlo { text-align: right; color: var(--ink-2); }
  .va-whi { font-weight: 600; }
  .va-wtrack { position: relative; height: 18px; }
  .va-wnorm { position: absolute; top: 1px; bottom: 1px; background: var(--rule); border-radius: 3px; }
  .va-wbar { position: absolute; top: 5px; height: 8px; border-radius: 4px; }
  .va-wgap { grid-column: 3 / 6; align-self: stretch; margin: 7px 0; position: relative; display: flex; align-items: center; justify-content: center; font-size: .75rem; color: var(--ink-2); border-radius: 4px; }
  .va-whead { font-size: .6875rem; color: var(--ink-2); }
  .va-waxis { position: relative; height: 16px; grid-column: 4; }
  .va-waxis span { position: absolute; transform: translateX(-50%); }
  .va-wdot { position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%; background: var(--note); transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--ground); }

  .va-roll { margin-top: var(--s-2); }
  .va-rr { padding: 10px 0 12px; border-top: 1px solid var(--rule); }
  .va-rr-h { display: flex; justify-content: space-between; align-items: baseline; font-size: .875rem; gap: 8px; }
  .va-rr-h b { font-weight: 600; }
  .va-rr-h span { color: var(--ink-2); font-size: .75rem; }
  .va-rr-h em { font-style: normal; font-weight: 700; font-size: .8125rem; }
  .va-pair { display: grid; grid-template-columns: 1fr 112px; gap: 5px 8px; align-items: center; margin-top: 6px; }
  .va-pair .t { height: 10px; position: relative; }
  .va-pair .t i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 4px 4px 0; min-width: 2px; }
  .va-pair .v { font-size: .8125rem; line-height: 1.15; }
  .va-pair .v small { display: block; color: var(--ink-2); font-size: .6875rem; }
  .va-invalid { margin-top: 6px; padding: 10px 12px; border-radius: 4px; font-size: .75rem; color: var(--ink); }

  .va-ev { margin-top: var(--s-2); }
  .va-evh, .va-evr { display: grid; grid-template-columns: 1.5fr .85fr .75fr .85fr; gap: 0 8px; align-items: baseline; }
  .va-evh { font-size: .6875rem; color: var(--ink-2); padding-bottom: 4px; }
  .va-evr { padding: 9px 0 8px; border-top: 1px solid var(--rule); font-size: .875rem; }
  .va-evr .s { font-weight: 600; line-height: 1.2; }
  .va-evr .s small { display: block; font-weight: 400; color: var(--ink-2); font-size: .6875rem; }
  .va-evr .n, .va-evh .n { text-align: right; }
  .va-evr .n small { color: var(--ink-2); font-size: .6875rem; }
  .va-evr .tot { font-weight: 700; }
  .va-evr .meter { grid-column: 1 / -1; height: 3px; margin-top: 7px; position: relative; }
  .va-evr .meter i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--rain-2); border-radius: 0 2px 2px 0; }
  .va-promo { margin-top: var(--s-6); padding: var(--s-4) 0 var(--s-2); border-top: 2px solid var(--rain-3); }
  .va-chart { width: 100%; margin-top: var(--s-2); }
  `;

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (n, p = 2) => Number(n).toFixed(p);
  const iso = (d) => d.toISOString().slice(0, 10);
  const parseISO = (s) => new Date(s + "T00:00:00Z");
  const addDays = (s, n) => iso(new Date(parseISO(s).getTime() + n * DAY));
  const shiftYear = (s, n) => { const d = parseISO(s); d.setUTCFullYear(d.getUTCFullYear() + n); return iso(d); };
  const fmtTZ = (ms, opts) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, ...opts }).format(new Date(ms));
  const laParts = (ms) => Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: TZ, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).formatToParts(new Date(ms)).map((p) => [p.type, p.value]));
  const laStr = (ms) => { const p = laParts(ms); return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`; };
  const laHour = (ms) => +laParts(ms).hour;
  const dow = (s) => ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][parseISO(s).getUTCDay()];
  const dayNum = (s) => +s.slice(8);
  const monShort = (s) => MONTHS[+s.slice(5, 7) - 1].slice(0, 3);
  const dur = (min) => { const h = Math.floor(min / 60), m = Math.round(min % 60); return h >= 24 ? `${Math.floor(h / 24)} d ${h % 24} h` : h ? `${h} h ${m} m` : `${m} m`; };
  const compass = (deg) => ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"][Math.round(deg / 22.5) % 16];
  const pct = (a, b) => (b ? Math.round(((a - b) / b) * 100) : null);
  const rangeLabel = (f, t) => `${dayNum(f)} ${monShort(f)} to ${dayNum(t)} ${monShort(t)} ${t.slice(0, 4)}`;

  function sumRange(map, from, to) {
    let sum = 0, missing = 0;
    for (let d = from; d <= to; d = addDays(d, 1)) { if (d in map) sum += map[d]; else missing++; }
    return { sum, missing };
  }

  // ---------- Now ----------
  function summary(c) {
    const t = c.temp_f;
    const tw = t < 32 ? "Freezing" : t < 45 ? "Cold" : t < 55 ? "Cool" : t < 68 ? "Mild" : t < 80 ? "Warm" : "Hot";
    const w = Math.max(c.wind_avg10m_mph ?? 0, c.wind_mph ?? 0);
    let s = w < 3 ? `${tw} and still` : w < 8 ? `${tw} with a light breeze` : w < 15 ? `${tw} and breezy` : `${tw} and windy`;
    if (c.rain_rate_in_hr > 0) s += `, rain falling at ${fmt(c.rain_rate_in_hr)} in/hr`;
    else if (c.rain_today_in > 0) s += ` after ${fmt(c.rain_today_in)} in of rain today`;
    else s += ", and dry all day";
    return s;
  }

  function nowSection(d, c) {
    const nowMs = Date.parse(c.timestamp);
    const raining = c.rain_rate_in_hr > 0;
    const pt = c.pressure_trend_3h_inhg;
    const ptxt = Math.abs(pt) < 0.03 ? "steady" : pt > 0 ? "rising" : "falling";
    const lastRainMs = c.last_rain ? Date.parse(c.last_rain) : null;
    const since = lastRainMs ? (nowMs - lastRainMs) / H : null;
    const w = c.wind_avg10m_mph;
    const tr = c.temp_trend_symbol === "↓" ? "Falling" : c.temp_trend_symbol === "↑" ? "Rising" : "Steady";
    return `
      <div class="va-station"><b>${d.meta.station_name}</b><span>${fmtTZ(nowMs, { weekday: "short", month: "short", day: "numeric" })}, ${fmtTZ(nowMs, { hour: "numeric", minute: "2-digit" })}</span></div>
      <h1 class="va-summary">${summary(c)}</h1>
      <p class="va-lastrain">${raining ? "Raining now." : lastRainMs ? `Last rain ${fmtTZ(lastRainMs, { hour: "numeric", minute: "2-digit" })}, ${since < 1 ? Math.round(since * 60) + " min" : since.toFixed(since < 10 ? 1 : 0) + " h"} ago.` : ""} Live from the station.</p>
      <div class="va-hero">
        <div class="va-temp">${fmt(c.temp_f, 1)}<sup>°F</sup><small>${tr} ${c.temp_trend_symbol || ""}</small></div>
        <dl class="va-kv">
          <div><dt>Feels like</dt><dd>${fmt(c.feels_like_f, 0)}°</dd></div>
          <div><dt>Wind</dt><dd>${w < 0.5 ? "Calm" : fmt(w, 0) + " mph " + compass(c.wind_dir)}</dd></div>
          <div><dt>Humidity</dt><dd>${c.humidity}%</dd></div>
          <div><dt>Pressure</dt><dd>${fmt(c.pressure_inhg, 2)} <i>${ptxt}</i></dd></div>
          <div><dt>Rain today</dt><dd>${fmt(c.rain_today_in)} in</dd></div>
        </dl>
      </div>`;
  }

  // ---------- Ribbon ----------
  function ribbonBlock(d, c) {
    const r = d.recent, step = (r.interval_min || 15) * MIN;
    const tEnd = Date.parse(c.timestamp), tStart = tEnd - 48 * H, span = tEnd - tStart;
    const X = (t) => ((t - tStart) / span) * 1000;
    const P = (t) => ((t - tStart) / span) * 100;
    const idx = [];
    r.t.forEach((t, i) => { if (t + step > tStart && t <= tEnd) idx.push(i); });
    let svg = "", hatch = "", rainIn = 0;
    const rampFor = (v) => (v <= 0 ? null : v < 0.03 ? "--rain-1" : v < 0.1 ? "--rain-2" : v < 0.2 ? "--rain-3" : "--rain-4");
    const segs = [];
    idx.forEach((i) => {
      const last = segs[segs.length - 1];
      if (last && r.t[i] - r.t[last[last.length - 1]] <= step * 1.5) last.push(i); else segs.push([i]);
    });
    idx.forEach((i) => {
      const t0 = Math.max(r.t[i], tStart), t1 = Math.min(r.t[i] + step, tEnd);
      const x0 = X(t0), w = X(t1) - x0 + 0.4;
      const day = r.solar_wm2[i] > 0;
      svg += `<rect x="${x0.toFixed(2)}" y="0" width="${w.toFixed(2)}" height="100" style="fill:${day ? "var(--sun-soft)" : "var(--night)"}"/>`;
      const rc = rampFor(r.rain_rate_in_hr[i]);
      if (rc) { svg += `<rect x="${x0.toFixed(2)}" y="0" width="${w.toFixed(2)}" height="100" style="fill:var(${rc})"/>`; rainIn += r.rain_rate_in_hr[i] * 0.25; }
    });
    let cursor = tStart;
    const gaps = [];
    segs.forEach((s) => {
      const a = Math.max(r.t[s[0]], tStart), b = Math.min(r.t[s[s.length - 1]] + step, tEnd);
      if (a - cursor > step * 0.5) gaps.push([cursor, a]);
      cursor = Math.max(cursor, b);
    });
    if (tEnd - cursor > step * 0.5) gaps.push([cursor, tEnd]);
    gaps.forEach(([a, b]) => { hatch += `<div class="va-hatch gap-hatch" style="left:${P(a).toFixed(2)}%;width:${(P(b) - P(a)).toFixed(2)}%"></div>`; });
    const pendingH = (tEnd - cursor) / H;
    const archivedH = gaps.reduce((s, [a, b]) => s - (b - a) / H, 48);

    const temps = idx.map((i) => r.temp_f[i]).concat([c.temp_f]);
    const lo = Math.min(...temps) - 2, hi = Math.max(...temps) + 2;
    const Y = (v) => 14 + (1 - (v - lo) / (hi - lo)) * 72;
    let line = "";
    segs.forEach((s) => {
      const pts = s.map((i) => `${X(Math.min(r.t[i] + step / 2, tEnd)).toFixed(2)},${Y(r.temp_f[i]).toFixed(2)}`).join(" ");
      line += `<polyline points="${pts}" fill="none" vector-effect="non-scaling-stroke" style="stroke:var(--surface);stroke-width:5;stroke-linejoin:round;stroke-linecap:round"/><polyline points="${pts}" fill="none" vector-effect="non-scaling-stroke" style="stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`;
    });
    let imax = idx[0], imin = idx[0];
    idx.forEach((i) => { if (r.temp_f[i] > r.temp_f[imax]) imax = i; if (r.temp_f[i] < r.temp_f[imin]) imin = i; });
    const lab = (i, cls) => `<span class="va-tlabel ${cls}" style="left:${Math.min(92, Math.max(6, P(r.t[i] + step / 2)))}%;top:${(Y(r.temp_f[i]) * 0.84).toFixed(1)}px">${fmt(r.temp_f[i], 0)}°</span>`;

    let ticks = "";
    for (let t = Math.ceil(tStart / H) * H; t < tEnd; t += H) {
      const h = laHour(t);
      if (h % 6) continue;
      const x = P(t);
      if (x < 5 || x > 84) continue;
      const hl = h === 0 ? "" : h === 12 ? "12p" : h < 12 ? h + "a" : h - 12 + "p";
      ticks += `<div class="va-tick" style="left:${x.toFixed(2)}%"><b>${h === 0 ? fmtTZ(t + 6 * H, { weekday: "short" }) : hl}</b></div>`;
    }
    const rs = ["--rain-1", "--rain-2", "--rain-3", "--rain-4"].map((v) => `<i style="background:var(${v})"></i>`).join("");
    const endMs = Date.parse(d.meta.archive_range[1]);
    const head = pendingH > 6
      ? `The archive covers ${Math.round(archivedH)} of the last 48 hours, and ${fmt(rainIn)} in of rain fell in them`
      : `${fmt(rainIn)} in of rain in the last 48 hours`;
    return `
      <div class="va-block" id="va-ribbon">
        <h2>${head}</h2>
        <p class="va-sub">Rain, daylight and temperature every 15 minutes for the 48 hours up to now. In the recorded hours the temperature ranged from ${fmt(r.temp_f[imin], 0)}° to ${fmt(r.temp_f[imax], 0)}°.</p>
        <div class="va-rib">
          <div class="va-band" role="img" aria-label="48-hour condition ribbon showing rain intensity, day and night, and temperature">
            <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">${svg}</svg>
            ${hatch}
            <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">${line}</svg>
            ${lab(imax, "")}${lab(imin, "lo")}
            <div class="va-nowdot" style="top:${(Y(c.temp_f) * 0.84).toFixed(1)}px"></div>
          </div>
          <div class="va-ticks">${ticks}<div class="va-nowlab">Now<br>${fmt(c.temp_f, 1)}°</div></div>
        </div>
        <div class="va-key">
          <span><i class="va-sw" style="background:var(--night)"></i>Night</span>
          <span><i class="va-sw" style="background:var(--sun-soft)"></i>Day</span>
          <span>Rain <span class="va-ramp">${rs}</span> light to heavy</span>
          <span><i class="va-sw gap-hatch"></i>Not archived</span>
        </div>
        <p class="va-foot">The archive ends ${fmtTZ(endMs, { weekday: "short", hour: "numeric", minute: "2-digit" })}, about ${Math.round(pendingH)} hours before the live reading, so only the live temperature is shown for that stretch. Source: station 15-minute record and live reading.</p>
      </div>`;
  }

  // ---------- Week ----------
  const DROP = `<svg viewBox="0 0 11 14" aria-hidden="true"><path d="M5.5 0C5.5 0 0 6 0 9a5.5 5.5 0 0 0 11 0C11 6 5.5 0 5.5 0z" style="fill:var(--rain-2)"/></svg>`;

  function weekBlock(d, c) {
    const rainMap = Object.fromEntries(d.daily_all.date.map((k, i) => [k, d.daily_all.rain_in[i]]).filter(([, v], i) => d.daily_all.coverage[i] >= 0.8 && v != null));
    const D = d.daily_all, dm = {};
    D.date.forEach((k, i) => { if (D.coverage[i] >= 0.8 && D.temp_min[i] != null) dm[k] = i; });
    const qt = (arr, q) => { const a = arr.slice().sort((x, y) => x - y); const p = (a.length - 1) * q, f = Math.floor(p); return a[f] + (a[Math.min(f + 1, a.length - 1)] - a[f]) * (p - f); };
    const allIdx = Object.fromEntries(D.date.map((k, i) => [k, i]));
    const normal = (k) => {
      const lows = [], highs = [], yr = +k.slice(0, 4);
      for (let y = 2023; y < yr; y++) for (let o = -7; o <= 7; o++) { const kk = addDays(shiftYear(k, y - yr), o); if (kk in dm) { lows.push(D.temp_min[dm[kk]]); highs.push(D.temp_max[dm[kk]]); } }
      return lows.length >= 10 ? { n: lows.length, lo: qt(lows, 0.25), hi: qt(highs, 0.75) } : null;
    };
    const lastDay = d.daily.date[d.daily.date.length - 1];
    const days = [];
    for (let i = 0; i < 10; i++) days.push(addDays(lastDay, -i));
    const month = +lastDay.slice(5, 7);
    const norms = Object.fromEntries(days.map((k) => [k, normal(k)]));
    const vals = [c.temp_f];
    days.forEach((k) => { if (k in dm) vals.push(D.temp_min[dm[k]], D.temp_max[dm[k]]); if (norms[k]) vals.push(norms[k].lo, norms[k].hi); });
    const sLo = Math.floor(Math.min(...vals) / 5) * 5, sHi = Math.ceil(Math.max(...vals) / 5) * 5;
    const X = (v) => ((v - sLo) / (sHi - sLo)) * 100;
    const stops = [[28, "--temp-cold"], [45, "--temp-cool"], [55, "--temp-mild"], [68, "--temp-warm"], [88, "--temp-hot"]]
      .map(([t, v]) => `var(${v}) ${Math.max(0, Math.min(100, X(t))).toFixed(1)}%`).join(",");
    const grad = `linear-gradient(to right, ${stops})`;
    let ticks = "";
    for (let t = Math.ceil(sLo / 10) * 10; t <= sHi; t += 10) ticks += `<span style="left:${X(t).toFixed(1)}%">${t}°</span>`;
    const head = `<div class="va-whead"><div class="va-waxis">${ticks}</div></div>`;

    const rowFor = (k) => {
      const i = dm[k], hasT = i !== undefined, hasR = k in rainMap;
      const rain = hasR ? rainMap[k] : null;
      const rainEl = hasR ? `<div class="va-wrain ${rain > 0 ? "" : "zero"}">${rain > 0 ? DROP + fmt(rain) : "—"}</div>` : `<div class="va-wrain zero gap-hatch" style="align-self:stretch;margin:7px 0;border-radius:3px;justify-content:center;font-size:.75rem">n/a</div>`;
      const dayEl = `<div class="va-wday">${dow(k)}<small>${dayNum(k)} ${monShort(k)}</small></div>`;
      if (!hasT) return `<div class="va-wrow">${dayEl}${rainEl}<div class="va-wgap gap-hatch">No temperature record</div></div>`;
      const lo = D.temp_min[i], hi = D.temp_max[i];
      const left = X(lo), w = Math.max(X(hi) - X(lo), 1.5);
      const bg = `background-image:${grad};background-size:${(10000 / w).toFixed(1)}% 100%;background-position:${w >= 100 ? 0 : ((left / (100 - w)) * 100).toFixed(1)}% 0`;
      return `<div class="va-wrow">${dayEl}${rainEl}<div class="va-wlo">${fmt(lo, 0)}°</div>
        <div class="va-wtrack">${norms[k] ? `<div class="va-wnorm" style="left:${X(norms[k].lo).toFixed(1)}%;width:${(X(norms[k].hi) - X(norms[k].lo)).toFixed(1)}%"></div>` : ""}<div class="va-wbar" style="left:${left.toFixed(1)}%;width:${w.toFixed(1)}%;${bg}"></div></div>
        <div class="va-whi">${fmt(hi, 0)}°</div></div>`;
    };
    const todayKey = laStr(Date.parse(c.timestamp)).slice(0, 10);
    const todayRow = `<div class="va-wrow">
      <div class="va-wday">Today<small>${dayNum(todayKey)} ${monShort(todayKey)}</small></div>
      <div class="va-wrain ${c.rain_today_in > 0 ? "" : "zero"}">${c.rain_today_in > 0 ? DROP + fmt(c.rain_today_in) : "—"}</div>
      <div class="va-wgap gap-hatch">Range not archived yet<span class="va-wdot" style="left:${(((X(c.temp_f) / 100) * 0.7 + 0.15) * 100).toFixed(0)}%;display:none"></span></div></div>`;

    const rains = days.filter((k) => k in rainMap).map((k) => [k, rainMap[k]]);
    const wetCount = rains.filter(([, v]) => v > 0).length;
    const total10 = rains.reduce((a, [, v]) => a + v, 0);
    const wettest = rains.slice().sort((a, b) => b[1] - a[1])[0];
    const highs = days.filter((k) => k in dm).map((k) => [k, D.temp_max[dm[k]]]).sort((a, b) => b[1] - a[1]);
    const headline = wetCount === 0 ? `Ten dry days in a row`
      : wetCount <= 3 ? `Only ${wetCount} of the last 10 days had rain, and ${dow(wettest[0])} ${dayNum(wettest[0])} brought most of it: ${fmt(wettest[1])} in`
      : `${wetCount} of the last 10 days brought rain, ${fmt(total10)} in in all`;
    return `
      <div class="va-block" id="va-week">
        <h2>${headline}</h2>
        <p class="va-sub">The last ten archived days: rain, and the day's low-to-high range against a faint band for how ${MONTHS[month - 1]} usually runs. The warmest high was ${fmt(highs[0][1], 0)}° on ${dow(highs[0][0])} ${dayNum(highs[0][0])}.</p>
        <div class="va-week">${head}${todayRow}${days.map(rowFor).join("")}</div>
        <p class="va-foot">Grey band: the usual range for that date, from the 25th percentile of daily lows to the 75th of daily highs, taken from the same date plus or minus a week in earlier years. Hatched: under 80% of readings archived. Source: station archive.</p>
      </div>`;
  }

  // ---------- Rain ----------
  function rainSection(d, c) {
    const rm = Object.fromEntries(d.rain.daily_all.date.map((k, i) => [k, d.rain.daily_all.rain_in[i]]));
    const last = d.rain.daily_all.date[d.rain.daily_all.date.length - 1];
    const y = +last.slice(0, 4), mo = last.slice(5, 7), cut = +last.slice(8);
    const years = [];
    for (let yy = 2023; yy <= y; yy++) {
      const s = sumRange(rm, `${yy}-${mo}-01`, `${yy}-${mo}-${String(cut).padStart(2, "0")}`);
      years.push({ year: yy, total: s.sum, missing: s.missing });
    }
    const cur = years[years.length - 1];
    const prior = years.filter((v) => v.year !== y && v.missing === 0);
    const wetter = prior.filter((v) => v.total > cur.total).length;
    const mname = MONTHS[+mo - 1];
    const minPrior = Math.min(...prior.map((v) => v.total));
    let headline;
    if (!prior.length) headline = `${mname} rain so far: ${fmt(cur.total)} in`;
    else if (wetter === 0) headline = `${mname} is already wetter than any ${mname} since ${Math.min(...prior.map((v) => v.year))}`;
    else if (wetter === prior.length) headline = `This is the driest ${mname} yet at this point: ${fmt(cur.total)} in, against ${fmt(minPrior)} in in the next-driest year`;
    else headline = `${mname} has been wetter than ${prior.length - wetter} of ${prior.length} earlier ${mname}s so far`;

    const wins = [7, 30, 90, 365].map((n) => {
      const from = addDays(last, -(n - 1));
      const pFrom = shiftYear(from, -1), pTo = shiftYear(last, -1);
      const a = sumRange(rm, from, last), b = sumRange(rm, pFrom, pTo);
      return { n, from, pFrom, pTo, a, b, invalid: a.missing > 0 || b.missing > 0 };
    });
    const label = (n) => (n === 365 ? "Last 365 days" : `Last ${n} days`);
    const rows = wins.map((w) => {
      if (w.invalid) {
        return `<div class="va-rr"><div class="va-rr-h"><b>${label(w.n)}</b><span>not compared</span></div>
          <div class="va-invalid gap-hatch">${w.a.missing + w.b.missing} days are missing from the archive across the two windows (${w.a.missing} this year, ${w.b.missing} a year earlier), so the totals are not like for like.</div></div>`;
      }
      const mx = Math.max(w.a.sum, w.b.sum, 0.01), p = pct(w.a.sum, w.b.sum);
      const delta = p === null ? "" : `<em style="color:${p < 0 ? "var(--note)" : "var(--ink)"}">${p > 0 ? "+" : "−"}${Math.abs(p)}%</em>`;
      return `<div class="va-rr"><div class="va-rr-h"><b>${label(w.n)}</b>${delta}</div>
        <div class="va-pair"><div class="t"><i style="width:${(w.a.sum / mx) * 100}%;background:var(--rain-3)"></i></div><div class="v"><b>${fmt(w.a.sum)} in</b><small>${rangeLabel(w.from, last).replace(/ \d{4}$/, "")}</small></div>
        <div class="t"><i style="width:${(w.b.sum / mx) * 100}%;background:var(--rain-1)"></i></div><div class="v">${fmt(w.b.sum)} in<small>year before</small></div></div></div>`;
    }).join("");
    const valid = wins.filter((w) => !w.invalid);
    const below = valid.filter((w) => w.a.sum < w.b.sum).length;
    const rollHead = !valid.length ? `Rolling windows can't be compared: the archive has gaps`
      : below === valid.length ? `Every window we can compare is drier than a year ago`
      : below === 0 ? `Every window we can compare is wetter than a year ago` : `Recent rain is mixed against last year: ${below} of ${valid.length} comparable windows are drier`;
    const gapYears = years.filter((v) => v.missing > 0 && v.year !== y);
    const live = Math.max(0, c.rain_month_in - cur.total);
    return { html: `
      <section id="va-rain" class="va-sec">
        <p class="va-eyebrow">Rain</p>
        <h2>${headline}</h2>
        <p class="va-sub">Cumulative rain from 1 ${mname} through the ${dayNum(last)}th, one line per year. The gauge has added ${fmt(live)} in since the archive ended, so the live ${mname} total is ${fmt(c.rain_month_in)} in.</p>
        <div id="va-chart-mtd" class="va-chart" style="height:230px"></div>
        <p class="va-foot">${gapYears.length ? "Left out: " + gapYears.map((v) => v.year + " has " + v.missing + " missing days").join("; ") + ". " : ""}Source: station archive, daily rain totals.</p>
        <div class="va-block" style="margin-top:var(--s-6)">
          <h3>${rollHead}</h3>
          <p class="va-sub">Totals for the windows ending ${dayNum(last)} ${monShort(last)}, against the same dates a year earlier.</p>
          <div class="va-roll">${rows}</div>
          <p class="va-foot">Dark bar: this year. Light bar: a year earlier. A comparison is withheld when either window includes days missing from the archive.</p>
        </div>
      </section>`, years, cut, mo, rm };
  }

  // ---------- Events ----------
  const startMs = (e) => Date.parse(e.start_time);
  const sortedEvents = (d) => d.rain_events.slice().sort((a, b) => a.start_time.localeCompare(b.start_time));

  function eventChartBlock(d, c, promoted) {
    const latest = sortedEvents(d).pop();
    const raining = c.rain_rate_in_hr > 0;
    const head = promoted ? `It is raining: ${fmt(c.rain_event_in)} in so far in this event` : `The latest event, drop by drop`;
    return `
      <div class="${promoted ? "va-promo" : "va-block"}">
        ${promoted ? '<p class="va-eyebrow" style="color:var(--rain-3)">Rain event under way</p>' : ""}
        <${promoted ? "h2" : "h3"}>${head}</${promoted ? "h2" : "h3"}>
        <p class="va-sub">Started ${fmtTZ(startMs(latest), { weekday: "long", hour: "numeric", minute: "2-digit" })} and peaked at ${fmt(latest.max_hourly_rate)} in/hr. The archive holds ${fmt(latest.total_rainfall)} in of it; the live gauge reads ${fmt(c.rain_event_in)} in for the event so far, which includes rain after the archive ended.</p>
        <div id="va-chart-event" class="va-chart" style="height:200px"></div>
        <p class="va-foot">The catalog marks this event open because the archive stops at ${fmtTZ(Date.parse(latest.end_time), { hour: "numeric", minute: "2-digit" })}. The gauge ${raining ? "is reading rain now" : "reads no rain falling now"}. Source: Event Catalog, 5-minute readings.</p>
      </div>`;
  }

  function eventsSection(d, c, promoted) {
    const evs = sortedEvents(d);
    const recent = evs.slice(-5).reverse();
    const maxTot = Math.max(...recent.map((e) => e.total_rainfall), 0.01);
    const rowFor = (e) => `<div class="va-evr">
      <div class="s">${fmtTZ(startMs(e), { weekday: "short", month: "short", day: "numeric" })}<small>${fmtTZ(startMs(e), { hour: "numeric", minute: "2-digit" })}${e.ongoing ? ", open at archive end" : ""}</small></div>
      <div class="n">${dur(e.duration_minutes)}</div><div class="n tot">${fmt(e.total_rainfall)} in</div><div class="n">${fmt(e.max_hourly_rate)}<small> in/hr</small></div>
      <div class="meter"><i style="width:${(e.total_rainfall / maxTot) * 100}%"></i></div></div>`;
    const cutoff = Date.parse(c.timestamp) - 60 * DAY;
    const in60 = evs.filter((e) => startMs(e) >= cutoff);
    const big = in60.slice().sort((a, b) => b.total_rainfall - a.total_rainfall)[0];
    const sum60 = in60.reduce((a, e) => a + e.total_rainfall, 0);
    const headline = in60.length
      ? `${in60.length} rain events in 60 days, ${big.total_rainfall >= 0.5 ? `the biggest ${fmt(big.total_rainfall)} in on ${fmtTZ(startMs(big), { month: "short", day: "numeric" })}` : `none above ${fmt(big.total_rainfall)} in`}`
      : `No rain events in the last 60 days`;
    return `
      <section id="va-events" class="va-sec">
        <p class="va-eyebrow">Events</p>
        <h2>${headline}</h2>
        <p class="va-sub">Together they dropped ${fmt(sum60)} in. Here are the five most recent in the Event Catalog, newest first.</p>
        <div class="va-ev"><div class="va-evh"><span>Started</span><span class="n">Length</span><span class="n">Total</span><span class="n">Peak rate</span></div>${recent.map(rowFor).join("")}</div>
        <p class="va-foot">Peak rate is the highest hourly rate within the event. Source: Event Catalog.</p>
        ${promoted ? "" : eventChartBlock(d, c, false)}
      </section>`;
  }

  // ---------- Solar ----------
  function solarSection(d, c) {
    const DA = d.daily_all;
    const sm = Object.fromEntries(DA.date.map((k, i) => [k, DA.solar_kwh_m2[i]]).filter(([, v], i) => DA.coverage[i] >= 0.8 && v != null));
    const last = DA.date[DA.date.length - 1];
    const from = addDays(last, -29), pFrom = shiftYear(from, -1), pTo = shiftYear(last, -1);
    const a = sumRange(sm, from, last), b = sumRange(sm, pFrom, pTo);
    const invalid = a.missing > 0 || b.missing > 0;
    const p = pct(a.sum, b.sum);
    const headline = invalid ? `Thirty-day sunshine can't be compared with last year: archive days are missing`
      : `The last 30 days caught ${Math.abs(p)}% ${p >= 0 ? "more" : "less"} sun than the same 30 days last year`;
    const prof = d.solar.hourly_profile_by_month;
    const month = +last.slice(5, 7);
    const hrs = prof.filter((x) => x.month === month).sort((x, y) => x.hour - y.hour);
    const tot = hrs.reduce((s, x) => s + x.kwh_m2_per_day, 0);
    const peak = hrs.reduce((m, x) => (x.kwh_m2_per_day > m.kwh_m2_per_day ? x : m), hrs[0]);
    const core = hrs.filter((x) => x.hour >= 10 && x.hour < 14).reduce((s, x) => s + x.kwh_m2_per_day, 0);
    const hh = (h) => (h === 12 ? "noon" : h < 12 ? h + " am" : h - 12 + " pm");
    const mn = MONTHS[month - 1];
    return { html: `
      <section id="va-solar" class="va-sec">
        <p class="va-eyebrow">Solar</p>
        <h2>${headline}</h2>
        <p class="va-sub">${invalid ? `${a.missing + b.missing} days are missing across the two periods.` : `${fmt(a.sum, 1)} kWh/m² from ${dayNum(from)} ${monShort(from)} to ${dayNum(last)} ${monShort(last)}, against ${fmt(b.sum, 1)} kWh/m² over the same dates a year earlier.`}</p>
        <div id="va-chart-sol30" class="va-chart" style="height:230px"></div>
        <p class="va-foot">Bars: daily energy this year. Line: the same date last year. The Streamlit 30-day tile reads ${fmt(d.solar.tiles.last_30d.total_kwh, 1)} kWh/m², which does not match this window's ${fmt(a.sum, 1)}; it appears to cover a different span. Source: station archive.</p>
        <div class="va-block" style="margin-top:var(--s-6)">
          <h3>${mn} sunshine peaks at ${hh(peak.hour)}, and ${Math.round((core / tot) * 100)}% of it arrives between 10 am and 2 pm</h3>
          <p class="va-sub">Energy by hour on a typical ${mn} day, averaged over every ${mn} on record (${fmt(tot, 2)} kWh/m² a day), with July for contrast.</p>
          <div id="va-chart-hour" class="va-chart" style="height:200px"></div>
          <p class="va-foot">Hours are station-local time. Source: station archive.</p>
        </div>
      </section>`, sm, from, last, month, hrs };
  }

  // ---------- Plotly ----------
  function baseLayout(extra) {
    const font = { family: css("--sans").replace(/"/g, "") || "system-ui", size: 11, color: css("--ink-2") };
    const ax = { showgrid: false, zeroline: false, linecolor: css("--rule-strong"), linewidth: 1, ticks: "outside", ticklen: 3, tickcolor: css("--rule-strong"), fixedrange: true };
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font, dragmode: false, showlegend: false,
      margin: { l: 34, r: 12, t: 8, b: 30 }, hovermode: "x",
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--rule-strong"), font: { family: font.family, size: 12, color: css("--ink") } },
      xaxis: Object.assign({}, ax),
      yaxis: Object.assign({}, ax, { showgrid: true, gridcolor: css("--rule"), gridwidth: 1, showline: false, ticks: "" }),
    }, extra || {});
  }
  const CFG = { displayModeBar: false, responsive: true };
  const mk = (id, traces, layout) => { const el = document.getElementById(id); if (el && window.Plotly) Plotly.newPlot(el, traces, layout, CFG); };
  const ax = (k, o) => Object.assign(baseLayout()[k], o);

  function drawCharts(d, c, ctx) {
    const rain3 = css("--rain-3"), ink = css("--ink"), ink2 = css("--ink-2"), ink3 = css("--ink-3"), note = css("--note"), sun = css("--sun");
    // month-to-date cumulative
    const { years, cut, rm, mo } = ctx.rain;
    const traces = [], ends = [];
    years.forEach((v) => {
      if (v.missing > 0) return;
      const xs = [], ys = []; let acc = 0;
      for (let dd = 1; dd <= cut; dd++) { acc += rm[`${v.year}-${mo}-${String(dd).padStart(2, "0")}`] || 0; xs.push(dd); ys.push(+acc.toFixed(3)); }
      const cur = v.year === years[years.length - 1].year;
      traces.push({ x: xs, y: ys, mode: "lines", type: "scatter", line: { color: cur ? rain3 : ink3, width: cur ? 3 : 1.5, shape: "hv" }, hovertemplate: `${v.year}: %{y:.2f} in<extra></extra>` });
      ends.push({ v, y: acc, cur });
    });
    const ymax = Math.max(...ends.map((e) => e.y), 0.5), gap = ymax * 0.075;
    ends.sort((a, b) => a.y - b.y);
    let prev = -1e9;
    ends.forEach((e) => { e.ly = Math.max(e.y, prev + gap); prev = e.ly; });
    const ann = ends.map((e) => ({ x: cut, y: e.ly, xanchor: "left", yanchor: "middle", showarrow: false, xshift: 6, text: `${e.cur ? "<b>" : ""}${e.v.year}  ${e.y.toFixed(2)} in${e.cur ? "</b>" : ""}`, font: { color: e.cur ? note : ink2, size: 11 } }));
    mk("va-chart-mtd", traces, baseLayout({ margin: { l: 38, r: 92, t: 8, b: 34 }, annotations: ann, hovermode: "closest",
      xaxis: ax("xaxis", { range: [1, cut], tickvals: [1, 8, 15, 22, cut], title: { text: "day of month", font: { size: 10 }, standoff: 2 } }),
      yaxis: ax("yaxis", { range: [0, ymax * 1.1], tickformat: ".1f" }) }));

    // latest event curve
    const ev = sortedEvents(d).pop();
    if (ev && ev.series) {
      const xs = ev.series.t.map(laStr), ys = ev.series.accum_in;
      mk("va-chart-event", [{ x: xs, y: ys, type: "scatter", mode: "lines", fill: "tozeroy", fillcolor: css("--rain-0"), line: { color: rain3, width: 2.5, shape: "hv" }, hovertemplate: "%{y:.2f} in so far<extra></extra>" }],
        baseLayout({ margin: { l: 38, r: 74, t: 12, b: 30 },
          xaxis: ax("xaxis", { tickformat: "%-I%p", nticks: 5, range: [xs[0], xs[xs.length - 1]] }),
          yaxis: ax("yaxis", { rangemode: "tozero", tickformat: ".2f" }),
          annotations: [{ x: xs[xs.length - 1], y: ys[ys.length - 1], xanchor: "left", xshift: 6, showarrow: false, text: `<b>${ys[ys.length - 1].toFixed(2)} in</b><br>archive ends`, font: { color: note, size: 11 }, align: "left" }] }));
    }

    // solar 30 days vs last year
    const S = ctx.solar, xs = [], ya = [], yb = [];
    for (let k = S.from; k <= S.last; k = addDays(k, 1)) { xs.push(k); ya.push(k in S.sm ? S.sm[k] : null); const pk = shiftYear(k, -1); yb.push(pk in S.sm ? S.sm[pk] : null); }
    const imaxOf = (arr) => arr.reduce((m, v, i) => (v != null && v > (arr[m] ?? -1) ? i : m), 0);
    mk("va-chart-sol30", [
      { x: xs, y: ya, type: "bar", marker: { color: sun }, hovertemplate: "%{y:.2f} kWh/m²<extra></extra>" },
      { x: xs, y: yb, type: "scatter", mode: "lines", line: { color: ink, width: 1.5 }, hovertemplate: "last year %{y:.2f}<extra></extra>" },
    ], baseLayout({ margin: { l: 34, r: 12, t: 22, b: 30 }, bargap: 0.25, hovermode: "x unified",
      xaxis: ax("xaxis", { tickformat: "%-d %b", nticks: 5 }),
      yaxis: ax("yaxis", { rangemode: "tozero", tickformat: ".1f" }),
      annotations: [
        { x: xs[imaxOf(ya)], y: ya[imaxOf(ya)], text: "<b>this year</b>", showarrow: false, yshift: 10, font: { color: ink, size: 11 } },
        { x: xs[xs.length - 3], y: yb[xs.length - 3], text: "last year", showarrow: false, yshift: -14, font: { color: ink, size: 11 } },
      ] }));

    // hour-of-day profile
    const hrs = S.hrs, jul = d.solar.hourly_profile_by_month.filter((x) => x.month === 7).sort((a, b) => a.hour - b.hour);
    const pk = (arr) => arr.reduce((m, x) => (x.kwh_m2_per_day > m.kwh_m2_per_day ? x : m), arr[0]);
    const mn = MONTHS[S.month - 1];
    mk("va-chart-hour", [
      { x: jul.map((x) => x.hour), y: jul.map((x) => x.kwh_m2_per_day), type: "scatter", mode: "lines", line: { color: ink3, width: 1.5, shape: "spline" }, hovertemplate: "July %{y:.2f}<extra></extra>" },
      { x: hrs.map((x) => x.hour), y: hrs.map((x) => x.kwh_m2_per_day), type: "scatter", mode: "lines", fill: "tozeroy", fillcolor: css("--sun-soft"), line: { color: sun, width: 2.5, shape: "spline" }, hovertemplate: `${mn} %{y:.2f}<extra></extra>` },
    ], baseLayout({ margin: { l: 38, r: 12, t: 22, b: 30 },
      xaxis: ax("xaxis", { range: [4, 21], tickvals: [6, 9, 12, 15, 18], ticktext: ["6a", "9a", "12p", "3p", "6p"] }),
      yaxis: ax("yaxis", { rangemode: "tozero", tickformat: ".2f" }),
      annotations: [
        { x: pk(hrs).hour, y: pk(hrs).kwh_m2_per_day, text: `<b>${mn}</b>`, showarrow: false, yshift: 10, font: { color: ink, size: 11 } },
        { x: pk(jul).hour, y: pk(jul).kwh_m2_per_day, text: "July", showarrow: false, yshift: 10, font: { color: ink2, size: 11 } },
      ] }));
  }

  // ---------- render ----------
  function render(root, data) {
    if (!document.getElementById("va-style")) {
      const st = document.createElement("style"); st.id = "va-style"; st.textContent = CSS; document.head.appendChild(st);
    }
    const c = data.current;
    root.className = "va";
    const promoted = c.rain_rate_in_hr > 0;
    const rainS = rainSection(data, c);
    const solarS = solarSection(data, c);
    root.innerHTML = `
      <nav class="va-nav" aria-label="Sections"><div class="va-nav-in">
        <button data-t="va-now" aria-current="true">Now</button><button data-t="va-rain">Rain</button><button data-t="va-events">Events</button><button data-t="va-solar">Solar</button>
      </div></nav>
      <main class="va-main">
        <section id="va-now">
          ${nowSection(data, c)}
          ${promoted ? eventChartBlock(data, c, true) : ""}
          ${ribbonBlock(data, c)}
          ${weekBlock(data, c)}
        </section>
        ${rainS.html}
        ${eventsSection(data, c, promoted)}
        ${solarS.html}
      </main>`;
    const btns = [...root.querySelectorAll(".va-nav button")];
    const secs = btns.map((b) => document.getElementById(b.dataset.t));
    btns.forEach((b, i) => b.addEventListener("click", () => {
      const y = i === 0 ? 0 : secs[i].getBoundingClientRect().top + window.scrollY - 52;
      window.scrollTo({ top: y, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }));
    const onScroll = () => {
      let cur = 0;
      secs.forEach((s, i) => { if (s.getBoundingClientRect().top <= 90) cur = i; });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) cur = secs.length - 1;
      btns.forEach((b, i) => b.setAttribute("aria-current", i === cur ? "true" : "false"));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    try { drawCharts(data, c, { rain: rainS, solar: solarS }); } catch (e) { console.error(e); }
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => location.reload());
  }

  window.LOOKOUT_VARIANT = { name: "Today", render };
})();
