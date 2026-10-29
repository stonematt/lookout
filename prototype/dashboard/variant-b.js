// PROTOTYPE variant B, "Almanac": comparison leads, real tabs.
// Every headline is computed from data.json. Archive gaps render as .gap-hatch, never as computed values.
(function () {
  "use strict";

  /* ---------- styles ---------- */
  const CSS = `
.vb{max-width:1240px;margin:0 auto;padding:0 16px 96px;font-family:var(--sans);color:var(--ink);font-variant-numeric:tabular-nums}
.vb button{font:inherit;color:inherit;cursor:pointer}
.vb h1,.vb h2,.vb h3,.vb p{margin:0}
.vb-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:16px 0 8px;flex-wrap:wrap}
.vb-station{font-family:var(--serif);font-size:var(--step-1);font-weight:600;letter-spacing:-.01em}
.vb-stamp{font-size:var(--step--1);color:var(--ink-2);display:flex;align-items:center;gap:8px}
.vb-dot{width:8px;height:8px;border-radius:50%;background:var(--rain-2);display:inline-block}
.vb-tabs{position:sticky;top:0;z-index:20;display:flex;gap:4px;overflow-x:auto;background:var(--ground);border-bottom:1px solid var(--rule);margin:0 -16px 24px;padding:0 16px;scrollbar-width:none}
.vb-tabs::-webkit-scrollbar{display:none}
.vb-tab{flex:0 0 auto;background:none;border:0;padding:0 14px;min-height:46px;font-size:var(--step-0);font-weight:500;color:var(--ink-2);box-shadow:inset 0 -3px 0 transparent;border-radius:0}
.vb-tab:hover{color:var(--ink)}
.vb-tab[aria-selected=true]{color:var(--ink);font-weight:700;box-shadow:inset 0 -3px 0 var(--rain-3)}
.vb-panel[hidden]{display:none}
.vb-grid{display:grid;gap:40px 48px;grid-template-columns:minmax(0,1fr)}
@media (min-width:1024px){
 .vb{padding:0 32px 96px}.vb-tabs{margin:0 -32px 32px;padding:0 32px}
 .vb-grid.two{grid-template-columns:minmax(0,5fr) minmax(0,7fr)}
 .vb-grid.two-r{grid-template-columns:minmax(0,7fr) minmax(0,5fr)}
 .vb-grid.three{grid-template-columns:repeat(3,minmax(0,1fr));gap:32px}
 .vb-span{grid-column:1/-1}
}
.vb h2{font-family:var(--serif);font-size:var(--step-2);line-height:1.15;font-weight:600;letter-spacing:-.015em;text-wrap:balance}
.vb h3{font-family:var(--serif);font-size:var(--step-1);line-height:1.2;font-weight:600;text-wrap:balance}
.vb .sub{font-size:var(--step--1);color:var(--ink-2);margin-top:6px;line-height:1.45;max-width:68ch}
.vb .src{font-size:.75rem;color:var(--ink-2);margin-top:10px;line-height:1.45;max-width:68ch}
.vb .block{padding-top:4px}
.vb .now{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap}
.vb .temp{font-size:clamp(3.5rem,14vw,var(--step-4));font-weight:600;line-height:.9;letter-spacing:-.04em}
.vb .temp sup{font-size:.4em;vertical-align:top;position:relative;top:.15em;font-weight:500;margin-left:2px}
.vb .lead{font-family:var(--serif);font-size:var(--step-1);line-height:1.3;margin-top:12px;text-wrap:balance;max-width:34ch}
.vb .facts{display:grid;grid-template-columns:repeat(3,auto);gap:10px 20px;margin-top:16px;justify-content:start}
.vb .facts div{font-size:var(--step--1);color:var(--ink-2)}
.vb .facts b{display:block;font-size:var(--step-0);color:var(--ink);font-weight:600}
.vb .strip-wrap{margin-top:18px}
.vb .strip-wrap .claim{font-size:var(--step-0);font-weight:600;line-height:1.3}
.vb .strip-wrap .claim span{font-weight:400;color:var(--ink-2)}
.vb .strip{width:100%;display:block;margin-top:6px;overflow:visible}
.vb .strip text{font-family:var(--sans);font-size:11px;fill:var(--ink-2)}
.vb .strip .lab{font-weight:600;fill:var(--note);font-size:12px}
.vb .seg{display:inline-flex;border:1px solid var(--rule-strong);border-radius:8px;overflow:hidden;margin-top:12px}
.vb .seg button{background:var(--surface);border:0;padding:0 14px;min-height:40px;font-size:var(--step--1);font-weight:500;color:var(--ink-2)}
.vb .seg button+button{border-left:1px solid var(--rule-strong)}
.vb .seg button[aria-pressed=true]{background:var(--ink);color:var(--ground);font-weight:600}
.vb .alm{display:grid;gap:3px;margin-top:14px;align-items:stretch}
.vb .alm .h{font-size:.6875rem;color:var(--ink-2);text-align:center;padding-bottom:2px}
.vb .alm .y{font-size:var(--step--1);color:var(--ink-2);display:flex;align-items:center;font-weight:600}
.vb .cell{border:0;border-radius:4px;min-height:44px;padding:0;font-size:.6875rem;font-weight:500;display:flex;align-items:center;justify-content:center;position:relative;line-height:1}
.vb .cell.none{background:none;cursor:default;border:1px dashed var(--rule)}
.vb .cell.gap{color:var(--ink-2);border:1px solid var(--rule)}
.vb .cell.cur{outline:2px solid var(--note);outline-offset:1px;z-index:1}
.vb .cell:focus-visible{outline:2px solid var(--rain-2);outline-offset:2px}
.vb .cell.sel{box-shadow:0 0 0 2px var(--ink)}
.vb .readout{min-height:2.6em;font-size:var(--step--1);color:var(--ink);margin-top:10px;line-height:1.4}
.vb .key{display:flex;align-items:center;gap:8px;font-size:.75rem;color:var(--ink-2);margin-top:8px;flex-wrap:wrap}
.vb .key i{display:inline-block;width:72px;height:8px;border-radius:2px}
.vb .key .hatch{width:16px;height:12px;border-radius:2px;border:1px solid var(--rule)}
.vb table{width:100%;border-collapse:collapse;margin-top:12px;font-size:var(--step--1)}
.vb th{font-weight:600;color:var(--ink-2);text-align:right;padding:6px 6px;border-bottom:1px solid var(--rule-strong);font-size:.75rem;line-height:1.2}
.vb th:first-child,.vb td:first-child{text-align:left;padding-left:0}
.vb td{padding:9px 6px;text-align:right;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
.vb td.gap-hatch{color:var(--ink-2);text-align:center}
.vb td.d{color:var(--ink-2)}
.vb td b{font-weight:600}
.vb caption{caption-side:top;text-align:left;font-weight:600;font-size:var(--step-0)}
.vb .ev tr[aria-current=true] td{background:var(--rain-0)}
.vb .ev tr.pin td{border-bottom:2px solid var(--rule-strong)}
.vb .ev td{white-space:nowrap}.vb .ev td,.vb .ev th{padding-left:3px;padding-right:3px}.vb .ev .q{padding:2px 6px}.vb .ev .live{font-size:0}.vb .ev .live::before{width:9px;height:9px}.vb .ev td:last-child{white-space:normal}
.vb .rowbtn{background:none;border:0;padding:8px 0;text-align:left;min-height:44px;display:flex;align-items:center;gap:6px;font-weight:600}
.vb .live{color:var(--note);font-weight:700;font-size:.6875rem;letter-spacing:.04em;text-transform:uppercase;display:inline-flex;align-items:center;gap:4px}
.vb .live::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--note)}
.vb .q{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--rule-strong);font-size:.6875rem;font-weight:600}
.vb .q.low{border-style:dashed;color:var(--ink-2)}
.vb .more{margin-top:12px;background:var(--surface);border:1px solid var(--rule-strong);border-radius:8px;min-height:44px;padding:0 16px;font-weight:600}
.vb .chart{width:100%;margin-top:8px}
.vb .cal{display:grid;grid-auto-flow:column;grid-template-rows:repeat(7,1fr);gap:3px;margin-top:6px}
.vb .cal .c{aspect-ratio:1;min-width:0;min-height:0;border:0;border-radius:3px;padding:0}
.vb .cal .c.cur{outline:2px solid var(--note);outline-offset:1px}
.vb .calm{display:grid;gap:3px;font-size:.6875rem;color:var(--ink-2);margin-top:14px;height:1em}
.vb .calm span{white-space:nowrap}
.vb .calwrap{display:grid;grid-template-columns:28px minmax(0,1fr);gap:6px}
.vb .dows{display:grid;grid-template-rows:repeat(7,1fr);gap:3px;font-size:.6875rem;color:var(--ink-2);margin-top:6px}
.vb .dows span{display:flex;align-items:center}
.vb .note-live{border-top:1px solid var(--rule);padding-top:10px;margin-top:16px;font-size:var(--step--1);color:var(--ink-2);line-height:1.45;max-width:68ch}
.vb .sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
`;

  /* ---------- helpers ---------- */
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MONL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const DAY = 864e5;
  const idx = (s) => { const [y, m, d] = s.slice(0, 10).split("-").map(Number); return Math.floor(Date.UTC(y, m - 1, d) / DAY); };
  const dt = (i) => new Date(i * DAY);
  const iso = (i) => dt(i).toISOString().slice(0, 10);
  const fmtDay = (i) => `${MON[dt(i).getUTCMonth()]} ${dt(i).getUTCDate()}`;
  const daysIn = (y, m) => new Date(Date.UTC(y, m, 0)).getUTCDate(); // m 1-based
  const addYears = (i, k) => { const d = dt(i); return Math.floor(Date.UTC(d.getUTCFullYear() + k, d.getUTCMonth(), d.getUTCDate()) / DAY); };
  const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2);
  const sgn = (v, s) => (v > 0 ? "+" : v < 0 ? "−" : "") + s;
  const PT = (opts) => new Intl.DateTimeFormat("en-US", Object.assign({ timeZone: "America/Los_Angeles" }, opts));
  const fmtPT = (ms, o) => PT(o || { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(ms));
  const ptString = (ms) => { // plotly-friendly local wall time
    const p = Object.fromEntries(PT({ year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }).formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
    return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
  };
  const fmtDur = (m) => (m < 60 ? `${Math.round(m)}m` : m < 1440 ? `${Math.floor(m / 60)}h ${Math.round(m % 60)}m` : `${Math.floor(m / 1440)}d ${Math.round((m % 1440) / 60)}h`);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const isDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
  const $ = (r, s) => r.querySelector(s);

  function hex2rgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  function ramp(names, t) { // names: token names; returns {bg, fg}
    t = Math.max(0, Math.min(1, t));
    const cols = names.map((n) => hex2rgb(css(n)));
    const x = t * (cols.length - 1), i = Math.min(cols.length - 2, Math.floor(x)), f = x - i;
    const c = cols[i].map((v, k) => Math.round(v + (cols[i + 1][k] - v) * f));
    const lum = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
    return { bg: `rgb(${c.join(",")})`, fg: lum < 0.55 ? "#fff" : "#0f1a2b" };
  }
  const RAIN = ["--rain-0", "--rain-1", "--rain-2", "--rain-3", "--rain-4"];
  const TEMP = ["--temp-cold", "--temp-cool", "--temp-mild", "--temp-warm", "--temp-hot"];
  const SUN = ["--sun-soft", "--sun", "--temp-warm", "--temp-hot"];

  function plotBase(extra) {
    const sans = css("--sans"), ink3 = css("--ink-3"), rule = css("--rule");
    const L = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
      font: { family: sans, size: 12, color: css("--ink-2") }, margin: { l: 38, r: 14, t: 10, b: 30 },
      xaxis: { showgrid: false, zeroline: false, linecolor: rule, tickcolor: rule, tickfont: { color: ink3 }, fixedrange: true },
      yaxis: { gridcolor: rule, zeroline: false, showline: false, tickfont: { color: ink3 }, fixedrange: true, gridwidth: 1 },
      hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--rule-strong"), font: { family: sans, size: 12, color: css("--ink") } },
    };
    return Object.assign(L, extra || {});
  }
  const PCFG = { displayModeBar: false, responsive: true };

  /* ---------- data prep ---------- */
  function prep(D) {
    const P = { D };
    const mapOf = (dates, vals) => { const m = new Map(); dates.forEach((d, i) => { if (vals[i] != null) m.set(idx(d), vals[i]); }); return m; };
    const A = D.daily_all, ok = A.coverage.map((c) => c >= 0.8); // coverage < 0.8 = invalid day
    const mapOk = (vals) => { const m = new Map(); A.date.forEach((d, i) => { if (ok[i] && vals[i] != null) m.set(idx(d), vals[i]); }); return m; };
    P.rain = mapOk(A.rain_in);
    P.sol = mapOk(A.solar_kwh_m2);
    P.tmp = mapOk(A.temp_mean);
    P.first = idx(A.date[0]);
    P.last = idx(A.date[A.date.length - 1]);
    P.today = idx(D.current.timestamp);
    P.tmpFirst = P.first;
    return P;
  }
  // window over a day map: sum or mean; valid only when every day is present
  function win(map, endI, n, mode) {
    let s = 0, c = 0;
    for (let i = endI - n + 1; i <= endI; i++) { const v = map.get(i); if (v != null) { s += v; c++; } }
    const valid = c >= Math.ceil(0.9 * n); // up to 10% of days may be excluded; totals are prorated and marked
    return { valid, est: valid && c < n, v: valid ? (mode === "mean" ? s / c : (s * n) / c) : null, have: c, n };
  }
  // calendar month aggregate with status
  function month(P, map, y, m, mode, live) {
    const ms = Math.floor(Date.UTC(y, m - 1, 1) / DAY), me = ms + daysIn(y, m) - 1;
    const lo = map === P.tmp ? P.tmpFirst : P.first;
    if (me < lo || ms > P.last + 1) return { st: "none", why: "Before the archive began (Feb 15, 2023)" };
    let s = 0, c = 0;
    const a = Math.max(ms, lo), b = Math.min(me, P.last);
    for (let i = a; i <= b; i++) { const v = map.get(i); if (v != null) { s += v; c++; } }
    const expect = b - a + 1;
    const cur = me > P.last; // current month, still running
    if (live && cur) { s += live; c += 1; }
    if (ms < lo) return { st: "gap", why: `Archive starts ${fmtDay(lo)} mid-month`, partial: c, have: c, sum: s };
    if (c < Math.ceil(0.9 * expect) && !(live && cur)) return { st: "gap", why: `${expect - c} of ${expect} days missing or under 80% station coverage`, partial: s, have: c, sum: s };
    const est = !cur && c < expect;
    const v = mode === "mean" ? s / (c - (live && cur ? 1 : 0) || 1) : est ? (s * expect) / c : s;
    return { st: cur ? "cur" : "ok", v, days: c, sum: s, est, miss: expect - c, why: cur ? `Month still running, ${c} days so far` : "" };
  }

  /* ---------- text composition ---------- */
  function pctFrom5(v, tbm, m) {
    const pts = [tbm.min[m], tbm.q1[m], tbm.median[m], tbm.q3[m], tbm.max[m]];
    const lv = [0, 25, 50, 75, 100];
    if (v <= pts[0]) return 0; if (v >= pts[4]) return 100;
    for (let i = 0; i < 4; i++) if (v <= pts[i + 1]) return lv[i] + (lv[i + 1] - lv[i]) * (v - pts[i]) / (pts[i + 1] - pts[i]);
    return 100;
  }
  function tempWord(t) { return t < 32 ? "Freezing" : t < 45 ? "Cold" : t < 55 ? "Cool" : t < 65 ? "Mild" : t < 80 ? "Warm" : "Hot"; }
  function summary(c, rainStats) {
    const hum = c.humidity >= 90 ? "damp" : c.humidity <= 35 ? "dry" : "";
    const wind = c.wind_avg10m_mph < 3 ? "still" : c.wind_avg10m_mph < 10 ? "lightly breezy" : "windy";
    const w = [tempWord(c.temp_f), hum, wind].filter(Boolean);
    const lead = w.length > 1 ? w.slice(0, -1).join(", ") + " and " + w[w.length - 1] : w[0];
    let r;
    if (c.rain_rate_in_hr > 0) r = `Raining now at ${f2(c.rain_rate_in_hr)} in/hr, ${f2(c.rain_today_in)} in so far today.`;
    else if (c.rain_today_in > 0) r = `${f2(c.rain_today_in)} in of rain today, the last of it ${rainStats.time_since_rain.replace("h", " hours")} ago.`;
    else r = "No rain recorded today.";
    return `${lead}. ${r}`;
  }

  /* ---------- strips (SVG, drawn at real pixel width) ---------- */
  function svgEl(w, h, inner) { return `<svg class="strip" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img">${inner}</svg>`; }
  function drawTempStrip(host, o) {
    const w = host.clientWidth || 320, h = 104, pad = 14, y = 56;
    const lo = Math.floor((Math.min(o.now, ...o.vals) - 2) / 10) * 10, hi = Math.ceil((Math.max(o.now, ...o.vals) + 2) / 10) * 10;
    const X = (v) => pad + (w - 2 * pad) * (v - lo) / (hi - lo);
    let s = `<line x1="${pad}" x2="${w - pad}" y1="${y + 22}" y2="${y + 22}" stroke="var(--rule-strong)"/>`;
    for (let t = lo; t <= hi; t += 10) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y + 22}" y2="${y + 27}" stroke="var(--rule-strong)"/><text x="${X(t)}" y="${y + 40}" text-anchor="${t === lo ? "start" : t === hi ? "end" : "middle"}">${t}°</text>`;
    let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    o.vals.forEach((v) => { s += `<circle cx="${X(v)}" cy="${y + (rnd() - 0.5) * 30}" r="3.5" fill="var(--temp-cool)" opacity=".6"/>`; });
    const nx = Math.min(w - pad - 50, Math.max(pad + 50, X(o.now)));
    s += `<line x1="${X(o.now)}" x2="${X(o.now)}" y1="${y - 30}" y2="${y + 22}" stroke="var(--note)" stroke-width="2.5"/><text class="lab" x="${nx}" y="${y - 36}" text-anchor="middle">today ${f1(o.now)}° so far</text>`;
    host.innerHTML = svgEl(w, h, s);
  }
  function drawRainStrip(host, o) {
    const w = host.clientWidth || 320, h = 104, pad = 14, y = 56;
    const mx = Math.max(o.now, ...o.vals) * 1.02, S = Math.sqrt;
    const X = (v) => pad + (w - 2 * pad) * S(v) / S(mx);
    let s = "";
    [0, 0.1, 0.5, 1, 2].filter((t) => t <= mx).forEach((t) => { s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y + 22}" y2="${y + 27}" stroke="var(--rule-strong)"/><text x="${X(t)}" y="${y + 40}" text-anchor="${t === 0 ? "start" : t === 1 && mx < 1.3 ? "end" : "middle"}">${t}${t === 0 ? " in" : ""}</text>`; });
    s += `<line x1="${pad}" x2="${w - pad}" y1="${y + 22}" y2="${y + 22}" stroke="var(--rule-strong)"/>`;
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    o.vals.forEach((v) => { s += `<circle cx="${X(v)}" cy="${y + (rnd() - 0.5) * 30}" r="3.5" fill="var(--rain-2)" opacity=".5"/>`; });
    const nx = Math.min(w - pad - 40, Math.max(pad + 40, X(o.now)));
    s += `<line x1="${X(o.now)}" x2="${X(o.now)}" y1="${y - 30}" y2="${y + 22}" stroke="var(--note)" stroke-width="2.5"/><text class="lab" x="${nx}" y="${y - 36}" text-anchor="middle">today ${f2(o.now)} in so far</text>`;
    host.innerHTML = svgEl(w, h, s);
  }

  /* ---------- almanac grid ---------- */
  function almanacGrid(host, cfg) {
    const { years, cell, stops, fmt, unit, readout, nameOf } = cfg;
    const cells = years.map((y) => MON.map((_, mi) => cell(y, mi + 1)));
    const valid = cells.flat().filter((c) => c.st === "ok").map((c) => c.v);
    const lo = cfg.lo != null ? cfg.lo : 0, hi = Math.max(...valid), curIdx = cfg.cur;
    const tf = cfg.curve || ((t) => t);
    let html = `<div class="alm" style="grid-template-columns:34px repeat(12,minmax(0,1fr))"><div></div>${MON.map((m) => `<div class="h">${m[0]}<span class="sr">${m}</span></div>`).join("")}`;
    years.forEach((y, yi) => {
      html += `<div class="y">${y}</div>`;
      cells[yi].forEach((c, mi) => {
        const id = `${y}-${mi}`;
        const label = `${MONL[mi]} ${y}`;
        let cls = "cell", style = "", txt = "", info = "";
        if (c.st === "none") { cls += " none"; info = `${label}: ${c.why}.`; }
        else if (c.st === "gap") { cls += " gap gap-hatch"; txt = ""; info = `${label}: not a valid month. ${c.why}. ${c.partial != null ? "Recorded so far " + fmt(c.partial) + " " + unit + " (not comparable)." : ""}`; }
        else {
          const r = ramp(stops, tf((c.v - lo) / (hi - lo || 1)));
          style = `background:${r.bg};color:${r.fg}`;
          if (c.st === "cur") { cls += " cur"; style = `background:repeating-linear-gradient(135deg,rgba(255,255,255,.35) 0 3px,transparent 3px 7px),${r.bg};color:${r.fg}`; info = `${label}: ${fmt(c.v)} ${unit} so far. ${c.why}, so not comparable with full months.`; }
          else info = `${label}: ${c.est ? "about " : ""}${fmt(c.v)} ${unit}${c.est ? ` (${c.miss} low-coverage day${c.miss > 1 ? "s" : ""} excluded${unit.includes("mean") ? "" : ", total prorated"})` : ""}`;
          txt = `<span class="val">${c.est ? "≈" : ""}${fmt(c.v)}</span>`;
        }
        html += `<button type="button" class="${cls}" style="${style}" data-i="${id}" data-info="${info.replace(/"/g, "&quot;")}" aria-label="${info.replace(/"/g, "&quot;")}"${c.st === "none" ? " tabindex=-1" : ""}>${txt}</button>`;
      });
    });
    html += `</div>`;
    host.innerHTML = html;
    // value text: hide on narrow phones to avoid cramped numerals; readout carries it
    const narrow = host.clientWidth < 520;
    host.querySelectorAll(".val").forEach((v) => (v.style.display = narrow ? "none" : ""));
    const show = (b) => { if (b.dataset.info) { readout.textContent = b.dataset.info; host.querySelectorAll(".sel").forEach((x) => x.classList.remove("sel")); b.classList.add("sel"); } };
    host.querySelectorAll(".cell").forEach((b) => { b.addEventListener("click", () => show(b)); b.addEventListener("mouseenter", () => show(b)); b.addEventListener("focus", () => show(b)); });
    return { lo, hi };
  }

  /* ---------- panels ---------- */
  function build(root, D) {
    const P = prep(D);
    const C = D.current;
    root.innerHTML = "";
    root.classList.add("vb");
    const style = document.createElement("style"); style.textContent = CSS; root.appendChild(style);

    const stamp = fmtPT(Date.parse(C.timestamp), { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    const archEnd = P.last;
    const top = document.createElement("div");
    top.innerHTML = `<div class="vb-top"><h1 class="vb-station">${D.meta.station_name}</h1><div class="vb-stamp"><span class="vb-dot" aria-hidden="true"></span>Live reading ${stamp}</div></div>`;
    root.appendChild(top);

    const TABS = [["overview", "Overview"], ["rain", "Rain"], ["events", "Events"], ["solar", "Solar"]];
    const tabs = document.createElement("div"); tabs.className = "vb-tabs"; tabs.setAttribute("role", "tablist"); tabs.setAttribute("aria-label", "Dashboard sections");
    tabs.innerHTML = TABS.map(([k, l]) => `<button class="vb-tab" role="tab" id="tab-${k}" aria-controls="panel-${k}" aria-selected="false" tabindex="-1" data-k="${k}">${l}</button>`).join("");
    root.appendChild(tabs);
    const panels = {};
    TABS.forEach(([k]) => { const p = document.createElement("section"); p.className = "vb-panel"; p.id = `panel-${k}`; p.setAttribute("role", "tabpanel"); p.setAttribute("aria-labelledby", `tab-${k}`); p.hidden = true; root.appendChild(p); panels[k] = p; });

    const drawn = {}; const redraws = [];
    const R = { overview: () => overview(panels.overview, P, redraws), rain: () => rainTab(panels.rain, P), events: () => eventsTab(panels.events, P), solar: () => solarTab(panels.solar, P, redraws) };

    function select(k, focus) {
      if (!R[k]) k = "overview";
      TABS.forEach(([t]) => { const on = t === k; const b = $(tabs, `[data-k=${t}]`); b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; panels[t].hidden = !on; });
      if (!drawn[k]) { drawn[k] = true; if (!window.Plotly && k !== "overview" && k !== "solar") { panels[k].innerHTML = "<p class=sub>Charts need Plotly.js, which did not load (offline?). Reload when connected.</p>"; } else R[k](); } else if (window.Plotly) panels[k].querySelectorAll(".js-plotly-plot").forEach((g) => Plotly.Plots.resize(g));
      redraws.forEach((f) => f());
      if (focus) $(tabs, `[data-k=${k}]`).focus();
      if (location.hash !== "#" + k) history.replaceState(null, "", "#" + k);
    }
    tabs.addEventListener("click", (e) => { const b = e.target.closest(".vb-tab"); if (b) select(b.dataset.k); });
    tabs.addEventListener("keydown", (e) => {
      const ks = TABS.map((t) => t[0]); const cur = ks.indexOf(document.activeElement.dataset ? document.activeElement.dataset.k : "");
      let n = -1;
      if (e.key === "ArrowRight") n = (cur + 1) % ks.length; else if (e.key === "ArrowLeft") n = (cur - 1 + ks.length) % ks.length; else if (e.key === "Home") n = 0; else if (e.key === "End") n = ks.length - 1;
      if (n >= 0) { e.preventDefault(); select(ks[n], true); }
    });
    window.addEventListener("hashchange", () => select(location.hash.slice(1)));
    select(location.hash.slice(1) || "overview");

    let rt; new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => redraws.forEach((f) => f()), 120); }).observe(root);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => build(root, D));
  }

  /* ----- Overview ----- */
  function overview(el, P, redraws) {
    const D = P.D, C = D.current, tbm = D.overview.temp_by_month;
    const m0 = dt(P.today).getUTCMonth(), dd = dt(P.today).getUTCDate();
    const todayStr = C.timestamp.slice(0, 10), rt = D.recent;
    const tv = rt.t.map((ms, i) => ({ d: ptString(ms).slice(0, 10), v: rt.temp_f[i] })).filter((x) => x.d === todayStr && x.v != null).map((x) => x.v);
    const nowT = tv.length ? tv.reduce((a, b) => a + b, 0) / tv.length : C.temp_f;
    const tvals = [];
    for (let k = 1; k <= 3; k++) for (let o = -7; o <= 7; o++) { const v = P.tmp.get(addYears(P.today, -k) + o); if (v != null) tvals.push(v); }
    const pct = tvals.length ? 100 * tvals.filter((v) => v < nowT).length / tvals.length : 50;
    const tword = pct < 50 ? Math.round(100 - pct) : Math.round(pct);
    const tmed = tvals.slice().sort((a, b) => a - b)[Math.floor(tvals.length / 2)];

    // rain: same date +-7 days in earlier years, valid days only
    const vals = []; let nDays = 0;
    for (let k = 1; k <= 3; k++) for (let o = -7; o <= 7; o++) { const i = addYears(P.today, -k) + o; if (i >= P.first && i <= P.last) { nDays++; const v = P.rain.get(i); if (v != null) vals.push(v); } }
    const nowRain = C.rain_today_in;
    const wetter = vals.length ? Math.round(100 * vals.filter((v) => v < nowRain).length / vals.length) : null;
    const dry = vals.filter((v) => v === 0).length;
    const span = dd <= 10 ? "early" : dd <= 20 ? "mid" : "late";
    const winLbl = `${fmtDay(addYears(P.today, -1) - 7)} – ${fmtDay(addYears(P.today, -1) + 7)}`;

    // October-to-date vs same days earlier years
    const mtd = C.rain_month_in; const mtdPrev = [];
    for (let k = 1; k <= 3; k++) { const a = Math.floor(Date.UTC(dt(P.today).getUTCFullYear() - k, m0, 1) / DAY), b = addYears(P.today, -k); const w = win(P.rain, b, b - a + 1, "sum"); mtdPrev.push({ y: dt(P.today).getUTCFullYear() - k, w }); }
    const okPrev = mtdPrev.filter((x) => x.w.valid);
    const beat = okPrev.filter((x) => mtd > x.w.v).length;
    const monthName = MONL[m0];
    let head;
    if (okPrev.length && beat === okPrev.length) head = `${monthName} is already wetter than any ${monthName} on record: ${f2(mtd)} in by the ${dd}th`;
    else if (okPrev.length && beat === 0) head = `${monthName} is running drier than every earlier ${monthName}: ${f2(mtd)} in by the ${dd}th, against ${okPrev.map((x) => f1(x.w.v)).join(", ")} in`;
    else head = `${monthName} so far: ${f2(mtd)} in by the ${dd}th, wetter than ${beat} of ${okPrev.length} earlier ${monthName}s at this point`;

    el.innerHTML = `
<div class="vb-grid two">
  <div>
    <div class="block">
      <div class="now"><div class="temp">${f1(C.temp_f)}<sup>°F</sup></div></div>
      <p class="lead">${summary(C, D.rain.stats)}</p>
      <div class="facts">
        <div>Feels like<b>${f1(C.feels_like_f)}°</b></div><div>Wind<b>${f1(C.wind_avg10m_mph)} mph</b></div><div>Humidity<b>${C.humidity}%</b></div>
        <div>Rain today<b>${f2(C.rain_today_in)} in</b></div><div>Dew point<b>${f1(C.dew_point_f)}°</b></div><div>Pressure<b>${f2(C.pressure_inhg)} inHg ${C.pressure_trend_symbol}</b></div>
      </div>
    </div>
    <div class="block" style="margin-top:36px">
      <h2>Now against normal: ${pct < 50 ? "cooler" : "warmer"} than usual, ${wetter != null && wetter >= 50 ? "wetter" : "drier"} than most ${span}-${monthName} days</h2>
      <div class="strip-wrap"><p class="claim">${pct < 50 ? "Cooler" : "Warmer"} than ${tword}% of ${span}-${monthName} days <span>(today averages ${f1(nowT)}°F so far, ${f1(C.temp_f)}°F now; the median day is ${f1(tmed)}°F; ${tvals.length} recorded days, 2023–25)</span></p><div id="vb-ts"></div></div>
      <div class="strip-wrap"><p class="claim">${wetter == null ? "No earlier-year rain record for this date" : `Wetter than ${wetter}% of ${span}-${monthName} days`} <span>(${vals.length} recorded days, ${winLbl} in 2023–25; ${dry} were dry)</span></p><div id="vb-rs"></div></div>
      <p class="src">Source: lookout post station. Temperature compares today’s mean so far with each daily mean within seven days of this date, 2023–25. Rain compares today’s total so far with each daily total within seven days of this date, 2023–25; days inside archive gaps are excluded, not counted as dry.</p>
    </div>
  </div>
  <div>
    <h2 id="vb-alh"></h2>
    <p class="sub" id="vb-als"></p>
    <div class="seg" role="group" aria-label="Almanac measure"><button type="button" data-m="rain" aria-pressed="true">Rain total</button><button type="button" data-m="temp" aria-pressed="false">Mean temperature</button></div>
    <div id="vb-alm"></div>
    <div class="key" id="vb-key"></div>
    <p class="readout" id="vb-ro" aria-live="polite">Tap or hover a month for its value.</p>
    <p class="src" id="vb-als2"></p>
  </div>
  <div class="vb-span" id="vb-roll"></div>
</div>`;
    const draw = () => {
      if (el.hidden) return;
      drawTempStrip($(el, "#vb-ts"), { vals: tvals, now: nowT });
      drawRainStrip($(el, "#vb-rs"), { vals, now: nowRain });
      drawAlmanac();
    };
    let mode = "rain";
    function drawAlmanac() {
      const host = $(el, "#vb-alm"), ro = $(el, "#vb-ro"); ro.textContent = "Tap or hover a month for its value.";
      const years = [2023, 2024, 2025, 2026];
      const curY = dt(P.today).getUTCFullYear();
      const live = mode === "rain" ? C.rain_today_in : 0;
      const cell = (y, m) => { const c = mode === "rain" ? month(P, P.rain, y, m, "sum", live) : month(P, P.tmp, y, m, "mean"); c.cur = y === curY && m - 1 === m0; return c; };
      const cfg = { years, cell, readout: ro, unit: mode === "rain" ? "in rain" : "°F mean", fmt: mode === "rain" ? (v) => v.toFixed(1) : (v) => Math.round(v) + "°", stops: mode === "rain" ? RAIN : TEMP, lo: mode === "rain" ? 0 : undefined, curve: mode === "rain" ? Math.sqrt : (t) => t, cur: m0 };
      if (mode === "temp") { const v = years.flatMap((y) => MON.map((_, i) => cell(y, i + 1))).filter((c) => c.st === "ok").map((c) => c.v); cfg.lo = Math.min(...v); }
      const { lo, hi } = almanacGrid(host, cfg);
      $(el, "#vb-key").innerHTML = mode === "rain"
        ? `<span>${lo.toFixed(0)} in</span><i style="background:linear-gradient(90deg,var(--rain-0),var(--rain-2),var(--rain-4))"></i><span>${hi.toFixed(1)} in</span><span class="hatch gap-hatch" style="display:inline-block"></span><span>gap or partial month, not comparable</span>`
        : `<span>${Math.round(lo)}°</span><i style="background:linear-gradient(90deg,var(--temp-cold),var(--temp-mild),var(--temp-warm),var(--temp-hot))"></i><span>${Math.round(hi)}°</span><span class="hatch gap-hatch" style="display:inline-block"></span><span>no valid month</span>`;
      // headline
      const all = years.flatMap((y) => MON.map((_, i) => ({ y, m: i + 1, c: cell(y, i + 1) })));
      const ok = all.filter((x) => x.c.st === "ok");
      const hp = $(el, "#vb-alh"), sp = $(el, "#vb-als"), s2 = $(el, "#vb-als2");
      if (mode === "rain") {
        const peak = ok.reduce((a, b) => (b.c.v > a.c.v ? b : a));
        const octs = ok.filter((x) => x.m - 1 === m0).sort((a, b) => b.c.v - a.c.v);
        hp.textContent = head;
        sp.textContent = `Every month since 2023 by rain total. The wettest full month was ${MONL[peak.m - 1]} ${peak.y} at ${f1(peak.c.v)} in; full ${monthName}s: ${octs.map((x) => `${x.y} ${f1(x.c.v)}`).join(", ") || "none yet"} in.`;
        s2.textContent = `Source: station archive, daily rain totals. Outlined month is the current one (live total, hatched). Hatched grey: the archive has missing or low-coverage days or starts mid-month (Feb 2023), so no total is computed. Archive ends ${fmtDay(P.last)}; the live reading is about a day ahead.`;
      } else {
        const hot = ok.reduce((a, b) => (b.c.v > a.c.v ? b : a)), cold = ok.reduce((a, b) => (b.c.v < a.c.v ? b : a));
        hp.textContent = ok.length ? `${MONL[hot.m - 1]} ${hot.y} was the warmest month on file at ${f1(hot.c.v)}°F; ${MONL[cold.m - 1]} ${cold.y} the coolest at ${f1(cold.c.v)}°F` : "No complete month of temperature";
        sp.textContent = "Mean of daily mean temperature. Months with a gap or low-coverage day are hatched, not computed.";
        s2.textContent = `Source: station archive, daily mean temperature. Hatched grey: no valid month (gap, low-coverage day, or still running). ${monthName} ${curY} is outlined.`;
      }
    }
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { mode = b.dataset.m; el.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b)); drawAlmanac(); }));
    rollingTable($(el, "#vb-roll"), P);
    redraws.push(draw);
  }

  function rollingTable(host, P) {
    const N = [7, 30, 90, 365];
    const metrics = [
      { k: "tmp", name: "Temperature, mean", map: P.tmp, mode: "mean", fmt: (v) => f1(v) + "°", dfmt: (a, b) => sgn(a - b, Math.abs(a - b).toFixed(1) + "°"), whyNo: "Archive gap, low-coverage day, or before the archive began" },
      { k: "rain", name: "Rain, total", map: P.rain, mode: "sum", fmt: (v) => f2(v) + " in", dfmt: (a, b) => (b === 0 ? "n/a" : sgn(a - b, Math.round(Math.abs((a - b) / b) * 100) + "%")), whyNo: "Archive gap or before the archive began" },
      { k: "sol", name: "Solar energy, total (kWh/m\u00b2)", map: P.sol, mode: "sum", fmt: (v) => f1(v), dfmt: (a, b) => (b === 0 ? "n/a" : sgn(a - b, Math.round(Math.abs((a - b) / b) * 100) + "%")), whyNo: "Archive gap or before the archive began" },
    ];
    const calc = metrics.map((m) => N.map((n) => {
      const cur = win(m.map, P.last, n, m.mode), ly = win(m.map, addYears(P.last, -1), n, m.mode);
      const prior = [1, 2, 3].map((k) => win(m.map, addYears(P.last, -k), n, m.mode)).filter((w) => w.valid);
      const avg = prior.length ? prior.reduce((s, w) => s + w.v, 0) / prior.length : null;
      return { n, cur, ly, avg, na: prior.length };
    }));
    // headline: biggest valid rain delta vs earlier-year average
    const r = calc[1].filter((x) => x.cur.valid && x.avg != null && x.avg > 0).map((x) => ({ n: x.n, d: (x.cur.v - x.avg) / x.avg, v: x.cur.v, a: x.avg }));
    const big = r.sort((a, b) => Math.abs(b.d) - Math.abs(a.d))[0];
    const head = big ? `The last ${big.n} days brought ${f2(big.v)} in of rain, ${Math.round(Math.abs(big.d) * 100)}% ${big.d < 0 ? "less" : "more"} than the same ${big.n} days in earlier years` : "Rolling windows";
    let html = `<h2>${head}</h2><p class="sub">Each window ends ${fmtDay(P.last)}, the last day in the archive, and is set against the same window a year earlier and the average of earlier years. A window missing more than 10% of its days is shown hatched, not computed; a ≈ marks a prorated total with a few low-coverage days excluded.</p><div class="vb-grid three" style="margin-top:16px">`;
    metrics.forEach((m, mi) => {
      html += `<table><caption>${m.name}</caption><thead><tr><th>Window</th><th>This</th><th>Last yr</th><th>Δ</th><th>Avg</th><th>Δ</th></tr></thead><tbody>`;
      calc[mi].forEach((x) => {
        const na = (why) => `<td class="gap-hatch" title="${why}">n/a<span class="sr"> not valid: ${why}</span></td>`;
        const th = `<td><b>${x.n} d</b></td>`;
        const cur = x.cur.valid ? `<td><b>${x.cur.est ? "≈" : ""}${m.fmt(x.cur.v)}</b></td>` : na(m.whyNo);
        const ly = x.ly.valid ? `<td>${x.ly.est ? "≈" : ""}${m.fmt(x.ly.v)}</td>` : na(m.whyNo);
        const d1 = x.cur.valid && x.ly.valid ? `<td class="d">${m.dfmt(x.cur.v, x.ly.v)}</td>` : na(m.whyNo);
        const av = x.avg != null ? `<td>${m.fmt(x.avg)}<span class="sr"> average of ${x.na} years</span></td>` : na(m.whyNo);
        const d2 = x.cur.valid && x.avg != null ? `<td class="d">${m.dfmt(x.cur.v, x.avg)}</td>` : na(m.whyNo);
        html += `<tr>${th}${cur}${ly}${d1}${av}${d2}</tr>`;
      });
      html += `</tbody></table>`;
    });
    html += `</div><p class="src">Source: station archive daily values through ${fmtDay(P.last)}. Avg = mean of the same window ending on this date in each earlier year that is valid (the 365-day average of rain and solar rests on one or two years; 2023 is too short). Solar in kWh/m².</p>`;
    host.innerHTML = html;
  }

  /* ----- Rain ----- */
  function waCum(P, wy) {
    const start = Math.floor(Date.UTC(wy - 1, 9, 1) / DAY), end = Math.floor(Date.UTC(wy, 8, 30) / DAY);
    const base = Math.floor(Date.UTC(2000, 9, 1) / DAY);
    const pts = []; let cum = 0;
    for (let i = start; i <= Math.min(end, P.today); i++) {
      const d = dt(i); let mo = d.getUTCMonth(), da = d.getUTCDate(); if (mo === 1 && da === 29) da = 28;
      const x = Math.floor(Date.UTC(mo >= 9 ? 2000 : 2001, mo, da) / DAY) - base;
      let v = P.rain.get(i), gap = false;
      if (i === P.today) { v = P.D.current.rain_today_in; }
      else if (v == null) gap = true;
      cum += v || 0;
      pts.push({ x, y: +cum.toFixed(3), gap, i });
    }
    return { wy, pts };
  }
  function rainTab(el, P) {
    const D = P.D, curWY = dt(P.today).getUTCMonth() >= 9 ? dt(P.today).getUTCFullYear() + 1 : dt(P.today).getUTCFullYear();
    const wys = [curWY - 3, curWY - 2, curWY - 1, curWY].map((w) => waCum(P, w));
    const cur = wys[3], off = cur.pts.length - 1;
    const at = (w) => { const p = w.pts[off]; return p ? { v: p.y, gap: w.pts.slice(0, off + 1).some((q) => q.gap) } : null; };
    const prev = wys.slice(0, 3).map((w) => ({ wy: w.wy, a: at(w) })).filter((x) => x.a && !x.a.gap);
    const cv = cur.pts[off].y, beat = prev.filter((x) => cv > x.a.v).length;
    const days = off + 1;
    const head = !prev.length ? "Not enough history" : beat === prev.length ? `Water year ${curWY} has started wetter than any before it: ${f2(cv)} in in ${days} days` : beat === 0 ? `Water year ${curWY} has started drier than every earlier one: ${f2(cv)} in in ${days} days, against ${prev.map((x) => f1(x.a.v)).join(", ")} in` : `Water year ${curWY} has ${f2(cv)} in after ${days} days, wetter than ${beat} of ${prev.length} earlier starts`;
    // monthly totals
    const months = [];
    for (let k = 23; k >= 0; k--) { const d = new Date(Date.UTC(dt(P.today).getUTCFullYear(), dt(P.today).getUTCMonth() - k, 1)); months.push({ y: d.getUTCFullYear(), m: d.getUTCMonth() + 1 }); }
    const mm = months.map((o) => ({ ...o, c: month(P, P.rain, o.y, o.m, "sum", o.y === dt(P.today).getUTCFullYear() && o.m - 1 === dt(P.today).getUTCMonth() ? D.current.rain_today_in : 0) }));
    const okm = mm.filter((x) => x.c.st === "ok");
    const pk = okm.reduce((a, b) => (b.c.v > a.c.v ? b : a));
    const dry = mm.filter((x) => x.c.st === "ok" && x.c.v < 0.5);
    el.innerHTML = `
<div class="vb-grid two-r">
  <div><h2>${head}</h2><p class="sub">Cumulative rain from October 1, one line per water year. The bold line is the year in progress.</p>
    <div class="seg" role="group" aria-label="Range"><button type="button" data-r="q" aria-pressed="true">First 3 months</button><button type="button" data-r="y" aria-pressed="false">Full water year</button></div>
    <div id="vb-cum" class="chart" style="height:340px"></div>
    <p class="src">Source: station archive, daily rain totals, plus today’s live total. Dotted segments are days missing or under 80% station coverage (largest: Apr–Jun 2025, Feb 2026), so those years’ totals are understated. Water year 2023 is omitted: the archive begins Feb 15, 2023.</p></div>
  <div><h2>${MONL[pk.m - 1]} ${pk.y} was the wettest full month in two years, with ${f1(pk.c.v)} in; ${dry.length} full months stayed under half an inch</h2><p class="sub">Monthly rain totals, last 24 months. Hatched bars are months with archive gaps or still running.</p>
    <div id="vb-mon" class="chart" style="height:300px"></div>
    <p class="src">Source: station archive, daily rain totals.</p></div>
</div>`;
    const rainCols = [css("--rain-1"), css("--rain-2"), css("--rain-3")];
    function drawCum(range) {
      const tv = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335], tt = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"];
      const xmax = range === "q" ? 92 : 365;
      const tr = [], ann = [];
      wys.forEach((w, wi) => {
        const isCur = wi === 3; const col = isCur ? css("--ink") : rainCols[wi]; let seg = [], gapState = null;
        const flush = () => { if (seg.length > 1 || (seg.length === 1 && isCur)) tr.push({ type: "scatter", mode: "lines", x: seg.map((p) => p.x), y: seg.map((p) => p.y), line: { color: col, width: isCur ? 4 : 2, dash: gapState ? "dot" : "solid", shape: "hv" }, hoverinfo: gapState ? "skip" : "text", text: seg.map((p) => `WY${w.wy}: ${f2(p.y)} in by ${fmtDay(p.i)}`), cliponaxis: false }); seg = []; };
        w.pts.forEach((p) => { if (gapState === null) gapState = p.gap; if (p.gap !== gapState) { const last = seg[seg.length - 1]; flush(); gapState = p.gap; if (last) seg.push(last); } seg.push(p); });
        flush();
        const vis = w.pts.filter((p) => p.x <= xmax); const lp = vis[vis.length - 1];
        if (lp) ann.push({ x: lp.x, y: lp.y, text: `WY${w.wy} ${f1(lp.y)}`, wy: w.wy, cur: isCur });
      });
      ann.sort((a, b) => a.y - b.y); const ymax = Math.max(...ann.map((a) => a.y)) * 1.12;
      for (let i = 1; i < ann.length; i++) { const gap = ymax * 0.07; if (ann[i].y - ann[i].ty0 < 0) { } ann[i].ty = Math.max(ann[i].y, (ann[i - 1].ty != null ? ann[i - 1].ty : ann[i - 1].y) + gap); } if (ann[0]) ann[0].ty = ann[0].y;
      const annotations = ann.map((a) => ({ x: a.x, y: a.ty != null ? a.ty : a.y, xanchor: range === "q" ? "left" : "right", yanchor: "middle", text: a.text, showarrow: false, xshift: range === "q" ? 6 : -4, font: { size: 12, color: css("--ink"), family: css("--sans") } }));
      Plotly.react($(el, "#vb-cum"), tr, plotBase({ margin: { l: 38, r: range === "q" ? 92 : 14, t: 16, b: 30 }, xaxis: { range: [0, range === "q" ? 92 : 365], tickvals: tv.filter((v) => v <= xmax), ticktext: tt.filter((_, i) => tv[i] <= xmax), showgrid: false, linecolor: css("--rule"), tickfont: { color: css("--ink-3") }, fixedrange: true, zeroline: false }, yaxis: { title: { text: "inches, cumulative", font: { size: 11, color: css("--ink-3") } }, gridcolor: css("--rule"), zeroline: false, range: [0, ymax * 1.08], fixedrange: true, tickfont: { color: css("--ink-3") } }, annotations, hovermode: "closest" }), PCFG);
    }
    let range = "q"; drawCum(range);
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { range = b.dataset.r; el.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b)); drawCum(range); }));
    // monthly bars
    const rule = css("--rule-strong"), surf = css("--ground");
    Plotly.react($(el, "#vb-mon"), [{
      type: "bar", x: mm.map((o) => `${o.y}-${String(o.m).padStart(2, "0")}-15`), y: mm.map((o) => (o.c.v != null ? o.c.v : o.c.partial || 0)),
      marker: { color: mm.map((o) => (o.c.st === "ok" ? css("--rain-3") : o.c.st === "cur" ? css("--rain-2") : css("--rule-strong"))), pattern: { shape: mm.map((o) => (o.c.st === "ok" ? "" : "/")), fgcolor: css("--rain-3"), bgcolor: surf, size: 5, solidity: 0.35 } },
      hovertemplate: mm.map((o) => `${MON[o.m - 1]} ${o.y}: %{y:.2f} in${o.c.st === "ok" ? "" : " (" + (o.c.st === "cur" ? "month still running" : "gap, not a valid total") + ")"}<extra></extra>`),
    }], plotBase({ margin: { l: 34, r: 8, t: 8, b: 30 }, bargap: 0.25, xaxis: { type: "date", tickformat: "%b\n%Y", dtick: "M3", showgrid: false, linecolor: css("--rule"), tickfont: { color: css("--ink-3") }, fixedrange: true }, yaxis: { gridcolor: css("--rule"), zeroline: false, fixedrange: true, tickfont: { color: css("--ink-3") }, ticksuffix: "" }, hovermode: "closest" }), PCFG);
  }

  /* ----- Events ----- */
  function eventsTab(el, P) {
    const D = P.D, ev = D.rain_events.slice().sort((a, b) => (a.ongoing === b.ongoing ? Date.parse(b.start_time) - Date.parse(a.start_time) : a.ongoing ? -1 : 1));
    const byTotal = D.rain_events.slice().sort((a, b) => b.total_rainfall - a.total_rainfall);
    const rank = (e) => byTotal.findIndex((x) => x.event_id === e.event_id) + 1;
    let sel = ev[0], limit = 20;
    const curN = D.current;
    el.innerHTML = `
<div class="vb-grid">
  <div><h2 id="vb-eh"></h2><p class="sub" id="vb-es"></p><div id="vb-ec" class="chart" style="height:280px"></div><p class="src" id="vb-esrc"></p></div>
  <div><div id="vb-et"></div></div>
</div>`;
    function curve() {
      const e = sel, s = e.series, x = s.t.map(ptString), tt = e.total_rainfall;
      const r = rank(e);
      $(el, "#vb-eh").textContent = `${f2(tt)} in over ${fmtDur(e.duration_minutes)}: the ${r}${["th", "st", "nd", "rd"][r % 100 > 10 && r % 100 < 14 ? 0 : Math.min(r % 10, 4) % 4]} wettest of ${D.rain_events.length} catalogued events${e.max_hourly_rate ? ", peaking at " + f2(e.max_hourly_rate) + " in/hr" : ""}`;
      $(el, "#vb-es").textContent = `Accumulation of the event that began ${fmtPT(Date.parse(e.start_time))}${e.ongoing ? "; the catalog still marks it ongoing" : ""}. Data quality: ${e.quality_rating}, ${Math.round(e.data_completeness * 100)}% complete.`;
      $(el, "#vb-esrc").textContent = "Source: Event Catalog, 5-minute station readings, Pacific time.";
      const tr = [{ type: "scatter", mode: "lines", x, y: s.accum_in, line: { color: css("--rain-3"), width: 3, shape: "hv" }, fill: "tozeroy", fillcolor: css("--rain-0"), hovertemplate: "%{x|%b %d %H:%M}: %{y:.2f} in<extra></extra>" }];
      const lastI = x.length - 1;
      const ann = [{ x: x[lastI], y: s.accum_in[lastI], text: `${f2(s.accum_in[lastI])} in`, showarrow: false, xanchor: "right", yanchor: "bottom", yshift: 6, font: { size: 12, color: css("--ink") } }];
      if (e.ongoing) {
        tr.push({ type: "scatter", mode: "markers", x: [x[lastI]], y: [s.accum_in[lastI]], marker: { color: css("--note"), size: 11, line: { color: css("--ground"), width: 2 } }, hoverinfo: "skip" });
        ann.push({ x: x[lastI], y: s.accum_in[lastI], text: "archive ends", showarrow: false, xanchor: "right", yanchor: "top", yshift: -10, font: { size: 11, color: css("--note") } });
      }
      Plotly.react($(el, "#vb-ec"), tr, plotBase({ margin: { l: 38, r: 14, t: 12, b: 30 }, annotations: ann, yaxis: { gridcolor: css("--rule"), zeroline: false, rangemode: "tozero", fixedrange: true, tickfont: { color: css("--ink-3") }, ticksuffix: " in" }, xaxis: { showgrid: false, linecolor: css("--rule"), tickfont: { color: css("--ink-3") }, fixedrange: true, zeroline: false, hoverformat: "%b %d %H:%M" }, hovermode: "x" }), PCFG);
    }
    function table() {
      const rows = ev.slice(0, limit);
      const q = (e) => { const bad = e.quality_rating === "poor" || e.flags.low_completeness || !e.usable_for_analysis; return `<span class="q ${bad || e.flags.has_gaps ? "low" : ""}">${e.quality_rating}${e.flags.has_gaps ? " · gaps" : ""}${e.flags.interrupted ? " · cut" : ""}</span>`; };
      let h = `<table class="ev"><caption class="sr">Rain events, newest first, ongoing event pinned</caption><thead><tr><th>Start</th><th>Duration</th><th>Total (in)</th><th>Peak (in/h)</th><th>Quality</th></tr></thead><tbody>`;
      rows.forEach((e) => {
        h += `<tr class="${e.ongoing ? "pin" : ""}" ${e.event_id === sel.event_id ? 'aria-current="true"' : ""}><td><button type="button" class="rowbtn" data-id="${e.event_id}">${e.ongoing ? '<span class="live">live</span>' : ""}${fmtPT(Date.parse(e.start_time), { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).replace(",", "").replace(/ ([AP])M/, (m, x) => x.toLowerCase())}</button></td><td>${fmtDur(e.duration_minutes)}</td><td><b>${f2(e.total_rainfall)}</b></td><td>${e.max_hourly_rate ? f2(e.max_hourly_rate) : "–"}</td><td>${q(e)}</td></tr>`;
      });
      h += `</tbody></table>`;
      const pinned = ev[0].ongoing;
      const note = pinned ? `<p class="note-live"><span class="live">Live</span> The catalog still marks the top event as ongoing, as of the archive’s last reading (${fmtPT(Date.parse(ev[0].end_time))}). The live station shows ${curN.rain_rate_in_hr > 0 ? "rain falling now" : "no rain falling now"}, last rain ${D.rain.stats.time_since_rain.replace("h", " h")} ago, and an event total of ${f2(curN.rain_event_in)} in against the catalog’s ${f2(ev[0].total_rainfall)} in. The catalog lags the live reading by about a day.</p>` : "";
      h += (limit < ev.length ? `<button type="button" class="more">Show ${Math.min(40, ev.length - limit)} more of ${ev.length}</button>` : "") + `<p class="src">Source: Event Catalog (${ev.length} events since Feb 2023). Peak rate is the highest hourly rate in the event. Dashed quality chips mark events with gaps or low completeness. Select a row to chart it.</p>` + note;
      $(el, "#vb-et").innerHTML = h;
      $(el, "#vb-et").querySelectorAll(".rowbtn").forEach((b) => b.addEventListener("click", () => { sel = ev.find((e) => e.event_id === b.dataset.id); curve(); table(); window.scrollTo({ top: el.getBoundingClientRect().top + scrollY - 70, behavior: "smooth" }); }));
      const more = $(el, ".more"); if (more) more.addEventListener("click", () => { limit += 40; table(); });
    }
    curve(); table();
  }

  /* ----- Solar ----- */
  function solarTab(el, P, redraws) {
    const D = P.D, H = D.solar.heatmap_15min_120d;
    const day = new Map(); H.date.forEach((d, i) => { const k = idx(d); day.set(k, (day.get(k) || 0) + H.kwh_m2[i]); });
    const keys = [...day.keys()].sort((a, b) => a - b), k0 = keys[0], k1 = keys[keys.length - 1];
    const missing = []; for (let i = k0; i <= k1; i++) if (!day.has(i)) missing.push(i);
    const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const first14 = keys.slice(0, 14).map((k) => day.get(k)), last14 = keys.slice(-14).map((k) => day.get(k));
    const pkK = keys.reduce((a, b) => (day.get(b) > day.get(a) ? b : a));
    const drop = Math.round((1 - avg(last14) / avg(first14)) * 100);
    const h1 = `Daily sun is down ${drop}% since late June: ${f2(avg(last14))} kWh/m² a day lately against ${f2(avg(first14))} then`;
    // month x year
    const years = [2023, 2024, 2025, 2026], m0 = dt(P.today).getUTCMonth(), curY = dt(P.today).getUTCFullYear();
    const cell = (y, m) => { const c = month(P, P.sol, y, m, "sum", 0); return c; };
    const lastFull = (() => { for (let k = 1; k < 24; k++) { const d = new Date(Date.UTC(curY, m0 - k, 1)); const a = cell(d.getUTCFullYear(), d.getUTCMonth() + 1), b = cell(d.getUTCFullYear() - 1, d.getUTCMonth() + 1); if (a.st === "ok" && b.st === "ok") return { d, a, b }; } })();
    const h2 = lastFull ? `${MONL[lastFull.d.getUTCMonth()]} ${lastFull.d.getUTCFullYear()} collected ${f1(lastFull.a.v)} kWh/m², ${Math.round(Math.abs(lastFull.a.v / lastFull.b.v - 1) * 100)}% ${lastFull.a.v < lastFull.b.v ? "less" : "more"} than the month a year before` : "Month by month";
    el.innerHTML = `
<div class="vb-grid two">
  <div><h2>${h1}</h2><p class="sub">Daily solar energy, last ${keys.length} days with data (${fmtDay(k0)} – ${fmtDay(k1)}). Each square is a day, weeks run left to right.</p>
    <div id="vb-cal"></div><p class="readout" id="vb-cro" aria-live="polite">Tap or hover a day for its value.</p>
    <div class="key"><span>0</span><i style="background:linear-gradient(90deg,var(--sun-soft),var(--sun),var(--temp-warm),var(--temp-hot))"></i><span id="vb-chi"></span><span class="hatch gap-hatch" style="display:inline-block"></span><span>no data</span></div>
    <p class="src">Source: station solar radiation, 15-minute periods summed per day (heatmap data). Peak day ${fmtDay(pkK)} at ${f1(day.get(pkK))} kWh/m². ${missing.length ? missing.length + " day(s) missing, shown hatched." : "No missing days in this window."} Archive ends ${fmtDay(P.last)}; today is not included.</p></div>
  <div><h2>${h2}</h2><p class="sub">Monthly solar energy by year, kWh/m². Hover or tap a month for its value.</p>
    <div id="vb-sm"></div><div class="key"><span>low</span><i style="background:linear-gradient(90deg,var(--sun-soft),var(--sun),var(--temp-warm),var(--temp-hot))"></i><span>high</span><span class="hatch gap-hatch" style="display:inline-block"></span><span>gap or partial month</span></div>
    <p class="readout" id="vb-smro" aria-live="polite">Tap or hover a month for its value.</p>
    <p class="src">Source: station archive, daily solar energy. Hatched grey: the archive has missing or low-coverage days or starts mid-month (Feb 2023). The outlined month is still running.</p></div>
</div>`;
    const maxV = Math.max(...keys.map((k) => day.get(k)));
    $(el, "#vb-chi").textContent = f1(maxV) + " kWh/m²";
    function drawCal() {
      const host = $(el, "#vb-cal");
      const startDow = (dt(k0).getUTCDay() + 6) % 7; const gridStart = k0 - startDow;
      const weeks = Math.ceil((k1 - gridStart + 1) / 7);
      let c = `<div class="calwrap"><div class="dows" aria-hidden="true">${["Mon", "", "Wed", "", "Fri", "", "Sun"].map((d) => `<span>${d}</span>`).join("")}</div><div><div class="calm" style="grid-template-columns:repeat(${weeks},minmax(0,1fr))">`;
      for (let w = 0; w < weeks; w++) { const d0 = gridStart + w * 7; let lab = ""; for (let o = 0; o < 7; o++) { if (dt(d0 + o).getUTCDate() === 1 && d0 + o >= k0) lab = MON[dt(d0 + o).getUTCMonth()]; } if (w === 0 && !lab) lab = MON[dt(k0).getUTCMonth()]; c += `<span>${lab}</span>`; }
      c += `</div><div class="cal" style="grid-template-columns:repeat(${weeks},minmax(0,1fr))">`;
      for (let i = gridStart; i < gridStart + weeks * 7; i++) {
        if (i < k0 || i > k1) { c += `<div class="c"></div>`; continue; }
        const v = day.get(i); const lbl = `${MON[dt(i).getUTCMonth()]} ${dt(i).getUTCDate()}`;
        if (v == null) c += `<button type="button" class="c gap-hatch" data-info="${lbl}: no data, not computed" aria-label="${lbl}: no data"></button>`;
        else c += `<button type="button" class="c" style="background:${ramp(SUN, v / maxV).bg}" data-info="${lbl}: ${f2(v)} kWh/m²" aria-label="${lbl}: ${f2(v)} kWh per square metre"></button>`;
      }
      c += `</div></div></div>`;
      host.innerHTML = c;
      const ro = $(el, "#vb-cro");
      host.querySelectorAll(".c[data-info]").forEach((b) => { const s = () => (ro.textContent = b.dataset.info); b.addEventListener("click", s); b.addEventListener("mouseenter", s); b.addEventListener("focus", s); });
      const sm = $(el, "#vb-sm"), smro = $(el, "#vb-smro");
      almanacGrid(sm, { years, cell: (y, m) => { const r = cell(y, m); r.cur = y === curY && m - 1 === m0; return r; }, readout: smro, unit: "kWh/m²", fmt: (v) => Math.round(v) + "", stops: SUN, lo: 0, cur: m0 });
    }
    redraws.push(() => { if (!el.hidden) drawCal(); });
  }

  window.LOOKOUT_VARIANT = {
    name: "Almanac",
    render(root, data) { build(root, data); },
  };
})();
