// PROTOTYPE variant C: "What's happening". Phone-app shell, bottom nav (left rail >=1024px),
// state-adaptive Now view. One visual grammar everywhere: NOW drawn bold, USUAL as a soft band.
// Charts are hand-built SVG (measured width) so hatching, direct labels and gaps are exact.
(function () {
  const TZ = "America/Los_Angeles";
  const H = 3600000, D = 86400000;
  const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];

  /* ---------- tiny helpers ---------- */
  const fmt = (ms, o) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, ...o }).format(ms);
  const dShort = (ms) => fmt(ms, { month: "short", day: "numeric" });
  const dLong = (ms) => fmt(ms, { weekday: "short", month: "short", day: "numeric" });
  const tShort = (ms) => fmt(ms, { hour: "numeric", minute: "2-digit" }).replace(" ", " ").toLowerCase();
  const isoShort = (iso) => dShort(Date.parse(iso + "T12:00:00Z") + 0).replace(/^/, "");
  const dayMs = (iso) => Date.parse(iso + "T12:00:00Z");
  const dayShort = (iso) => new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" }).format(dayMs(iso));
  const dayDow = (iso) => new Intl.DateTimeFormat("en-US", { timeZone: "UTC", weekday: "short" }).format(dayMs(iso));
  const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2);
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const quant = (arr, p) => {
    const s = [...arr].sort((a, b) => a - b);
    if (!s.length) return NaN;
    const i = (s.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
    return s[lo] + (s[hi] - s[lo]) * (i - lo);
  };
  const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  function niceTicks(max, n) {
    const raw = max / (n || 4), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const st = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += st) out.push(+v.toFixed(6));
    return out;
  }
  const ordinal = (n) => { const s = ["th","st","nd","rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
  const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join("");
  const T = (x, y, s, cls, anchor) =>
    `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" class="${cls || "tk"}"${anchor ? ` text-anchor="${anchor}"` : ""}>${s}</text>`;
  let uidN = 0;

  /* ---------- styles ---------- */
  const CSS = `
.vc{--proto-bar:48px;--nav-h:64px;--vc-sun-ink:color-mix(in srgb,var(--sun) 62%,var(--ink));
  --vc-hot:color-mix(in srgb,var(--temp-hot) 85%,var(--ink));font-family:var(--sans);color:var(--ink);min-height:100vh;
  background:var(--ground);}
.vc *{box-sizing:border-box}
.vc-top{display:flex;justify-content:space-between;align-items:flex-start;gap:var(--s-3);padding:max(var(--s-4),env(safe-area-inset-top)) var(--s-4) var(--s-2);max-width:1040px;margin:0 auto}
.vc-id b{font-weight:600;font-size:var(--step-0);letter-spacing:.01em}
.vc-id span{display:block;font-size:var(--step--1);color:var(--ink-2);margin-top:2px}
.vc-proto{display:flex;flex-direction:column;align-items:flex-end;gap:2px;font-size:11px;color:var(--ink-2);border:1px dashed var(--rule-strong);border-radius:8px;padding:4px 6px}
.vc-proto select{font:inherit;font-size:12px;color:var(--ink);background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:2px 4px}
.vc-main{max-width:1040px;margin:0 auto;padding:var(--s-3) var(--s-4) calc(var(--proto-bar) + var(--nav-h) + var(--s-6))}
.vc-main[hidden]{display:none}
.vc-h{font-family:var(--serif);font-weight:600;font-size:var(--step-2);line-height:1.15;margin:var(--s-3) 0 var(--s-2);text-wrap:balance;max-width:28ch;letter-spacing:-.01em}
.vc-sub{margin:0 0 var(--s-4);font-size:.9375rem;line-height:1.45;color:var(--ink-2);max-width:62ch}
.vc-facts{display:flex;flex-wrap:wrap;gap:var(--s-2) var(--s-5);margin:0 0 var(--s-3);padding:var(--s-2) 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.vc-facts div{display:flex;align-items:baseline;gap:6px}
.vc-facts dt{font-size:var(--step--1);color:var(--ink-2);order:2}
.vc-facts dd{margin:0;font-weight:600;font-size:var(--step-1);order:1}
.vc-chart{width:100%;min-height:200px}
.vc-svg{display:block;max-width:100%;overflow:visible}
.vc-foot{font-size:var(--step--1);line-height:1.4;color:var(--ink-2);margin:var(--s-2) 0 0;max-width:70ch}
.vc-why{margin:var(--s-3) 0 0;font-size:var(--step--1);color:var(--ink-2)}
.vc-why summary{cursor:pointer;width:max-content}
.vc-why ol{margin:var(--s-2) 0 0;padding:0;list-style:none;display:grid;gap:4px}
.vc-why li{display:flex;gap:8px;align-items:baseline}
.vc-why li i{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--ink-3);flex:none;transform:translateY(0)}
.vc-why li[data-met="1"] i{background:var(--rain-3);border-color:var(--rain-3)}
.vc-why li[data-lead="1"]{color:var(--ink);font-weight:600}
.vc-sec{margin-top:var(--s-6)}
.vc-sec h2{font-family:var(--serif);font-weight:600;font-size:var(--step-1);margin:0 0 var(--s-1);letter-spacing:-.005em}
.vc-sec .vc-sub{margin-bottom:var(--s-3)}
.vc-grid{display:grid;gap:0 var(--s-6)}
.vc-read{display:grid;grid-template-columns:1fr 1fr;margin:0;border-top:1px solid var(--rule)}
.vc-read div{padding:var(--s-3) 0;border-bottom:1px solid var(--rule)}
.vc-read div:nth-child(odd){padding-right:var(--s-3)}
.vc-read dt{font-size:var(--step--1);color:var(--ink-2)}
.vc-read dd{margin:2px 0 0;font-size:var(--step-1);font-weight:600;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.vc-read dd small{font-size:var(--step--1);font-weight:400;color:var(--ink-2)}
.vc-read svg{width:16px;height:16px;stroke:currentColor;fill:none;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round;flex:none}
.vc-ctx{display:grid;gap:var(--s-4)}
.vc-ctx-row{display:grid;grid-template-columns:4.6rem 1fr;gap:2px var(--s-3);align-items:center}
.vc-ctx-l{font-size:.875rem;font-weight:600}
.vc-ctx-track{position:relative;height:14px;border-radius:7px;background:var(--rule);overflow:visible}
.vc-ctx-fill{position:absolute;left:0;top:0;bottom:0;border-radius:7px;background:var(--rain-3)}
.vc-ctx-tick{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);border-radius:1px}
.vc-ctx-bad{position:absolute;inset:0;border-radius:7px;border:1px dashed var(--rule-strong)}
.vc-ctx-v{grid-column:2;font-size:.8125rem;color:var(--ink-2)}
.vc-ctx-v b{color:var(--ink);font-weight:600}
.vc-rows{list-style:none;margin:0;padding:0;border-top:1px solid var(--rule)}
.vc-row{display:grid;grid-template-columns:1fr auto;gap:2px var(--s-3);padding:var(--s-3) 0;border-bottom:1px solid var(--rule);align-items:center}
.vc-row b{font-weight:600}
.vc-row .vc-bar{grid-column:1/3;height:6px;border-radius:3px;background:var(--rule);position:relative;margin-top:4px}
.vc-row .vc-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--rain-3)}
.vc-row small{color:var(--ink-2);font-size:var(--step--1)}
.vc-tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:5px;border:1px solid var(--rule-strong);color:var(--ink-2);margin-left:6px;vertical-align:1px}
.vc-nav{position:fixed;z-index:50;left:0;right:0;bottom:var(--proto-bar);height:var(--nav-h);display:flex;background:var(--surface);border-top:1px solid var(--rule);box-shadow:0 -2px 10px rgba(15,26,43,.06)}
.vc-nav button{all:unset;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;font:500 12px/1 var(--sans);color:var(--ink-2);cursor:pointer;position:relative}
.vc-nav button:focus-visible{outline:2px solid var(--rain-2);outline-offset:-4px;border-radius:10px}
.vc-nav svg{width:24px;height:24px;stroke:currentColor;fill:none;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round}
.vc-nav button[aria-current="page"]{color:var(--rain-3);font-weight:600}
.vc-nav button[aria-current="page"]::before{content:"";position:absolute;top:0;left:30%;right:30%;height:2px;background:var(--rain-3);border-radius:0 0 2px 2px}
.vc-svg .ax{stroke:var(--rule);stroke-width:1;fill:none}
.vc-svg .ax2{stroke:var(--rule-strong);stroke-width:1;fill:none}
.vc-svg .tk{font:11px var(--sans);fill:var(--ink-2);font-variant-numeric:tabular-nums}
.vc-svg .lb{font:600 12px var(--sans);fill:var(--ink);paint-order:stroke;stroke:var(--ground);stroke-width:4px;stroke-linejoin:round}
.vc-svg .lb2{font:12px var(--sans);fill:var(--ink-2);paint-order:stroke;stroke:var(--ground);stroke-width:4px;stroke-linejoin:round}
.vc-svg .nt{font:600 12.5px var(--sans);fill:var(--note);paint-order:stroke;stroke:var(--ground);stroke-width:4px;stroke-linejoin:round}
.vc-svg .now{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:3}
.vc-svg .thin{fill:none;stroke-width:1.5}
@keyframes vcdraw{to{stroke-dashoffset:0}}
.vc-anim .now{stroke-dasharray:1;stroke-dashoffset:1;animation:vcdraw 1.1s cubic-bezier(.16,1,.3,1) .1s forwards}
@media (prefers-reduced-motion:reduce){.vc-anim .now{animation:none;stroke-dasharray:none;stroke-dashoffset:0}}
.vc ::selection{background:var(--rain-1)}
@media (min-width:1024px){
  .vc{--proto-bar:0px}
  .vc-nav{top:0;right:auto;bottom:0;left:0;width:96px;height:auto;flex-direction:column;justify-content:flex-start;padding-top:var(--s-6);gap:var(--s-2);border-top:0;border-right:1px solid var(--rule);box-shadow:none}
  .vc-nav button{flex:none;height:72px}
  .vc-nav button[aria-current="page"]::before{top:18%;bottom:18%;left:0;right:auto;width:2px;height:auto;border-radius:0 2px 2px 0}
  .vc-top,.vc-main{margin-left:calc(96px + max(0px,(100vw - 96px - 1040px)/2));margin-right:auto;max-width:944px;width:calc(100% - 96px - var(--s-6))}
  .vc-top{padding-left:var(--s-6)} .vc-main{padding:var(--s-4) var(--s-6) 72px}
  .vc-h{font-size:var(--step-3);max-width:24ch}
  .vc-grid{grid-template-columns:1fr 1fr}
  .vc-sec{margin-top:var(--s-7)}
  .vc-grid .vc-sec{margin-top:var(--s-6)}
}
`;

  const ICONS = {
    now: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="2.5"/>',
    rain: '<path d="M12 3.5C12 3.5 5.5 10.8 5.5 15a6.5 6.5 0 0 0 13 0C18.5 10.8 12 3.5 12 3.5z"/>',
    events: '<path d="M9 6.5h11M9 12h11M9 17.5h11"/><circle cx="4.5" cy="6.5" r="1.2"/><circle cx="4.5" cy="12" r="1.2"/><circle cx="4.5" cy="17.5" r="1.2"/>',
    solar: '<circle cx="12" cy="12" r="3.8"/><path d="M12 2.8v2.4M12 18.8v2.4M2.8 12h2.4M18.8 12h2.4M5.5 5.5l1.7 1.7M16.8 16.8l1.7 1.7M5.5 18.5l1.7-1.7M16.8 7.2l1.7-1.7"/>',
    up: '<path d="M12 19V5M6 11l6-6 6 6"/>',
    down: '<path d="M12 5v14M6 13l6 6 6-6"/>',
    flat: '<path d="M4 12h16M14 6l6 6-6 6"/>',
  };
  const icon = (n) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[n]}</svg>`;

  /* ---------- the model: everything computed from data.json ---------- */
  function buildModel(data) {
    const cur = data.current, M = {};
    M.cur = cur;
    M.nowMs = cur.dateutc;
    M.archEnd = Date.parse(data.meta.archive_range[1]);
    M.lagH = (M.nowMs - M.archEnd) / H;
    // archive daily rain + gaps
    const da = data.rain.daily_all;
    M.rainDay = new Map(da.date.map((d, i) => [d, da.rain_in[i]]));
    M.dates = da.date;
    M.missing = (a, b) => { // missing archive days strictly inside [a,b]
      const out = []; for (let t = dayMs(a); t <= dayMs(b); t += D) { const k = new Date(t).toISOString().slice(0, 10); if (!M.rainDay.has(k)) out.push(k); } return out;
    };
    M.gapSpans = []; // contiguous missing runs
    for (let i = 1; i < M.dates.length; i++) {
      const a = dayMs(M.dates[i - 1]), b = dayMs(M.dates[i]);
      if (b - a > D * 1.5) M.gapSpans.push({ from: new Date(a + D).toISOString().slice(0, 10), to: new Date(b - D).toISOString().slice(0, 10) });
    }
    // events
    M.events = data.rain_events.slice().sort((a, b) => Date.parse(a.start_time) - Date.parse(b.start_time));
    const last = M.events[M.events.length - 1];
    M.lastRain = Date.parse(cur.last_rain);
    M.hoursSinceRain = (M.nowMs - M.lastRain) / H;
    M.raining = cur.rain_rate_in_hr > 0;
    const liveTail = last && last.ongoing && cur.rain_event_in > 0;
    const mkEv = (e, tail) => {
      const s = Date.parse(e.start_time), ser = e.series || { t: [], accum_in: [] };
      const archEndMs = ser.t.length ? ser.t[ser.t.length - 1] : Date.parse(e.end_time);
      const o = { e, start: s, series: ser, tail, peak: e.max_hourly_rate, archTotal: e.total_rainfall, archEndMs };
      if (tail) { o.total = cur.rain_event_in; o.endMs = M.lastRain; }
      else { o.total = e.total_rainfall; o.endMs = Date.parse(e.end_time); }
      o.durH = Math.max(0.25, (o.endMs - s) / H);
      return o;
    };
    M.liveEv = liveTail ? mkEv(last, true) : null;
    const completed = M.events.filter((e) => !e.ongoing && e.series && e.series.t && e.series.t.length);
    M.pastEv = mkEv(completed[completed.length - 1], false);
    M.pool = completed; // events with curves, completed
    M.rainLead = M.raining || (M.hoursSinceRain <= 12 && cur.rain_event_in > 0);
    // temperature normal
    const mi = Number(fmt(M.nowMs, { month: "numeric" })) - 1;
    const tb = data.overview.temp_by_month;
    M.month = mi; M.monthName = MONTHS[mi];
    M.tq1 = tb.q1[mi]; M.tmed = tb.median[mi]; M.tq3 = tb.q3[mi];
    M.tdep = cur.temp_f - M.tmed;
    M.tempLead = Math.abs(M.tdep) >= 10;
    M.solarLead = cur.solar_wm2 >= 400;
    return M;
  }

  /* ---------- rain event: curve + usual band ---------- */
  function accumAt(ser, start, h) {
    let v = 0;
    for (let i = 0; i < ser.t.length; i++) { if ((ser.t[i] - start) / H <= h + 1e-9) v = ser.accum_in[i]; else break; }
    return v;
  }
  function similarBand(M, E) {
    const tryRatio = (lo, hi) => M.pool.filter((e) => { const r = e.duration_minutes / 60 / E.durH; return r >= lo && r <= hi && e.start_time !== E.e.start_time; });
    let sim = tryRatio(0.6, 1.6);
    if (sim.length < 6) sim = tryRatio(0.4, 2.5);
    const xmax = Math.max(E.durH, E.tail ? (M.nowMs - E.start) / H : 0);
    const hs = [];
    for (let h = 0; h <= xmax + 1e-9; h += 0.5) hs.push(h);
    const lo = [], med = [], hi = [];
    hs.forEach((h) => {
      const vals = sim.map((e) => accumAt(e.series, Date.parse(e.start_time), h));
      lo.push(quant(vals, 0.25)); med.push(quant(vals, 0.5)); hi.push(quant(vals, 0.75));
    });
    const finals = sim.map((e) => e.total_rainfall);
    return { n: sim.length, hs, lo, med, hi, finals, medFinal: quant(finals, 0.5), xmax,
      pct: finals.length ? finals.filter((v) => v <= E.total).length / finals.length : 0.5 };
  }
  function durText(h) { const m = Math.round(h * 60); return m >= 1440 ? `${Math.floor(m / 1440)} d ${Math.floor((m % 1440) / 60)} h` : m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} m` : `${m} m`; }

  function rainEventChart(M, E, B, label) {
    return (w, anim) => {
      const h = w < 600 ? 250 : 310, m = { t: 12, r: w < 600 ? 14 : 24, b: 28, l: 34 };
      const iw = w - m.l - m.r, ih = h - m.t - m.b;
      const nowH = (M.nowMs - E.start) / H, xmax = Math.max(B.xmax, E.durH);
      const ymax = Math.max(E.total, Math.max(...B.hi)) * 1.18 || 0.5;
      const x = lin(0, xmax, 0, iw), y = lin(0, ymax, ih, 0);
      const id = "vh" + ++uidN;
      let s = `<defs>${hatchDef(id)}</defs><g transform="translate(${m.l},${m.t})">`;
      niceTicks(ymax, 4).forEach((v) => { s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v.toFixed(v < 1 ? 2 : 1), "tk", "end"); });
      const step = xmax > 30 ? 6 : xmax > 12 ? 3 : 2;
      for (let hh = 0; hh <= xmax; hh += step) s += T(x(hh), ih + 18, hh + " h", "tk", hh === 0 ? "start" : "middle");
      // usual band + median
      const top = B.hs.map((hh, i) => [x(hh), y(B.hi[i])]), bot = B.hs.map((hh, i) => [x(hh), y(B.lo[i])]).reverse();
      s += `<path d="${path(top)}L${path(bot).slice(1)}Z" style="fill:var(--rain-0);opacity:.95"/>`;
      s += `<path d="${path(B.hs.map((hh, i) => [x(hh), y(B.med[i])]))}" class="thin" style="stroke:var(--rain-2);stroke-dasharray:4 3"/>`;
      // archive-lag hatch
      const aH = (E.archEndMs - E.start) / H;
      if (E.tail) {
        s += `<rect x="${x(aH)}" y="0" width="${Math.max(0, x(Math.min(nowH, xmax)) - x(aH))}" height="${ih}" style="fill:url(#${id});opacity:.85"/>`;
      }
      // now: archived curve
      const pts = [[0, 0]].concat(E.series.t.map((t, i) => [(t - E.start) / H, E.series.accum_in[i]]));
      let np = pts.map((p) => [x(p[0]), y(p[1])]);
      if (!E.tail) np.push([x(E.durH), y(E.total)]);
      s += `<path d="${path(np)}" class="now" pathLength="1" style="stroke:var(--rain-3)"/>`;
      if (E.tail) {
        const lr = (E.endMs - E.start) / H;
        s += `<path d="M${x(aH)},${y(pts[pts.length - 1][1])}L${x(lr)},${y(E.total)}" class="thin" style="stroke:var(--rain-3);stroke-width:2;stroke-dasharray:1 5;stroke-linecap:round"/>`;
        s += `<path d="M${x(lr)},${y(E.total)}L${x(Math.min(nowH, xmax))},${y(E.total)}" class="now" style="stroke:var(--rain-3)"/>`;
        s += `<circle cx="${x(lr)}" cy="${y(E.total)}" r="5" style="fill:var(--rain-3);stroke:var(--ground);stroke-width:2"/>`;
        s += T(x(lr) - 8, y(E.total) - 12, `${f2(E.total)} in`, "nt", "end");
        const hw = x(Math.min(nowH, xmax)) - x(aH);
        if (hw > 90) s += T(x(aH) + 8, ih - 30, "Not yet in the archive:", "lb2") + T(x(aH) + 8, ih - 14, "only the live total is known", "lb2");
      } else {
        s += `<circle cx="${x(E.durH)}" cy="${y(E.total)}" r="5" style="fill:var(--rain-3);stroke:var(--ground);stroke-width:2"/>`;
        s += T(x(E.durH) - 8, y(E.total) - 12, `${f2(E.total)} in`, "nt", "end");
      }
      s += T(4, 12, `Soft band: middle half of ${B.n} past events of similar length`, "lb2");
      s += T(x(B.hs[Math.floor(B.hs.length * 0.3)]), y(B.med[Math.floor(B.hs.length * 0.3)]) + 18, `median ${f2(B.medFinal)} in`, "lb2");
      s += "</g>";
      return `<svg class="vc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
    };
  }
  const hatchDef = (id) =>
    `<pattern id="${id}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" style="fill:var(--gap-b)"/><rect width="3" height="8" style="fill:var(--gap-a)"/></pattern>`;

  /* ---------- temperature lead ---------- */
  function tempChart(M, data, label) {
    return (w) => {
      const r = data.recent, h = w < 600 ? 250 : 310, m = { t: 12, r: w < 600 ? 44 : 60, b: 28, l: 34 };
      const iw = w - m.l - m.r, ih = h - m.t - m.b;
      const t0 = r.t[0], t1 = M.nowMs;
      const lo = Math.min(...r.temp_f, M.cur.temp_f, M.tq1) - 3, hi = Math.max(...r.temp_f, M.tq3) + 3;
      const x = lin(t0, t1, 0, iw), y = lin(lo, hi, ih, 0);
      const warm = M.tdep > 0, col = warm ? "var(--vc-hot)" : "var(--temp-cold)";
      const id = "vh" + ++uidN;
      let s = `<defs>${hatchDef(id)}</defs><g transform="translate(${m.l},${m.t})">`;
      niceTicks(hi - lo, 4).forEach((dv) => {}); // keep deterministic
      for (let v = Math.ceil(lo / 10) * 10; v <= hi; v += 10) s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v + "°", "tk", "end");
      s += `<rect x="0" y="${y(M.tq3)}" width="${iw}" height="${y(M.tq1) - y(M.tq3)}" style="fill:var(--rain-0);opacity:.95"/>`;
      s += `<line x1="0" x2="${iw}" y1="${y(M.tmed)}" y2="${y(M.tmed)}" class="thin" style="stroke:var(--rain-2);stroke-dasharray:4 3"/>`;
      // day ticks at local midnight
      for (let t = Math.ceil(t0 / H) * H; t <= t1; t += H) {
        if (fmt(t, { hour: "numeric", hour12: false }).replace("24", "0") === "0" || fmt(t, { hour: "2-digit", hourCycle: "h23" }) === "00") {
          s += `<line class="ax" x1="${x(t)}" x2="${x(t)}" y1="${ih}" y2="${ih + 5}"/>` + T(x(t) + 3, ih + 18, fmt(t, { weekday: "short" }), "tk");
        }
      }
      const ae = r.t[r.t.length - 1];
      s += `<rect x="${x(ae)}" y="0" width="${x(t1) - x(ae)}" height="${ih}" style="fill:url(#${id});opacity:.85"/>`;
      const pts = r.t.map((t, i) => [x(t), y(r.temp_f[i])]);
      s += `<path d="${path(pts)}" class="now" pathLength="1" style="stroke:${col}"/>`;
      s += `<circle cx="${x(t1)}" cy="${y(M.cur.temp_f)}" r="5.5" style="fill:${col};stroke:var(--ground);stroke-width:2"/>`;
      s += T(x(t1) + 10, y(M.cur.temp_f) + 4, `${f1(M.cur.temp_f)}°`, "nt");
      s += T(4, y(M.tq1) + 15, `Soft band: middle half of ${M.monthName} readings since 2023`, "lb2");
      s += T(x(ae) - 6, y(M.tmed) - 6, `median ${f1(M.tmed)}°`, "lb2", "end");
      const hw = x(t1) - x(ae);
      if (hw > 70) s += T(x(ae) + 6, ih - 8, "not yet archived", "lb2");
      s += "</g>";
      return `<svg class="vc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
    };
  }

  /* ---------- solar: a day against the usual profile ---------- */
  function solarModel(data, M) {
    const p = data.solar.periods_15min_7d, days = {};
    p.start.forEach((st, i) => {
      const k = st.slice(0, 10), mins = Number(st.slice(11, 13)) * 60 + Number(st.slice(14, 16));
      (days[k] = days[k] || []).push({ min: mins, w: p.kwh_m2[i] * 4000, kwh: p.kwh_m2[i] });
    });
    const full = Object.keys(days).sort().filter((k) => days[k].length >= 90);
    full.forEach((k) => (days[k].total = sum(days[k].map((q) => q.kwh))));
    const monthOf = (k) => Number(k.slice(5, 7));
    const prof = (mo) => data.solar.hourly_profile_by_month.filter((q) => q.month === mo).sort((a, b) => a.hour - b.hour);
    const sd = data.solar.daily_all, byMonth = {};
    sd.date.forEach((d, i) => { if (sd.kwh_m2[i] != null) (byMonth[d.slice(0, 7)] = byMonth[d.slice(0, 7)] || []).push(sd.kwh_m2[i]); });
    const monthVals = (mo, notYear) => sd.date.map((d, i) => [d, sd.kwh_m2[i]]).filter(([d, v]) => v != null && Number(d.slice(5, 7)) === mo && Number(d.slice(0, 4)) !== notYear).map(([, v]) => v);
    const usualDay = (mo) => sum(prof(mo).map((q) => q.kwh_m2_per_day));
    return { days, full, prof, monthOf, monthVals, usualDay, sd };
  }
  function solarDayChart(S, key, M, label) {
    return (w) => {
      const h = w < 600 ? 246 : 306, m = { t: 28, r: 16, b: 28, l: 40 };
      const iw = w - m.l - m.r, ih = h - m.t - m.b;
      const day = S.days[key], prof = S.prof(S.monthOf(key));
      const usual = prof.map((q) => ({ hr: q.hour + 0.5, w: q.kwh_m2_per_day * 1000 }));
      const ymax = Math.max(...day.map((q) => q.w), ...usual.map((q) => q.w)) * 1.15 || 100;
      const x = lin(5, 20, 0, iw), y = lin(0, ymax, ih, 0);
      let s = `<g transform="translate(${m.l},${m.t})">`;
      niceTicks(ymax, 4).forEach((v) => { s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v, "tk", "end"); });
      for (let hr = 6; hr <= 20; hr += 2) s += T(x(hr), ih + 18, hr === 12 ? "noon" : (hr % 12 || 12) + (hr < 12 ? "a" : "p"), "tk", "middle");
      const up = usual.filter((q) => q.hr >= 5 && q.hr <= 20);
      s += `<path d="${path(up.map((q) => [x(q.hr), y(q.w)]))}L${x(up[up.length - 1].hr)},${ih}L${x(up[0].hr)},${ih}Z" style="fill:var(--sun-soft);opacity:.95"/>`;
      const pts = day.filter((q) => q.min / 60 >= 5 && q.min / 60 <= 20).map((q) => [x(q.min / 60 + 0.125), y(q.w)]);
      s += `<path d="${path(pts)}" class="now" pathLength="1" style="stroke:var(--vc-sun-ink)"/>`;
      const peak = day.reduce((a, b) => (b.w > a.w ? b : a));
      s += `<circle cx="${x(peak.min / 60 + 0.125)}" cy="${y(peak.w)}" r="4.5" style="fill:var(--vc-sun-ink);stroke:var(--ground);stroke-width:2"/>`;
      s += T(x(peak.min / 60 + 0.125) + 8, y(peak.w) - 8, `peak ${Math.round(peak.w)} W/m²`, "nt");
      s += T(0, -12, `W/m² · soft area: average ${MONTHS[S.monthOf(key) - 1]} day, 2023 to now`, "lb2");
      if (M && M.cur.solar_wm2 > 0) {
        const lh = Number(fmt(M.nowMs, { hour: "numeric", hourCycle: "h23" })) + Number(fmt(M.nowMs, { minute: "numeric" })) / 60;
        s += `<circle cx="${x(lh)}" cy="${y(M.cur.solar_wm2)}" r="5" style="fill:var(--note);stroke:var(--ground);stroke-width:2"/>`;
      }
      s += "</g>";
      return `<svg class="vc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
    };
  }
  function solar14Chart(S, M, label) {
    return (w) => {
      const sd = S.sd, n = 14, N = sd.date.length;
      const dates = sd.date.slice(N - n), vals = sd.kwh_m2.slice(N - n);
      const mo = Number(dates[n - 1].slice(5, 7)), yr = Number(dates[n - 1].slice(0, 4));
      const mv = S.monthVals(mo, yr), p25 = quant(mv, 0.25), p75 = quant(mv, 0.75), p50 = quant(mv, 0.5);
      const h = w < 600 ? 210 : 250, m = { t: 12, r: 12, b: 34, l: 34 };
      const iw = w - m.l - m.r, ih = h - m.t - m.b;
      const ymax = Math.max(...vals, p75) * 1.15, y = lin(0, ymax, ih, 0);
      const slot = iw / (n + 1), bw = Math.max(6, slot * 0.62), id = "vh" + ++uidN;
      let s = `<defs>${hatchDef(id)}</defs><g transform="translate(${m.l},${m.t})">`;
      niceTicks(ymax, 4).forEach((v) => { s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v.toFixed(1), "tk", "end"); });
      s += `<rect x="0" y="${y(p75)}" width="${iw}" height="${y(p25) - y(p75)}" style="fill:var(--sun-soft);opacity:.95"/>`;
      s += `<line x1="0" x2="${iw}" y1="${y(p50)}" y2="${y(p50)}" class="thin" style="stroke:var(--vc-sun-ink);stroke-dasharray:4 3"/>`;
      dates.forEach((d, i) => {
        const cx = slot * (i + 0.5), v = vals[i];
        s += `<rect x="${cx - bw / 2}" y="${y(v)}" width="${bw}" height="${ih - y(v)}" rx="2" style="fill:var(--vc-sun-ink)"/>`;
        if (i % 2 === (n - 1) % 2) s += T(cx, ih + 16, dayShort(d).replace(" ", " "), "tk", "middle");
      });
      const cx = slot * (n + 0.5);
      s += `<rect x="${cx - bw / 2}" y="0" width="${bw}" height="${ih}" rx="2" style="fill:url(#${id})"/>` + T(cx, ih + 16, dayShort(new Date(dayMs(dates[n - 1]) + D).toISOString().slice(0, 10)).replace(" ", " "), "tk", "end");
      s += T(4, y(p75) - 6, `soft band: middle half of ${MONTHS[mo - 1]} days`, "lb2");
      s += "</g>";
      return `<svg class="vc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
    };
  }

  /* ---------- calm: dry spells ---------- */
  function drySpells(M) {
    const dates = M.dates, spells = [];
    let run = 0, start = null, prev = null;
    const close = (endIso, valid) => { if (run > 0 && valid) spells.push({ len: run, start, end: endIso }); };
    for (let i = 0; i < dates.length; i++) {
      const d = dates[i], r = M.rainDay.get(d);
      const brokeGap = prev && dayMs(d) - dayMs(prev) > D * 1.5;
      if (brokeGap) { run = 0; start = null; } // spell touching a gap is not valid, drop it
      if (r < 0.01) { if (run === 0) start = d; run++; }
      else { if (run > 0) close(prev, true); run = 0; start = null; }
      prev = d;
    }
    return spells; // completed spells only (a trailing open spell is excluded)
  }
  function calmChart(M, C, label) {
    return (w) => {
      const h = w < 600 ? 160 : 170, m = { t: 8, r: 16, b: 8, l: 12 }, iw = w - m.l - m.r;
      const xmax = Math.max(C.n, C.p90 * 1.15, 12), x = lin(0, xmax, 0, iw);
      const id = "vh" + ++uidN;
      let s = `<defs>${hatchDef(id)}</defs><g transform="translate(${m.l},${m.t})">`;
      s += `<rect x="${x(C.p25)}" y="0" width="${x(C.p75) - x(C.p25)}" height="34" rx="3" style="fill:var(--rain-0)"/>`;
      s += `<line x1="${x(C.p50)}" x2="${x(C.p50)}" y1="-2" y2="36" class="thin" style="stroke:var(--rain-2);stroke-dasharray:4 3"/>`;
      s += `<rect x="0" y="9" width="${x(C.n)}" height="16" rx="8" style="fill:var(--rain-3)"/>`;
      s += x(C.n) + 8 > iw - 70 ? `<text x="${x(C.n) - 10}" y="22" text-anchor="end" style="font:600 12.5px var(--sans);fill:var(--surface)">${C.nLabel}</text>` : T(x(C.n) + 8, 22, `${C.nLabel}`, "nt");
      s += T(x(C.p25), 52, `usual spell: ${C.p25.toFixed(0)} to ${C.p75.toFixed(0)} days (median ${C.p50.toFixed(0)})`, "lb2");
      // 56-day strip
      const cells = C.strip, cw = iw / cells.length, y0 = 78;
      cells.forEach((c, i) => {
        const fill = c.v == null ? `url(#${id})` : c.v < 0.01 ? "var(--surface)" : c.v < 0.1 ? "var(--rain-0)" : c.v < 0.3 ? "var(--rain-1)" : c.v < 0.75 ? "var(--rain-2)" : c.v < 1.5 ? "var(--rain-3)" : "var(--rain-4)";
        s += `<rect x="${(i * cw + 0.5).toFixed(1)}" y="${y0}" width="${(cw - 1.5).toFixed(1)}" height="40" rx="2" style="fill:${fill};stroke:var(--rule);stroke-width:1"/>`;
      });
      s += T(0, y0 + 58, dayShort(cells[0].d), "tk") + T(iw, y0 + 58, dayShort(cells[cells.length - 1].d) + (C.stripNote || ""), "tk", "end");
      s += T(0, y0 - 8, "Last 8 weeks, one cell per day. Darker blue means more rain.", "lb2");
      s += "</g>";
      return `<svg class="vc-svg" width="${w}" height="${h + 14}" viewBox="0 0 ${w} ${h + 14}" role="img" aria-label="${label}">${s}</svg>`;
    };
  }

  /* ---------- the app ---------- */
  function render(root, data) {
    if (!document.getElementById("vc-style")) {
      const st = document.createElement("style"); st.id = "vc-style"; st.textContent = CSS; document.head.appendChild(st);
    }
    const M = buildModel(data), S = solarModel(data, M), spells = drySpells(M);
    const cur = M.cur, qp = new URLSearchParams(location.search);
    let view = ["now", "rain", "events", "solar"].includes(qp.get("view")) ? qp.get("view") : "now";
    let forced = ["rain", "temp", "solar", "calm"].includes(qp.get("state")) ? qp.get("state") : "auto";
    let charts = {}, animate = true;

    /* ----- ranking rules ----- */
    function rules() {
      return [
        { k: "rain", text: `Rain event live or ended in the last 12 h: ${M.raining ? "raining now" : M.rainLead ? `last rain ${durText(M.hoursSinceRain)} ago, ${f2(cur.rain_event_in)} in in the event` : `last rain ${durText(M.hoursSinceRain)} ago`}`, met: M.rainLead },
        { k: "temp", text: `Temperature at least 10° from the ${M.monthName} median: ${M.tdep >= 0 ? "+" : "−"}${f1(Math.abs(M.tdep))}° now`, met: M.tempLead },
        { k: "solar", text: `Sunny daytime (at least 400 W/m²): ${Math.round(cur.solar_wm2)} W/m² now`, met: M.solarLead },
        { k: "calm", text: "Otherwise: nothing unusual, days since last rain", met: !M.rainLead && !M.tempLead && !M.solarLead },
      ];
    }
    const autoState = () => (M.rainLead ? "rain" : M.tempLead ? "temp" : M.solarLead ? "solar" : "calm");

    /* ----- lead builders ----- */
    function leadRain() {
      const real = M.liveEv && M.rainLead;
      const E = real ? M.liveEv : M.pastEv;
      const B = similarBand(M, E);
      const word = B.pct < 0.25 ? "lighter than most events this long" : B.pct <= 0.75 ? "a typical event for its length" : B.pct <= 0.9 ? "heavier than most events this long" : "among the heaviest for its length";
      const ongoing = real && M.raining;
      const verb = ongoing ? "so far" : real ? "in total" : "in total";
      const h1 = `${f2(E.total)} in ${verb}, ${word}`;
      const since = real && !ongoing ? `Rain stopped ${durText(M.hoursSinceRain)} ago, at ${tShort(M.lastRain)}.` : ongoing ? "Still raining." : `Event of ${dLong(E.start)}.`;
      const sub = `${since} Past events of similar length (${B.n} with recorded curves) ended at a median ${f2(B.medFinal)} in. The bold line is this event, the soft band is the usual spread.`;
      const facts = [
        [durText(E.durH), "duration"],
        [(E.peak != null ? f2(E.peak) : "–") + " in/hr", E.tail ? "peak, archived part" : "peak rate"],
        [`${Math.round(B.pct * 100)}th`, "percentile of similar events"],
      ];
      const foot = E.tail
        ? `Source: station archive to ${dShort(M.archEnd)}, ${tShort(M.archEnd)} (${f1(M.lagH)} h behind the live reading), then the live event total. The ${f2(E.archTotal)} in already archived is part of the ${f2(E.total)} in; the shape of the curve after the archive ends is not known, so it is hatched, not drawn.`
        : `Source: station archive, event catalog (${E.e.data_point_count} readings).`;
      return { h1, sub, facts, foot, chart: rainEventChart(M, E, B, h1), tag: "rain" };
    }
    function leadTemp() {
      const dep = M.tdep, abs = f1(Math.abs(dep));
      const out = cur.temp_f < M.tq1 ? "below" : cur.temp_f > M.tq3 ? "above" : "inside";
      const h1 = out === "inside" ? `${f1(cur.temp_f)}° is inside the usual ${M.monthName} range` : `${f1(cur.temp_f)}°, ${abs}° ${dep < 0 ? "below" : "above"} the ${M.monthName} median`;
      const sub = `${out === "inside" ? "Within" : "Outside"} the middle half of ${M.monthName} readings (${f1(M.tq1)}° to ${f1(M.tq3)}°). Feels like ${f1(cur.feels_like_f)}°, dew point ${f1(cur.dew_point_f)}°.`;
      const facts = [[f1(cur.temp_f) + "°", "now"], [(dep > 0 ? "+" : "−") + abs + "°", `vs ${M.monthName} median`], [f1(M.tmed) + "°", "median"]];
      const foot = `Source: station archive, last 7 days of 15-minute readings to ${dShort(M.archEnd)}, ${tShort(M.archEnd)}; the final point is the live reading, ${f1(M.lagH)} h later.`;
      return { h1, sub, facts, foot, chart: tempChart(M, data, h1), tag: "temp" };
    }
    function leadSolar() {
      const best = S.full.slice(-7).reduce((a, k) => (S.days[k].total > S.days[a].total ? k : a), S.full[S.full.length - 1]);
      const mo = S.monthOf(best), U = S.usualDay(mo), tot = S.days[best].total, ratio = tot / U;
      const h1 = `${dayDow(best)} ${dayShort(best)} was the sunniest of the week: ${f2(tot)} kWh/m², ${ratio >= 1 ? f1(ratio) + "× " : Math.round(ratio * 100) + "% of "}a usual ${MONTHS[mo - 1]} day`;
      const sub = `Irradiance through the day in bold, the average ${MONTHS[mo - 1]} day behind it. ${cur.solar_wm2 >= 400 ? `Right now ${Math.round(cur.solar_wm2)} W/m².` : "It is not sunny right now, so this replays the best recent day."}`;
      const facts = [[f2(tot) + " kWh/m²", "that day"], [f2(U) + " kWh/m²", `usual ${MONTHS[mo - 1]} day`], [Math.round(Math.max(...S.days[best].map((q) => q.w))) + " W/m²", "peak"]];
      const foot = `Source: station archive, 15-minute solar energy (last 7 days); usual = hourly average for ${MONTHS[mo - 1]} across the whole archive.`;
      return { h1, sub, facts, foot, chart: solarDayChart(S, best, cur.solar_wm2 >= 400 ? M : null, h1), tag: "solar" };
    }
    function leadCalm() {
      const completed = spells.map((s) => s.len);
      const real = M.hoursSinceRain >= 24;
      let nVal, endIso, label, replayNote = "";
      if (real) { nVal = M.hoursSinceRain / 24; endIso = null; }
      else {
        const sp = spells.filter((s) => s.len >= 5).slice(-1)[0] || spells[spells.length - 1];
        nVal = sp.len; endIso = sp.end;
        replayNote = ` Replaying the last long dry spell (${dayShort(sp.start)} to ${dayShort(sp.end)}), since it rained ${durText(M.hoursSinceRain)} ago.`;
      }
      const C = { n: nVal, p25: quant(completed, 0.25), p50: quant(completed, 0.5), p75: quant(completed, 0.75), p90: quant(completed, 0.9) };
      C.nLabel = real ? `${f1(nVal)} days` : `${nVal} days`;
      const pct = Math.round((completed.filter((v) => v <= nVal).length / completed.length) * 100);
      const h1 = real ? `${f1(nVal)} days since the last rain, longer than ${pct}% of dry spells here` : `${nVal} dry days in a row, longer than ${pct}% of dry spells here`;
      const sub = `Nothing unusual outside right now: ${f1(cur.temp_f)}°, ${cur.humidity}% humidity. A dry spell is any run of days under 0.01 in.${replayNote}`;
      const facts = [[C.nLabel, real ? "since last rain" : "dry spell"], [C.p50.toFixed(0) + " days", "median dry spell"], [Math.max(...completed) + " days", "longest on record"]];
      // 56-day strip ending at endIso (or at archive end)
      const lastKey = endIso || M.dates[M.dates.length - 1];
      const cells = [];
      const extra = real ? 1 : 0;
      for (let i = 55; i >= 0; i--) { const t = dayMs(lastKey) - (i - extra) * D; const d = new Date(t).toISOString().slice(0, 10); cells.push({ d, v: M.rainDay.has(d) ? M.rainDay.get(d) : null }); }
      C.strip = cells; C.stripNote = real ? " (live day, not yet archived)" : "";
      const foot = `Source: station archive, daily rain totals${M.gapSpans.length ? `; ${M.gapSpans.length} archive gaps exist, and spells touching a gap are left out of the comparison` : ""}.`;
      return { h1, sub, facts, foot, chart: calmChart(M, C, h1), tag: "calm" };
    }
    const LEADS = { rain: leadRain, temp: leadTemp, solar: leadSolar, calm: leadCalm };

    /* ----- shared blocks ----- */
    function readout() {
      const dir = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"][Math.round(cur.wind_dir / 22.5) % 16];
      const ts = cur.temp_trend_symbol, tIcon = ts === "↑" ? "up" : ts === "↓" ? "down" : "flat";
      const pt = cur.pressure_trend_3h_inhg, pIcon = pt > 0.02 ? "up" : pt < -0.02 ? "down" : "flat";
      const pWord = Math.abs(pt) < 0.02 ? "steady" : pt > 0 ? (pt > 0.06 ? "rising" : "rising slowly") : pt < -0.06 ? "falling" : "falling slowly";
      return `<dl class="vc-read">
        <div><dt>Temperature</dt><dd>${f1(cur.temp_f)}°F ${icon(tIcon)}</dd></div>
        <div><dt>Feels like</dt><dd>${f1(cur.feels_like_f)}°F</dd></div>
        <div><dt>Wind</dt><dd>${f1(cur.wind_avg10m_mph)} mph <small>${dir}, gust ${f1(cur.wind_gust_mph)}</small></dd></div>
        <div><dt>Humidity</dt><dd>${cur.humidity}%<small>dew pt ${f1(cur.dew_point_f)}°</small></dd></div>
        <div><dt>Pressure</dt><dd>${cur.pressure_inhg.toFixed(2)} inHg ${icon(pIcon)}<small>${pWord}</small></dd></div>
        <div><dt>Rain today</dt><dd>${f2(cur.rain_today_in)} in<small>${f2(cur.rain_week_in)} in this week</small></dd></div>
      </dl>`;
    }
    function context(title, sub) {
      const rows = data.rain.rolling_context.filter((r) => [7, 30, 90, 365].includes(r.window_days));
      const html = rows.map((r) => {
        const miss = M.missing(r.period_start.slice(0, 10), r.period_end.slice(0, 10)).length;
        const max = Math.max(r.total, r.normal) * 1.2, lab = r.window_days + " days";
        if (miss > 0) {
          return `<div class="vc-ctx-row"><div class="vc-ctx-l">${lab}</div><div class="vc-ctx-track gap-hatch" style="background-color:transparent"><i class="vc-ctx-bad"></i></div>
          <div class="vc-ctx-v"><b>Not comparable.</b> ${miss} day${miss > 1 ? "s" : ""} missing from the archive in this window, so ${f2(r.total)} in is a floor, not a total.</div></div>`;
        }
        const pct = Math.round(r.anomaly_pct);
        return `<div class="vc-ctx-row"><div class="vc-ctx-l">${lab}</div><div class="vc-ctx-track"><i class="vc-ctx-fill" style="width:${(r.total / max) * 100}%"></i><i class="vc-ctx-tick" style="left:calc(${(r.normal / max) * 100}% - 1px)"></i></div>
          <div class="vc-ctx-v"><b>${f2(r.total)} in</b>, ${Math.abs(pct)}% ${pct < 0 ? "below" : "above"} the usual ${f2(r.normal)} in</div></div>`;
      }).join("");
      return `<section class="vc-sec"><h2>${title}</h2><p class="vc-sub">${sub}</p><div class="vc-ctx">${html}</div>
        <p class="vc-foot">Bar is the total, tick is usual for a window this long. Archive through ${dShort(M.archEnd)}; today's rain is not included.</p></section>`;
    }

    /* ----- views ----- */
    function viewNow() {
      const st = forced === "auto" ? autoState() : forced;
      const L = LEADS[st](), rs = rules();
      const whyLead = forced === "auto" ? st : st;
      const why = `<details class="vc-why"><summary>Why this leads${forced !== "auto" ? " (forced by the prototype picker)" : ""}</summary><ol>${rs.map((r) => `<li data-met="${r.met ? 1 : 0}" data-lead="${r.k === whyLead ? 1 : 0}"><i></i><span>${r.text}${r.met ? ", met" : ""}</span></li>`).join("")}</ol></details>`;
      charts.lead = L.chart;
      return `<section class="vc-lead"><h1 class="vc-h">${L.h1}</h1><p class="vc-sub">${L.sub}</p>
        <dl class="vc-facts">${L.facts.map(([v, k]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
        <div class="vc-chart" data-c="lead"></div><p class="vc-foot">${L.foot}</p>${why}</section>
        <div class="vc-grid"><section class="vc-sec"><h2>Right now</h2><p class="vc-sub">Live reading, ${tShort(M.nowMs)}.</p>${readout()}</section>
        ${context("Rain in context", "Rolling totals against what is usual for each window length.")}</div>`;
    }
    function viewRain() {
      const yoy = data.rain.yoy_cumulative, yr = String(new Date(M.nowMs).getUTCFullYear());
      const doyOf = (iso) => (Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10)) - Date.UTC(+iso.slice(0, 4), 0, 0)) / D;
      const cumAt = (y, doy) => { const o = yoy[y]; let v = 0; for (let i = 0; i < o.day_of_year.length; i++) { if (o.day_of_year[i] <= doy) v = o.cum_in[i]; else break; } return v; };
      const lastDoy = doyOf(M.dates[M.dates.length - 1]);
      const c26 = cumAt(yr, lastDoy), c24 = cumAt("2024", lastDoy), c25 = cumAt("2025", lastDoy);
      const gap26 = M.gapSpans.filter((g) => g.from.startsWith(yr)), gap25 = M.gapSpans.filter((g) => g.from.startsWith("2025"));
      const lo = Math.min(c24, c25), hi = Math.max(c24, c25);
      const rel = c26 < lo ? "drier than both" : c26 > hi ? "wetter than both" : "inside the range of";
      const h1 = `${yr} has ${f1(c26)} in of rain so far, ${rel} 2024 and 2025 at this date`;
      const sub = `By ${dShort(dayMs(M.dates[M.dates.length - 1]))}: 2024 had ${f1(c24)} in, 2025 had at least ${f1(c25)} in. 2023 is left out because the archive starts in February.`;
      charts.rain = (w) => {
        const h = w < 600 ? 270 : 330, m = { t: 12, r: 16, b: 58, l: 34 }, iw = w - m.l - m.r, ih = h - m.t - m.b;
        const ymax = Math.max(...yoy["2024"].cum_in, ...yoy["2025"].cum_in, c26) * 1.08;
        const x = lin(1, 366, 0, iw), y = lin(0, ymax, ih, 0), id = "vh" + ++uidN;
        let s = `<defs>${hatchDef(id)}</defs><g transform="translate(${m.l},${m.t})">`;
        niceTicks(ymax, 4).forEach((v) => { s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v, "tk", "end"); });
        const days = []; for (let d = 1; d <= 365; d += 3) days.push(d); days.push(365);
        const topB = days.map((d) => [x(d), y(Math.max(cumAt("2024", d), cumAt("2025", d)))]), botB = days.map((d) => [x(d), y(Math.min(cumAt("2024", d), cumAt("2025", d)))]).reverse();
        s += `<path d="${path(topB)}L${path(botB).slice(1)}Z" style="fill:var(--rain-0);opacity:.95"/>`;
        const cm = new Date(Date.UTC(2025, 0, 1)); // month ticks
        for (let mo = 0; mo < 12; mo++) { const d = (Date.UTC(2025, mo, 1) - Date.UTC(2025, 0, 0)) / D; s += T(x(d) + 2, ih + 16, MONTHS[mo].slice(0, 3), "tk"); s += `<line class="ax" x1="${x(d)}" x2="${x(d)}" y1="${ih}" y2="${ih + 4}"/>`; }
        const pts = []; for (let d = 1; d <= lastDoy; d += 1) pts.push([x(d), y(cumAt(yr, d))]);
        s += `<path d="${path(pts)}" class="now" pathLength="1" style="stroke:var(--rain-3)"/>`;
        s += `<circle cx="${x(lastDoy)}" cy="${y(c26)}" r="5" style="fill:var(--rain-3);stroke:var(--ground);stroke-width:2"/>`;
        s += T(x(lastDoy) - 8, y(c26) - 12, `${yr}: ${f1(c26)} in`, "nt", "end");
        s += T(4, 14, "Soft band: the range between 2024 and 2025", "lb2");
        s += T(x(300), y(Math.max(c24, c25)) - 10, `earlier years: ${f1(lo)} to ${f1(hi)}+ in`, "lb2", "end");
        // gap strip
        s += `<text x="0" y="${ih + 50}" class="tk">Missing archive days</text>`;
        const gs = gap25.map((g) => ({ g, o: 0 })).concat(gap26.map((g) => ({ g, o: 0 })));
        M.gapSpans.filter((g) => g.from.startsWith(yr) || g.from.startsWith("2025")).forEach((g) => {
          const a = doyOf(g.from), b = doyOf(g.to) + 1;
          s += `<rect x="${x(a)}" y="${ih + 24}" width="${Math.max(3, x(b) - x(a))}" height="9" style="fill:url(#${id});stroke:var(--rule-strong);stroke-width:.75"/>`;
        });
        s += T(iw, ih + 50, "(2025 and 2026 gaps)", "tk", "end");
        s += "</g>";
        return `<svg class="vc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${h1}">${s}</svg>`;
      };
      const miss25 = sum(gap25.map((g) => M.missing(g.from, g.to).length)), miss26 = sum(gap26.map((g) => M.missing(g.from, g.to).length));
      const foot = `Source: station archive, daily rain, cumulative from 1 January. 2025 is missing ${miss25} days and ${yr} is missing ${miss26}, so both lines are floors; the verdict holds only because ${yr} is still below them. The live rain-year counter reads ${f2(cur.rain_year_in)} in against ${f2(c26)} in from the archive.`;
      return `<section class="vc-lead"><h1 class="vc-h">${h1}</h1><p class="vc-sub">${sub}</p><div class="vc-chart" data-c="rain"></div><p class="vc-foot">${foot}</p></section>
        ${context("Recent rain against usual", "The last 7, 30, 90 and 365 days.")}`;
    }
    function viewEvents() {
      const live = M.liveEv, all = M.events;
      const rowsData = all.map((e) => ({ e, start: Date.parse(e.start_time), end: Date.parse(e.end_time), total: e === (live && live.e) ? live.total : e.total_rainfall, live: e === (live && live.e), dur: (e === (live && live.e) ? live.durH * 60 : e.duration_minutes) }));
      const latest = rowsData[rowsData.length - 1];
      const yr = new Date(latest.start).getUTCFullYear();
      const thisYear = rowsData.filter((r) => new Date(r.start).getUTCFullYear() === yr && r.total > 0);
      const rank = thisYear.filter((r) => r.total > latest.total).length + 1;
      const h1 = `${f2(latest.total)} in makes the ${dShort(latest.start)} event the ${ordinal(rank)} wettest of ${thisYear.length} this year`;
      const prevBigger = [...rowsData].reverse().find((r) => r !== latest && r.total >= latest.total);
      const sub = `${latest.live ? "Includes rain since the archive ended. " : ""}${prevBigger ? `The last event at least this wet was ${dShort(prevBigger.start)} (${f2(prevBigger.total)} in).` : "No earlier event in the record was this wet."} Dots are past events by duration and total; the soft band is the usual spread for each duration.`;
      charts.events = (w) => {
        const pool = rowsData.filter((r) => !r.live && r.total > 0 && r.dur > 0 && !r.e.ongoing);
        const h = w < 600 ? 260 : 320, m = { t: 12, r: 16, b: 34, l: 38 }, iw = w - m.l - m.r, ih = h - m.t - m.b;
        const xmin = Math.log10(0.5), xmax = Math.log10(Math.max(...pool.map((r) => r.dur / 60), latest.dur / 60) * 1.1);
        const ymaxV = Math.max(...pool.map((r) => r.total), latest.total) * 1.05, sq = Math.sqrt;
        const x = (hrs) => lin(xmin, xmax, 0, iw)(Math.log10(hrs)), y = (v) => lin(0, sq(ymaxV), ih, 0)(sq(v));
        let s = `<g transform="translate(${m.l},${m.t})">`;
        [0.1, 0.5, 1, 2].filter((v) => v < ymaxV).forEach((v) => { s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v, "tk", "end"); });
        [1, 3, 6, 12, 24, 48, 96].filter((v) => Math.log10(v) <= xmax).forEach((v) => { s += `<line class="ax" x1="${x(v)}" x2="${x(v)}" y1="${ih}" y2="${ih + 4}"/>` + T(x(v), ih + 17, v >= 24 ? v / 24 + " d" : v + " h", "tk", "middle"); });
        const sorted = [...pool].sort((a, b) => a.dur - b.dur), nb = 6, size = Math.ceil(sorted.length / nb), bins = [];
        for (let i = 0; i < sorted.length; i += size) { const g = sorted.slice(i, i + size); if (g.length >= 4) bins.push({ x: x(quant(g.map((r) => r.dur / 60), 0.5)), lo: quant(g.map((r) => r.total), 0.25), hi: quant(g.map((r) => r.total), 0.75), med: quant(g.map((r) => r.total), 0.5) }); }
        s += `<path d="${path(bins.map((b) => [b.x, y(b.hi)]))}L${path(bins.map((b) => [b.x, y(b.lo)]).reverse()).slice(1)}Z" style="fill:var(--rain-0);opacity:.95"/>`;
        s += `<path d="${path(bins.map((b) => [b.x, y(b.med)]))}" class="thin" style="stroke:var(--rain-2);stroke-dasharray:4 3"/>`;
        pool.forEach((r) => (s += `<circle cx="${x(r.dur / 60)}" cy="${y(r.total)}" r="2.6" style="fill:var(--rain-2);opacity:.45"/>`));
        s += `<circle cx="${x(latest.dur / 60)}" cy="${y(latest.total)}" r="6.5" style="fill:var(--rain-3);stroke:var(--ground);stroke-width:2.5"/>`;
        s += T(x(latest.dur / 60) - 12, y(latest.total) - 10, `${dShort(latest.start)}: ${f2(latest.total)} in`, "nt", "end");
        s += T(4, 14, "in of rain (square-root scale), by event duration (log scale)", "lb2");
        s += "</g>";
        return `<svg class="vc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${h1}">${s}</svg>`;
      };
      const maxT = Math.max(...rowsData.slice(-8).map((r) => r.total));
      const list = rowsData.slice(-8).reverse().map((r) => {
        const e = r.e, gaps = e.flags && e.flags.has_gaps;
        const when = `${dShort(r.start)}${dShort(r.end) !== dShort(r.start) ? " to " + dShort(r.end) : ""}`;
        return `<li class="vc-row"><div><b>${when}</b>${r.live ? '<span class="vc-tag">live total</span>' : ""}${gaps ? '<span class="vc-tag gap-hatch">gaps in data</span>' : ""}</div><div><b>${f2(r.total)} in</b></div>
          <small>${durText(r.dur / 60)}, peak ${e.max_hourly_rate != null ? f2(e.max_hourly_rate) : "–"} in/hr${r.live ? " (archived part)" : ""}</small><small></small><div class="vc-bar"><i style="width:${(r.total / maxT) * 100}%"></i></div></li>`;
      }).join("");
      const foot = `Source: station archive, Event Catalog (${all.length} events since ${dShort(Date.parse(all[0].start_time))} ${new Date(Date.parse(all[0].start_time)).getUTCFullYear()}). The newest event is still open in the catalog (${f2(live ? live.archTotal : 0)} in archived); its total is the live reading.`;
      return `<section class="vc-lead"><h1 class="vc-h">${h1}</h1><p class="vc-sub">${sub}</p><div class="vc-chart" data-c="events"></div><p class="vc-foot">${foot}</p></section>
        <section class="vc-sec"><h2>Recent events</h2><p class="vc-sub">The last eight, newest first.</p><ul class="vc-rows">${list}</ul></section>`;
    }
    function viewSolar() {
      const key = S.full[S.full.length - 1], mo = S.monthOf(key), U = S.usualDay(mo), tot = S.days[key].total, ratio = tot / U;
      const h1 = `${dayDow(key)} ${dayShort(key)} brought ${f2(tot)} kWh/m², ${ratio >= 1 ? f1(ratio) + "× " : Math.round(ratio * 100) + "% of "}a usual ${MONTHS[mo - 1]} day`;
      const sd = S.sd, N = sd.date.length, last14 = sd.kwh_m2.slice(N - 14), mv = S.monthVals(mo, +key.slice(0, 4));
      const mean14 = sum(last14) / 14, medM = quant(mv, 0.5);
      const sub = `Latest full day in the archive. In bold, the day's irradiance; behind it, the average ${MONTHS[mo - 1]} day. ${cur.solar_wm2 > 0 ? "" : "It is dark now."} The last 14 days averaged ${f2(mean14)} kWh/m² a day against a ${MONTHS[mo - 1]} median of ${f2(medM)}.`;
      charts.solar = solarDayChart(S, key, null, h1);
      charts.solar14 = solar14Chart(S, M, "Daily solar energy, last 14 days");
      const foot = `Source: station archive, 15-minute solar energy, to ${dShort(M.archEnd)}. ${dShort(M.nowMs)} has no archived energy yet.`;
      return `<section class="vc-lead"><h1 class="vc-h">${h1}</h1><p class="vc-sub">${sub}</p><div class="vc-chart" data-c="solar"></div><p class="vc-foot">${foot}</p></section>
        <section class="vc-sec"><h2>Last 14 days of energy</h2><p class="vc-sub">Daily kWh/m², bars against the middle half of ${MONTHS[mo - 1]} days.</p><div class="vc-chart" data-c="solar14"></div>
        <p class="vc-foot">Source: station archive, daily solar energy. The hatched bar is today, not yet archived.</p></section>`;
    }

    /* ----- shell ----- */
    root.innerHTML = `<div class="vc"><header class="vc-top"><div class="vc-id"><b>lookout post</b><span id="vcstamp"></span></div>
      <label class="vc-proto" id="vcproto">Prototype state<select aria-label="Prototype state"><option value="auto">Auto (by rules)</option><option value="rain">Rain</option><option value="temp">Temperature</option><option value="solar">Solar</option><option value="calm">Calm</option></select></label></header>
      <main class="vc-main" id="vcmain"></main>
      <nav class="vc-nav" aria-label="Views">${[["now", "Now"], ["rain", "Rain"], ["events", "Events"], ["solar", "Solar"]].map(([k, l]) => `<button data-v="${k}">${icon(k)}<span>${l}</span></button>`).join("")}</nav></div>`;
    const el = root.querySelector(".vc"), mainEl = root.querySelector("#vcmain"), sel = root.querySelector("select");
    sel.value = forced;
    root.querySelector("#vcstamp").textContent = `${dLong(M.nowMs)}, ${tShort(M.nowMs)}. Archive ends ${tShort(M.archEnd)} yesterday.`;
    sel.onchange = () => { forced = sel.value; animate = true; show(); };
    root.querySelectorAll(".vc-nav button").forEach((b) => (b.onclick = () => { view = b.dataset.v; animate = true; show(); window.scrollTo(0, 0); }));

    let lastW = 0;
    function draw() {
      mainEl.querySelectorAll(".vc-chart").forEach((c) => {
        const fn = charts[c.dataset.c]; if (!fn) return;
        const w = Math.floor(c.clientWidth); if (!w) return;
        c.innerHTML = fn(w);
      });
      el.classList.toggle("vc-anim", animate); lastW = mainEl.clientWidth;
      animate = false;
    }
    function show() {
      charts = {};
      mainEl.innerHTML = { now: viewNow, rain: viewRain, events: viewEvents, solar: viewSolar }[view]();
      root.querySelectorAll(".vc-nav button").forEach((b) => (b.getAttribute("data-v") === view ? b.setAttribute("aria-current", "page") : b.removeAttribute("aria-current")));
      root.querySelector("#vcproto").hidden = view !== "now";
      draw();
    }
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (mainEl.clientWidth !== lastW) draw(); }, 120); });
    show();
  }

  window.LOOKOUT_VARIANT = { name: "What's happening", render };
})();
