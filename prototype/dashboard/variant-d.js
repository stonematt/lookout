// PROTOTYPE variant D, "Converged": Almanac structure (B) led by the rules-based hero (C),
// with a temperature-across-periods chart and A's ten-day range list.
// Every headline is computed from data.json. Archive gaps render as .gap-hatch, never as computed values.
(function () {
  "use strict";

  /* ---------- styles ---------- */
  const CSS = `
.vd{max-width:1240px;margin:0 auto;padding:0 16px 96px;font-family:var(--sans);color:var(--ink);font-variant-numeric:normal}
.vd :is(table,.facts,.vd-facts dd,.vd-nowstats,.alm,.cell,.cal,.dows,.vd-wrow,.vd-whead,.readout,.key,.temp,.vd-stamp,.vd-ribbon){font-variant-numeric:tabular-nums}
.vd .js-plotly-plot :is(.xtick,.ytick) text{font-variant-numeric:tabular-nums}
.vd button{font:inherit;color:inherit;cursor:pointer}
.vd h1,.vd h2,.vd h3,.vd p{margin:0}
.vd-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:16px 0 8px;flex-wrap:wrap}
.vd-station{font-family:var(--serif);font-size:var(--step-1);font-weight:600;letter-spacing:-.01em}
.vd-stamp{font-size:var(--step--1);color:var(--ink-2);display:flex;align-items:center;gap:8px}
.vd-dot{width:8px;height:8px;border-radius:50%;background:var(--rain-2);display:inline-block}
.vd-tabs{position:sticky;top:0;z-index:20;display:flex;gap:4px;overflow-x:auto;background:var(--ground);border-bottom:1px solid var(--rule);margin:0 -16px 24px;padding:0 16px;scrollbar-width:none}
.vd-tabs::-webkit-scrollbar{display:none}
.vd-tab{flex:0 0 auto;background:none;border:0;padding:0 14px;min-height:46px;font-size:var(--step-0);font-weight:500;color:var(--ink-2);box-shadow:inset 0 -3px 0 transparent;border-radius:0}
.vd-tab:hover{color:var(--ink)}
.vd-tab[aria-selected=true]{color:var(--ink);font-weight:700;box-shadow:inset 0 -3px 0 var(--rain-3)}
.vd-panel[hidden]{display:none}
.vd-grid{display:grid;gap:40px 48px;grid-template-columns:minmax(0,1fr)}
@media (min-width:1024px){
 .vd{padding:0 32px 96px}.vd-tabs{margin:0 -32px 32px;padding:0 32px}
 .vd-grid.two{grid-template-columns:minmax(0,5fr) minmax(0,7fr)}
 .vd-grid.two-r{grid-template-columns:minmax(0,7fr) minmax(0,5fr)}
 .vd-grid.three{grid-template-columns:repeat(3,minmax(0,1fr));gap:32px}
 .vd-span{grid-column:1/-1}
}
.vd h2{font-family:var(--serif);font-size:var(--step-2);line-height:1.15;font-weight:600;letter-spacing:-.015em;text-wrap:balance}
.vd h3{font-family:var(--serif);font-size:var(--step-1);line-height:1.2;font-weight:600;text-wrap:balance}
.vd .sub{font-size:var(--step--1);color:var(--ink-2);margin-top:6px;line-height:1.45;max-width:68ch}
.vd .src{font-size:.75rem;color:var(--ink-2);margin-top:10px;line-height:1.45;max-width:68ch}
.vd .block{padding-top:4px}
.vd .now{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap}
.vd .temp{font-size:clamp(3.5rem,14vw,var(--step-4));font-weight:600;line-height:.9;letter-spacing:-.04em}
.vd .temp sup{font-size:.4em;vertical-align:top;position:relative;top:.15em;font-weight:500;margin-left:2px}
.vd .lead{font-family:var(--serif);font-size:var(--step-1);line-height:1.3;margin-top:12px;text-wrap:balance;max-width:34ch}
.vd .facts{display:grid;grid-template-columns:repeat(3,auto);gap:10px 20px;margin-top:16px;justify-content:start}
.vd .facts div{font-size:var(--step--1);color:var(--ink-2)}
.vd .facts b{display:block;font-size:var(--step-0);color:var(--ink);font-weight:600}
.vd .strip-wrap{margin-top:18px}
.vd .strip-wrap .claim{font-size:var(--step-0);font-weight:600;line-height:1.3}
.vd .strip-wrap .claim span{font-weight:400;color:var(--ink-2)}
.vd .strip{width:100%;display:block;margin-top:6px;overflow:visible}
.vd .strip text{font-family:var(--sans);font-size:11px;fill:var(--ink-2)}
.vd .strip .lab{font-weight:600;fill:var(--note);font-size:12px}
.vd .seg{display:inline-flex;border:1px solid var(--rule-strong);border-radius:8px;overflow:hidden;margin-top:12px}
.vd .seg button{background:var(--surface);border:0;padding:0 14px;min-height:40px;font-size:var(--step--1);font-weight:500;color:var(--ink-2)}
.vd .seg button+button{border-left:1px solid var(--rule-strong)}
.vd .seg button[aria-pressed=true]{background:var(--ink);color:var(--ground);font-weight:600}
.vd .alm{display:grid;gap:3px;margin-top:14px;align-items:stretch}
.vd .alm .h{font-size:.6875rem;color:var(--ink-2);text-align:center;padding-bottom:2px}
.vd .alm .y{font-size:var(--step--1);color:var(--ink-2);display:flex;align-items:center;font-weight:600}
.vd .cell{border:0;border-radius:4px;min-height:44px;padding:0;font-size:.6875rem;font-weight:500;display:flex;align-items:center;justify-content:center;position:relative;line-height:1}
.vd .cell.none{background:none;cursor:default;border:1px dashed var(--rule)}
.vd .cell.gap{color:var(--ink-2);border:1px solid var(--rule)}
.vd .cell.cur{outline:2px solid var(--note);outline-offset:1px;z-index:1}
.vd .cell:focus-visible{outline:2px solid var(--rain-2);outline-offset:2px}
.vd .cell.sel{box-shadow:0 0 0 2px var(--ink)}
.vd .readout{min-height:2.6em;font-size:var(--step--1);color:var(--ink);margin-top:10px;line-height:1.4}
.vd .key{display:flex;align-items:center;gap:8px;font-size:.75rem;color:var(--ink-2);margin-top:8px;flex-wrap:wrap}
.vd .key i{display:inline-block;width:72px;height:8px;border-radius:2px}
.vd .key .hatch{width:16px;height:12px;border-radius:2px;border:1px solid var(--rule)}
.vd table{width:100%;border-collapse:collapse;margin-top:12px;font-size:var(--step--1)}
.vd th{font-weight:600;color:var(--ink-2);text-align:right;padding:6px 6px;border-bottom:1px solid var(--rule-strong);font-size:.75rem;line-height:1.2}
.vd th:first-child,.vd td:first-child{text-align:left;padding-left:0}
.vd td{padding:9px 6px;text-align:right;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
.vd td.gap-hatch{color:var(--ink-2);text-align:center}
.vd td.d{color:var(--ink-2)}
.vd td b{font-weight:600}
.vd caption{caption-side:top;text-align:left;font-weight:600;font-size:var(--step-0)}
.vd .ev tr[aria-current=true] td{background:var(--rain-0)}
.vd .ev tr.pin td{border-bottom:2px solid var(--rule-strong)}
.vd .ev td{white-space:nowrap}.vd .ev td,.vd .ev th{padding-left:3px;padding-right:3px}.vd .ev .q{padding:2px 6px}.vd .ev .live{font-size:0}.vd .ev .live::before{width:9px;height:9px}.vd .ev td:last-child{white-space:nowrap}
@media (max-width:480px){.vd .ev{font-size:.8125rem}.vd .ev .q{padding:1px 5px;font-size:.6875rem}.vd .ev .rowbtn{gap:4px}}
.vd .rowbtn{background:none;border:0;padding:8px 0;text-align:left;min-height:44px;display:flex;align-items:center;gap:6px;font-weight:600}
.vd .live{color:var(--note);font-weight:700;font-size:.6875rem;letter-spacing:.04em;text-transform:uppercase;display:inline-flex;align-items:center;gap:4px}
.vd .live::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--note)}
.vd .q{display:inline-block;white-space:nowrap;padding:2px 8px;border-radius:999px;border:1px solid var(--rule-strong);font-size:.6875rem;font-weight:600}
.vd .q.low{border-style:dashed;color:var(--ink-2)}
.vd .more{margin-top:12px;background:var(--surface);border:1px solid var(--rule-strong);border-radius:8px;min-height:44px;padding:0 16px;font-weight:600}
.vd .chart{width:100%;margin-top:8px}
.vd .cal{display:grid;grid-auto-flow:column;grid-template-rows:repeat(7,1fr);gap:3px;margin-top:6px}
.vd .cal .c{aspect-ratio:1;min-width:0;min-height:0;border:0;border-radius:3px;padding:0}
.vd .cal .c.cur{outline:2px solid var(--note);outline-offset:1px}
.vd .calm{display:grid;gap:3px;font-size:.6875rem;color:var(--ink-2);margin-top:14px;height:1em}
.vd .calm span{white-space:nowrap}
.vd .calwrap{display:grid;grid-template-columns:28px minmax(0,1fr);gap:6px}
.vd .dows{display:grid;grid-template-rows:repeat(7,1fr);gap:3px;font-size:.6875rem;color:var(--ink-2);margin-top:6px}
.vd .dows span{display:flex;align-items:center}
.vd .note-live{border-top:1px solid var(--rule);padding-top:10px;margin-top:16px;font-size:var(--step--1);color:var(--ink-2);line-height:1.45;max-width:68ch}
.vd .sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}

.vd .vd-nowline{display:flex;align-items:center;gap:6px 18px;flex-wrap:wrap;margin:0 0 20px}
.vd .vd-nowline .temp{font-size:clamp(2.75rem,12vw,3.5rem)}
.vd .vd-nowline .lead{margin:0;font-size:var(--step-0);max-width:44ch;flex:1 1 220px}
.vd .vd-grid.vd-next,.vd .vd-next{margin-top:48px}
.vd .vd-hero-h{font-family:var(--serif);font-weight:600;font-size:var(--step-2);line-height:1.12;letter-spacing:-.015em;text-wrap:balance;max-width:26ch}
@media (min-width:1024px){.vd .vd-hero-h{font-size:var(--step-3)}.vd .vd-grid.half{grid-template-columns:repeat(2,minmax(0,1fr));gap:32px}}
.vd .vd-grid>*{min-width:0}
.vd .vd-facts{display:grid;grid-template-columns:repeat(3,minmax(0,auto));justify-content:space-between;gap:8px 12px;margin:14px 0 6px;padding:8px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
@media (min-width:600px){.vd .vd-facts dd{font-size:var(--step-1)}}
.vd .vd-facts div{display:flex;flex-direction:column;gap:0}
.vd .vd-facts dt{font-size:var(--step--1);color:var(--ink-2);order:2}
.vd .vd-facts dd{margin:0;font-weight:600;font-size:var(--step-0);order:1;white-space:nowrap}
.vd .vd-chart{width:100%;min-height:200px;margin-top:8px}
.vd .vd-svg{display:block;max-width:100%;overflow:visible}
.vd .vd-svg .ax{stroke:var(--rule);stroke-width:1;fill:none}
.vd .vd-svg .tk{font:11px var(--sans);fill:var(--ink-2);font-variant-numeric:tabular-nums}
.vd .vd-svg :is(.lb2,.nt){font-variant-numeric:normal}
.vd .vd-svg .lb2{font:12px var(--sans);fill:var(--ink-2);paint-order:stroke;stroke:var(--ground);stroke-width:4px;stroke-linejoin:round}
.vd .vd-svg .nt{font:600 12.5px var(--sans);fill:var(--note);paint-order:stroke;stroke:var(--ground);stroke-width:4px;stroke-linejoin:round}
.vd .vd-svg .now{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:3}
.vd .vd-svg .thin{fill:none;stroke-width:1.5}
@keyframes vddraw{to{stroke-dashoffset:0}}
.vd .vd-anim .now{stroke-dasharray:1;stroke-dashoffset:1;animation:vddraw 1.1s cubic-bezier(.16,1,.3,1) .1s forwards}
@media (prefers-reduced-motion:reduce){.vd .vd-anim .now{animation:none;stroke-dasharray:none;stroke-dashoffset:0}}
.vd{--vd-sun-ink:color-mix(in srgb,var(--sun) 62%,var(--ink));--vd-hot:color-mix(in srgb,var(--temp-hot) 85%,var(--ink))}
.vd .vd-why{margin:16px 0 0;font-size:var(--step--1);color:var(--ink-2)}
.vd .vd-why summary,.vd .vd-nums summary{cursor:pointer;width:max-content;min-height:32px;display:flex;align-items:center}
.vd .vd-why ol{margin:8px 0 0;padding:0;list-style:none;display:grid;gap:6px}
.vd .vd-why li{display:flex;gap:8px;align-items:baseline}
.vd .vd-why li i{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--ink-3);flex:none}
.vd .vd-why li[data-met="1"] i{background:var(--rain-3);border-color:var(--rain-3)}
.vd .vd-why li[data-lead="1"]{color:var(--ink);font-weight:600}
.vd .vd-proto{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--ink-2);border:1px dashed var(--rule-strong);border-radius:8px;padding:4px 8px;margin:0 0 4px;width:max-content}
.vd .vd-proto[hidden]{display:none}
.vd .vd-proto select{font:inherit;font-size:12px;color:var(--ink);background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:2px 4px}
.vd .vd-nums{font-size:var(--step--1);color:var(--ink-2);margin-top:6px}
.vd .vd-nums td.gap-hatch{text-align:center}
.vd .vd-week{margin-top:12px}
.vd .vd-wrow,.vd .vd-whead{display:grid;grid-template-columns:50px 30px 1fr 30px;gap:0 8px;align-items:center}
.vd .vd-wrow{min-height:40px;border-top:1px solid var(--rule);font-size:.875rem}
.vd .vd-wday{font-weight:600;line-height:1.15}
.vd .vd-wday small{display:block;font-weight:400;color:var(--ink-2);font-size:.6875rem}
.vd .vd-wlo{text-align:right;color:var(--ink-2)}
.vd .vd-whi{font-weight:600}
.vd .vd-wtrack{position:relative;height:18px}
.vd .vd-wnorm{position:absolute;top:1px;bottom:1px;background:var(--rule);border-radius:3px}
.vd .vd-wbar{position:absolute;top:5px;height:8px;border-radius:4px}
.vd .vd-wgap{grid-column:2 / 5;align-self:stretch;margin:6px 0;display:flex;align-items:center;justify-content:center;font-size:.75rem;color:var(--ink-2);border-radius:4px;text-align:center}
.vd .vd-whead{font-size:.6875rem;color:var(--ink-2)}
.vd .vd-waxis{position:relative;height:16px;grid-column:3}
.vd .vd-waxis span{position:absolute;transform:translateX(-50%)}
.vd .vd-nowstats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 12px;margin:12px 0 0;padding:8px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);max-width:520px}
.vd .vd-nowstats div{display:flex;flex-direction:column}
.vd .vd-nowstats dt{font-size:.8125rem;color:var(--ink-2);order:2}
.vd .vd-nowstats dd{margin:0;font-weight:650;font-size:var(--step-1);order:1;white-space:nowrap}
.vd .vd-ribbon{margin:12px 0 0;max-width:760px}
.vd .vd-ribbon .claim{font-size:.875rem;line-height:1.35;color:var(--ink-2);margin:0 0 6px;max-width:60ch}
.vd .vd-rband{position:relative;height:52px;border-radius:4px;overflow:hidden;background:var(--night)}
.vd .vd-rband svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.vd .vd-rhatch{position:absolute;top:0;bottom:0}
.vd .vd-rdot{position:absolute;right:3px;width:9px;height:9px;border-radius:50%;background:var(--note);box-shadow:0 0 0 2px var(--ground);transform:translateY(-50%)}
.vd .vd-rticks{position:relative;height:30px;margin-top:3px;font-size:.75rem;color:var(--ink-2)}
.vd .vd-rtick{position:absolute;top:0;transform:translateX(-50%);white-space:nowrap;line-height:1.3}
.vd .vd-rtick::before{content:"";display:block;width:1px;height:4px;background:var(--rule-strong);margin:0 auto 1px}
.vd .vd-rnow{position:absolute;right:0;top:0;font-size:.75rem;font-weight:650;color:var(--note);text-align:right;line-height:1.3}
.vd .vd-rkey{display:flex;flex-wrap:wrap;gap:2px 12px;font-size:.75rem;color:var(--ink-2);margin-top:2px}
.vd .vd-rkey span{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.vd .vd-rkey i{display:inline-block;width:12px;height:9px;border-radius:2px;border:1px solid var(--rule)}
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
  // one time format everywhere: "9:36 am"; dates "Oct 27, 11:00 pm"
  const amp = (t) => t.replace(/[\u202f\u00a0]/g, " ").replace(/ (AM|PM)\b/g, (m, x) => " " + x.toLowerCase());
  const fmtPT = (ms, o) => amp(PT(o || { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(ms)));
  const ptString = (ms) => { // plotly-friendly local wall time
    const p = Object.fromEntries(PT({ year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }).formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
    return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
  };
  // one duration format everywhere: "1 d 10 h", "5 h 12 m", "45 m"
  const fmtDur = (m) => { m = Math.round(m); if (m < 60) return `${m} m`; if (m < 1440) return `${Math.floor(m / 60)} h ${m % 60} m`; return `${Math.floor(m / 1440)} d ${Math.floor((m % 1440) / 60)} h`; };
  // years named in comparisons, newest first: "3.5 in (2025) and 4.5 in (2024)"
  const namedList = (items, fmt, unit) => { const a = items.slice().sort((x, y) => y.label - x.label).map((x) => `${fmt(x.v)} ${unit} (${x.label})`); return a.length > 1 ? a.slice(0, -1).join(", ") + " and " + a[a.length - 1] : a[0] || ""; };
  const yearsList = (ys) => { const a = ys.slice().sort((x, y) => y - x); return a.length > 1 ? a.slice(0, -1).join(", ") + " and " + a[a.length - 1] : String(a[0] || ""); };
  // wrap a long SVG label into lines that fit maxW (12px text, about 6.3px a character)
  const wrapLines = (text, maxW) => { const per = Math.max(12, Math.floor(maxW / 6.3)); const out = []; let line = ""; text.split(" ").forEach((w) => { if (line && (line + " " + w).length > per) { out.push(line); line = w; } else line = line ? line + " " + w : w; }); if (line) out.push(line); return out; };
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const isDark = () => {
    const t = document.documentElement.getAttribute("data-theme");
    return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const $ = (r, s) => r.querySelector(s);

  function hex2rgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  function ramp(names, t) { // names: token names; returns {bg, fg}
    t = Math.max(0, Math.min(1, t));
    const cols = names.map((n) => hex2rgb(css(n)));
    const x = t * (cols.length - 1), i = Math.min(cols.length - 2, Math.floor(x)), f = x - i;
    const c = cols[i].map((v, k) => Math.round(v + (cols[i + 1][k] - v) * f));
    const rl = (a) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(a[0]) + 0.7152 * f(a[1]) + 0.0722 * f(a[2]); };
    const L = rl(c), cw = 1.05 / (L + 0.05), cd = (L + 0.05) / (rl([20, 22, 26]) + 0.05); // WCAG contrast vs white / vs near-black
    return { bg: `rgb(${c.join(",")})`, fg: cw >= cd ? "#fff" : "#14161a" };
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
    // current month: `live` is the station's own month-to-date total, used as is so every view shows one value
    if (live && cur && mode === "sum") { const dn = dt(P.today).getUTCDate(); return { st: "cur", v: live, days: dn, sum: live, est: false, miss: 0, why: `Month still running, ${dn} days so far` }; }
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
    else if (c.rain_today_in > 0) r = `${f2(c.rain_today_in)} in of rain today, the last of it ${fmtDur((c.dateutc - Date.parse(c.last_rain)) / 60000)} ago.`;
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

  /* ---------- prototype data fix: an event still flagged ongoing 48 h after its last rain is closed ---------- */
  function normEvents(D) {
    const now = D.current.dateutc;
    D.rain_events.forEach((e) => { e.ongoing = !!e.ongoing; if (e.ongoing && now - Date.parse(e.end_time) > 48 * 36e5) { e.ongoing = false; if (e.flags) e.flags.ongoing = false; } });
  }

  /* ---------- 48-hour condition ribbon (ported from variant A, Paper tokens) ---------- */
  function ribbon48(D) {
    const c = D.current, r = D.recent, H = 36e5, step = (r.interval_min || 15) * 6e4;
    const tEnd = Date.parse(c.timestamp), tStart = tEnd - 48 * H, span = tEnd - tStart;
    const X = (t) => ((t - tStart) / span) * 1000, Pc = (t) => ((t - tStart) / span) * 100;
    const idxs = []; r.t.forEach((t, i) => { if (t + step > tStart && t <= tEnd && r.temp_f[i] != null) idxs.push(i); });
    if (!idxs.length) return "";
    const rampFor = (v) => (v <= 0 ? null : v < 0.03 ? "--rain-1" : v < 0.1 ? "--rain-2" : v < 0.2 ? "--rain-3" : "--rain-4");
    let svg = "", wet = 0;
    idxs.forEach((i) => {
      const x0 = X(Math.max(r.t[i], tStart)), w = X(Math.min(r.t[i] + step, tEnd)) - x0 + 0.4;
      svg += `<rect x="${x0.toFixed(2)}" y="0" width="${w.toFixed(2)}" height="100" style="fill:${r.solar_wm2[i] > 0 ? "var(--sun-soft)" : "var(--night)"}"/>`;
      const rc = rampFor(r.rain_rate_in_hr[i]);
      if (rc) { svg += `<rect x="${x0.toFixed(2)}" y="0" width="${w.toFixed(2)}" height="100" style="fill:var(${rc})"/>`; wet += step / H; }
    });
    const segs = []; idxs.forEach((i) => { const l = segs[segs.length - 1]; if (l && r.t[i] - r.t[l[l.length - 1]] <= step * 1.5) l.push(i); else segs.push([i]); });
    let cursor = tStart, hatch = ""; const gaps = [];
    segs.forEach((sg) => { const a = Math.max(r.t[sg[0]], tStart), b = Math.min(r.t[sg[sg.length - 1]] + step, tEnd); if (a - cursor > step * 0.5) gaps.push([cursor, a]); cursor = Math.max(cursor, b); });
    if (tEnd - cursor > step * 0.5) gaps.push([cursor, tEnd]);
    gaps.forEach(([a, b]) => { hatch += `<div class="vd-rhatch gap-hatch" style="left:${Pc(a).toFixed(2)}%;width:${(Pc(b) - Pc(a)).toFixed(2)}%"></div>`; });
    const archivedH = gaps.reduce((acc, [a, b]) => acc - (b - a) / H, 48);
    const temps = idxs.map((i) => r.temp_f[i]).concat([c.temp_f]);
    const lo = Math.min(...temps) - 2, hi = Math.max(...temps) + 2, Y = (v) => 12 + (1 - (v - lo) / (hi - lo)) * 76;
    let line = "";
    segs.forEach((sg) => { const pts = sg.map((i) => `${X(Math.min(r.t[i] + step / 2, tEnd)).toFixed(2)},${Y(r.temp_f[i]).toFixed(2)}`).join(" "); line += `<polyline points="${pts}" fill="none" vector-effect="non-scaling-stroke" style="stroke:var(--ground);stroke-width:4.5;stroke-linejoin:round;stroke-linecap:round"/><polyline points="${pts}" fill="none" vector-effect="non-scaling-stroke" style="stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`; });
    const tlo = Math.round(Math.min(...idxs.map((i) => r.temp_f[i]))), thi = Math.round(Math.max(...idxs.map((i) => r.temp_f[i])));
    let ticks = "";
    for (let t = Math.ceil(tStart / H) * H; t < tEnd; t += H) {
      const hr = Number(PT({ hour: "numeric", hourCycle: "h23" }).format(new Date(t)));
      if (hr % 6) continue;
      const x = Pc(t); if (x < 5 || x > 76) continue;
      ticks += `<div class="vd-rtick" style="left:${x.toFixed(2)}%">${hr === 0 ? PT({ weekday: "short" }).format(new Date(t + 6 * H)) : hr === 12 ? "noon" : hr < 12 ? hr + " am" : hr - 12 + " pm"}</div>`;
    }
    const rs = ["--rain-1", "--rain-2", "--rain-3", "--rain-4"].map((v) => `<i style="background:var(${v});width:9px;border-radius:0;border:0"></i>`).join("");
    return `<div class="vd-ribbon" id="vd-ribbon"><p class="claim">${tlo}° to ${thi}° over the last 48 hours, with rain in ${Math.round(wet)} of the ${Math.round(archivedH)} archived hours</p>
<div class="vd-rband" role="img" aria-label="48-hour condition ribbon: rain intensity, day and night, and temperature, ${tlo} to ${thi} degrees"><svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">${svg}</svg>${hatch}<svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">${line}</svg><div class="vd-rdot" style="top:${(Y(c.temp_f) * 0.52).toFixed(1)}px"></div></div>
<div class="vd-rticks">${ticks}<div class="vd-rnow">now ${f1(c.temp_f)}°</div></div>
<div class="vd-rkey"><span><i style="background:var(--night)"></i>night</span><span><i style="background:var(--sun-soft)"></i>day</span><span>rain ${rs}</span><span><i class="gap-hatch"></i>not archived yet</span></div></div>`;
  }

  /* ---------- panels ---------- */
  function build(root, D) {
    normEvents(D);
    const P = prep(D);
    const C = D.current;
    root.innerHTML = "";
    root.classList.add("vd");
    const style = document.createElement("style"); style.textContent = CSS; root.appendChild(style);

    const stamp = fmtPT(Date.parse(C.timestamp), { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    const archEnd = P.last;
    const top = document.createElement("div");
    top.innerHTML = `<div class="vd-top"><h1 class="vd-station">${D.meta.station_name}</h1><div class="vd-stamp"><span class="vd-dot" aria-hidden="true"></span>Live reading ${stamp}</div></div><label class="vd-proto" id="vd-proto">Prototype: lead with <select aria-label="Prototype lead state"><option value="auto">Auto (by rules)</option><option value="rain">Rain</option><option value="temp">Temperature</option><option value="solar">Solar</option><option value="calm">Calm</option></select></label>`;
    root.appendChild(top);
    const pick = $(top, "#vd-proto select"); pick.value = ["rain", "temp", "solar", "calm"].includes(new URLSearchParams(location.search).get("state")) ? new URLSearchParams(location.search).get("state") : "auto";
    pick.addEventListener("change", () => P.onState && P.onState(pick.value));

    const TABS = [["overview", "Overview"], ["rain", "Rain"], ["events", "Rain Events"], ["solar", "Solar"]];
    const tabs = document.createElement("div"); tabs.className = "vd-tabs"; tabs.setAttribute("role", "tablist"); tabs.setAttribute("aria-label", "Dashboard sections");
    tabs.innerHTML = TABS.map(([k, l]) => `<button class="vd-tab" role="tab" id="tab-${k}" aria-controls="panel-${k}" aria-selected="false" tabindex="-1" data-k="${k}">${l}</button>`).join("");
    root.appendChild(tabs);
    const panels = {};
    TABS.forEach(([k]) => { const p = document.createElement("section"); p.className = "vd-panel"; p.id = `panel-${k}`; p.setAttribute("role", "tabpanel"); p.setAttribute("aria-labelledby", `tab-${k}`); p.hidden = true; root.appendChild(p); panels[k] = p; });

    const drawn = {}; const redraws = [];
    const R = { overview: () => overview(panels.overview, P, redraws), rain: () => rainTab(panels.rain, P), events: () => eventsTab(panels.events, P, redraws), solar: () => solarTab(panels.solar, P, redraws) };

    function select(k, focus) {
      if (!R[k]) k = "overview";
      TABS.forEach(([t]) => { const on = t === k; const b = $(tabs, `[data-k=${t}]`); b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; panels[t].hidden = !on; });
      if (!drawn[k]) { drawn[k] = true; if (!window.Plotly && k !== "overview" && k !== "solar") { panels[k].innerHTML = "<p class=sub>Charts need Plotly.js, which did not load (offline?). Reload when connected.</p>"; } else R[k](); } else if (window.Plotly) panels[k].querySelectorAll(".js-plotly-plot").forEach((g) => Plotly.Plots.resize(g));
      $(top, "#vd-proto").hidden = k !== "overview";
      redraws.forEach((f) => f());
      if (focus) $(tabs, `[data-k=${k}]`).focus();
      if (location.hash !== "#" + k) history.replaceState(null, "", "#" + k);
    }
    tabs.addEventListener("click", (e) => { const b = e.target.closest(".vd-tab"); if (b) select(b.dataset.k); });
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
  const DOWS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const dowOf = (i) => DOWS[dt(i).getUTCDay()];
  const qtl = (arr, q) => { const a = arr.slice().sort((x, y) => x - y); const p = (a.length - 1) * q, f = Math.floor(p); return a[f] + (a[Math.min(f + 1, a.length - 1)] - a[f]) * (p - f); };

  const T = (x, y, s, cls, anchor) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" class="${cls || "tk"}"${anchor ? ` text-anchor="${anchor}"` : ""}>${s}</text>`;
  /* temperature across periods: model */
  function periodModel(P) {
    const D = P.D, R = D.overview.temp_ranges, A = D.daily_all, now = D.current.temp_f;
    const cov = new Map(); A.date.forEach((d, i) => cov.set(idx(d), A.coverage[i]));
    const anchor = idx((R._anchor || {}).archive_end || A.date[A.date.length - 1]);
    const lagged = P.today > anchor;
    const defs = [["today", "today"], ["yesterday", "yesterday"], ["last 7d", "7 days"], ["last 30d", "30 days"], ["last 90d", "90 days"], ["last 365d", "365 days"]];
    const rows = defs.map(([k, name]) => {
      const w = R[k]; if (!w) return null;
      const s = idx(w.start), e = idx(w.end);
      const days = []; // local days the window leans on
      if (k === "today") days.push(e); else if (k === "yesterday") days.push(s); else for (let i = s + 1; i <= e; i++) days.push(i);
      const bad = k === "today" ? 0 : days.filter((i) => !(cov.get(i) >= 0.8)).length;
      const share = days.length ? bad / days.length : 0;
      let label = name, sub = "";
      if (k === "today") { label = lagged ? fmtDay(e) : "Today"; sub = lagged ? "latest day, so far" : "so far"; }
      else if (k === "yesterday") { label = lagged ? fmtDay(s) : "Yesterday"; sub = lagged ? "day before" : fmtDay(s); }
      else sub = `${fmtDay(s + 1)}${e - s > 300 ? " ’" + String(dt(s + 1).getUTCFullYear()).slice(2) : ""} – ${fmtDay(e)}`;
      const has = w.min != null && w.p25 != null;
      const valid = has && share <= 0.1;
      return { k, label, sub, min: w.min, max: w.max, p25: w.p25, p75: w.p75, n: w.n, valid, est: valid && bad > 0, bad, nd: days.length, why: !has ? "No readings yet in this window" : share > 0.1 ? `${bad} of ${days.length} days missing or under 80% station coverage` : "" };
    }).filter(Boolean);
    const lagTxt = lagged ? `Windows end at the last archived reading (${fmtDay(anchor)}, ${fmtPT(Date.parse(R._anchor.archive_end), { hour: "numeric", minute: "2-digit" })}); the live reading is from ${fmtDay(P.today)}.` : "Windows end at the latest archived reading.";
    // headline: where now sits against the middle half of each window
    const rel = (r) => (now < r.p25 ? "below" : now > r.p75 ? "above" : "inside");
    const longs = rows.filter((r) => ["last 7d", "last 30d", "last 90d", "last 365d"].includes(r.k) && r.valid);
    const word = (x) => (x === "below" ? "colder than three in four readings" : x === "above" ? "warmer than three in four readings" : "inside the middle half");
    let head;
    if (!longs.length) head = "No window has enough valid record to place the current temperature";
    else {
      const rs = longs.map((r) => ({ r, x: rel(r) }));
      if (rs.every((o) => o.x === rs[0].x)) head = `${f1(now)}° is ${word(rs[0].x)} in every window from ${rs[0].r.k.replace("last ", "").replace("d", " days")} to ${rs[rs.length - 1].r.k.replace("last ", "").replace("d", " days")}`;
      else { const a = rs[0], z = rs[rs.length - 1]; head = `${f1(now)}° is ${word(a.x)} over the last ${a.r.k.replace("last ", "").replace("d", " days")}, but ${word(z.x)} over the past ${z.r.k === "last 365d" ? "year" : z.r.k.replace("last ", "").replace("d", " days")}`; }
    }
    return { rows, now, head, lagTxt, anchor };
  }
  function drawPeriods(host, M) {
    const w = Math.max(280, host.clientWidth || 340), narrow = w < 520;
    const labW = narrow ? 82 : 118, rowH = narrow ? 42 : 46, top = 38, bot = 26, padR = narrow ? 10 : 16;
    const vals = M.rows.filter((r) => r.valid).flatMap((r) => [r.min, r.max]).concat(M.now);
    const lo = Math.floor((Math.min(...vals) - 7) / 10) * 10, hi = Math.ceil((Math.max(...vals) + 7) / 10) * 10;
    const iw = w - labW - padR, X = (v) => labW + (iw * (v - lo)) / (hi - lo);
    const h = top + M.rows.length * rowH + bot, id = "vdp" + (++drawPeriods.n || (drawPeriods.n = 1));
    const step = hi - lo > 80 ? 20 : 10;
    let s = `<defs><pattern id="${id}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" style="fill:var(--gap-b)"/><rect width="3" height="8" style="fill:var(--gap-a)"/></pattern></defs>`;
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) s += `<line class="ax" x1="${X(t)}" x2="${X(t)}" y1="${top - 6}" y2="${top + M.rows.length * rowH}"/>` + T(X(t), top + M.rows.length * rowH + 16, t + "°", "tk", "middle");
    M.rows.forEach((r, i) => {
      const cy = top + i * rowH + rowH / 2;
      if (i) s += `<line class="ax" x1="0" x2="${w - padR}" y1="${top + i * rowH}" y2="${top + i * rowH}" style="stroke-opacity:.6"/>`;
      s += `<text x="0" y="${cy - 1}" style="font:600 ${narrow ? 12.5 : 13.5}px var(--sans);fill:var(--ink)">${r.label}</text>` + `<text x="0" y="${cy + 13}" style="font:${narrow ? 10.5 : 11.5}px var(--sans);fill:var(--ink-2)">${r.sub}${r.est ? " ≈" : ""}</text>`;
      if (!r.valid) {
        s += `<rect x="${labW}" y="${cy - 11}" width="${iw}" height="22" rx="4" style="fill:url(#${id});stroke:var(--rule-strong);stroke-width:.75"/>`;
        s += T(labW + iw / 2, cy + 4, narrow ? "n/a: gap in record" : "n/a: " + r.why, "lb2", "middle");
        return;
      }
      s += `<g role="img" aria-label="${r.label}: low ${f1(r.min)}, middle half ${f1(r.p25)} to ${f1(r.p75)}, high ${f1(r.max)} degrees"><title>${r.label}: ${f1(r.min)}° to ${f1(r.max)}°, middle half ${f1(r.p25)}° to ${f1(r.p75)}°</title>`;
      s += `<line x1="${X(r.min)}" x2="${X(r.max)}" y1="${cy}" y2="${cy}" style="stroke:var(--ink-3);stroke-width:1.5"/>`;
      s += `<circle cx="${X(r.min)}" cy="${cy}" r="2.6" style="fill:var(--ink-3)"/><circle cx="${X(r.max)}" cy="${cy}" r="2.6" style="fill:var(--ink-3)"/>`;
      s += `<rect x="${X(r.p25)}" y="${cy - 6}" width="${Math.max(2, X(r.p75) - X(r.p25))}" height="12" rx="3" style="fill:var(--temp-cool)"/>`;
      s += T(X(r.min) - 6, cy + 4, Math.round(r.min) + "°", "tk", "end") + T(X(r.max) + 6, cy + 4, Math.round(r.max) + "°", "tk", "start");
      s += `</g>`;
    });
    const nx = X(M.now), lx = Math.min(w - 52, Math.max(labW + 30, nx));
    s += `<line x1="${nx}" x2="${nx}" y1="${top - 10}" y2="${top + M.rows.length * rowH}" style="stroke:var(--note);stroke-width:2.25"/>`;
    s += `<circle cx="${nx}" cy="${top - 10}" r="4" style="fill:var(--note)"/>` + T(lx, top - 20, `now ${f1(M.now)}°`, "nt", "middle");
    host.innerHTML = `<svg class="vd-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="group" aria-label="Temperature range by period">${s}</svg>`;
  }

  /* A's ten-day range list: each day's low-to-high bar against the usual range for that date */
  function tenDay(P) {
    const D = P.D, A = D.daily_all, C = D.current;
    const dm = new Map(); A.date.forEach((d, i) => { if (A.coverage[i] >= 0.8 && A.temp_min[i] != null && A.temp_max[i] != null) dm.set(idx(d), i); });
    const normal = (k) => {
      const lows = [], highs = [], yr = dt(k).getUTCFullYear();
      for (let y = 2023; y < yr; y++) for (let o = -7; o <= 7; o++) { const kk = addYears(k, y - yr) + o; if (dm.has(kk)) { lows.push(A.temp_min[dm.get(kk)]); highs.push(A.temp_max[dm.get(kk)]); } }
      return lows.length >= 10 ? { n: lows.length, lo: qtl(lows, 0.25), hi: qtl(highs, 0.75) } : null;
    };
    const last = idx(A.date[A.date.length - 1]);
    const days = []; for (let i = 0; i < 10; i++) days.push(last - i);
    const norms = new Map(days.map((k) => [k, normal(k)]));
    const have = days.filter((k) => dm.has(k));
    const vv = [C.temp_f]; have.forEach((k) => { const i = dm.get(k); vv.push(A.temp_min[i], A.temp_max[i]); const n = norms.get(k); if (n) vv.push(n.lo, n.hi); });
    const sLo = Math.floor(Math.min(...vv) / 5) * 5, sHi = Math.ceil(Math.max(...vv) / 5) * 5;
    const X = (v) => ((v - sLo) / (sHi - sLo)) * 100;
    const stops = [[28, "--temp-cold"], [45, "--temp-cool"], [55, "--temp-mild"], [68, "--temp-warm"], [88, "--temp-hot"]].map(([t, v]) => `var(${v}) ${Math.max(0, Math.min(100, X(t))).toFixed(1)}%`).join(",");
    let ticks = ""; for (let t = Math.ceil(sLo / 10) * 10; t <= sHi; t += 10) ticks += `<span style="left:${X(t).toFixed(1)}%">${t}°</span>`;
    const rows = days.map((k) => {
      const dayEl = `<div class="vd-wday">${dowOf(k)}<small>${fmtDay(k)}</small></div>`;
      if (!dm.has(k)) return `<div class="vd-wrow">${dayEl}<div class="vd-wgap gap-hatch">n/a: no valid record (archive gap)</div></div>`;
      const i = dm.get(k), lo = A.temp_min[i], hi = A.temp_max[i], n = norms.get(k);
      const left = X(lo), wd = Math.max(X(hi) - X(lo), 1.5);
      const bg = `background-image:linear-gradient(to right, ${stops});background-size:${(10000 / wd).toFixed(1)}% 100%;background-position:${wd >= 100 ? 0 : ((left / (100 - wd)) * 100).toFixed(1)}% 0`;
      return `<div class="vd-wrow">${dayEl}<div class="vd-wlo">${Math.round(lo)}°</div><div class="vd-wtrack">${n ? `<div class="vd-wnorm" style="left:${X(n.lo).toFixed(1)}%;width:${(X(n.hi) - X(n.lo)).toFixed(1)}%"></div>` : ""}<div class="vd-wbar" style="left:${left.toFixed(1)}%;width:${wd.toFixed(1)}%;${bg}"></div></div><div class="vd-whi">${Math.round(hi)}°</div></div>`;
    });
    const lagRow = P.today > last ? `<div class="vd-wrow">${`<div class="vd-wday">${dowOf(P.today)}<small>${fmtDay(P.today)}</small></div>`}<div class="vd-wgap gap-hatch">not archived yet; ${f1(C.temp_f)}° now</div></div>` : "";
    // finding
    let warm = 0, cold = 0, nn = 0, hot = null, cool = null;
    have.forEach((k) => { const i = dm.get(k), n = norms.get(k), hi = A.temp_max[i], lo = A.temp_min[i]; if (!hot || hi > hot.v) hot = { v: hi, k }; if (!cool || lo < cool.v) cool = { v: lo, k }; if (n) { nn++; if (hi > n.hi) warm++; if (lo < n.lo) cold++; } });
    const parts = []; if (warm) parts.push(`${warm} topped the usual high`); if (cold) parts.push(`${cold} dipped below the usual low`);
    const head = !nn ? "Ten-day range" : parts.length ? `Of the last ${days.length} days, ${parts.join(" and ")} for their date` : `All ${nn} valid days of the last ${days.length} stayed inside the usual range for their date`;
    const missing = days.length - have.length;
    const html = `<h2>${head}</h2><p class="sub">Each bar runs from the day’s low to its high. ${hot ? `The warmest high was ${Math.round(hot.v)}° on ${dowOf(hot.k)} ${fmtDay(hot.k)}; the coolest low was ${Math.round(cool.v)}° on ${dowOf(cool.k)} ${fmtDay(cool.k)}.` : ""}</p>
<div class="vd-week"><div class="vd-whead"><div class="vd-waxis">${ticks}</div></div>${lagRow}${rows.join("")}</div>
<p class="src">Shaded band: the usual range for that date, from the 25th percentile of daily lows to the 75th of daily highs, taken from the same date plus or minus a week in earlier years. Hatched: under 80% of readings archived${missing ? `, so ${missing} day${missing > 1 ? "s are" : " is"} not computed` : ""}. Source: station archive.</p>`;
    return html;
  }

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
    const prevNamed = namedList(okPrev.map((x) => ({ v: x.w.v, label: x.y })), f1, "in"), byDay = `${MON[m0]} ${dd}`;
    if (okPrev.length && beat === okPrev.length) head = `${monthName} is already wetter than any earlier ${monthName}: ${f2(mtd)} in of rain by ${byDay}, against ${prevNamed}`;
    else if (okPrev.length && beat === 0) head = `${monthName} is running drier than every earlier ${monthName}: ${f2(mtd)} in of rain by ${byDay}, against ${prevNamed}`;
    else head = `${monthName} so far: ${f2(mtd)} in of rain by ${byDay}, wetter than ${beat} of ${okPrev.length} earlier ${monthName}s at this point (${prevNamed})`;

    const periodM = periodModel(P);
    const wins = periodM.rows.filter((r) => r.valid);
    let forced = ["rain", "temp", "solar", "calm"].includes(new URLSearchParams(location.search).get("state")) ? new URLSearchParams(location.search).get("state") : "auto";
    P.forced = () => forced;
    el.innerHTML = `
<div class="vd-nowline" style="margin-bottom:0"><div class="temp">${f1(C.temp_f)}<sup>°F</sup></div><p class="lead">${summary(C, D.rain.stats)}</p></div>
<dl class="vd-nowstats" aria-label="Right now"><div><dt>feels like</dt><dd>${f1(C.feels_like_f)}°</dd></div><div><dt>wind</dt><dd>${f1(C.wind_avg10m_mph)} mph</dd></div><div><dt>rain today</dt><dd>${f2(C.rain_today_in)} in</dd></div></dl>
${ribbon48(D)}
<div class="vd-grid two-r" style="margin-top:28px">
  <section id="vd-hero" aria-label="What is happening now"></section>
  <div>
    <h3>Also right now</h3>
    <div class="facts">
      <div>Humidity<b>${C.humidity}%</b></div><div>Dew point<b>${f1(C.dew_point_f)}°</b></div><div>Pressure<b>${f2(C.pressure_inhg)} inHg ${C.pressure_trend_symbol}</b></div>
    </div>
    <div id="vd-why"></div>
  </div>
</div>
<div class="vd-grid two-r vd-next">
  <section id="vd-per">
    <h2>${periodM.head}</h2>
    <p class="sub">Each row is a window of the station record: the thin line runs from its lowest to its highest reading, the thick bar covers the middle half of readings (25th to 75th percentile), and the line marked now is the current temperature.</p>
    <div id="vd-pchart" class="chart"></div>
    <details class="vd-nums"><summary>Show the numbers</summary><table><thead><tr><th>Window</th><th>Low</th><th>25th</th><th>75th</th><th>High</th></tr></thead><tbody>${periodM.rows.map((r) => (r.valid ? `<tr><td><b>${r.label}</b></td><td>${f1(r.min)}°</td><td>${f1(r.p25)}°</td><td>${f1(r.p75)}°</td><td>${f1(r.max)}°</td></tr>` : `<tr><td><b>${r.label}</b></td><td class="gap-hatch" colspan="4" title="${r.why}">n/a: ${r.why}</td></tr>`)).join("")}</tbody></table></details>
    <p class="src">Source: station archive, 5-minute readings, Pacific time. ${periodM.lagTxt} A window with more than 10% of its days missing or under 80% station coverage is hatched, not computed; ≈ marks a window with a few such days. The percentiles are over readings, not over days.</p>
  </section>
  <section id="vd-ten">${tenDay(P)}</section>
</div>
<div class="vd-grid two vd-next">
  <div class="block">
    <h2>Now against normal: ${pct < 50 ? "cooler" : "warmer"} than usual, ${wetter != null && wetter >= 50 ? "wetter" : "drier"} than most ${span}-${monthName} days</h2>
    <div class="strip-wrap"><p class="claim">${pct < 50 ? "Cooler" : "Warmer"} than ${tword}% of ${span}-${monthName} days <span>(today averages ${f1(nowT)}°F so far, ${f1(C.temp_f)}°F now; the median day is ${f1(tmed)}°F; ${tvals.length} recorded days, 2023–25)</span></p><div id="vd-ts"></div></div>
    <div class="strip-wrap"><p class="claim">${wetter == null ? "No earlier-year rain record for this date" : `Wetter than ${wetter}% of ${span}-${monthName} days`} <span>(${vals.length} recorded days, ${winLbl} in 2023–25; ${dry} were dry)</span></p><div id="vd-rs"></div></div>
    <p class="src">Source: lookout post station. Temperature compares today’s mean so far with each daily mean within seven days of this date, 2023–25. Rain compares today’s total so far with each daily total within seven days of this date, 2023–25; days inside archive gaps are excluded, not counted as dry.</p>
  </div>
  <div>
    <h2 id="vd-alh"></h2>
    <p class="sub" id="vd-als"></p>
    <div class="seg" role="group" aria-label="Almanac measure"><button type="button" data-m="rain" aria-pressed="true">Rain total</button><button type="button" data-m="temp" aria-pressed="false">Mean temperature</button></div>
    <div id="vd-alm"></div>
    <div class="key" id="vd-key"></div>
    <p class="readout" id="vd-ro" aria-live="polite">Tap or hover a month for its value.</p>
    <p class="src" id="vd-als2"></p>
  </div>
</div>
<div class="vd-next" id="vd-roll"></div>`;
    // hero: C's state-adaptive lead, rules, and bold-now-over-soft-usual chart
    let H = null, heroFirst = true;
    function renderHero() {
      H = HERO.hero(D, forced);
      const L = H.L;
      $(el, "#vd-hero").innerHTML = `<h2 class="vd-hero-h">${L.h1}</h2><p class="sub">${L.sub}</p>
<dl class="vd-facts">${L.facts.map(([v, k]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
<div class="vd-chart" id="vd-hchart"></div><p class="src">${L.foot}</p>`;
      $(el, "#vd-why").innerHTML = `<details class="vd-why"><summary>Why this leads${forced !== "auto" ? " (forced by the prototype picker)" : ""}</summary><ol>${H.rules.map((r) => `<li data-met="${r.met ? 1 : 0}" data-lead="${r.k === H.st ? 1 : 0}"><i></i><span>${r.text}${r.met ? ", met" : ""}</span></li>`).join("")}</ol></details>`;
      heroFirst = true;
    }
    function drawHero() {
      const host = $(el, "#vd-hchart"); if (!host || !H) return;
      const w = Math.floor(host.clientWidth); if (!w) return;
      host.innerHTML = H.L.chart(w);
      host.classList.toggle("vd-anim", heroFirst); heroFirst = false;
    }
    renderHero();
    P.onState = (v) => { forced = v; const u = new URL(location.href); if (v === "auto") u.searchParams.delete("state"); else u.searchParams.set("state", v); history.replaceState(null, "", u); renderHero(); drawHero(); };
    const draw = () => {
      if (el.hidden) return;
      drawHero();
      drawPeriods($(el, "#vd-pchart"), periodM);
      drawTempStrip($(el, "#vd-ts"), { vals: tvals, now: nowT });
      drawRainStrip($(el, "#vd-rs"), { vals, now: nowRain });
      drawAlmanac();
    };
    let mode = "rain";
    function drawAlmanac() {
      const host = $(el, "#vd-alm"), ro = $(el, "#vd-ro"); ro.textContent = "Tap or hover a month for its value.";
      const years = [2023, 2024, 2025, 2026];
      const curY = dt(P.today).getUTCFullYear();
      const live = mode === "rain" ? C.rain_month_in : 0;
      const cell = (y, m) => { const c = mode === "rain" ? month(P, P.rain, y, m, "sum", live) : month(P, P.tmp, y, m, "mean"); c.cur = y === curY && m - 1 === m0; return c; };
      const cfg = { years, cell, readout: ro, unit: mode === "rain" ? "in rain" : "°F mean", fmt: mode === "rain" ? (v) => v.toFixed(1) : (v) => Math.round(v) + "°", stops: mode === "rain" ? RAIN : TEMP, lo: mode === "rain" ? 0 : undefined, curve: mode === "rain" ? Math.sqrt : (t) => t, cur: m0 };
      if (mode === "temp") { const v = years.flatMap((y) => MON.map((_, i) => cell(y, i + 1))).filter((c) => c.st === "ok").map((c) => c.v); cfg.lo = Math.min(...v); }
      const { lo, hi } = almanacGrid(host, cfg);
      $(el, "#vd-key").innerHTML = mode === "rain"
        ? `<span>${lo.toFixed(0)} in</span><i style="background:linear-gradient(90deg,var(--rain-0),var(--rain-2),var(--rain-4))"></i><span>${hi.toFixed(1)} in</span><span class="hatch gap-hatch" style="display:inline-block"></span><span>gap or partial month, not comparable</span><span>≈ prorated, a few low-coverage days</span>`
        : `<span>${Math.round(lo)}°</span><i style="background:linear-gradient(90deg,var(--temp-cold),var(--temp-mild),var(--temp-warm),var(--temp-hot))"></i><span>${Math.round(hi)}°</span><span class="hatch gap-hatch" style="display:inline-block"></span><span>no valid month</span>`;
      // headline
      const all = years.flatMap((y) => MON.map((_, i) => ({ y, m: i + 1, c: cell(y, i + 1) })));
      const ok = all.filter((x) => x.c.st === "ok" && !x.c.est); // full months only: a ≈ month is never called full
      const hp = $(el, "#vd-alh"), sp = $(el, "#vd-als"), s2 = $(el, "#vd-als2");
      if (mode === "rain") {
        const peak = ok.reduce((a, b) => (b.c.v > a.c.v ? b : a));
        const octs = ok.filter((x) => x.m - 1 === m0).sort((a, b) => b.c.v - a.c.v);
        hp.textContent = head;
        sp.textContent = `Every month since 2023 by rain total. The wettest full month was ${MONL[peak.m - 1]} ${peak.y} at ${f1(peak.c.v)} in; full ${monthName}s: ${octs.map((x) => `${x.y} ${f1(x.c.v)}`).join(", ") || "none yet"} in. ≈ marks a prorated month with a few low-coverage days.`;
        s2.textContent = `Source: station archive, daily rain totals. Outlined month is the current one (live total, hatched). Hatched: the archive has missing or low-coverage days or starts mid-month (Feb 2023), so no total is computed. A ≈ month has a few low-coverage days and is prorated, so it is never called a full month. Archive ends ${fmtDay(P.last)}; the live reading is about a day ahead.`;
      } else {
        const hot = ok.reduce((a, b) => (b.c.v > a.c.v ? b : a)), cold = ok.reduce((a, b) => (b.c.v < a.c.v ? b : a));
        hp.textContent = ok.length ? `${MONL[hot.m - 1]} ${hot.y} was the warmest full month on file at ${f1(hot.c.v)}°F; ${MONL[cold.m - 1]} ${cold.y} the coolest at ${f1(cold.c.v)}°F` : "No complete month of temperature";
        sp.textContent = "Mean of daily mean temperature. Months with a gap or low-coverage day are hatched, not computed.";
        s2.textContent = `Source: station archive, daily mean temperature. Hatched: no valid month (gap, low-coverage day, or still running). ${monthName} ${curY} is outlined.`;
      }
    }
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { mode = b.dataset.m; el.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b)); drawAlmanac(); }));
    rollingTable($(el, "#vd-roll"), P);
    redraws.push(draw);
  }

  function rollingTable(host, P) {
    const N = [7, 30, 90, 365];
    const metrics = [
      { k: "rain", name: "Rain, total", map: P.rain, mode: "sum", fmt: (v) => f2(v) + " in", dfmt: (a, b) => (b === 0 ? "n/a" : sgn(a - b, Math.round(Math.abs((a - b) / b) * 100) + "%")), whyNo: "Archive gap or before the archive began" },
      { k: "sol", name: "Solar energy, total (kWh/m\u00b2)", map: P.sol, mode: "sum", fmt: (v) => f1(v), dfmt: (a, b) => (b === 0 ? "n/a" : sgn(a - b, Math.round(Math.abs((a - b) / b) * 100) + "%")), whyNo: "Archive gap or before the archive began" },
    ];
    const calc = metrics.map((m) => N.map((n) => {
      const cur = win(m.map, P.last, n, m.mode), ly = win(m.map, addYears(P.last, -1), n, m.mode);
      const prior = [1, 2, 3].map((k) => win(m.map, addYears(P.last, -k), n, m.mode)).filter((w) => w.valid);
      const avg = prior.length ? prior.reduce((s, w) => s + w.v, 0) / prior.length : null;
      return { n, cur, ly, avg, na: prior.length, ys: [1, 2, 3].filter((k) => win(m.map, addYears(P.last, -k), n, m.mode).valid).map((k) => dt(addYears(P.last, -k)).getUTCFullYear()) };
    }));
    // headline: biggest valid rain delta vs earlier-year average
    const r = calc[0].filter((x) => x.cur.valid && x.avg != null && x.avg > 0).map((x) => ({ n: x.n, d: (x.cur.v - x.avg) / x.avg, v: x.cur.v, a: x.avg, est: x.cur.est, ys: x.ys }));
    const big = r.sort((a, b) => Math.abs(b.d) - Math.abs(a.d))[0];
    const head = big ? `The last ${big.n} days brought ${big.est ? "about " : ""}${f2(big.v)} in of rain, ${Math.round(Math.abs(big.d) * 100)}% ${big.d < 0 ? "less" : "more"} than the average of the same ${big.n} days in ${yearsList(big.ys)}` : "Rolling windows";
    let html = `<h2>${head}</h2><p class="sub">Rain and solar over rolling windows (temperature is in the periods chart above). Each window ends ${fmtDay(P.last)}, the last day in the archive, and is set against the same window a year earlier and the average of earlier years. A window missing more than 10% of its days is shown hatched, not computed; a ≈ marks a prorated total with a few low-coverage days excluded.</p><div class="vd-grid half" style="margin-top:16px">`;
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
      if (i === P.today) { v = P.D.current.rain_today_in; if (mo === 9) v = Math.max(0, P.D.current.rain_month_in - cum); } // in October the water year equals the month to date
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
    const prevNamed = namedList(prev.map((x) => ({ v: x.a.v, label: x.wy })), f1, "in");
    const head = !prev.length ? "Not enough history" : beat === prev.length ? `Water year ${curWY} has started wetter than any before it: ${f2(cv)} in of rain in ${days} days, against ${prevNamed}` : beat === 0 ? `Water year ${curWY} has started drier than every earlier one: ${f2(cv)} in of rain in ${days} days, against ${prevNamed}` : `Water year ${curWY} has ${f2(cv)} in of rain after ${days} days, wetter than ${beat} of ${prev.length} earlier starts (${prevNamed})`;
    // monthly totals
    const months = [];
    for (let k = 23; k >= 0; k--) { const d = new Date(Date.UTC(dt(P.today).getUTCFullYear(), dt(P.today).getUTCMonth() - k, 1)); months.push({ y: d.getUTCFullYear(), m: d.getUTCMonth() + 1 }); }
    const mm = months.map((o) => ({ ...o, c: month(P, P.rain, o.y, o.m, "sum", o.y === dt(P.today).getUTCFullYear() && o.m - 1 === dt(P.today).getUTCMonth() ? D.current.rain_month_in : 0) }));
    const okm = mm.filter((x) => x.c.st === "ok" && !x.c.est); // full months only: a ≈ month is never called full
    const pk = okm.reduce((a, b) => (b.c.v > a.c.v ? b : a));
    const dry = okm.filter((x) => x.c.v < 0.5);
    el.innerHTML = `
<div class="vd-grid two-r">
  <div><h2>${head}</h2><p class="sub">Cumulative rain from October 1, one line per water year. The bold line is the year in progress.</p>
    <div class="seg" role="group" aria-label="Range"><button type="button" data-r="q" aria-pressed="true">First 3 months</button><button type="button" data-r="y" aria-pressed="false">Full water year</button></div>
    <div id="vd-cum" class="chart" style="height:340px"></div>
    <p class="src">Source: station archive, daily rain totals, plus today’s live total. Dotted segments are days missing or under 80% station coverage (largest: Apr–Jun 2025, Feb 2026), so those years’ totals are understated. Water year 2023 is omitted: the archive begins Feb 15, 2023.</p></div>
  <div><h2>${MONL[pk.m - 1]} ${pk.y} was the wettest full month in two years, with ${f1(pk.c.v)} in; ${dry.length} full months stayed under half an inch</h2><p class="sub">Monthly rain totals, last 24 months. Hatched bars are months with archive gaps or still running; a ≈ bar is prorated for a few low-coverage days and is not a full month.</p>
    <div id="vd-mon" class="chart" style="height:300px"></div>
    <p class="src">Source: station archive, daily rain totals, and the station’s own month-to-date total for the month in progress.</p></div>
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
      Plotly.react($(el, "#vd-cum"), tr, plotBase({ margin: { l: 38, r: range === "q" ? 92 : 14, t: 16, b: 30 }, xaxis: { range: [0, range === "q" ? 92 : 365], tickvals: tv.filter((v) => v <= xmax), ticktext: tt.filter((_, i) => tv[i] <= xmax), showgrid: false, linecolor: css("--rule"), tickfont: { color: css("--ink-3") }, fixedrange: true, zeroline: false }, yaxis: { title: { text: "inches, cumulative", font: { size: 11, color: css("--ink-3") } }, gridcolor: css("--rule"), zeroline: false, range: [0, ymax * 1.08], fixedrange: true, tickfont: { color: css("--ink-3") } }, annotations, hovermode: "closest" }), PCFG);
    }
    let range = "q"; drawCum(range);
    el.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { range = b.dataset.r; el.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b)); drawCum(range); }));
    // monthly bars
    const rule = css("--rule-strong"), surf = css("--ground");
    Plotly.react($(el, "#vd-mon"), [{
      type: "bar", x: mm.map((o) => `${o.y}-${String(o.m).padStart(2, "0")}-15`), y: mm.map((o) => (o.c.v != null ? o.c.v : o.c.partial || 0)),
      text: mm.map((o) => (o.c.est ? "≈" : "")), textposition: "outside", cliponaxis: false, textfont: { color: css("--ink-2"), size: 12 },
      marker: { color: mm.map((o) => (o.c.st === "ok" && !o.c.est ? css("--rain-3") : o.c.st === "cur" || o.c.est ? css("--rain-2") : css("--rule-strong"))), pattern: { shape: mm.map((o) => (o.c.st === "ok" && !o.c.est ? "" : "/")), fgcolor: css("--rain-3"), bgcolor: surf, size: 5, solidity: 0.35 } },
      hovertemplate: mm.map((o) => `${MON[o.m - 1]} ${o.y}: ${o.c.est ? "about " : ""}%{y:.2f} in${o.c.st === "ok" && !o.c.est ? "" : " (" + (o.c.st === "cur" ? "month still running" : o.c.est ? "prorated, a few low-coverage days" : "gap, not a valid total") + ")"}<extra></extra>`),
    }], plotBase({ margin: { l: 34, r: 8, t: 8, b: 30 }, bargap: 0.25, xaxis: { type: "date", tickformat: "%b\n%Y", dtick: "M3", showgrid: false, linecolor: css("--rule"), tickfont: { color: css("--ink-3") }, fixedrange: true }, yaxis: { gridcolor: css("--rule"), zeroline: false, fixedrange: true, tickfont: { color: css("--ink-3") }, ticksuffix: "" }, hovermode: "closest" }), PCFG);
  }

  /* ----- Events ----- */
  function eventsTab(el, P, redraws) {
    const D = P.D, ev = D.rain_events.slice().sort((a, b) => (a.ongoing === b.ongoing ? Date.parse(b.start_time) - Date.parse(a.start_time) : a.ongoing ? -1 : 1));
    const LV = HERO.model(D).liveEv; // the one live-event object, shared with the hero
    const totOf = (e) => (LV && LV.e === e ? LV.total : e.total_rainfall), durOf = (e) => (LV && LV.e === e ? LV.durMin : e.duration_minutes);
    const byTotal = D.rain_events.slice().sort((a, b) => totOf(b) - totOf(a));
    const rank = (e) => byTotal.findIndex((x) => x.event_id === e.event_id) + 1;
    let sel = ev[0], limit = 20;
    const curN = D.current;
    el.innerHTML = `
<div class="vd-grid">
  <div><h2 id="vd-sh"></h2><p class="sub" id="vd-ss"></p><div id="vd-esc" class="vd-chart"></div><p class="src">Source: Event Catalog (${D.rain_events.length} events since Feb 2023), station archive. Events without a duration or total are left out. Select a row below to move the highlighted dot.</p></div>
  <div class="vd-grid two-r">
  <div><h2 id="vd-eh"></h2><p class="sub" id="vd-es"></p><div id="vd-ec" class="chart" style="height:280px"></div><p class="src" id="vd-esrc"></p></div>
  <div><div id="vd-et"></div></div>
  </div>
</div>`;
    function scatter() {
      const S = HERO.eventsScatter(D, sel.event_id);
      $(el, "#vd-sh").textContent = S.h1; $(el, "#vd-ss").textContent = S.sub;
      const host = $(el, "#vd-esc"), w = Math.floor(host.clientWidth);
      if (w) host.innerHTML = S.chart(w);
    }
    redraws.push(() => { if (!el.hidden) scatter(); });
    function curve() {
      const e = sel, s = e.series, x = s.t.map(ptString), tt = totOf(e), isLive = !!(LV && LV.e === e);
      const r = rank(e);
      $(el, "#vd-eh").textContent = `${f2(tt)} in of rain over ${fmtDur(durOf(e))}: the ${r}${["th", "st", "nd", "rd"][r % 100 > 10 && r % 100 < 14 ? 0 : Math.min(r % 10, 4) % 4]} wettest of ${D.rain_events.length} catalogued events${e.max_hourly_rate ? ", peaking at " + f2(e.max_hourly_rate) + " in/hr" : ""}`;
      $(el, "#vd-es").textContent = `Accumulation of the event that began ${fmtPT(Date.parse(e.start_time))}${isLive ? "; it is still open in the catalog" : ""}. Data quality: ${e.quality_rating}, ${Math.round(e.data_completeness * 100)}% complete.`;
      $(el, "#vd-esrc").textContent = isLive ? `Source: live station for the total and duration, Event Catalog curve. ${f2(LV.archTotal)} in archived to ${HERO.dShort(Date.parse(D.meta.archive_range[1]))}, ${HERO.tShort(Date.parse(D.meta.archive_range[1]))}; the dotted line is not recorded in the archive.` : "Source: Event Catalog, 5-minute station readings, Pacific time.";
      const tr = [{ type: "scatter", mode: "lines", x, y: s.accum_in, line: { color: css("--rain-3"), width: 3, shape: "hv" }, fill: "tozeroy", fillcolor: css("--rain-0"), hovertemplate: "%{x|%b %d %H:%M}: %{y:.2f} in<extra></extra>" }];
      const lastI = x.length - 1;
      let ann = [{ x: x[lastI], y: s.accum_in[lastI], text: `${f2(s.accum_in[lastI])} in`, showarrow: false, xanchor: "right", yanchor: "bottom", yshift: 6, font: { size: 12, color: css("--ink") } }];
      if (isLive) {
        const xe = ptString(LV.endMs);
        tr.push({ type: "scatter", mode: "lines", x: [x[lastI], xe], y: [s.accum_in[lastI], tt], line: { color: css("--rain-3"), width: 3, dash: "dot" }, hoverinfo: "skip" });
        tr.push({ type: "scatter", mode: "markers", x: [xe], y: [tt], marker: { color: css("--note"), size: 11, line: { color: css("--ground"), width: 2 } }, hoverinfo: "skip" });
        ann = [{ x: xe, y: tt, text: `${f2(tt)} in`, showarrow: false, xanchor: "right", yanchor: "bottom", yshift: 8, xshift: -6, font: { size: 12, color: css("--ink") } }];
      }
      Plotly.react($(el, "#vd-ec"), tr, plotBase({ margin: { l: 56, r: 14, t: 12, b: 30 }, annotations: ann, yaxis: { gridcolor: css("--rule"), zeroline: false, rangemode: "tozero", fixedrange: true, tickfont: { color: css("--ink-3") }, ticksuffix: " in" }, xaxis: { showgrid: false, linecolor: css("--rule"), tickfont: { color: css("--ink-3") }, fixedrange: true, zeroline: false, hoverformat: "%b %d %H:%M" }, hovermode: "x" }), PCFG);
    }
    function table() {
      const rows = ev.slice(0, limit);
      const q = (e) => { const bad = e.quality_rating === "poor" || e.flags.low_completeness || !e.usable_for_analysis; return `<span class="q ${bad || e.flags.has_gaps ? "low" : ""}">${e.quality_rating}${e.flags.has_gaps ? " · gaps" : ""}${e.flags.interrupted ? " · cut" : ""}</span>`; };
      let h = `<table class="ev"><caption class="sr">Rain events, newest first, ongoing event pinned</caption><thead><tr><th>Start</th><th>Duration</th><th>Total (in)</th><th>Peak (in/h)</th><th>Quality</th></tr></thead><tbody>`;
      rows.forEach((e) => {
        h += `<tr class="${e.ongoing ? "pin" : ""}" ${e.event_id === sel.event_id ? 'aria-current="true"' : ""}><td>${e.series ? `<button type="button" class="rowbtn" data-id="${e.event_id}">` : '<span class="rowbtn" title="No stored curve for this older event">'}${e.ongoing ? '<span class="live">live</span>' : ""}${fmtPT(Date.parse(e.start_time), { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).replace(",", "")}${e.series ? "</button>" : "</span>"}</td><td>${fmtDur(durOf(e))}</td><td><b>${f2(totOf(e))}</b></td><td>${e.max_hourly_rate ? f2(e.max_hourly_rate) : "–"}</td><td>${q(e)}</td></tr>`;
      });
      h += `</tbody></table>`;
      const pinned = ev[0].ongoing;
      const note = pinned && LV ? `<p class="note-live"><span class="live">Live</span> The top event is still open in the catalog. The live station shows ${curN.rain_rate_in_hr > 0 ? "rain falling now" : "no rain falling now"}, last rain ${fmtDur((curN.dateutc - LV.endMs) / 60000)} ago at ${fmtPT(LV.endMs, { hour: "numeric", minute: "2-digit" })}, and an event total of ${f2(LV.total)} in over ${fmtDur(LV.durMin)}. Archived: ${f2(LV.archTotal)} in to ${fmtPT(Date.parse(ev[0].end_time))}.</p>` : "";
      h += (limit < ev.length ? `<button type="button" class="more">Show ${Math.min(40, ev.length - limit)} more of ${ev.length}</button>` : "") + `<p class="src">Source: Event Catalog (${ev.length} events since Feb 2023). Peak rate is the highest hourly rate in the event. Dashed quality chips mark events with gaps or low completeness. Select a row to chart it (older events keep no stored curve).</p>` + note;
      $(el, "#vd-et").innerHTML = h;
      $(el, "#vd-et").querySelectorAll(".rowbtn").forEach((b) => b.addEventListener("click", () => { sel = ev.find((e) => e.event_id === b.dataset.id); curve(); scatter(); table(); window.scrollTo({ top: el.getBoundingClientRect().top + scrollY - 70, behavior: "smooth" }); }));
      const more = $(el, ".more"); if (more) more.addEventListener("click", () => { limit += 40; table(); });
    }
    curve(); table(); scatter();
  }

  /* ----- Solar ----- */
  function solarTab(el, P, redraws) {
    const D = P.D, H = D.solar.heatmap_15min_120d;
    const day = new Map(); H.date.forEach((d, i) => { const k = idx(d); day.set(k, (day.get(k) || 0) + H.kwh_m2[i]); });
    const keys = [...day.keys()].filter((k) => k <= P.last).sort((a, b) => a - b), k0 = keys[0], k1 = keys[keys.length - 1]; // only days the archive really covers
    const missing = []; for (let i = k0; i <= k1; i++) if (!day.has(i)) missing.push(i);
    const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const first14 = keys.slice(0, 14).map((k) => day.get(k)), last14 = keys.slice(-14).map((k) => day.get(k));
    const pkK = keys.reduce((a, b) => (day.get(b) > day.get(a) ? b : a));
    const drop = Math.round((1 - avg(last14) / avg(first14)) * 100);
    const h1 = `Daily sun is down ${drop}% since ${fmtDay(k0)}: ${f2(avg(last14))} kWh/m² a day lately against ${f2(avg(first14))} then`;
    // month x year
    const years = [2023, 2024, 2025, 2026], m0 = dt(P.today).getUTCMonth(), curY = dt(P.today).getUTCFullYear();
    const cell = (y, m) => { const c = month(P, P.sol, y, m, "sum", 0); return c; };
    const lastFull = (() => { for (let k = 1; k < 24; k++) { const d = new Date(Date.UTC(curY, m0 - k, 1)); const a = cell(d.getUTCFullYear(), d.getUTCMonth() + 1), b = cell(d.getUTCFullYear() - 1, d.getUTCMonth() + 1); if (a.st === "ok" && b.st === "ok" && !a.est && !b.est) return { d, a, b }; } })();
    const h2 = lastFull ? `${MONL[lastFull.d.getUTCMonth()]} ${lastFull.d.getUTCFullYear()} collected ${f1(lastFull.a.v)} kWh/m², ${Math.round(Math.abs(lastFull.a.v / lastFull.b.v - 1) * 100)}% ${lastFull.a.v < lastFull.b.v ? "less" : "more"} than ${MONL[lastFull.d.getUTCMonth()]} ${lastFull.d.getUTCFullYear() - 1}` : "Month by month";
    const SD = HERO.solarDay(D);
    el.innerHTML = `
<section><h2>${SD.h1}</h2><p class="sub">${SD.sub}</p><div id="vd-sday" class="vd-chart"></div><p class="src">${SD.foot}</p></section>
<div class="vd-grid two vd-next">
  <div><h2>${h1}</h2><p class="sub">Daily solar energy, last ${keys.length} days with data (${fmtDay(k0)} – ${fmtDay(k1)}). Each square is a day, weeks run left to right.</p>
    <div id="vd-cal"></div><p class="readout" id="vd-cro" aria-live="polite">Tap or hover a day for its value.</p>
    <div class="key"><span>0</span><i style="background:linear-gradient(90deg,var(--sun-soft),var(--sun),var(--temp-warm),var(--temp-hot))"></i><span id="vd-chi"></span><span class="hatch gap-hatch" style="display:inline-block"></span><span>no data</span></div>
    <p class="src">Source: station solar radiation, 15-minute periods summed per day (heatmap data). Peak day ${fmtDay(pkK)} at ${f1(day.get(pkK))} kWh/m². ${missing.length ? missing.length + " day(s) missing, shown hatched." : "No missing days in this window."} Archive ends ${fmtDay(P.last)}; today is not included.</p></div>
  <div><h2>${h2}</h2><p class="sub">Monthly solar energy by year, kWh/m². Hover or tap a month for its value.</p>
    <div id="vd-sm"></div><div class="key"><span>low</span><i style="background:linear-gradient(90deg,var(--sun-soft),var(--sun),var(--temp-warm),var(--temp-hot))"></i><span>high</span><span class="hatch gap-hatch" style="display:inline-block"></span><span>gap or partial month</span></div>
    <p class="readout" id="vd-smro" aria-live="polite">Tap or hover a month for its value.</p>
    <p class="src">Source: station archive, daily solar energy. Hatched: the archive has missing or low-coverage days or starts mid-month (Feb 2023). The outlined month is still running.</p></div>
</div>`;
    const maxV = Math.max(...keys.map((k) => day.get(k)));
    $(el, "#vd-chi").textContent = f1(maxV) + " kWh/m²";
    function drawCal() {
      const host = $(el, "#vd-cal");
      const startDow = (dt(k0).getUTCDay() + 6) % 7; const gridStart = k0 - startDow;
      const weeks = Math.ceil((k1 - gridStart + 1) / 7);
      let c = `<div class="calwrap"><div class="dows" aria-hidden="true">${["Mon", "", "Wed", "", "Fri", "", "Sun"].map((d) => `<span>${d}</span>`).join("")}</div><div><div class="calm" style="grid-template-columns:repeat(${weeks},minmax(0,1fr))">`;
      for (let w = 0; w < weeks; w++) { const d0 = gridStart + w * 7; let lab = ""; for (let o = 0; o < 7; o++) { if (dt(d0 + o).getUTCDate() === 1 && d0 + o >= k0 && d0 + o <= k1) lab = MON[dt(d0 + o).getUTCMonth()]; } if (w === 0 && !lab) lab = MON[dt(k0).getUTCMonth()]; c += `<span>${lab}</span>`; }
      c += `</div><div class="cal" style="grid-template-columns:repeat(${weeks},minmax(0,1fr))">`;
      for (let i = gridStart; i < gridStart + weeks * 7; i++) {
        if (i < k0 || i > k1) { c += `<div class="c"></div>`; continue; }
        const v = day.get(i); const lbl = `${MON[dt(i).getUTCMonth()]} ${dt(i).getUTCDate()}`;
        if (v == null) c += `<button type="button" class="c gap-hatch" data-info="${lbl}: no data, not computed" aria-label="${lbl}: no data"></button>`;
        else c += `<button type="button" class="c" style="background:${ramp(SUN, v / maxV).bg}" data-info="${lbl}: ${f2(v)} kWh/m²" aria-label="${lbl}: ${f2(v)} kWh per square metre"></button>`;
      }
      c += `</div></div></div>`;
      host.innerHTML = c;
      const ro = $(el, "#vd-cro");
      host.querySelectorAll(".c[data-info]").forEach((b) => { const s = () => (ro.textContent = b.dataset.info); b.addEventListener("click", s); b.addEventListener("mouseenter", s); b.addEventListener("focus", s); });
      const sm = $(el, "#vd-sm"), smro = $(el, "#vd-smro");
      almanacGrid(sm, { years, cell: (y, m) => { const r = cell(y, m); r.cur = y === curY && m - 1 === m0; return r; }, readout: smro, unit: "kWh/m²", fmt: (v) => Math.round(v) + "", stops: SUN, lo: 0, cur: m0 });
    }
    redraws.push(() => { if (!el.hidden) { const hh = $(el, "#vd-sday"), w = Math.floor(hh.clientWidth); if (w) hh.innerHTML = SD.chart(w); drawCal(); } });
  }

  /* ---------- hero: ported from variant C (model, rules, charts) ---------- */
  const HERO = (function () {
  const TZ = "America/Los_Angeles";
  const H = 3600000, D = 86400000;
  const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];

  /* ---------- tiny helpers ---------- */
  const fmt = (ms, o) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, ...o }).format(ms);
  const dShort = (ms) => fmt(ms, { month: "short", day: "numeric" });
  const dLong = (ms) => fmt(ms, { weekday: "short", month: "short", day: "numeric" });
  const tShort = (ms) => fmt(ms, { hour: "numeric", minute: "2-digit" }).replace(/[\u202f\u00a0]/g, " ").replace(/ (AM|PM)\b/g, (m, x) => " " + x.toLowerCase());
  const TL = (x, y, text, maxW, cls, anchor) => wrapLines(text, maxW).map((l, i) => T(x, y + i * 15, l, cls, anchor)).join("");
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
      o.durMin = o.durH * 60;
      o.archDurMin = e.duration_minutes;
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
  // the one comparison set for ranking an event: closed events with recorded curves of similar duration
  function similarSet(M, durH, startIso) {
    const tryRatio = (lo, hi) => M.pool.filter((e) => { const r = e.duration_minutes / 60 / durH; return r >= lo && r <= hi && e.start_time !== startIso; });
    let sim = tryRatio(0.6, 1.6);
    if (sim.length < 6) sim = tryRatio(0.4, 2.5);
    return sim;
  }
  function rankAmong(M, total, durH, startIso) {
    const sim = similarSet(M, durH, startIso), finals = sim.map((e) => e.total_rainfall);
    return { sim, n: sim.length, finals, medFinal: quant(finals, 0.5), pct: finals.length ? finals.filter((v) => v < total).length / finals.length : 0.5 };
  }
  function similarBand(M, E) {
    const R = rankAmong(M, E.total, E.durH, E.e.start_time), sim = R.sim;
    const xmax = Math.max(E.durH, E.tail ? (M.nowMs - E.start) / H : 0);
    const hs = [];
    for (let h = 0; h <= xmax + 1e-9; h += 0.5) hs.push(h);
    const lo = [], med = [], hi = [];
    hs.forEach((h) => {
      const vals = sim.map((e) => accumAt(e.series, Date.parse(e.start_time), h));
      lo.push(quant(vals, 0.25)); med.push(quant(vals, 0.5)); hi.push(quant(vals, 0.75));
    });
    return { n: R.n, hs, lo, med, hi, finals: R.finals, medFinal: R.medFinal, xmax, pct: R.pct };
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
      s += TL(4, 12, `Soft band: middle half of ${B.n} past events of similar length`, iw - 8, "lb2");
      s += T(x(B.hs[Math.floor(B.hs.length * 0.3)]), y(B.med[Math.floor(B.hs.length * 0.3)]) + 18, `median ${f2(B.medFinal)} in`, "lb2");
      s += "</g>";
      return `<svg class="vd-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
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
      const warm = M.tdep > 0, col = warm ? "var(--vd-hot)" : "var(--temp-cold)";
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
      s += TL(4, y(M.tq1) + 15, `Soft band: middle half of ${M.monthName} readings since 2023`, iw - 8, "lb2");
      s += T(x(ae) - 6, y(M.tmed) - 6, `median ${f1(M.tmed)}°`, "lb2", "end");
      const hw = x(t1) - x(ae);
      if (hw > 70) s += T(x(ae) + 6, ih - 8, "not yet archived", "lb2");
      s += "</g>";
      return `<svg class="vd-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
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
    const sdAll = data.solar.daily_all, lastDay = data.meta.archive_range[1].slice(0, 10), nKeep = sdAll.date.filter((d) => d <= lastDay).length;
    const sd = { date: sdAll.date.slice(0, nKeep), kwh_m2: sdAll.kwh_m2.slice(0, nKeep) }, byMonth = {};
    sd.date.forEach((d, i) => { if (sd.kwh_m2[i] != null) (byMonth[d.slice(0, 7)] = byMonth[d.slice(0, 7)] || []).push(sd.kwh_m2[i]); });
    const monthVals = (mo, notYear) => sd.date.map((d, i) => [d, sd.kwh_m2[i]]).filter(([d, v]) => v != null && Number(d.slice(5, 7)) === mo && Number(d.slice(0, 4)) !== notYear).map(([, v]) => v);
    const usualDay = (mo) => sum(prof(mo).map((q) => q.kwh_m2_per_day));
    return { days, full, prof, monthOf, monthVals, usualDay, sd };
  }
  function solarDayChart(S, key, M, label) {
    return (w) => {
      const titleTxt = `W/m² · soft area: average ${MONTHS[S.monthOf(key) - 1]} day, 2023 to now`, tl = wrapLines(titleTxt, w - 40 - 16 - 8);
      const h = (w < 600 ? 246 : 306) + (tl.length - 1) * 15, m = { t: 28 + (tl.length - 1) * 15, r: 16, b: 28, l: 40 };
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
      s += `<path d="${path(pts)}" class="now" pathLength="1" style="stroke:var(--vd-sun-ink)"/>`;
      const peak = day.reduce((a, b) => (b.w > a.w ? b : a));
      s += `<circle cx="${x(peak.min / 60 + 0.125)}" cy="${y(peak.w)}" r="4.5" style="fill:var(--vd-sun-ink);stroke:var(--ground);stroke-width:2"/>`;
      s += T(x(peak.min / 60 + 0.125) + 8, y(peak.w) - 8, `peak ${Math.round(peak.w)} W/m²`, "nt");
      s += tl.map((l, i) => T(0, -12 - (tl.length - 1 - i) * 15, l, "lb2")).join("");
      if (M && M.cur.solar_wm2 > 0) {
        const lh = Number(fmt(M.nowMs, { hour: "numeric", hourCycle: "h23" })) + Number(fmt(M.nowMs, { minute: "numeric" })) / 60;
        s += `<circle cx="${x(lh)}" cy="${y(M.cur.solar_wm2)}" r="5" style="fill:var(--note);stroke:var(--ground);stroke-width:2"/>`;
      }
      s += "</g>";
      return `<svg class="vd-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
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
      s += `<line x1="0" x2="${iw}" y1="${y(p50)}" y2="${y(p50)}" class="thin" style="stroke:var(--vd-sun-ink);stroke-dasharray:4 3"/>`;
      dates.forEach((d, i) => {
        const cx = slot * (i + 0.5), v = vals[i];
        s += `<rect x="${cx - bw / 2}" y="${y(v)}" width="${bw}" height="${ih - y(v)}" rx="2" style="fill:var(--vd-sun-ink)"/>`;
        if (i % 2 === (n - 1) % 2) s += T(cx, ih + 16, dayShort(d).replace(" ", " "), "tk", "middle");
      });
      const cx = slot * (n + 0.5);
      s += `<rect x="${cx - bw / 2}" y="0" width="${bw}" height="${ih}" rx="2" style="fill:url(#${id})"/>` + T(cx, ih + 16, dayShort(new Date(dayMs(dates[n - 1]) + D).toISOString().slice(0, 10)).replace(" ", " "), "tk", "end");
      s += T(4, y(p75) - 6, `soft band: middle half of ${MONTHS[mo - 1]} days`, "lb2");
      s += "</g>";
      return `<svg class="vd-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${s}</svg>`;
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
      const m = { t: 8, r: 16, b: 8, l: 12 }, iw = w - m.l - m.r;
      const usualTxt = `usual spell: ${C.p25.toFixed(0)} to ${C.p75.toFixed(0)} days (median ${C.p50.toFixed(0)})`, stripTxt = "Last 8 weeks, one cell per day. Darker cells mean more rain.";
      const ul = wrapLines(usualTxt, iw - 8), sl = wrapLines(stripTxt, iw - 8);
      const y1 = 52 + ul.length * 15 + 4, y0 = y1 + sl.length * 15 + 2, h = y0 + 58 - 14;
      const xmax = Math.max(C.n, C.p90 * 1.15, 12), x = lin(0, xmax, 0, iw);
      const id = "vh" + ++uidN;
      let s = `<defs>${hatchDef(id)}</defs><g transform="translate(${m.l},${m.t})">`;
      s += `<rect x="${x(C.p25)}" y="0" width="${x(C.p75) - x(C.p25)}" height="34" rx="3" style="fill:var(--rain-0)"/>`;
      s += `<line x1="${x(C.p50)}" x2="${x(C.p50)}" y1="-2" y2="36" class="thin" style="stroke:var(--rain-2);stroke-dasharray:4 3"/>`;
      s += `<rect x="0" y="9" width="${x(C.n)}" height="16" rx="8" style="fill:var(--rain-3)"/>`;
      s += x(C.n) + 8 > iw - 70 ? `<text x="${x(C.n) - 10}" y="22" text-anchor="end" style="font:600 12.5px var(--sans);fill:var(--surface)">${C.nLabel}</text>` : T(x(C.n) + 8, 22, `${C.nLabel}`, "nt");
      s += TL(4, 52, usualTxt, iw - 8, "lb2");
      // 56-day strip
      const cells = C.strip, cw = iw / cells.length;
      cells.forEach((c, i) => {
        const fill = c.v == null ? `url(#${id})` : c.v < 0.01 ? "var(--surface)" : c.v < 0.1 ? "var(--rain-0)" : c.v < 0.3 ? "var(--rain-1)" : c.v < 0.75 ? "var(--rain-2)" : c.v < 1.5 ? "var(--rain-3)" : "var(--rain-4)";
        s += `<rect x="${(i * cw + 0.5).toFixed(1)}" y="${y0}" width="${(cw - 1.5).toFixed(1)}" height="40" rx="2" style="fill:${fill};stroke:var(--rule);stroke-width:1"/>`;
      });
      s += T(0, y0 + 58, dayShort(cells[0].d), "tk") + T(iw, y0 + 58, dayShort(cells[cells.length - 1].d) + (C.stripNote || ""), "tk", "end");
      s += TL(0, y1 + 11, stripTxt, iw - 8, "lb2");
      s += "</g>";
      return `<svg class="vd-svg" width="${w}" height="${h + 14}" viewBox="0 0 ${w} ${h + 14}" role="img" aria-label="${label}">${s}</svg>`;
    };
  }

  /* ---------- assembled hero + the C charts reused on other tabs ---------- */
  function hero(data, forced) {
    const M = buildModel(data), S = solarModel(data, M), spells = drySpells(M);
    const cur = M.cur;
    /* ----- ranking rules ----- */
    function rules() {
      return [
        { k: "rain", text: `Rain event live or ended in the last 12 h: ${M.raining ? "raining now" : M.rainLead ? `last rain ${durText(M.hoursSinceRain)} ago, event total ${f2(M.liveEv ? M.liveEv.total : cur.rain_event_in)} in` : `last rain ${durText(M.hoursSinceRain)} ago`}`, met: M.rainLead },
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
      const verb = ongoing ? "so far" : "in total";
      const h1 = `${f2(E.total)} in of rain ${verb}, ${word}`;
      const since = real && !ongoing ? `Rain stopped ${durText(M.hoursSinceRain)} ago, at ${tShort(M.lastRain)}.` : ongoing ? "Still raining." : `Event of ${dLong(E.start)}.`;
      const sub = `${since} Past events of similar length (${B.n} with recorded curves) ended at a median ${f2(B.medFinal)} in. The bold line is this event, the soft band is the usual spread.`;
      const facts = [
        [durText(E.durH), "duration"],
        [(E.peak != null ? f2(E.peak) : "–") + " in/hr", E.tail ? "peak, archived part" : "peak rate"],
        [`${Math.round(B.pct * 100)}%`, "of similar events were lighter"],
      ];
      const foot = E.tail
        ? `Source: live station for the total and duration, station archive for the curve. ${f2(E.archTotal)} in archived to ${dShort(M.archEnd)}, ${tShort(M.archEnd)}; the curve after that is not known, so it is hatched, not drawn.`
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
    const auto = autoState();
    const st = forced && forced !== "auto" ? forced : auto;
    return { M, st, auto, L: LEADS[st](), rules: rules() };
  }

  // Events tab: duration x total scatter with the usual spread, highlighting the selected event
  function eventsScatter(data, selId) {
    const M = buildModel(data), live = M.liveEv;
    const rowsData = M.events.map((e) => ({ e, start: Date.parse(e.start_time), end: Date.parse(e.end_time), total: live && e === live.e ? live.total : e.total_rainfall, live: !!(live && e === live.e), dur: live && e === live.e ? live.durH * 60 : e.duration_minutes }));
    const sel = rowsData.find((r) => r.e.event_id === selId) || rowsData[rowsData.length - 1];
    const yr = new Date(sel.start).getUTCFullYear();
    const thisYear = rowsData.filter((r) => new Date(r.start).getUTCFullYear() === yr && r.total > 0);
    const rank = thisYear.filter((r) => r.total > sel.total).length + 1;
    const pool = rowsData.filter((r) => !r.live && r.total > 0 && r.dur > 0 && !r.e.ongoing);
    const RK = rankAmong(M, sel.total, sel.dur / 60, sel.e.start_time); // same comparison set as the hero
    const heavier = RK.n >= 6 ? Math.round(100 * RK.pct) : null;
    const h1 = `${f2(sel.total)} in of rain on ${dShort(sel.start)} makes it the ${ordinal(rank)} wettest of ${thisYear.length} events in ${yr}${heavier != null ? `, heavier than ${heavier}% of events its length` : ""}`;
    const prevBigger = [...rowsData].reverse().find((r) => r !== sel && r.start < sel.start && r.total >= sel.total);
    const sub = `${prevBigger ? `The last earlier event at least this wet was ${dShort(prevBigger.start)} (${f2(prevBigger.total)} in).` : "No earlier event in the record was this wet."} Each dot is a past event by duration and total; the soft band is the usual spread (middle half) for events of that length.${sel.live ? " The selected event is still open in the catalog, so its total is the live reading." : ""}`;
    const chart = (w) => {
      const mm = { r: 16, b: 34, l: 44 }, titleTxt = "Rain in inches (square-root scale) by event duration (log scale)", tl = wrapLines(titleTxt, w - mm.l - mm.r - 8);
      const h = (w < 600 ? 260 : 320) + (tl.length - 1) * 15, m = { t: 12 + (tl.length - 1) * 15, ...mm }, iw = w - m.l - m.r, ih = h - m.t - m.b;
      const xmin = Math.log10(0.5), xmax = Math.log10(Math.max(...pool.map((r) => r.dur / 60), sel.dur / 60) * 1.1);
      const ymaxV = Math.max(...pool.map((r) => r.total), sel.total) * 1.05, sq = Math.sqrt;
      const x = (hrs) => lin(xmin, xmax, 0, iw)(Math.log10(hrs)), y = (v) => lin(0, sq(ymaxV), ih, 0)(sq(v));
      let s = `<g transform="translate(${m.l},${m.t})">`;
      [0.1, 0.5, 1, 2].filter((v) => v < ymaxV).forEach((v) => { s += `<line class="ax" x1="0" x2="${iw}" y1="${y(v)}" y2="${y(v)}"/>` + T(-6, y(v) + 4, v, "tk", "end"); });
      [1, 3, 6, 12, 24, 48, 96].filter((v) => Math.log10(v) <= xmax).forEach((v) => { s += `<line class="ax" x1="${x(v)}" x2="${x(v)}" y1="${ih}" y2="${ih + 4}"/>` + T(x(v), ih + 17, v >= 24 ? v / 24 + " d" : v + " h", "tk", "middle"); });
      const sorted = [...pool].sort((a, b) => a.dur - b.dur), nb = 6, size = Math.ceil(sorted.length / nb), bins = [];
      for (let i = 0; i < sorted.length; i += size) { const g = sorted.slice(i, i + size); if (g.length >= 4) bins.push({ x: x(quant(g.map((r) => r.dur / 60), 0.5)), lo: quant(g.map((r) => r.total), 0.25), hi: quant(g.map((r) => r.total), 0.75), med: quant(g.map((r) => r.total), 0.5) }); }
      s += `<path d="${path(bins.map((b) => [b.x, y(b.hi)]))}L${path(bins.map((b) => [b.x, y(b.lo)]).reverse()).slice(1)}Z" style="fill:var(--rain-0);opacity:.95"/>`;
      s += `<path d="${path(bins.map((b) => [b.x, y(b.med)]))}" class="thin" style="stroke:var(--rain-2);stroke-dasharray:4 3"/>`;
      pool.forEach((r) => (s += `<circle cx="${x(r.dur / 60)}" cy="${y(r.total)}" r="2.6" style="fill:var(--rain-2);opacity:.45"/>`));
      const sx = x(Math.max(0.5, sel.dur / 60)), left = sx < iw * 0.35;
      s += `<circle cx="${sx}" cy="${y(sel.total)}" r="6.5" style="fill:var(--rain-3);stroke:var(--ground);stroke-width:2.5"/>`;
      s += T(sx + (left ? 12 : -12), y(sel.total) - 10, `${dShort(sel.start)}: ${f2(sel.total)} in`, "nt", left ? "start" : "end");
      s += tl.map((l, i) => T(4, 14 - (tl.length - 1) * 15 + i * 15, l, "lb2")).join("");
      s += "</g>";
      return `<svg class="vd-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${h1}">${s}</svg>`;
    };
    return { h1, sub, chart };
  }

  // Solar tab: the latest full day's irradiance against the average day of that month
  function solarDay(data) {
    const M = buildModel(data), S = solarModel(data, M);
    const key = S.full[S.full.length - 1], mo = S.monthOf(key), U = S.usualDay(mo), tot = S.days[key].total, ratio = tot / U;
    const h1 = `${dayDow(key)} ${dayShort(key)} brought ${f2(tot)} kWh/m², ${ratio >= 1 ? f1(ratio) + "× " : Math.round(ratio * 100) + "% of "}a usual ${MONTHS[mo - 1]} day`;
    const sd = S.sd, N = sd.date.length, last14 = sd.kwh_m2.slice(N - 14), mv = S.monthVals(mo, +key.slice(0, 4));
    const sub = `Latest full day in the archive. In bold, the day's irradiance; behind it, the average ${MONTHS[mo - 1]} day. The last 14 days averaged ${f2(sum(last14) / 14)} kWh/m² a day against a ${MONTHS[mo - 1]} median of ${f2(quant(mv, 0.5))}.`;
    return { h1, sub, chart: solarDayChart(S, key, null, h1), foot: `Source: station archive, 15-minute solar energy. The archive ends ${dShort(M.archEnd)}, ${tShort(M.archEnd)}, so ${dShort(M.nowMs)} is not included.` };
  }

  return { hero, eventsScatter, solarDay, model: buildModel, dShort, tShort, durText, f1, f2 };
})();

  window.LOOKOUT_VARIANT = {
    name: "Converged",
    render(root, data) { build(root, data); },
  };
})();
