// Lookout widget for Scriptable (https://scriptable.app).
//
// Shows one station's current reading against its own history, in the view
// the snapshot picks (`widget_view`): the live rain event, a dry spell longer
// than usual, or now against the temperature ranges of today, 7, 30 and 365
// days. Python decides every number; this script only draws it.
//
// Widget parameter: a station name from the station index ("Salmon Run"),
// matched ignoring case and spaces. Leave it empty for All stations: the
// station with the most going on (rain event, then dry spell, then
// temperature ranges; the default station breaks a tie).
//
// The logic above `main` runs in Node too, for the tests
// (web/tests/scriptable-widget.test.ts). See widgets/scriptable/README.md.

const BASE_URL = "https://lookout.stonecover.com";
const REFRESH_MINUTES = 5;

const VIEW_RANK = { rain_event: 0, dry_spell: 1, temperature_ranges: 2 };
const RANGE_ROWS = [
  ["today", "Today"],
  ["7d", "7 days"],
  ["30d", "30 days"],
  ["365d", "365 days"],
];
const WINDOW_NAMES = {
  today: "today",
  yesterday: "yesterday",
  "7d": "the last 7 days",
  "30d": "the last 30 days",
  "90d": "the last 90 days",
  "365d": "the last 365 days",
};

// ---------- logic (no Scriptable APIs) ----------

/** A name as typed or stored, for matching: trimmed, single-spaced, lower case. */
function normalize(text) {
  return String(text ?? "")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

/** "A", "A or B", "A, B or C". */
function orList(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} or ${items[items.length - 1]}`;
}

/**
 * The station a widget parameter names.
 *
 * @param stations - The station index's stations, default first.
 * @param parameter - The widget parameter: a name, a MAC, or empty.
 * @returns `{all: true}`, `{station}`, or `{error}` naming the real stations.
 */
function matchStation(stations, parameter) {
  const wanted = normalize(parameter);
  if (wanted === "") return { all: true };
  const station = stations.find(
    (s) => normalize(s.name) === wanted || normalize(s.mac) === wanted,
  );
  if (station) return { station };
  const names = stations.map((s) => s.name ?? s.mac);
  return {
    error: `No station named "${String(parameter).trim()}". Try ${orList(names)}.`,
  };
}

/**
 * The entry whose view ranks first; the earliest in index order on a tie.
 *
 * @param entries - One per station, in index order, each with its `view`.
 * @returns The leading entry.
 */
function leadingStation(entries) {
  const rank = (e) => VIEW_RANK[e.view] ?? VIEW_RANK.temperature_ranges;
  return entries.reduce((best, e) => (rank(e) < rank(best) ? e : best));
}

/** Minutes as the site's exact short form: "45 m", "5 h 34 m", "1 d 1 h". */
function formatDuration(minutes) {
  const total = Math.round(minutes);
  const days = Math.floor(total / 1440);
  const hours = Math.floor((total % 1440) / 60);
  const mins = total % 60;
  if (days > 0) return hours > 0 ? `${days} d ${hours} h` : `${days} d`;
  if (hours > 0) return mins > 0 ? `${hours} h ${mins} m` : `${hours} h`;
  return `${mins} m`;
}

/** A number of days: whole when whole, else one decimal. */
function dayCount(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/** The view a dashboard names, or the default for one published before it. */
function viewOf(dashboard) {
  return dashboard.widget_view in VIEW_RANK
    ? dashboard.widget_view
    : "temperature_ranges";
}

function temperatureModel(dashboard) {
  const ctx = dashboard.temperature_context;
  const byKey = new Map(ctx.periods.map((p) => [p.status.key, p]));
  const rows = RANGE_ROWS.map(([key, label]) => {
    const p = byKey.get(key);
    const valid = p && p.status.status !== "invalid";
    return {
      key,
      label,
      low: valid ? p.low : null,
      q1: valid ? p.q1 : null,
      q3: valid ? p.q3 : null,
      high: valid ? p.high : null,
    };
  });
  const found = byKey.get(ctx.finding);
  let sentence = "No window complete enough to compare";
  if (ctx.now === null) sentence = "No current temperature";
  else if (found && found.position !== null) {
    sentence = `${found.position} the middle half of ${WINDOW_NAMES[ctx.finding] ?? ctx.finding}`;
    sentence = sentence[0].toUpperCase() + sentence.slice(1);
  }
  return { view: "temperature_ranges", now: ctx.now, rows, sentence };
}

function rainEventModel(dashboard) {
  const active = dashboard.active_rain_event;
  const events = dashboard.rain_events;
  const live = events.events.find((e) => e.live);
  const curve = events.open_curve
    ? events.open_curve.t_ms
        .map((t, i) => ({ t, inches: events.open_curve.event_in[i] }))
        .filter((p) => p.inches !== null)
    : [];
  return {
    view: "rain_event",
    now: active.total_in,
    totalIn: active.total_in,
    peakInHr: live ? live.peak_rate_in_hr : null,
    medianIn: events.all_time.median_in,
    curve,
    sentence: `${active.total_in.toFixed(2)} in over ${formatDuration(
      active.duration_minutes,
    )}, still raining`,
  };
}

function drySpellModel(dashboard) {
  const calm = dashboard.hero_lead.rules.find((r) => r.kind === "calm");
  const daysDry = Math.floor(calm.value);
  const usual = calm.reference;
  const longest = dashboard.rain_tab.dry_spells.longest;
  return {
    view: "dry_spell",
    now: daysDry,
    daysDry,
    usualDays: usual,
    longestDays: longest ? longest.days : null,
    sentence: `Dry for ${daysDry} days, longer than the usual ${dayCount(usual)}`,
  };
}

/**
 * What the widget draws for one station's dashboard.
 *
 * @param dashboard - The station's `/dashboard` document.
 * @returns The view, the station name, the sentence and the view's data.
 */
function widgetModel(dashboard) {
  const view = viewOf(dashboard);
  const calm = dashboard.hero_lead?.rules?.find((r) => r.kind === "calm");
  let model;
  if (view === "rain_event" && dashboard.active_rain_event)
    model = rainEventModel(dashboard);
  else if (
    view === "dry_spell" &&
    calm?.value != null &&
    calm?.reference != null
  )
    model = drySpellModel(dashboard);
  else model = temperatureModel(dashboard);
  return {
    ...model,
    name: dashboard.station.name ?? dashboard.station.mac,
    mac: dashboard.station.mac,
    tz: dashboard.station.tz,
    observedAtMs: dashboard.observed_at_ms,
  };
}

// ---------- drawing (Scriptable) ----------

const PALETTES = {
  dark: {
    tile: "#1a2129",
    ink: "#eef2f5",
    ink2: "#c3ccd5",
    ink3: "#aab5c0",
    rule: "#2a333d",
    ruleStrong: "#6b7886",
    now: "#f2a65e",
    rainLine: "#d6e9fb",
    rainBar: "#7fb0e0",
    band: "#606d7a",
    bandEdge: "#606d7a",
  },
  light: {
    tile: "#ffffff",
    ink: "#11161c",
    ink2: "#353e48",
    ink3: "#4d5763",
    rule: "#e1e5e9",
    ruleStrong: "#7d8995",
    now: "#a74b11",
    rainLine: "#1d4a7a",
    rainBar: "#3b6fa6",
    band: "#e8ecf0",
    bandEdge: "#7d8995",
  },
};

function palette() {
  return PALETTES[Device.isUsingDarkAppearance() ? "dark" : "light"];
}

async function getJSON(path) {
  const request = new Request(BASE_URL + path);
  request.timeoutInterval = 20;
  return request.loadJSON();
}

function timeText(ms, tz) {
  if (ms == null) return "";
  return new Date(ms)
    .toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      timeZone: tz,
    })
    .toLowerCase();
}

function addText(stack, text, font, color, lines = 1) {
  const t = stack.addText(text);
  t.font = font;
  t.textColor = new Color(color);
  t.lineLimit = lines;
  t.minimumScaleFactor = 0.7;
  return t;
}

function headline(model) {
  if (model.view === "rain_event") return `${model.totalIn.toFixed(2)} in`;
  if (model.view === "dry_spell") return `${model.daysDry} d`;
  return model.now === null ? "–" : `${Math.round(model.now)}°`;
}

function kicker(model) {
  if (model.view === "rain_event") return `Raining · ${model.name}`;
  if (model.view === "dry_spell") return `Dry · ${model.name}`;
  return model.name;
}

/** A chart image the width of the widget's content. */
function chart(model, width, height, c, small) {
  const dc = new DrawContext();
  dc.size = new Size(width, height);
  dc.opaque = false;
  dc.respectScreenScale = true;
  if (model.view === "rain_event") drawRain(dc, model, width, height, c);
  else if (model.view === "dry_spell")
    drawDry(dc, model, width, height, c, small);
  else drawRanges(dc, model, width, height, c, small);
  return dc.getImage();
}

function line(dc, x1, y1, x2, y2, color, width) {
  const p = new Path();
  p.move(new Point(x1, y1));
  p.addLine(new Point(x2, y2));
  dc.addPath(p);
  dc.setStrokeColor(new Color(color));
  dc.setLineWidth(width);
  dc.strokePath();
}

function box(dc, x, y, w, h, fill, edge) {
  const p = new Path();
  p.addRoundedRect(new Rect(x, y, Math.max(w, 2), h), 3, 3);
  dc.addPath(p);
  dc.setFillColor(new Color(fill));
  dc.fillPath();
  if (edge) {
    dc.addPath(p);
    dc.setStrokeColor(new Color(edge));
    dc.setLineWidth(1);
    dc.strokePath();
  }
}

function label(dc, text, x, y, w, color, size = 12, align = "left") {
  dc.setFont(Font.systemFont(size));
  dc.setTextColor(new Color(color));
  if (align === "right") dc.setTextAlignedRight();
  else dc.setTextAlignedLeft();
  dc.drawTextInRect(text, new Rect(x, y, w, size + 4));
}

function drawRanges(dc, model, width, height, c, small) {
  const rows = small ? model.rows.slice(0, 1) : model.rows;
  const known = rows.filter((r) => r.low !== null);
  const values = known.flatMap((r) => [r.low, r.high]).concat(model.now ?? []);
  if (values.length === 0) return;
  const lo = Math.min(...values) - 2;
  const hi = Math.max(...values) + 2;
  const left = small ? 4 : 64;
  const right = width - 4;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (right - left);
  const rowH = small ? height - 16 : height / rows.length;
  rows.forEach((r, i) => {
    const y = i * rowH + rowH / 2;
    if (!small) label(dc, r.label, 0, y - 8, 60, c.ink2);
    if (r.low === null) {
      label(dc, "not enough data", left, y - 8, right - left, c.ink3);
      return;
    }
    line(dc, x(r.low), y, x(r.high), y, c.ruleStrong, 1.5);
    box(dc, x(r.q1), y - 5, x(r.q3) - x(r.q1), 10, c.band, c.bandEdge);
  });
  if (small && known.length) {
    const r = known[0];
    label(dc, `${Math.round(r.low)}°`, 0, height - 15, 40, c.ink3);
    label(
      dc,
      `${Math.round(r.high)}°`,
      width - 40,
      height - 15,
      40,
      c.ink3,
      12,
      "right",
    );
  }
  if (model.now !== null) {
    const bottom = small ? height - 16 : height;
    line(dc, x(model.now), 0, x(model.now), bottom, c.now, 2.5);
  }
}

function drawRain(dc, model, width, height, c) {
  const pts = model.curve;
  const top = Math.max(model.totalIn, model.medianIn ?? 0) * 1.15 || 0.1;
  const plotH = height - 16;
  const y = (v) => plotH - (v / top) * (plotH - 4);
  line(dc, 0, plotH, width, plotH, c.ruleStrong, 1);
  if (model.medianIn !== null) {
    const my = y(model.medianIn);
    for (let x0 = 0; x0 < width; x0 += 11)
      line(dc, x0, my, Math.min(x0 + 6, width), my, c.ink2, 1.5);
    label(
      dc,
      `median ${model.medianIn.toFixed(2)} in`,
      0,
      Math.max(my - 17, 0),
      width,
      c.ink2,
    );
  }
  if (pts.length > 1) {
    const t0 = pts[0].t;
    const t1 = pts[pts.length - 1].t || t0 + 1;
    const x = (t) => ((t - t0) / Math.max(t1 - t0, 1)) * (width - 10);
    const p = new Path();
    p.move(new Point(x(pts[0].t), y(pts[0].inches)));
    for (let i = 1; i < pts.length; i++) {
      p.addLine(new Point(x(pts[i].t), y(pts[i - 1].inches)));
      p.addLine(new Point(x(pts[i].t), y(pts[i].inches)));
    }
    dc.addPath(p);
    dc.setStrokeColor(new Color(c.rainLine));
    dc.setLineWidth(2.5);
    dc.strokePath();
    const last = pts[pts.length - 1];
    dc.setFillColor(new Color(c.now));
    dc.fillEllipse(new Rect(x(last.t) - 4.5, y(last.inches) - 4.5, 9, 9));
  }
  const peak =
    model.peakInHr === null ? "" : `peak ${model.peakInHr.toFixed(2)} in/hr`;
  label(dc, peak, 0, height - 14, width, c.ink3);
}

function drawDry(dc, model, width, height, c, small) {
  const bars = [
    ["now", model.daysDry, c.ink],
    ["usual", model.usualDays, c.ruleStrong],
  ];
  if (model.longestDays !== null)
    bars.push(["record", model.longestDays, c.ruleStrong]);
  const shown = small ? bars.slice(0, 2) : bars;
  const max = Math.max(...shown.map((b) => b[1])) || 1;
  const left = 48;
  const rowH = height / shown.length;
  shown.forEach(([name, days, color], i) => {
    const y = i * rowH + rowH / 2;
    label(dc, name, 0, y - 8, left - 4, name === "now" ? c.now : c.ink2);
    const w = ((width - left - 40) * days) / max;
    box(dc, left, y - 6, w, 12, color);
    label(dc, `${dayCount(days)} d`, left + w + 6, y - 8, 40, c.ink3);
  });
}

function buildWidget(model, family) {
  const c = palette();
  const small = family === "small";
  const w = new ListWidget();
  w.backgroundColor = new Color(c.tile);
  w.setPadding(14, 16, 12, 16);
  w.url = `${BASE_URL}/?station=${encodeURIComponent(model.mac)}`;
  w.refreshAfterDate = new Date(Date.now() + REFRESH_MINUTES * 60_000);

  const head = w.addStack();
  head.layoutHorizontally();
  head.centerAlignContent();
  const left = head.addStack();
  left.layoutVertically();
  addText(left, kicker(model), Font.semiboldSystemFont(13), c.ink);
  if (!small)
    addText(left, model.sentence, Font.mediumSystemFont(13), c.ink, 2);
  head.addSpacer();
  if (!small) addText(head, headline(model), Font.lightSystemFont(30), c.ink);

  if (small) {
    addText(w, headline(model), Font.lightSystemFont(40), c.ink);
    addText(w, model.sentence, Font.mediumSystemFont(12), c.ink2, 2);
  }
  w.addSpacer(6);
  const width = small ? 126 : 306;
  const height = small ? 30 : 74;
  const img = w.addImage(chart(model, width, height, c, small));
  img.imageSize = new Size(width, height);
  w.addSpacer();
  addText(
    w,
    timeText(model.observedAtMs, model.tz),
    Font.systemFont(12),
    c.ink3,
  );
  return w;
}

function messageWidget(text) {
  const c = palette();
  const w = new ListWidget();
  w.backgroundColor = new Color(c.tile);
  w.setPadding(14, 16, 14, 16);
  addText(w, "Lookout", Font.semiboldSystemFont(13), c.ink);
  w.addSpacer(6);
  addText(w, text, Font.systemFont(13), c.ink2, 4);
  w.refreshAfterDate = new Date(Date.now() + REFRESH_MINUTES * 60_000);
  return w;
}

async function main() {
  let widget;
  try {
    const index = await getJSON("/stations");
    const choice = matchStation(index.stations, args.widgetParameter);
    let dashboard;
    if (choice.error) widget = messageWidget(choice.error);
    else if (choice.all) {
      const docs = await Promise.all(
        index.stations.map((s) =>
          getJSON(`/dashboard?station=${encodeURIComponent(s.mac)}`),
        ),
      );
      dashboard = leadingStation(
        docs.map((doc) => ({ doc, view: viewOf(doc) })),
      ).doc;
    } else {
      dashboard = await getJSON(
        `/dashboard?station=${encodeURIComponent(choice.station.mac)}`,
      );
    }
    if (dashboard)
      widget = buildWidget(
        widgetModel(dashboard),
        config.widgetFamily ?? "medium",
      );
  } catch (error) {
    widget = messageWidget(
      "Couldn't load Lookout. It will try again in a few minutes.",
    );
  }
  if (config.runsInWidget) Script.setWidget(widget);
  else await widget.presentMedium();
  Script.complete();
}

if (typeof Script === "undefined") {
  module.exports = {
    matchStation,
    leadingStation,
    widgetModel,
    formatDuration,
  };
} else {
  main();
}
