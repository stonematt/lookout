/**
 * The mosaic's last six tiles (#473, spec #460 stories 19, 24–27): Wind,
 * Pressure, Humidity, Sun, Daylight and Inside. Humidity carries the dew
 * point; Sun is the day's solar energy and Daylight sunrise, sunset and
 * day length. The moon and the full sentences are the zooms' (#476, #477).
 *
 * - Wind: the live speed, where it blows from and its gust; its mini chart
 *   a compass whose arrow points the way the wind blows.
 * - Pressure: the live barometer reading and the snapshot's 3-hour trend
 *   word (`barometer_trend`, decided in Python); its mini chart the last 24
 *   hours (`outdoor_trace`, #554), hourly, the newest hour as the now dot.
 * - Humidity: the live humidity and the dew point; its mini chart the last
 *   24 hours of humidity, the same way.
 * - Sun today: today's solar energy so far and yesterday's; its mini chart
 *   is the Solar tab's lead chart cropped, the last 24 hours over the usual
 *   range at each time of day.
 * - Daylight: the day's length and its change from yesterday; its mini
 *   chart an arc from sunrise to sunset, the sun's place in the day as the
 *   now dot. Not drawn for a station with no location (`sun_moon` null).
 * - Inside: one row per indoor sensor (`indoor`, #466): its temperature,
 *   humidity and 24-hour line. Not drawn for a station with none.
 *
 * Pressure and Humidity draw no usual band: the snapshot holds none for
 * either yet. A snapshot without `outdoor_trace` (null, or N−1) draws no
 * chart for them.
 *
 * Every value is the snapshot's; these only word, format and draw. A missing
 * live reading reads "n/a", never zero.
 *
 * Offers `windClauses`, `calmThreshold`, `windTile`, `pressureTile`,
 * `humidityTile`, `sunTile`, `daylightTile`, `insideTile`, `sensorRow` (one
 * indoor sensor's words), `lastIndex` and `traceMarks` (a series' line and
 * now dot), which the Inside zoom shares, and `outdoorChart` with
 * `PRESSURE_PAD` and `HUMIDITY_PAD` (an `outdoor_trace` series as a mini
 * chart), which the Pressure and Humidity zooms share.
 */
import { runs, scale } from "../charts/svg.js";
import { svgEl } from "../dom.js";
import {
  compassPoint,
  degrees,
  formatHours,
  formatTime,
  inHg,
  kwhPerM2,
  mph,
  percent,
} from "../format.js";
import { gapValue } from "../gaps.js";
import type { DashboardSnapshot, IndoorSensor, SolarWindow, SunMoon } from "../snapshot.js";
import type { MosaicRow, MosaicTile } from "./mosaic.js";
import { MINI_HEIGHT, ROW_LINE_HEIGHT, miniChart } from "./mosaic.js";

/** The strip under a mini chart that holds its end labels. */
const AXIS = 16;

/** Where the plot ends above the label strip. */
const PLOT_BOTTOM = MINI_HEIGHT - AXIS;

/** The baseline of a label in the strip. */
const LABEL_Y = MINI_HEIGHT - 2;

/** Points as an SVG `points` list: "x,y x,y". */
function pointList(points: readonly (readonly [number, number])[]): string {
  return points.map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
}

/** A line's points, split into runs at its nulls, as polyline point lists. */
function polylines(
  values: (number | null)[],
  x: (i: number) => number,
  y: (v: number) => number,
): string[] {
  return runs(values.length, (i) => values[i] !== null).map(([a, b]) => {
    const points: [number, number][] = [];
    for (let i = a; i <= b; i++) points.push([x(i), y(values[i]!)]);
    return pointList(points);
  });
}

/** A chart's two end labels in the strip under it. */
function endLabels(svg: SVGSVGElement, width: number, left: string, right: string): void {
  svg.append(
    svgEl("text", { class: "mosaic-label", x: 0, y: LABEL_Y }, left),
    svgEl("text", { class: "mosaic-label", x: width, y: LABEL_Y, "text-anchor": "end" }, right),
  );
}

/** The index of a series' last reading, or -1. */
export function lastIndex(values: (number | null)[]): number {
  for (let i = values.length - 1; i >= 0; i--) if (values[i] !== null) return i;
  return -1;
}

/**
 * A series' line, split into runs at its nulls, then its last reading as
 * the now dot: the Inside tile's row lines and the Inside zoom's 24 hours.
 *
 * @param values - The series; it holds a reading (`lastIndex` >= 0).
 * @param x - Index to x.
 * @param y - Value to y.
 * @param r - The now dot's radius.
 * @returns The polylines, then the dot.
 */
export function traceMarks(
  values: (number | null)[],
  x: (i: number) => number,
  y: (v: number) => number,
  r: number,
): SVGElement[] {
  const last = lastIndex(values);
  return [
    ...polylines(values, x, y).map((points) => svgEl("polyline", { class: "mosaic-line", points })),
    svgEl("circle", { class: "mosaic-now-dot", cx: x(last), cy: y(values[last]!), r }),
  ];
}

// ---------- Wind ----------

/** The compass: a dial, N at the top, an arrow the way the wind blows. */
function windChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const from = snapshot.current.wind_dir_deg;
  // A calm wind has no way to blow: no arrow (#598).
  if (from === null || isCalm(snapshot.current)) return null;
  const svg = miniChart(width);
  const r = MINI_HEIGHT / 2 - 2;
  const cx = width / 2;
  const cy = MINI_HEIGHT / 2;
  svg.append(
    svgEl("circle", { class: "mosaic-dial", cx, cy, r }),
    svgEl("text", { class: "mosaic-label", x: cx, y: cy - r + 13, "text-anchor": "middle" }, "N"),
  );
  // The wind blows toward the point opposite where it comes from.
  const to = ((from + 180) * Math.PI) / 180;
  const [dx, dy] = [Math.sin(to), -Math.cos(to)];
  // The head stops short of the rim, clear of the N.
  const head = [cx + dx * (r - 15), cy + dy * (r - 15)] as const;
  const tail = [cx - dx * (r - 12), cy - dy * (r - 12)] as const;
  const base = [cx + dx * (r - 24), cy + dy * (r - 24)] as const;
  const [px, py] = [-dy * 5, dx * 5];
  svg.append(
    svgEl("line", { class: "mosaic-arrow", x1: tail[0], y1: tail[1], x2: base[0], y2: base[1] }),
    svgEl("polygon", {
      class: "mosaic-arrow-head",
      points: pointList([head, [base[0] + px, base[1] + py], [base[0] - px, base[1] - py]]),
    }),
  );
  return svg;
}

/**
 * Whether the live wind is calm: a speed of exactly 0 (#598).
 *
 * TODO(#597): Python will null `wind_dir_deg` below `calm_below_mph`. Then
 * "calm" keys on a null direction with a known speed, not on a speed of 0,
 * and this test goes.
 *
 * @param c - The current conditions.
 * @returns True at a speed of exactly 0.
 */
export function isCalm(c: DashboardSnapshot["current"]): boolean {
  return c.wind_speed_mph === 0;
}

/**
 * A calm wind's gust clause: "no gusts" at a gust of 0, the gust when it is
 * above 0, "gust not known" with none.
 *
 * @param c - The current conditions.
 * @param gust - The `gust` clause from `windClauses`.
 * @returns The clause.
 */
export function calmGust(c: DashboardSnapshot["current"], gust: string | null): string {
  if (c.wind_gust_mph === null) return "gust not known";
  return c.wind_gust_mph > 0 && gust ? gust : "no gusts";
}

/**
 * The live wind's two clauses, each null when not known: "from the SE" and
 * "gusts to 2.5 mph". The tile's delta and the wind zoom's sentence share
 * them. A calm wind has no direction: `from` is null and `calm` is true.
 *
 * @param c - The current conditions.
 * @returns The clauses.
 */
export function windClauses(c: DashboardSnapshot["current"]): {
  calm: boolean;
  from: string | null;
  gust: string | null;
} {
  const calm = isCalm(c);
  return {
    calm,
    from: calm || c.wind_dir_deg === null ? null : `from the ${compassPoint(c.wind_dir_deg)}`,
    gust: c.wind_gust_mph === null ? null : `gusts to ${mph(c.wind_gust_mph)}`,
  };
}

/**
 * The calm threshold in words, as the snapshot gives it: "1 mph", "0.5 mph".
 *
 * @param value - `wind_rose.calm_below_mph`.
 * @returns The words.
 */
export function calmThreshold(value: number): string {
  return `${Number.isInteger(value) ? value : value.toFixed(1)} mph`;
}

/** Wind: the live speed, where it blows from and its gust. */
export const windTile: MosaicTile = {
  slug: "wind",
  depth: "wind",
  title: "Wind",
  zoomName: "wind",
  read(snapshot) {
    const c = snapshot.current;
    const { calm, from, gust } = windClauses(c);
    // Calm: the word, no direction and no unit (#598).
    if (calm) return { value: "calm", delta: calmGust(c, gust) };
    let delta: string;
    if (from && gust) delta = `${from}, ${gust}`;
    else if (from) delta = `${from}, gust not known`;
    else if (gust) delta = `direction not known, ${gust}`;
    else delta = "direction and gust not known";
    const speed = c.wind_speed_mph;
    return speed === null
      ? { value: "n/a", delta }
      : { value: mph(speed, { unit: false }), unit: "mph", delta };
  },
  chart: windChart,
  source(snapshot) {
    const live =
      "Speed and gust in mph. The direction is where the wind blows from, to the nearest of 16 " +
      "compass points; the compass arrow points the way it blows. A speed of 0 is calm: no direction, no arrow.";
    const rose = snapshot.wind_rose;
    if (!rose) return { line: "live speed, direction and gust.", method: live };
    return {
      line: "live speed, direction and gust; the rose from each window's readings.",
      method:
        `${live} The rose sorts each window's readings by the direction the wind came ` +
        "from: each direction's average wind and its highest gust. A reading under " +
        `${calmThreshold(rose.calm_below_mph)}, or with no direction, is calm and left out of ` +
        "the rose.",
    };
  },
};

// ---------- Pressure and Humidity: the last 24 hours ----------

/** The least room above and below the pressure line: 0.05 inHg. */
export const PRESSURE_PAD = 0.05;

/** The least room above and below the humidity line: 3 points. */
export const HUMIDITY_PAD = 3;

/**
 * One series of `outdoor_trace` (#554), the last 24 hours hourly, as a
 * mini chart: its line and its newest hour as the now dot, over "24 h" and
 * "now". The y range is the series' own, widened by `pad` either side, so a
 * steady day reads as a flat line in the middle.
 *
 * @param values - The series, or undefined when the snapshot has no trace.
 * @param width - The width to draw at.
 * @param pad - The least room above and below the line, in the series' unit.
 * @returns The chart, or null with no trace or no reading in it.
 */
export function outdoorChart(
  values: (number | null)[] | undefined,
  width: number,
  pad: number,
): SVGSVGElement | null {
  if (!values || values.length < 2) return null;
  const last = lastIndex(values);
  if (last < 0) return null;
  const numbers = values.filter((v): v is number => v !== null);
  const x = scale(0, values.length - 1, 2, width - 6);
  const y = scale(Math.min(...numbers) - pad, Math.max(...numbers) + pad, PLOT_BOTTOM - 4, 6);
  const svg = miniChart(width);
  svg.append(
    svgEl("line", { class: "mosaic-base", x1: 0, x2: width, y1: PLOT_BOTTOM, y2: PLOT_BOTTOM }),
  );
  svg.append(...traceMarks(values, x, y, 4));
  endLabels(svg, width, "24 h", "now");
  return svg;
}

// ---------- Pressure ----------

/** Pressure: the live barometer reading and its 3-hour trend word. */
export const pressureTile: MosaicTile = {
  slug: "pressure",
  depth: "pressure",
  title: "Pressure",
  zoomName: "pressure",
  read(snapshot) {
    const c = snapshot.current;
    const trend = c.barometer_trend;
    const delta = trend === null ? "3-hour trend not known" : `${trend} over 3 hours`;
    return c.barometer_inhg === null
      ? { value: "n/a", delta }
      : { value: inHg(c.barometer_inhg), unit: "inHg", delta };
  },
  chart: (snapshot, width) =>
    outdoorChart(snapshot.outdoor_trace?.pressure_inhg, width, PRESSURE_PAD),
  source: () => ({
    line: "live sea-level barometer and its last 24 hours; trend over 3 hours.",
    method:
      "The trend is the change over the last 3 hours: rising, falling or steady. The line is " +
      "the last 24 hours, each hour's mean.",
  }),
};

// ---------- Humidity ----------

/** Humidity: the live humidity and the dew point (the Air card's). */
export const humidityTile: MosaicTile = {
  slug: "humidity",
  depth: "humidity",
  title: "Humidity",
  zoomName: "humidity",
  read(snapshot) {
    const c = snapshot.current;
    return {
      value: c.humidity_pct === null ? "n/a" : percent(c.humidity_pct),
      delta: c.dew_point_f === null ? "dew point not known" : `dew point ${degrees(c.dew_point_f)}`,
    };
  },
  chart: (snapshot, width) =>
    outdoorChart(snapshot.outdoor_trace?.humidity_pct, width, HUMIDITY_PAD),
  source: () => ({
    line: "outdoor sensor and dew point.",
    method:
      "Relative humidity outdoors, and the dew point: the temperature at which that air would " +
      "be saturated. The line is the last 24 hours, each hour's mean.",
  }),
};

// ---------- Sun ----------

/** A solar window by key. */
function solarWindow(snapshot: DashboardSnapshot, key: string): SolarWindow | null {
  return snapshot.solar_tab?.windows.find((w) => w.key === key) ?? null;
}

/** A window's energy as its gap status allows, bare: "2.41", "≈ 61.2"; or null. */
function energyText(window: SolarWindow | null): string | null {
  if (!window || window.energy_kwh_m2 === null) return null;
  return gapValue(window.status, kwhPerM2(window.energy_kwh_m2, { unit: false }));
}

/** The last 24 hours of solar radiation over its usual range, now as the dot. */
function sunChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const trace = snapshot.solar_tab?.last_24h;
  if (!trace) return null;
  const values = trace.solar_wm2;
  const n = values.length;
  const last = lastIndex(values);
  if (n < 2 || last < 0) return null;
  const low = trace.usual_low_wm2;
  const high = trace.usual_high_wm2;
  const numbers = [...values, ...(high ?? [])].filter((v): v is number => v !== null);
  const x = scale(0, n - 1, 2, width - 6);
  const y = scale(0, Math.max(100, ...numbers), PLOT_BOTTOM, 6);
  const svg = miniChart(width);
  svg.append(
    svgEl("line", { class: "mosaic-base", x1: 0, x2: width, y1: PLOT_BOTTOM, y2: PLOT_BOTTOM }),
  );
  if (low && high) {
    for (const [a, b] of runs(n, (i) => low[i] !== null && high[i] !== null)) {
      const upper: [number, number][] = [];
      const lower: [number, number][] = [];
      for (let i = a; i <= b; i++) {
        upper.push([x(i), y(high[i]!)]);
        lower.unshift([x(i), y(low[i]!)]);
      }
      svg.append(
        svgEl("polygon", { class: "mosaic-sun-band", points: pointList([...upper, ...lower]) }),
      );
    }
  }
  for (const points of polylines(values, x, y)) {
    svg.append(svgEl("polyline", { class: "mosaic-sun-line", points }));
  }
  svg.append(svgEl("circle", { class: "mosaic-now-dot", cx: x(last), cy: y(values[last]!), r: 4 }));
  endLabels(svg, width, "24 h", "now");
  return svg;
}

/** Sun today: today's solar energy so far, with yesterday's whole day. */
export const sunTile: MosaicTile = {
  slug: "sun",
  depth: "sun",
  title: "Sun today",
  zoomName: "sun",
  shown: (snapshot) => snapshot.solar_tab !== null,
  read(snapshot) {
    const today = energyText(solarWindow(snapshot, "today"));
    const yesterday = energyText(solarWindow(snapshot, "yesterday"));
    const delta = yesterday === null ? null : `all of yesterday ${yesterday} kWh/m²`;
    return today === null
      ? { value: null, delta: delta ?? "not compared yet" }
      : { value: today, unit: "kWh/m²", delta };
  },
  chart: sunChart,
  source: () => ({
    line: "solar sensor, against yesterday and the last 30 days.",
    method:
      "Today's energy so far and yesterday's whole day, in kWh per m²; ≈ marks a total scaled " +
      "up over a little missing time. The line is the last 24 hours of solar radiation, over its " +
      "usual range at each time of day across the 30 days before today.",
  }),
};

// ---------- Daylight ----------

/** The daylight change from yesterday, in words. */
function daylightChange(block: SunMoon): string {
  const change = block.daylight_change_minutes;
  if (change === 0) return "the same as yesterday";
  const by = formatHours(Math.abs(change) * 60_000);
  return `${by} ${change > 0 ? "more" : "less"} than yesterday`;
}

/** An arc from sunrise to sunset; the day so far traced, the sun as the now dot. */
function daylightChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const block = snapshot.sun_moon;
  if (!block) return null;
  const pad = 6;
  // A point on the arc a share `t` of the way from sunrise to sunset.
  const point = (t: number): [number, number] => [
    pad + t * (width - 2 * pad),
    PLOT_BOTTOM - (PLOT_BOTTOM - 6) * Math.sin(Math.PI * t),
  ];
  // The arc from one share of the day to another, as polyline points.
  const arc = (from: number, to: number): string => {
    const steps = 40;
    const points: [number, number][] = [];
    for (let i = 0; i <= steps; i++) points.push(point(from + ((to - from) * i) / steps));
    return pointList(points);
  };
  const svg = miniChart(width);
  svg.append(
    svgEl("line", { class: "mosaic-base", x1: 0, x2: width, y1: PLOT_BOTTOM, y2: PLOT_BOTTOM }),
    svgEl("polyline", { class: "mosaic-arc", points: arc(0, 1) }),
  );
  const now = snapshot.observed_at_ms;
  const span = block.sunset_ms - block.sunrise_ms;
  if (now !== null && span > 0 && now >= block.sunrise_ms && now <= block.sunset_ms) {
    const t = (now - block.sunrise_ms) / span;
    const [sx, sy] = point(t);
    svg.append(
      svgEl("polyline", { class: "mosaic-arc-day", points: arc(0, t) }),
      svgEl("circle", { class: "mosaic-now-dot", cx: sx, cy: sy, r: 4 }),
    );
  }
  endLabels(svg, width, formatTime(block.sunrise_ms), formatTime(block.sunset_ms));
  return svg;
}

/** Daylight: the day's length and its change from yesterday. */
export const daylightTile: MosaicTile = {
  slug: "daylight",
  depth: "daylight",
  title: "Daylight",
  zoomName: "daylight",
  shown: (snapshot) => snapshot.sun_moon !== null,
  // The Sun and moon card said it in its body; with no tile, the footer does.
  absent: () => "Not known: the station reports no location.",
  read(snapshot) {
    const block = snapshot.sun_moon;
    if (!block) return { value: null, delta: "the station reports no location" };
    return {
      value: formatHours(block.daylight_minutes * 60_000, { noBreak: true }),
      delta: daylightChange(block),
      spoken: `sunrise ${formatTime(block.sunrise_ms)}, sunset ${formatTime(block.sunset_ms)}`,
    };
  },
  chart: daylightChart,
  source: () => ({
    line: "sunrise to sunset, against yesterday.",
    method:
      "Daylight runs from sunrise to sunset, against yesterday's. The arc is the day from " +
      "sunrise to sunset; the dot is where the sun is in it now.",
  }),
};

// ---------- Inside ----------

/** A sensor's last 24 hours, hourly, its newest hour as the now dot. */
function indoorLine(sensor: IndoorSensor, width: number): SVGSVGElement | null {
  const values = sensor.trace.temperature_f;
  const last = lastIndex(values);
  if (values.length < 2 || last < 0) return null;
  const numbers = values.filter((v): v is number => v !== null);
  const h = ROW_LINE_HEIGHT;
  const x = scale(0, values.length - 1, 2, width - 6);
  const y = scale(Math.min(...numbers) - 1, Math.max(...numbers) + 1, h - 4, 4);
  const svg = miniChart(width, h);
  svg.append(...traceMarks(values, x, y, 3.5));
  return svg;
}

/**
 * One indoor sensor as a row: its name, temperature, humidity and line. The
 * Inside zoom words its sensors with it too.
 *
 * @param sensor - The indoor sensor.
 * @returns The row.
 */
export function sensorRow(sensor: IndoorSensor): MosaicRow {
  return {
    name: sensor.name,
    value: degrees(sensor.temperature_f),
    aside:
      sensor.humidity_pct === null
        ? null
        : { text: percent(sensor.humidity_pct), spoken: "humidity" },
    line: (width) => indoorLine(sensor, width),
  };
}

/** Inside: one row per indoor sensor still reporting. */
export const insideTile: MosaicTile = {
  slug: "inside",
  depth: "inside",
  title: "Inside",
  zoomName: "inside",
  shown: (snapshot) => (snapshot.indoor?.length ?? 0) > 0,
  read: () => ({ value: null, delta: null }),
  rows: (snapshot): MosaicRow[] => (snapshot.indoor ?? []).map(sensorRow),
  source: () => ({
    line: "each sensor's last 24 hours.",
    method:
      "One row per indoor sensor still reporting: its temperature and humidity, and its last " +
      "24 hours as hourly means.",
  }),
};
