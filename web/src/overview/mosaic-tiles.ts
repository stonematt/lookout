/**
 * The mosaic's tile definitions (#472), in its fixed order (spec #460 story
 * 19): Temperature, Rain, Last 10 days, Dry spell, then the six from
 * `mosaic-conditions.ts` (#473): Wind, Pressure, Humidity, Sun, Daylight,
 * Inside.
 *
 * - Temperature: the live reading (the hero's), today's high, and the
 *   hero's own delta on usual (`nowDeltaWords`, the one snapshot value
 *   `temperature_context.usual_band.now_delta`); its mini chart is the usual
 *   and today rows of the window range chart, with the now rule.
 * - Rain since Oct 1: this water year's rain to date as a percent of the
 *   usual by this date (`rain_tab.usual_to_date`, #539); its mini chart is
 *   the water-year curves, cropped to the season so far.
 * - Last 10 days: the ten days' highs against their usual highs, a mean
 *   (`temperature_context.ten_day_highs`, #540); its mini chart is each
 *   day's low to high over the usual for its date.
 * - Dry spell: the days since it last rained against the usual dry spell
 *   (`dry_spell`, Release A; the calm lead rule for an older snapshot, the
 *   same numbers the Rain tab shows); its mini chart is the last 30 days'
 *   rain, a day to a square.
 *
 * Every value is the snapshot's; these only word, format and draw.
 *
 * Offers `MOSAIC_TILES` and each tile's definition.
 */
import { lastDays } from "../rain/charts.js";
import { calmRule, currentYear } from "../rain/text.js";
import { missingDay, scale } from "../charts/svg.js";
import { svgEl } from "../dom.js";
import {
  WET_DAY_IN,
  degrees,
  formatDayCount,
  formatDays,
  inches,
  percent,
  signed,
} from "../format.js";
import { gapValue, isComparable, svgHatch } from "../gaps.js";
import type { DashboardSnapshot } from "../snapshot.js";
import type { MosaicTile } from "./mosaic.js";
import {
  daylightTile,
  humidityTile,
  insideTile,
  pressureTile,
  sunTile,
  windTile,
} from "./mosaic-conditions.js";
import { MINI_HEIGHT, miniChart } from "./mosaic.js";
import { rampColour, shownDays, todayInList, todayPeriod } from "./temperature.js";
import { nowDeltaWords } from "./today-hero.js";

const SHORT_MONTH = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short" });

/** A mini chart's row label width: "usual", "today" in label type. */
const ROW_LABEL_PX = 46;

// ---------- Temperature ----------

/** The usual and today rows of the window range chart, with the now rule. */
function temperatureChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const today = todayPeriod(snapshot);
  const drawToday =
    today !== null &&
    isComparable(today.status) &&
    today.low !== null &&
    today.high !== null &&
    today.q1 !== null &&
    today.q3 !== null;
  const usual = todayInList(snapshot);
  const drawUsual = usual !== null && usual.usual_low !== null && usual.usual_high !== null;
  if (!drawToday && !drawUsual) return null;
  const now = snapshot.current.temperature_f;
  const numbers = [
    ...(drawToday ? [today.low!, today.high!] : []),
    ...(drawUsual ? [usual.usual_low!, usual.usual_high!] : []),
    ...(now === null ? [] : [now]),
  ];
  const lo = Math.min(...numbers) - 2;
  const hi = Math.max(...numbers) + 2;
  const x = scale(lo, hi, ROW_LABEL_PX + 6, width - 6);
  const svg = miniChart(width);
  const rows = { usual: 18, today: 46 };
  // A row is named only when it is drawn: no usual before an earlier year.
  const named: [string, number, boolean][] = [
    ["usual", rows.usual, drawUsual],
    ["today", rows.today, drawToday],
  ];
  for (const [name, y, drawn] of named) {
    if (drawn) svg.append(svgEl("text", { class: "mosaic-label", x: 0, y: y + 4 }, name));
  }
  if (drawUsual) {
    svg.append(
      svgEl("rect", {
        class: "mosaic-band",
        x: x(usual.usual_low!),
        y: rows.usual - 7,
        width: Math.max(2, x(usual.usual_high!) - x(usual.usual_low!)),
        height: 14,
        rx: 4,
      }),
    );
  }
  if (drawToday) {
    const y = rows.today;
    svg.append(
      svgEl("line", {
        class: "mosaic-whisker",
        x1: x(today.low!),
        x2: x(today.high!),
        y1: y,
        y2: y,
      }),
      svgEl("circle", { class: "mosaic-end", cx: x(today.low!), cy: y, r: 2.5 }),
      svgEl("circle", { class: "mosaic-end", cx: x(today.high!), cy: y, r: 2.5 }),
      svgEl("rect", {
        class: "mosaic-iqr",
        x: x(today.q1!),
        y: y - 6,
        width: Math.max(2, x(today.q3!) - x(today.q1!)),
        height: 12,
        rx: 3,
      }),
    );
  }
  if (now !== null) {
    svg.append(
      svgEl("line", { class: "mosaic-now", x1: x(now), x2: x(now), y1: 4, y2: MINI_HEIGHT - 4 }),
    );
  }
  return svg;
}

/** Temperature: the live reading, today's high, and the hero's delta on usual. */
export const temperatureTile: MosaicTile = {
  slug: "temperature",
  depth: "temperature",
  title: "Temperature",
  zoomName: "temperature",
  read(snapshot) {
    const t = snapshot.current.temperature_f;
    const today = todayPeriod(snapshot);
    const high =
      today && today.high !== null ? gapValue(today.status, degrees(today.high, 0)) : null;
    const parts = [high === null ? null : `high ${high}`, nowDeltaWords(snapshot)];
    const delta = parts.filter((p): p is string => p !== null).join(", ");
    return { value: t === null ? "n/a" : degrees(t), delta: delta || null };
  },
  chart: temperatureChart,
  source: () => ({
    what: "five-minute readings, station time, with the live reading as now",
    method:
      "The delta is the hero's: the live reading against the usual median for its hour. " +
      "The usual row runs from the median daily low to the median daily high for this date in " +
      "earlier years; today's row from its lowest to its highest reading, the bar its middle half.",
  }),
};

// ---------- Rain since Oct 1 ----------

/** The last place the crop shows: the season so far and a month on, at least a quarter. */
function cropEnd(dayIndex: number): number {
  return Math.min(364, Math.max(91, dayIndex + 30));
}

/** The water-year curves from Oct 1 to the crop's end, this year bold. */
function rainChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const tab = snapshot.rain_tab;
  const current = currentYear(tab);
  const end = cropEnd(tab.day_index);
  const shown = tab.water_years.filter((c) =>
    c.cumulative_in.slice(0, end + 1).some((v) => v !== null),
  );
  if (!shown.length) return null;
  const values = shown.flatMap((c) =>
    c.cumulative_in.slice(0, end + 1).filter((v): v is number => v !== null),
  );
  const axis = 14;
  const plotBottom = MINI_HEIGHT - axis - 2;
  const x = scale(0, end, 2, width - 6);
  const y = scale(0, Math.max(0.5, ...values), plotBottom, 6);
  const svg = miniChart(width);
  svg.append(
    svgEl("line", { class: "mosaic-base", x1: 0, x2: width, y1: plotBottom, y2: plotBottom }),
  );
  const endMonth = SHORT_MONTH.format(new Date(Date.UTC(2000, 9, 1 + end)));
  svg.append(
    svgEl("text", { class: "mosaic-label", x: 0, y: MINI_HEIGHT - 2 }, "Oct"),
    svgEl(
      "text",
      { class: "mosaic-label", x: width, y: MINI_HEIGHT - 2, "text-anchor": "end" },
      endMonth,
    ),
  );
  // Earlier years first, thin, so this year's line sits over them.
  for (const curve of [
    ...shown.filter((c) => c !== current),
    ...shown.filter((c) => c === current),
  ]) {
    const mine = curve === current;
    let run: string[] = [];
    const flush = () => {
      if (run.length > 1) {
        svg.append(
          svgEl("polyline", {
            class: mine ? "mosaic-curve mosaic-curve-now" : "mosaic-curve",
            points: run.join(" "),
          }),
        );
      }
      run = [];
    };
    curve.cumulative_in.slice(0, end + 1).forEach((v, i) => {
      if (v === null) flush();
      else run.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    });
    flush();
  }
  const today = current?.cumulative_in[tab.day_index];
  if (current && today !== null && today !== undefined) {
    svg.append(
      svgEl("circle", { class: "mosaic-now-dot", cx: x(tab.day_index), cy: y(today), r: 4 }),
    );
  }
  return svg;
}

/**
 * Rain since Oct 1: this water year to date, as a percent of the usual rain
 * by this date (`rain_tab.usual_to_date`, #539); the total alone when the
 * snapshot has no percent (null, or N−1).
 */
export const rainTile: MosaicTile = {
  slug: "rain",
  depth: "rain",
  title: "Rain since Oct 1",
  zoomName: "rain",
  read(snapshot) {
    const tab = snapshot.rain_tab;
    const current = currentYear(tab);
    if (!current || current.to_date_in === null) {
      return { value: null, delta: "not compared yet" };
    }
    const approx = current.prorated[tab.day_index] ? "≈ " : "";
    const share = tab.usual_to_date?.percent ?? null;
    return {
      value: `${approx}${inches(current.to_date_in, { unit: false })}`,
      unit: "in",
      delta: share === null ? null : `${percent(share)} of usual`,
    };
  },
  chart: rainChart,
  source: () => ({
    what: "daily totals with today's live total",
    method:
      "Rain since Oct 1, one line per water year, this year bold. The percent is this year's " +
      "total as a percent of the usual rain by this date: the median, over the earlier water " +
      "years that reach this date, of their rain since Oct 1. Feb 29 is folded into Feb 28 so " +
      "the years line up; ≈ marks a total scaled up over a few missing days.",
  }),
};

// ---------- Last 10 days ----------

/** Each day's low to high, standing over the usual for its date. */
function tenDaysChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const days = shownDays(snapshot);
  const numbers = days.flatMap((d) =>
    [d.low, d.high, d.usual_low, d.usual_high].filter((v): v is number => v !== null),
  );
  if (!numbers.length) return null;
  const slots = snapshot.temperature_context.ten_days.length;
  const step = width / Math.max(slots, 1);
  const y = scale(Math.min(...numbers) - 2, Math.max(...numbers) + 2, MINI_HEIGHT - 2, 2);
  const svg = miniChart(width);
  const hatch = svgHatch(svg);
  // Right-aligned, so today sits at the right edge as the days grow in.
  const offset = slots - days.length;
  days.forEach((d, i) => {
    const left = (offset + i) * step;
    if (d.usual_low !== null && d.usual_high !== null) {
      svg.append(
        svgEl("rect", {
          class: "mosaic-band-day",
          x: left,
          y: y(d.usual_high),
          width: step,
          height: Math.max(1, y(d.usual_low) - y(d.usual_high)),
        }),
      );
    }
    const barW = Math.max(3, Math.min(10, step * 0.45));
    const cx = left + step / 2;
    if (!isComparable(d.status) || d.low === null || d.high === null) {
      svg.append(
        svgEl("rect", {
          class: "mosaic-gap",
          x: cx - barW / 2,
          y: 2,
          width: barW,
          height: MINI_HEIGHT - 4,
          rx: barW / 2,
          fill: hatch,
        }),
      );
      return;
    }
    const bar = svgEl("rect", {
      class: "mosaic-day",
      x: cx - barW / 2,
      y: y(d.high),
      width: barW,
      height: Math.max(barW, y(d.low) - y(d.high)),
      rx: barW / 2,
    });
    bar.setAttribute("style", `fill:${rampColour(d.high)}`);
    svg.append(bar);
  });
  return svg;
}

/** The ten-day highs' side of usual, in words: "highs over usual". */
function highsWords(deltaF: number): string {
  return `highs ${deltaF > 0 ? "over" : deltaF < 0 ? "under" : "at"} usual`;
}

/**
 * Last 10 days: the last ten days' highs against their usual highs, a mean
 * (`temperature_context.ten_day_highs`, #540); not compared when the
 * snapshot has none (null, or N−1).
 */
export const tenDaysTile: MosaicTile = {
  slug: "last-10-days",
  depth: "last_10_days",
  title: "Last 10 days",
  zoomName: "last 10 days",
  read(snapshot) {
    const highs = snapshot.temperature_context.ten_day_highs ?? null;
    if (highs === null) return { value: null, delta: "not compared yet" };
    return {
      value: `${signed(highs.delta_f, 0, { plus: true })}°`,
      delta: highsWords(highs.delta_f),
    };
  },
  chart: tenDaysChart,
  bandUnderBars: true,
  source: () => ({
    what: "daily low and high",
    method:
      "The number is the mean of the last ten days' highs against their usual highs, over the " +
      "days that have both. Each bar runs from the day's low to its high, over the usual range " +
      "for its date: the median daily low to the median daily high over the days within a week " +
      "either side of it, in each earlier year whose record there is complete. Today is the day " +
      "so far.",
  }),
};

// ---------- Dry spell ----------

/**
 * The current dry spell and the usual one, in days: Release A's `dry_spell`,
 * or for an older snapshot the calm lead rule's (the Rain tab's numbers).
 */
function drySpellDays(snapshot: DashboardSnapshot): { since: number | null; usual: number | null } {
  const spell = snapshot.dry_spell;
  if (spell) return { since: spell.days_since_rain, usual: spell.usual_days };
  const calm = calmRule(snapshot);
  return { since: calm?.value ?? null, usual: calm?.reference ?? null };
}

/**
 * A day's square: a rain day by the Rain tab's own threshold (`WET_DAY_IN`),
 * else dry. One rain colour, with no second table of daily steps.
 */
const dayClass = (rainIn: number) => (rainIn < WET_DAY_IN ? "mosaic-dry" : "mosaic-wet");

/** The last 30 days' rain, ten squares to a row, today outlined. */
function dryChart(snapshot: DashboardSnapshot, width: number): SVGSVGElement | null {
  const recent = lastDays(snapshot.rain_tab.recent, 30);
  if (!recent.rain_in.length) return null;
  const gap = 3;
  const cols = 10;
  const rowsN = Math.ceil(recent.rain_in.length / cols);
  // The cells fill the tile's width and the chart's height, as near square
  // as a phone's half tile; wider on the grids.
  const cellW = (width - gap * (cols - 1)) / cols;
  const cellH = (MINI_HEIGHT - gap * (rowsN - 1)) / rowsN;
  const svg = miniChart(width);
  const hatch = svgHatch(svg);
  const last = recent.rain_in.length - 1;
  recent.rain_in.forEach((v, i) => {
    const attrs = {
      x: (i % cols) * (cellW + gap),
      y: Math.floor(i / cols) * (cellH + gap),
      width: cellW,
      height: cellH,
      rx: 3,
    };
    const reason = recent.reason[i] ?? null;
    if (v === null) {
      // A day not in the record is hatched; a day before it, or still to
      // come, is left blank.
      if (missingDay(reason))
        svg.append(svgEl("rect", { ...attrs, class: "mosaic-gap", fill: hatch }));
      else svg.append(svgEl("rect", { ...attrs, class: "mosaic-blank" }));
    } else {
      svg.append(svgEl("rect", { ...attrs, class: dayClass(v) }));
    }
    if (i === last) {
      svg.append(
        svgEl("rect", {
          ...attrs,
          x: attrs.x + 1,
          y: attrs.y + 1,
          width: cellW - 2,
          height: cellH - 2,
          class: "mosaic-today",
        }),
      );
    }
  });
  return svg;
}

/** Dry spell: the days since it last rained, against the usual dry spell. */
export const drySpellTile: MosaicTile = {
  slug: "dry-spell",
  depth: "dry_spell",
  title: "Dry spell",
  zoomName: "dry spell",
  read(snapshot) {
    const { since, usual } = drySpellDays(snapshot);
    if (since === null) return { value: null, delta: "no last rain on record" };
    return {
      value: formatDays(since),
      delta: usual === null ? null : `usual is ${formatDayCount(usual)}`,
    };
  },
  chart: dryChart,
  source: () => ({
    what: "daily totals; the last rain is the station's own record",
    method:
      `The time since the last rain day (at least ${WET_DAY_IN} in), against the usual dry spell: ` +
      "the median completed spell in the last 365 days. The squares are the last 30 days; a day " +
      "missing from the record is hatched, never read as dry.",
  }),
};

/** The mosaic's tiles, in its fixed order. */
export const MOSAIC_TILES: readonly MosaicTile[] = [
  temperatureTile,
  rainTile,
  tenDaysTile,
  drySpellTile,
  windTile,
  pressureTile,
  humidityTile,
  sunTile,
  daylightTile,
  insideTile,
];
