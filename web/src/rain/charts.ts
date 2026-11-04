/**
 * The Rain tab's three charts, drawn as SVG at their real pixel width
 * (DESIGN.md, "Chart grammar"): the daily bars, the water-year curves and
 * the hour-by-day heatmap.
 *
 * Shared grammar: hairline gridlines on y only, 12px tabular ticks, labels on
 * the marks haloed in the ground, the 135° hatch for a day not in the record
 * (with its reason in place where there is room), and one `now` mark for
 * the subject of the headline. Every chart has a live readout below it:
 * hover, tap, or focus the chart and use the arrow keys, and the readout
 * says what the mark under it means, so nothing relies on a title (phones
 * do not show one).
 */
import {
  WATER_MONTHS,
  WATER_YEAR_START_MONTH,
  addDays,
  daysBetween,
  dayText,
  hourText,
  inches,
} from "../format.js";
import {
  LABEL_CHAR_PX,
  NARROW,
  type AboveDot,
  type Readout,
  canvas,
  chartLabel,
  aboveDot,
  dayTicks,
  hatchMissingDays,
  runs,
  scrub,
  yAxis,
  yTickSteps,
  zeroTicks,
} from "../charts/svg.js";
import { svgEl } from "../dom.js";
import { dayStatus, gapSentence } from "../gaps.js";
import { rainStep } from "../overview/ribbon.js";
import type { RainTab, RecentRain, WaterYearCurve } from "../snapshot.js";
import { curveValue } from "./text.js";

/** A non-leap water year's first day: a place's date counts days from it. */
const REFERENCE_START = `2000-${String(WATER_YEAR_START_MONTH).padStart(2, "0")}-01`;

/** Month starts as places in a water year, derived from a non-leap reference year. */
const MONTH_PLACES = WATER_MONTHS.map((_, i) => {
  const month = ((WATER_YEAR_START_MONTH - 1 + i) % 12) + 1;
  const year = month >= WATER_YEAR_START_MONTH ? 2000 : 2001;
  return daysBetween(REFERENCE_START, `${year}-${String(month).padStart(2, "0")}-01`);
});

/** The daily bars' now dot: its radius and its gap above today's bar (DESIGN.md "Daily bars"). */
const NOW_R = 4;
const NOW_GAP = 4;
/** How far a label's baseline stands above a bar it passes over: its descender and halo. */
const LABEL_CLEAR = 6;
/** A label line's height, for stepping one label below another. */
const LABEL_LINE = 16;

/** Where today's now dot and its "today" label go on the daily bars. */
interface TodayMark extends AboveDot {
  /** The dot's centre. */
  cx: number;
  cy: number;
}

/**
 * Place today's mark: an r4 now dot 4px above today's bar, or above the
 * baseline when today is dry, so it never touches the fill (the Now Stands
 * Off Rule), and "today" above the dot, lifted clear of any taller bar under
 * it. None when today is not in the record (its day is hatched).
 *
 * @param recent - The days drawn.
 * @param x - A day's left edge.
 * @param y - A rain amount's height.
 * @param step - A day's width.
 * @param left - The plot's left edge.
 * @param right - The plot's right edge.
 * @param bottom - The baseline.
 * @returns The mark's place, or null.
 */
function todayMark(
  recent: RecentRain,
  x: (i: number) => number,
  y: (v: number) => number,
  step: number,
  left: number,
  right: number,
  bottom: number,
): TodayMark | null {
  const count = recent.rain_in.length;
  const today = recent.rain_in[count - 1];
  if (today === null || today === undefined) return null;
  const cx = x(count - 1) + step / 2;
  const cy = (today > 0 ? y(today) : bottom) - NOW_GAP - NOW_R;
  const at = aboveDot(cx, cy, "today", left, right, 12);
  let tallest = bottom;
  for (let i = Math.max(0, Math.floor((at.from - left) / step)); i < count && x(i) <= at.to; i++) {
    const value = recent.rain_in[i];
    if (value !== null && value !== undefined && value > 0) tallest = Math.min(tallest, y(value));
  }
  // The plot's top margin keeps the lifted label below aboveDot's minY.
  return { ...at, cx, cy, y: Math.min(at.y, tallest - LABEL_CLEAR) };
}

/**
 * Hatch the days not in the record; mark where the record begins, stepping
 * that label down a line where it would meet today's label.
 */
function hatchDays(
  svg: SVGSVGElement,
  recent: RecentRain,
  hatch: string,
  x: (i: number) => number,
  step: number,
  top: number,
  bottom: number,
  mark: TodayMark | null,
): void {
  hatchMissingDays(svg, recent.reason, hatch, x, step, top, bottom, "rain-label");
  const before = runs(recent.reason.length, (i) => recent.reason[i] === "before_archive");
  for (const [, b] of before) {
    if (b + 1 >= recent.reason.length) continue;
    const at = x(b + 1);
    svg.append(svgEl("line", { x1: at, x2: at, y1: top, y2: bottom, class: "rain-edge" }));
    const words = `record begins ${dayText(addDays(recent.first_day, b + 1))}`;
    const anchor = at > 140 ? "end" : "start";
    const lx = anchor === "end" ? at - 4 : at + 4;
    const wide = words.length * LABEL_CHAR_PX;
    const [from, to] = anchor === "end" ? [lx - wide, lx] : [lx, lx + wide];
    let ly = top + 14;
    // Step a line below today's label where the two would meet.
    if (mark && from < mark.to && to > mark.from && Math.abs(ly - mark.y) < LABEL_LINE)
      ly = mark.y + LABEL_LINE;
    svg.append(chartLabel(lx, ly, words, "rain-label", anchor));
  }
}

/**
 * The last `days` of the recent window, the same days as the snapshot
 * sent with their reasons and hours (the 30 / 90 switch slices, it never
 * decides).
 *
 * @param recent - The snapshot's recent days.
 * @param days - How many to keep, from today back.
 * @returns The window from its new first day.
 */
export function lastDays(recent: RecentRain, days: number): RecentRain {
  const offset = Math.max(0, recent.rain_in.length - days);
  if (offset === 0) return recent;
  return {
    ...recent,
    first_day: addDays(recent.first_day, offset),
    rain_in: recent.rain_in.slice(offset),
    reason: recent.reason.slice(offset),
    hourly_in: recent.hourly_in.slice(offset),
  };
}

/** One day's sentence: its total, or why it is not counted. */
function daySentence(recent: RecentRain, i: number, isToday: boolean): string {
  const day = addDays(recent.first_day, i);
  const reason = recent.reason[i];
  if (reason) return gapSentence(dayStatus(day, reason));
  const total = recent.rain_in[i] ?? 0;
  const when = `${dayText(day, true)}${isToday ? " (today, live)" : ""}`;
  return total > 0 ? `${when}: ${inches(total)} of rain.` : `${when}: no rain.`;
}

/**
 * The daily bars: one bar per day of the recent window, today marked.
 *
 * @param host - The figure to draw in; its contents are replaced.
 * @param recent - The snapshot's recent days.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawDailyBars(
  host: HTMLElement,
  recent: RecentRain,
  width: number,
  readout: Readout,
): void {
  const count = recent.rain_in.length;
  const height = 168;
  // Room above the top tick for today's dot and its label.
  const [left, right, top, bottom] = [40, width - 6, 32, height - 22];
  const values = recent.rain_in.filter((v): v is number => v !== null);
  const ticks = zeroTicks(Math.max(0.1, ...values), yTickSteps(width));
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / count;
  const x = (i: number) => left + i * step;
  const pad = step > 4 ? 1 : 0;

  const { svg, hatch } = canvas(
    width,
    height,
    `Rain each day, ${dayText(recent.first_day)} to today`,
    "rain-svg",
  );
  yAxis(svg, ticks, y, left, right, "rain", { text: " in", place: "tick" });
  const mark = todayMark(recent, x, y, step, left, right, bottom);
  hatchDays(svg, recent, hatch, x, step, top, bottom, mark);
  recent.rain_in.forEach((value, i) => {
    if (value === null || value <= 0) return;
    const isToday = i === count - 1;
    svg.append(
      svgEl("rect", {
        x: x(i) + pad / 2,
        y: y(value),
        width: Math.max(1, step - pad),
        height: bottom - y(value),
        class: isToday ? "rain-bar rain-bar-now" : "rain-bar",
      }),
    );
  });
  if (mark) {
    svg.append(svgEl("circle", { cx: mark.cx, cy: mark.cy, r: NOW_R, class: "rain-now-dot" }));
    svg.append(chartLabel(mark.x, mark.y, "today", "rain-now-label", mark.anchor));
  }
  dayTicks(svg, recent.first_day, count, x, step, bottom + 4, "rain");
  host.replaceChildren(svg);
  scrub(
    host,
    count,
    (px) => Math.floor((px - left) / step),
    (i) => readout(daySentence(recent, i, i === count - 1)),
  );
}

/**
 * The water-year curves: cumulative rain from Oct 1, one line per year, the
 * current year bold and marked "now".
 *
 * @param host - The figure to draw in; its contents are replaced.
 * @param tab - The Rain tab's data.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawYearCurves(
  host: HTMLElement,
  tab: RainTab,
  width: number,
  readout: Readout,
): void {
  const narrow = width < NARROW;
  const height = narrow ? 240 : 300;
  const shown = tab.water_years.filter((c) => c.cumulative_in.some((v) => v !== null));
  const current = tab.water_years[tab.water_years.length - 1];
  const earlier = shown.filter((c) => c !== current).reverse();
  // The right margin holds the earlier years' names, "2024 ≈ 48.6 in", sized
  // to the longest so none runs off the edge.
  const endText = (curve: WaterYearCurve) => {
    const value = curveValue(curve, curve.cumulative_in.length - 1);
    return value === null ? null : `${curve.water_year} ${value}`;
  };
  const longest = Math.max(0, ...earlier.map((c) => (endText(c) ?? "").length));
  const margin = longest ? Math.ceil(longest * LABEL_CHAR_PX) + 10 : 12;
  const [left, right, top, bottom] = [40, width - margin, 18, height - 24];
  const values = shown.flatMap((c) => c.cumulative_in.filter((v): v is number => v !== null));
  const ticks = zeroTicks(Math.max(1, ...values), yTickSteps(width));
  const yMax = ticks[ticks.length - 1] ?? 1;
  const y = (v: number) => bottom - ((bottom - top) * v) / yMax;
  const step = (right - left) / 364;
  const x = (place: number) => left + place * step;

  const { svg } = canvas(width, height, "Rain since Oct 1, one line per water year", "rain-svg");
  yAxis(svg, ticks, y, left, right, "rain", { text: " in", place: "tick" });
  // Every month is ticked and named to Sep, by its initial on a phone.
  MONTH_PLACES.forEach((place, i) => {
    svg.append(
      svgEl("line", { x1: x(place), x2: x(place), y1: bottom, y2: bottom + 4, class: "rain-axis" }),
    );
    const name = narrow ? WATER_MONTHS[i]!.charAt(0) : WATER_MONTHS[i];
    svg.append(
      svgEl("text", { x: x(place) + 2, y: bottom + 16, class: "rain-tick rain-month" }, name),
    );
  });

  const draw = (curve: WaterYearCurve, cls: string) => {
    const keep = (i: number) => curve.cumulative_in[i] !== null;
    for (const [a, b] of runs(curve.cumulative_in.length, keep)) {
      const points = [];
      for (let i = a; i <= b; i++)
        points.push(`${x(i).toFixed(1)},${y(curve.cumulative_in[i]!).toFixed(1)}`);
      if (points.length === 1) points.push(points[0]);
      svg.append(svgEl("polyline", { points: points.join(" "), class: cls }));
    }
  };

  // Earlier years: thin lines, named at the right edge, nudged apart.
  const ends: { y: number; text: string }[] = [];
  earlier.forEach((curve, i) => {
    draw(curve, `rain-year rain-year-${Math.min(i, 2)}`);
    const last = curve.cumulative_in.length - 1;
    const name = endText(curve);
    if (name !== null) {
      ends.push({ y: y(curve.cumulative_in[last]!), text: name });
      return;
    }
    // A line cut short by a gap is named where it stops.
    let stop = last;
    while (stop > 0 && curve.cumulative_in[stop] === null) stop--;
    const at = curve.cumulative_in[stop];
    if (at === null || at === undefined) return;
    const text = `${curve.water_year} ${curveValue(curve, stop)}${narrow ? "" : ", then not compared"}`;
    const room = text.length * LABEL_CHAR_PX;
    const anchor = x(stop) - 6 - room >= 0 ? "end" : "start";
    const lx = anchor === "end" ? x(stop) - 6 : Math.min(x(stop) + 6, width - room);
    svg.append(chartLabel(lx, Math.max(top + 12, y(at) - 8), text, "rain-label", anchor));
  });
  ends.sort((a, b) => a.y - b.y);
  let floor = -Infinity;
  for (const end of ends) {
    end.y = Math.max(end.y, floor + 15);
    floor = end.y;
    svg.append(chartLabel(right + 6, end.y + 4, end.text, "rain-label"));
  }

  // The current year: the bold line, its dot and "now" over it.
  if (current && shown.includes(current)) {
    draw(current, "rain-year rain-year-now");
    const place = current.cumulative_in.length - 1;
    const value = current.cumulative_in[place];
    if (value !== null && value !== undefined) {
      const cx = x(place);
      svg.append(svgEl("circle", { cx, cy: y(value), r: 5, class: "rain-now-dot" }));
      const text = `${current.water_year} now ${curveValue(current, place)}`;
      if (narrow) {
        // On a phone the label sits above its dot.
        const at = aboveDot(cx, y(value), text, 0, width, 12);
        svg.append(chartLabel(at.x, at.y, text, "rain-now-label", at.anchor));
      } else {
        const anchor = cx > (left + right) / 2 ? "end" : "start";
        svg.append(
          chartLabel(
            cx + (anchor === "end" ? -8 : 8),
            Math.max(top + 4, y(value) - 10),
            text,
            "rain-now-label",
            anchor,
          ),
        );
      }
    }
  }

  host.replaceChildren(svg);
  scrub(
    host,
    365,
    (px) => Math.round((px - left) / step),
    (place) => {
      const parts = [current, ...earlier]
        .filter((c): c is WaterYearCurve => Boolean(c))
        .filter((c) => place < c.cumulative_in.length)
        .map((c) => `${c.water_year} ${curveValue(c, place) ?? "not compared"}`);
      readout(
        `By ${dayText(addDays(REFERENCE_START, place))}: ${parts.length ? parts.join("; ") : "no year reaches this date yet"}.`,
      );
    },
  );
}

/** Hour rows: midnight at the top. */
const HOUR_TICKS = [0, 6, 12, 18];

/**
 * The heatmap: one column per day of the recent window, one row per hour of
 * the day, each cell shaded by its rain in four steps. A null hour the
 * snapshot says is still to come (`first_hour_to_come`, today) is left
 * blank; any other null hour has no readings in the record and is hatched,
 * never drawn as dry.
 *
 * @param host - The figure to draw in; its contents are replaced.
 * @param recent - The snapshot's recent days.
 * @param width - The width to draw at.
 * @param readout - Where position sentences go.
 */
export function drawHourGrid(
  host: HTMLElement,
  recent: RecentRain,
  width: number,
  readout: Readout,
): void {
  const count = recent.hourly_in.length;
  const row = width < NARROW ? 8 : 10;
  const [left, right, top] = [56, width - 6, 4];
  const bottom = top + 24 * row;
  const height = bottom + 22;
  const step = (right - left) / count;
  const x = (i: number) => left + i * step;
  const toCome = (i: number, h: number) => i === count - 1 && h >= recent.first_hour_to_come;

  const { svg, hatch } = canvas(
    width,
    height,
    `Rain by the hour, ${dayText(recent.first_day)} to today`,
    "rain-svg",
  );
  recent.hourly_in.forEach((hours, i) => {
    if (recent.reason[i] !== null) return;
    hours.forEach((rain, h) => {
      if (rain === null) {
        if (toCome(i, h)) return;
        svg.append(
          svgEl("rect", {
            x: x(i),
            y: top + h * row,
            width: Math.max(1, step - (step > 4 ? 1 : 0)),
            height: row - 1,
            class: "rain-cell rain-unknown",
            fill: hatch,
          }),
        );
        return;
      }
      const level = rainStep(rain, 60);
      svg.append(
        svgEl("rect", {
          x: x(i),
          y: top + h * row,
          width: Math.max(1, step - (step > 4 ? 1 : 0)),
          height: row - 1,
          class: level ? `rain-cell rain-step-${level}` : "rain-cell rain-dry",
        }),
      );
    });
  });
  hatchDays(svg, recent, hatch, x, step, top, bottom, null);
  HOUR_TICKS.forEach((h) => {
    svg.append(
      svgEl(
        "text",
        { x: left - 6, y: top + h * row + row, class: "rain-tick", "text-anchor": "end" },
        hourText(h),
      ),
    );
  });
  dayTicks(svg, recent.first_day, count, x, step, bottom + 4, "rain");
  host.replaceChildren(svg);
  scrub(
    host,
    count,
    (px) => Math.floor((px - left) / step),
    (i, h) => {
      const day = addDays(recent.first_day, i);
      const reason = recent.reason[i];
      if (reason) return readout(gapSentence(dayStatus(day, reason)));
      const rain = recent.hourly_in[i]?.[h];
      const when = `${dayText(day, true)}, ${hourText(h)}`;
      if (rain === null || rain === undefined) {
        return readout(
          toCome(i, h) ? `${when}: still to come.` : `${when}: no readings in the record.`,
        );
      }
      readout(rain > 0 ? `${when}: ${inches(rain)} of rain.` : `${when}: no rain.`);
    },
    24,
    (py) => Math.floor((py - top) / row),
  );
}
