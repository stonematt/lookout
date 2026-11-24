/**
 * The Rain tab's head, Rain tab C (#664, spec #647; boards RainC and
 * RainCDark): now, then the season, then usual. This file draws the first
 * two parts and the third's window strip; `index.ts`'s `renderYears` sets
 * them in order around the water-year verdict and its curves, for the tab's
 * first tile and the rain zoom alike (Mosaic decision 25).
 *
 * - The now part: last rain, or the now story while Python tells it
 *   (`nowLine`). With rain not leading it reads "Last rain 9 h ago" alone.
 * - The season part: wet hours over the last 7 and 30 days (the Overview
 *   meter's `wet_hours`) and since Oct 1 (`wet_hours_to_date`), each a bar
 *   of every hour in its span, the dark fill its wet hours, the hours the
 *   record cannot count hatched at the end, never drawn as dry. Since Oct 1
 *   carries usual wet hours to date as a dashed band on its bar.
 * - The window strip: each window's rain as a dot on its own usual range,
 *   the band's end values labelled, the snapshot's mark ("above usual
 *   range", "below usual range") in words beside the value; a window with
 *   no earlier year draws its dot with no band and says so.
 *
 * Every part is drawn only when the snapshot carries its fields (ADR-0008):
 * a version 4 document without them draws today's head. The site decides
 * nothing here: every word of judgement is the snapshot's.
 *
 * Offers `headParts`, `HeadPart`, `partLabel`, `markRaining`, `bandLabels`
 * and `windowDots`.
 */
import { inches } from "../format.js";
import { hatchGap } from "../gaps.js";
import { LABEL_CHAR_PX, drawSized } from "../charts/svg.js";
import { make, spoken, svgEl } from "../dom.js";
import { missingWords } from "../overview/wet-meter.js";
import { isShown } from "../overview/history.js";
import type { DashboardSnapshot, StripWindow, WetHoursToDate } from "../snapshot.js";
import {
  WINDOW_NAMES,
  nowLine,
  seasonLead,
  stripUsualWords,
  usualWetWords,
  windowValue,
} from "./text.js";

/** One part of the head: its small label, its lead in words, and what it draws under it. */
export interface HeadPart {
  /** "now", "season" or "verdict": the lead's `data-part` when it is not the tile's finding. */
  id: string;
  /** The boards' small label over the lead: "Now", "Wet hours". */
  label: string;
  lead: string;
  /** Drawn under the lead. */
  body: HTMLElement[];
}

/**
 * A head part's small label, over its lead.
 *
 * @param text - The label.
 * @param first - Whether it opens the head, with no part above to rule off.
 * @returns The label.
 */
export function partLabel(text: string, first = false): HTMLElement {
  const label = make("p", first ? "rain-part-label" : "rain-part-label rain-part-next", text);
  label.dataset.part = "label";
  return label;
}

/**
 * The head's parts before the water year, in order (#664): now, then the
 * season. Each is left out when the snapshot lacks its fields (ADR-0008).
 *
 * - Now: one line, the gauge's last rain or the now story (`nowLine`).
 * - Season: the wet hours in words, then a bar for each span the snapshot
 *   counts: 7 days, 30 days, since Oct 1.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The parts; none for a snapshot written before #657, #659 and #663.
 */
export function headParts(snapshot: DashboardSnapshot): HeadPart[] {
  const parts: HeadPart[] = [];
  const now = nowLine(snapshot);
  if (now !== null) parts.push({ id: "now", label: "Now", lead: now, body: [] });
  const season = seasonLead(snapshot);
  if (season !== null) {
    parts.push({ id: "season", label: "Wet hours", lead: season, body: [wetBars(snapshot)] });
  }
  return parts;
}

/**
 * Mark a now lead while it rains: a rain mark before the words, which say
 * "Raining" too (the Hue Is Never Alone Rule).
 *
 * @param lead - The now part's lead.
 * @param snapshot - The dashboard snapshot.
 */
export function markRaining(lead: HTMLElement, snapshot: DashboardSnapshot): void {
  if (snapshot.rain_now?.raining) lead.prepend(make("span", "rain-now-mark"));
}

/** One bar's part, sized as a share of its span's hours (at least 3px when it holds one). */
function share(className: string, hours: number, total: number): HTMLElement | null {
  if (hours <= 0 || total <= 0) return null;
  const node = make("span", className);
  node.style.setProperty("--share", `${(100 * hours) / total}%`);
  return node;
}

/** What one wet-hour bar draws. */
interface WetRow {
  name: string;
  hours: number;
  wet: number;
  countable: number;
  missing: number;
  /** The words under the bar, read together with " · ". */
  note: string[];
  /** Usual wet hours to date, as a span of hours on the bar. */
  usual?: [number, number] | null;
  key: string;
}

/** Usual wet hours to date as a span of this year's hours: the band, else the plain values' span. */
function usualSpan(wet: WetHoursToDate): [number, number] | null {
  const usual = wet.usual;
  if (!usual) return null;
  if (usual.low_hours !== null && usual.high_hours !== null) {
    return [usual.low_hours, usual.high_hours];
  }
  const values = usual.earlier.map((y) => y.usual_hours);
  return values.length ? [Math.min(...values), Math.max(...values)] : null;
}

/** One wet-hour row: name, bar, count, then its note, laid on the rows' shared grid. */
function wetRow(row: WetRow): HTMLElement {
  const name = make("span", "rain-wet-name", row.name);
  const bar = make("span", "rain-wet-bar");
  bar.dataset.part = "bar";
  const track = make("span", "rain-wet-track");
  const wet = share("rain-wet-fill", row.wet, row.hours);
  const missing = share("rain-wet-missing gap-hatch", row.missing, row.hours);
  if (wet) track.append(wet);
  if (missing) track.append(missing);
  bar.append(track);
  if (row.usual) {
    const [low, high] = row.usual;
    const band = make("span", "rain-wet-usual");
    band.dataset.part = "usual";
    band.style.setProperty("--from", `${(100 * low) / row.hours}%`);
    band.style.setProperty("--to", `${(100 * high) / row.hours}%`);
    bar.append(band);
  }
  const count = make("span", "rain-wet-count");
  count.dataset.part = "count";
  count.append(make("b", "", `${row.wet.toLocaleString("en-US")} h`));
  count.append(` of ${row.countable.toLocaleString("en-US")}`);
  const cells = [name, bar, count];
  if (row.note.length) {
    const note = make("span", "rain-wet-note");
    note.dataset.part = row.usual === undefined ? "missing" : "usual-words";
    if (row.usual) note.append(make("span", "rain-usual-swatch"));
    else if (row.missing > 0) note.append(make("span", "hatch-swatch"));
    const words = make("span", "");
    // A short phrase stays whole when the note wraps on a narrow phone.
    row.note.forEach((phrase, i) => {
      if (i > 0) words.append(" · ");
      words.append(make("span", phrase.length <= 24 ? "rain-wet-phrase" : "", phrase));
    });
    note.append(words);
    cells.push(note);
  }
  const wrap = make("div", "rain-wet-row");
  wrap.dataset.key = row.key;
  wrap.append(...cells);
  return wrap;
}

/**
 * The season part's bars: one for each span the snapshot counts, 7 days,
 * 30 days and since Oct 1, the last with usual wet hours to date.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The bars.
 */
function wetBars(snapshot: DashboardSnapshot): HTMLElement {
  const rows = make("div", "rain-wet-rows");
  rows.dataset.part = "wet-bars";
  const wet = snapshot.wet_hours;
  if (wet) {
    for (const [name, window, key] of [
      ["7 days", wet.last_7_days, "7d"],
      ["30 days", wet.last_30_days, "30d"],
    ] as const) {
      rows.append(
        wetRow({
          name,
          key,
          hours: window.hours,
          wet: window.wet_hours,
          countable: window.countable_hours,
          missing: window.missing_hours,
          note: missingWords(window),
        }),
      );
    }
  }
  const toDate = snapshot.rain_tab.wet_hours_to_date;
  if (toDate) {
    const row = wetRow({
      name: "Since Oct 1",
      key: "water_year",
      hours: toDate.hours,
      wet: toDate.wet_hours,
      countable: toDate.countable_hours,
      missing: toDate.missing_hours,
      note: [usualWetWords(toDate)],
      usual: usualSpan(toDate),
    });
    row.dataset.part = "wet-to-date";
    rows.append(row);
  }
  return rows;
}

/** The strip track's height: the track and dot, then the band's labels under it. */
const TRACK_H = 40;
/** The track's line, and the labels' baseline. */
const TRACK_Y = 12;
const LABEL_Y = 34;
/** Room at each end of the track for the dot's radius. */
const PAD = 7;
/** The gap between a band's end and its label, and between two labels. */
const LABEL_GAP = 4;

/** A label to place: its words and where it may sit. */
interface Placed {
  text: string;
  x: number;
  anchor: "start" | "end" | "middle";
}

/** A label's left and right edges, by its estimated width. */
function edges(label: Placed): [number, number] {
  const w = label.text.length * LABEL_CHAR_PX;
  if (label.anchor === "start") return [label.x, label.x + w];
  if (label.anchor === "end") return [label.x - w, label.x];
  return [label.x - w / 2, label.x + w / 2];
}

/** A label at `x`, anchored to the side it fits on within `width`. */
function fit(text: string, x: number, width: number): Placed {
  const w = text.length * LABEL_CHAR_PX;
  return x + w <= width ? { text, x, anchor: "start" } : { text, x: Math.max(w, x), anchor: "end" };
}

/**
 * The band's end values as labels under the track: the low one left of the
 * band, the high one right of it, else both as one label where they would
 * meet or run off the edge.
 *
 * @param window - The strip window.
 * @param x - A value's place.
 * @param width - The track's width.
 * @returns The labels, none for a window with no range.
 */
export function bandLabels(window: StripWindow, x: (v: number) => number, width: number): Placed[] {
  const { low_in: low, high_in: high } = window;
  if (window.years === 0 || low === null || high === null) {
    return [{ text: "no earlier year", x: 0, anchor: "start" }];
  }
  if (low === high) return [fit(inches(low), x(low), width)];
  const lowLabel: Placed = {
    text: inches(low, { unit: false }),
    x: x(low) - LABEL_GAP,
    anchor: "end",
  };
  const highLabel: Placed = { text: inches(high), x: x(high) + LABEL_GAP, anchor: "start" };
  const [lowLeft, lowRight] = edges(lowLabel);
  const [highLeft, highRight] = edges(highLabel);
  if (lowLeft >= 0 && highRight <= width && lowRight + LABEL_GAP <= highLeft) {
    return [lowLabel, highLabel];
  }
  const both = `${inches(low, { unit: false })}–${inches(high)}`;
  const half = (both.length * LABEL_CHAR_PX) / 2;
  const mid = Math.min(width - half, Math.max(half, (x(low) + x(high)) / 2));
  return [{ text: both, x: mid, anchor: "middle" }];
}

/**
 * One window's track: its usual band (a tick when the earlier years agree),
 * its rain as a dot, and the band's end values under it.
 *
 * @param host - The track's box; its contents are replaced.
 * @param window - The strip window, with its rain known.
 * @param width - The width to draw at.
 */
function drawTrack(host: HTMLElement, window: StripWindow, width: number): void {
  const rain = window.rain_in ?? 0;
  const top = Math.max(rain, window.high_in ?? 0, window.low_in ?? 0);
  // Each window on its own scale from 0: a week's rain and a year's differ a hundredfold.
  const max = top > 0 ? top * 1.1 : 0.1;
  const x = (v: number) => PAD + ((width - 2 * PAD) * v) / max;
  const svg = svgEl("svg", {
    class: "rain-svg rain-strip-svg",
    width,
    height: TRACK_H,
    viewBox: `0 0 ${width} ${TRACK_H}`,
    "aria-hidden": "true",
  });
  svg.append(
    svgEl("line", { x1: 0, x2: width, y1: TRACK_Y, y2: TRACK_Y, class: "rain-strip-line" }),
  );
  const { low_in: low, high_in: high } = window;
  if (window.years > 0 && low !== null && high !== null) {
    if (low === high) {
      svg.append(
        svgEl("line", {
          x1: x(low),
          x2: x(low),
          y1: TRACK_Y - 8,
          y2: TRACK_Y + 8,
          class: "rain-strip-tick",
        }),
      );
    } else {
      svg.append(
        svgEl("rect", {
          x: x(low),
          y: TRACK_Y - 7,
          width: Math.max(2, x(high) - x(low)),
          height: 14,
          rx: 3,
          class: "rain-strip-band",
        }),
      );
    }
  }
  svg.append(svgEl("circle", { cx: x(rain), cy: TRACK_Y, r: 5.5, class: "rain-strip-dot" }));
  for (const label of bandLabels(window, x, width)) {
    svg.append(
      svgEl(
        "text",
        { x: label.x, y: LABEL_Y, "text-anchor": label.anchor, class: "rain-tick" },
        label.text,
      ),
    );
  }
  host.replaceChildren(svg);
}

/**
 * The window strip (#640, #664): each window the record covers, in the
 * snapshot's order, as its rain on its usual range. A window it covers but
 * cannot compare keeps its row, its value hatched; one it does not cover
 * yet is left out (#470).
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The strip, or null for a snapshot without `window_strip` (the
 *   tab then draws its number strip) or with no window covered.
 */
export function windowDots(snapshot: DashboardSnapshot): HTMLElement | null {
  const strip = snapshot.rain_tab.window_strip;
  if (!strip) return null;
  const rows = strip
    .map((item) => ({ item, window: snapshot.rain_tab.windows.find((w) => w.key === item.key) }))
    .filter(({ window }) => window && isShown(window.status));
  if (!rows.length) return null;
  const part = make("div", "rain-strip-part");
  part.dataset.part = "windows";
  part.append(partLabel("Each window against its usual", true));
  const list = make("ul", "rain-strip");
  for (const { item, window } of rows) {
    const value = windowValue(window!);
    const row = make("li", "rain-strip-row");
    row.dataset.key = item.key;
    if (item.mark) row.dataset.mark = item.mark;
    const head = make("p", "rain-strip-head");
    const name = make("span", "rain-strip-name", WINDOW_NAMES[item.key] ?? item.key);
    const figure = make("span", "rain-strip-value");
    head.append(name, figure);
    row.append(head);
    if (value === null || item.rain_in === null) {
      hatchGap(figure, window!.status, { compact: true });
      list.append(row);
      continue;
    }
    figure.append(make("b", "", value));
    if (item.mark) figure.append(" · ", make("span", "rain-strip-mark", item.mark));
    figure.append(spoken(` (${stripUsualWords(item)})`));
    const track = make("div", "rain-strip-track");
    row.append(track);
    drawSized(track, (width) => drawTrack(track, item, width));
    list.append(row);
  }
  list.setAttribute("aria-label", "Each window's rain against its usual range");
  const key = make("p", "rain-strip-key");
  const dot = make("span", "rain-strip-key-item");
  dot.append(make("span", "rain-strip-key-dot"), "the window's rain");
  const band = make("span", "rain-strip-key-item");
  band.append(
    make("span", "rain-strip-key-band"),
    "the same window in earlier years: their middle half from 4 years, else their range",
  );
  key.append(dot, band);
  part.append(list, key);
  return part;
}
