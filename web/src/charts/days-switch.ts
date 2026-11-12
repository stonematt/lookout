/**
 * The 30 / 90 switch (DESIGN.md, "Daily bars" and "Rain-by-hour grid"): a
 * chart of days drawn under 520px of chart width (the Chart Width Rule)
 * shows its last 30 days, with a segmented switch to show all 90. From
 * 520px the chart shows 90 and the switch is hidden. The switch slices the
 * days the snapshot sent; it is display state, never a decision.
 *
 * Offers `drawDays`, which draws a chart of days through one switch, and
 * `segmented`, the segmented control under it, which the wind zoom's
 * window control shares.
 */
import "./days-switch.css";

import { NARROW, PHONE_DAYS, drawSized } from "./svg.js";
import { make } from "../dom.js";

/** A chart's 30 / 90 switch. */
interface DaysSwitch {
  /** The segmented control; mount it above its chart. */
  el: HTMLElement;
  /**
   * How many days a chart drawn at `width` shows, hiding the switch at
   * desktop chart width or when there are no more than 30 days. Call it
   * from the chart's draw.
   */
  days(width: number): number;
  /** The days chosen on a phone, 30 until the switch is pressed. */
  chosen(): number;
}

/** One segment of a segmented control. */
export interface Segment {
  /** What it picks: "30", "24h". */
  key: string;
  /** Its words: "30 days", "24 h". */
  text: string;
}

/**
 * A segmented control (DESIGN.md "Segmented control"): a named group of
 * buttons, the chosen one `aria-pressed`. Pressing another chooses it; the
 * caller redraws.
 *
 * @param className - The group's class: "days-seg", and any of the caller's.
 * @param label - The group's accessible name.
 * @param segments - The buttons, in order.
 * @param chosen - The key chosen now.
 * @param choose - Called with a newly pressed segment's key.
 * @returns The group, and its buttons, in order, for the caller's data.
 */
export function segmented(
  className: string,
  label: string,
  segments: readonly Segment[],
  chosen: () => string,
  choose: (key: string) => void,
): { el: HTMLElement; buttons: HTMLButtonElement[] } {
  const el = make("div", className);
  el.setAttribute("role", "group");
  el.setAttribute("aria-label", label);
  const press = () => {
    buttons.forEach((b, i) =>
      b.setAttribute("aria-pressed", String(segments[i]!.key === chosen())),
    );
  };
  const buttons = segments.map(({ key, text }) => {
    const button = make("button", "", text);
    button.type = "button";
    button.onclick = () => {
      if (chosen() === key) return;
      choose(key);
      press();
    };
    el.append(button);
    return button;
  });
  press();
  return { el, buttons };
}

/**
 * A 30 / 90 switch for one chart.
 *
 * @param label - The control's accessible name, e.g. "Days of rain shown".
 * @param onChange - Redraws the chart; called when a segment is pressed.
 * @param all - The days the snapshot sends.
 * @returns The switch.
 */
function daysSwitch(label: string, onChange: () => void, all = 90): DaysSwitch {
  let chosen = Math.min(PHONE_DAYS, all);
  const counts = [PHONE_DAYS, all];
  const { el, buttons } = segmented(
    "days-seg",
    label,
    counts.map((days) => ({ key: String(days), text: `${days} days` })),
    () => String(chosen),
    (key) => {
      chosen = Number(key);
      onChange();
    },
  );
  buttons.forEach((b, i) => (b.dataset.days = String(counts[i])));
  el.hidden = true;
  return {
    el,
    days(width: number): number {
      // A record of a month or less has nothing to switch between.
      const phone = width < NARROW && all > PHONE_DAYS;
      el.hidden = !phone;
      return phone ? chosen : all;
    },
    chosen: () => chosen,
  };
}

/** What `drawDays` draws, and the words that follow the days shown. */
export interface DaysChart {
  /** The days the snapshot sends. */
  all: number;
  /** The chart's accessible name, given the days it shows. */
  name: (days: number) => string;
  /** The tile's standfirst, which follows the days shown; none by default. */
  standfirst?: HTMLElement | null;
  /** The standfirst's words, given the days shown; with `standfirst`. */
  lede?: (days: number) => string;
  /** Draws the chart's last `days` at a width. */
  draw: (days: number, width: number) => void;
}

/**
 * Draw a chart of days at its width, through a 30 / 90 switch mounted just
 * above it: the last 30 at phone chart width until the switch shows 90.
 *
 * @param chart - The chart's figure, already in its tile.
 * @param spec - What to draw.
 * @returns Draws it again (when the selected day changes).
 */
export function drawDays(chart: HTMLElement, spec: DaysChart): () => void {
  const span = daysSwitch("Days shown", () => redraw(), spec.all);
  chart.before(span.el);
  const redraw = drawSized(chart, (width) => {
    const days = span.days(width);
    chart.setAttribute("aria-label", `${spec.name(days)}. Use the arrow keys to read each mark.`);
    if (spec.standfirst && spec.lede) spec.standfirst.textContent = spec.lede(days);
    spec.draw(days, width);
  });
  return redraw;
}
