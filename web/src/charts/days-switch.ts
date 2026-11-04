/**
 * The 30 / 90 switch (DESIGN.md, "Daily bars" and "Rain-by-hour grid"): a
 * chart of days drawn under 520px of chart width (the Chart Width Rule)
 * shows its last 30 days, with a segmented switch to show all 90. From
 * 520px the chart shows 90 and the switch is hidden. The switch slices the
 * days the snapshot sent; it is display state, never a decision.
 *
 * Offers `drawDays`, which draws a chart of days through one switch.
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
  const el = make("div", "days-seg");
  el.setAttribute("role", "group");
  el.setAttribute("aria-label", label);
  el.hidden = true;
  const buttons = [PHONE_DAYS, all].map((days) => {
    const button = make("button", "", `${days} days`);
    button.type = "button";
    button.dataset.days = String(days);
    button.onclick = () => {
      if (chosen === days) return;
      chosen = days;
      press();
      onChange();
    };
    el.append(button);
    return button;
  });
  const press = () => {
    for (const b of buttons)
      b.setAttribute("aria-pressed", String(b.dataset.days === String(chosen)));
  };
  press();
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
  /** The tile's standfirst, which follows the days shown; null for none. */
  standfirst: HTMLElement | null;
  /** The standfirst's words, given the days shown. */
  lede: (days: number) => string;
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
    if (spec.standfirst) spec.standfirst.textContent = spec.lede(days);
    spec.draw(days, width);
  });
  return redraw;
}
