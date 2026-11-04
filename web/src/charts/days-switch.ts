/**
 * The 30 / 90 switch (DESIGN.md, "Daily bars" and "Rain-by-hour grid"): a
 * chart of days drawn under 520px of chart width (the Chart Width Rule)
 * shows its last 30 days, with a segmented switch to show all 90. From
 * 520px the chart shows 90 and the switch is hidden. The switch slices the
 * days the snapshot sent; it is display state, never a decision.
 *
 * Offers `daysSwitch`.
 */
import "./days-switch.css";

import { NARROW, PHONE_DAYS } from "./svg.js";
import { make } from "../dom.js";

/** A chart's 30 / 90 switch. */
export interface DaysSwitch {
  /** The segmented control; mount it above its chart. */
  el: HTMLElement;
  /**
   * How many days a chart drawn at `width` shows, hiding the switch at
   * desktop chart width. Call it from the chart's draw.
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
export function daysSwitch(label: string, onChange: () => void, all = 90): DaysSwitch {
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
      const phone = width < NARROW;
      el.hidden = !phone;
      return phone ? chosen : all;
    },
    chosen: () => chosen,
  };
}
