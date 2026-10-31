/**
 * The Overview tab, composed from sections in the spec's order (#66, "Front
 * end"): the first screen, the hero lead with "Why this leads", temperature
 * across periods, the 10-day high-low list, "now against normal" strips, the
 * almanac grid, and the rolling-window tables.
 *
 * A section is a module under `web/src/overview/` exporting an
 * `OverviewSection`. To add one, import it here and add it to `SECTIONS` at
 * its place in that order: one import line and one list line. Each section
 * renders into its own `<section>` host, already attached to the page, so a
 * chart can measure its width when it draws.
 */
import "./overview.css";

import type { DashboardSnapshot } from "../snapshot.js";
import { firstScreen } from "./first-screen.js";
import { heroLead } from "./hero.js";

/** One block of the Overview. */
export interface OverviewSection {
  /** A stable id: the host's `data-section`, for tests and styles. */
  id: string;
  /** The host's accessible name. */
  label: string;
  /**
   * Fill the host from the snapshot.
   *
   * @param host - An empty `<section>`, attached to the panel.
   * @param snapshot - The dashboard snapshot.
   */
  render(host: HTMLElement, snapshot: DashboardSnapshot): void;
}

/** The Overview's sections, in the spec's order. */
export const SECTIONS: readonly OverviewSection[] = [
  firstScreen, // #78
  heroLead, // #79 the hero lead and "Why this leads"
  // #81 temperature across periods, the 10-day list, "now against normal"
  // #82 the almanac grid and the rolling-window rain tables
];

/**
 * Render the Overview into its tab panel, replacing what was there.
 *
 * @param panel - The Overview tab's panel.
 * @param snapshot - The dashboard snapshot.
 * @param sections - The sections to render, in order.
 */
export function renderOverview(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  sections: readonly OverviewSection[] = SECTIONS,
): void {
  panel.replaceChildren();
  for (const section of sections) {
    const host = document.createElement("section");
    host.className = "ov-section";
    host.dataset.section = section.id;
    host.setAttribute("aria-label", section.label);
    panel.append(host);
    section.render(host, snapshot);
  }
}
