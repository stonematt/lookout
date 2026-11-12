/**
 * The "More in <tab>" panel a zoom ends in (spec #460 stories 36–38; board
 * `RainZoom`): one link to the tab that holds the rest of the subject, its
 * name over a line saying what is there, with a chevron.
 *
 * The panel's own heading (the shell's `h3`, "More in Rain") moves into the
 * link, so the whole card is one target named by the heading and its line.
 * Following it changes the route to the tab, which closes the zoom; the
 * shell and the tab shell do the rest.
 *
 * Offers `morePanel` and `MoreTab`, the tab it links to.
 */
import "./more.css";

import { make } from "../../dom.js";
import type { ZoomPanel } from "../zoom.js";

/** The tab a "More in" panel links to. */
export interface MoreTab {
  /** The tab's hash, without `#`: "rain". */
  hash: string;
  /** The tab's name: "Rain". */
  name: string;
}

/**
 * A zoom's closing panel: "More in <tab>", linking to the tab.
 *
 * @param tab - The tab to link to.
 * @param line - What else the tab holds, e.g. "Each day of the last 90, dry
 *   spells, rain by the hour".
 * @returns The panel, full width.
 */
export function morePanel(tab: MoreTab, line: string): ZoomPanel {
  return {
    id: "more",
    title: `More in ${tab.name}`,
    wide: true,
    render(host) {
      const link = make("a", "zoom-more");
      link.href = `#${tab.hash}`;
      const title = host.querySelector(".zoom-panel-title");
      if (title) link.append(title);
      // A space between the heading and its line, so the link's text and
      // name read as two phrases; the grid draws no whitespace.
      link.append(" ", make("span", "zoom-more-line", line));
      const chevron = make("span", "zoom-more-chevron", "›");
      chevron.setAttribute("aria-hidden", "true");
      link.append(chevron);
      host.append(link);
    },
  };
}
