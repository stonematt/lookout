/**
 * The Overview's sources footer (prototype decision 22, #472): one footer at
 * the foot of the Overview listing, for each tile drawn, what it reads and
 * how its comparison is made, and for a tile the snapshot leaves out
 * (Daylight on a station with no location), why: `TileSection.absent`,
 * #477. The Overview's tiles carry no "Source · …"
 * fold and no method paragraph; the Rain, Rain events and Solar tabs keep
 * theirs. No archive time is repeated here: the header's freshness line
 * says it once.
 *
 * Each line comes from its section's `source` (`TileSection.source`, in
 * `tiles.ts`); a mosaic tile declares it in its definition
 * (`MosaicTile.source`, `mosaic.ts`), so a new tile brings its own line.
 *
 * The footer ends with the release version line (#536): `releaseLine()` as
 * `p.ov-sources-version`, after the list.
 *
 * Offers `sourcesFooter`.
 */
import "./sources.css";

import { releaseLine } from "../release.js";
import type { DashboardSnapshot } from "../snapshot.js";
import type { TileSection } from "../tiles.js";

/** One footer line: a drawn section's source, or why a section is not drawn. */
interface Line {
  name: string;
  what: string;
  method: string | null;
}

/** A section's footer line, or null for none. */
function lineOf(snapshot: DashboardSnapshot, section: TileSection): Line | null {
  if (section.shown && !section.shown(snapshot)) {
    const why = section.absent?.(snapshot) ?? null;
    return why === null ? null : { name: section.label, what: why, method: null };
  }
  return section.source ? { name: section.label, ...section.source(snapshot) } : null;
}

/**
 * The sources footer: for each section drawn, its source; for a section not
 * drawn because the snapshot holds nothing for it, why (`absent`, #477:
 * Daylight on a station with no location), in its place.
 *
 * @param snapshot - The dashboard snapshot.
 * @param sections - The Overview's sections, in order, drawn or not.
 * @returns The `footer`, or null when no section has a line.
 */
export function sourcesFooter(
  snapshot: DashboardSnapshot,
  sections: readonly TileSection[],
): HTMLElement | null {
  const lines = sections
    .map((section) => lineOf(snapshot, section))
    .filter((line): line is Line => line !== null);
  if (!lines.length) return null;
  const footer = document.createElement("footer");
  footer.className = "ov-sources";
  footer.dataset.part = "sources";
  const heading = document.createElement("h2");
  heading.className = "ov-sources-heading";
  heading.id = "overview-sources";
  heading.textContent = "Sources and method";
  footer.setAttribute("aria-labelledby", heading.id);
  const list = document.createElement("dl");
  list.className = "ov-sources-list";
  for (const line of lines) {
    const item = document.createElement("div");
    item.dataset.source = line.name;
    const term = document.createElement("dt");
    term.textContent = line.name;
    const what = document.createElement("dd");
    what.className = "ov-sources-what";
    what.textContent = line.what;
    item.append(term, what);
    if (line.method !== null) {
      const method = document.createElement("dd");
      method.className = "ov-sources-method";
      method.textContent = line.method;
      item.append(method);
    }
    list.append(item);
  }
  const version = document.createElement("p");
  version.className = "ov-sources-version";
  version.textContent = releaseLine();
  footer.append(heading, list, version);
  return footer;
}
