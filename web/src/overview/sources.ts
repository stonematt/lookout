/**
 * The Overview's sources footer (prototype decision 22, #472): one footer at
 * the foot of the Overview listing, for each tile drawn, what it reads and
 * how its comparison is made. The Overview's tiles carry no "Source · …"
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

/**
 * The sources footer for the sections drawn.
 *
 * @param snapshot - The dashboard snapshot.
 * @param sections - The sections the Overview drew, in order.
 * @returns The `footer`, or null when no section names a source.
 */
export function sourcesFooter(
  snapshot: DashboardSnapshot,
  sections: readonly TileSection[],
): HTMLElement | null {
  const lines = sections.flatMap((section) =>
    section.source ? [{ name: section.label, ...section.source(snapshot) }] : [],
  );
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
    const method = document.createElement("dd");
    method.className = "ov-sources-method";
    method.textContent = line.method;
    item.append(term, what, method);
    list.append(item);
  }
  const version = document.createElement("p");
  version.className = "ov-sources-version";
  version.textContent = releaseLine();
  footer.append(heading, list, version);
  return footer;
}
