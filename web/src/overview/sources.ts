/**
 * The Overview's sources footer (prototype decision 22, #472; made short,
 * #604): one footer at the foot of the Overview. For each tile drawn, one
 * short line naming its source and comparison, at most about 12 words; for a
 * tile the snapshot leaves out (Daylight on a station with no location), why
 * (`TileSection.absent`, #477). The method paragraphs, #595's percentile rule
 * among them, sit behind one closed disclosure after the lines, "How these
 * are worked out". The Overview's tiles and zooms carry no "Source · …" fold
 * and no method paragraph; the Rain, Rain events and Solar tabs keep theirs.
 * No archive time is repeated here: the header's freshness line says it once.
 *
 * Each entry comes from its section's `source` (`TileSection.source`, in
 * `tiles.ts`): `line` for the short line, `method` for the disclosure. A
 * mosaic tile declares it in its definition (`MosaicTile.source`,
 * `mosaic.ts`), so a new tile brings its own.
 *
 * The footer ends with the release version line (#536): `releaseLine()` as
 * `p.ov-sources-version`, after the disclosure.
 *
 * Offers `sourcesFooter`.
 */
import "./sources.css";

import { releaseLine } from "../release.js";
import type { DashboardSnapshot } from "../snapshot.js";
import { fold, type TileSection } from "../tiles.js";

/** One tile's entry: its short line, and its method or null for none. */
interface Entry {
  name: string;
  line: string;
  method: string | null;
}

/** A section's entry, or null for none. */
function entryOf(snapshot: DashboardSnapshot, section: TileSection): Entry | null {
  if (section.shown && !section.shown(snapshot)) {
    const why = section.absent?.(snapshot) ?? null;
    return why === null ? null : { name: section.label, line: why, method: null };
  }
  return section.source ? { name: section.label, ...section.source(snapshot) } : null;
}

/** A `dl` row: the tile's name as `dt`, the words as `dd`. */
function row(name: string, text: string, className: string): HTMLElement {
  const item = document.createElement("div");
  item.dataset.source = name;
  const term = document.createElement("dt");
  term.textContent = name;
  const body = document.createElement("dd");
  body.className = className;
  body.textContent = text;
  item.append(term, body);
  return item;
}

/** A `dl` of rows. */
function list(className: string, rows: readonly HTMLElement[]): HTMLDListElement {
  const dl = document.createElement("dl");
  dl.className = className;
  dl.append(...rows);
  return dl;
}

/**
 * The sources footer: a short line for each section drawn, or, for a
 * section not drawn because the snapshot holds nothing for it, why; then the
 * methods behind one closed disclosure; then the version line.
 *
 * @param snapshot - The dashboard snapshot.
 * @param sections - The Overview's sections, in order, drawn or not.
 * @returns The `footer`, or null when no section has a line.
 */
export function sourcesFooter(
  snapshot: DashboardSnapshot,
  sections: readonly TileSection[],
): HTMLElement | null {
  const entries = sections
    .map((section) => entryOf(snapshot, section))
    .filter((entry): entry is Entry => entry !== null);
  if (!entries.length) return null;
  const footer = document.createElement("footer");
  footer.className = "ov-sources";
  footer.dataset.part = "sources";
  const heading = document.createElement("h2");
  heading.className = "ov-sources-heading";
  heading.id = "overview-sources";
  heading.textContent = "Sources";
  footer.setAttribute("aria-labelledby", heading.id);
  const lines = list(
    "ov-sources-list",
    entries.map((entry) => row(entry.name, entry.line, "ov-sources-line")),
  );
  footer.append(heading, lines);
  const methods = entries
    .filter((entry): entry is Entry & { method: string } => entry.method !== null)
    .map((entry) => row(entry.name, entry.method, "ov-sources-method"));
  if (methods.length) {
    const how = fold("How these are worked out", [list("ov-sources-methods", methods)], "how");
    how.classList.add("ov-sources-how");
    footer.append(how);
  }
  const version = document.createElement("p");
  version.className = "ov-sources-version";
  version.textContent = releaseLine();
  footer.append(version);
  return footer;
}
