/**
 * The tile: the one block shape every tab's sections take (DESIGN.md "Tile",
 * "Source note", "Layout", spec #332).
 *
 * A tile runs kicker, finding, standfirst, chart or stats, then its folded
 * lines: "Why this leads" where a tile has one, then "Source". This module
 * offers the parts and the grid they sit in:
 *
 * - `TileSection`: what a tab declares per section, in its registry: its id,
 *   accessible name, kicker, column span, whether it belongs to a phone tab's
 *   fixed top, whether it is the now tile, and its `render`.
 * - `renderTiles`: renders a registry into a tab panel as a `.tile-grid` of
 *   `.tile` hosts, each opening with its kicker. The grid's columns follow
 *   the page layout switch (`layout.ts`): one on a phone, two from 760px,
 *   four from 1180px; a tile spans its declared columns, at most two in the
 *   two-column grid. On a phone (DESIGN.md "The Phone First-Screen Rule",
 *   "The Finding Row") the fixed top and any section kept open render as
 *   tiles; every other section is a finding row, a native `details` sharing
 *   one group name per tab, grouped in one tile. Closed, a row is its kicker
 *   over its finding with a chevron; open, the rest of the section. The page
 *   keeps one row open at a time in script too, where a browser does not
 *   honour `details name=`.
 * - `tileFinding`: the tile's finding, its `h2`, which a section's `render`
 *   appends first.
 * - `fold`: a closed native disclosure in Source type, a 32px target.
 * - `sourceNote`: the folded source: closed, "Source · <what>, archived
 *   <time>", with a hatched chip (one per reason) when the archive lags the
 *   live reading by more than 15 minutes or a gap hides part of the chart;
 *   open, the full method paragraph (`.ov-source`).
 *
 * `tiles.css` also carries `.ov-source` and `.ov-sr` (screen-reader-only
 * text), so a tab that renders tiles needs no other stylesheet for them.
 * - `archiveLagMs`, `lagCaveat`: the archive lag the chip names, from the
 *   snapshot's own two times, never the browser clock.
 *
 * Every value here comes from the snapshot: the archive time, the reading
 * time and the gap statuses Python decided. A tile states findings in the
 * live value; the lag lives only in its source note (DESIGN.md "The Live
 * Beats Archive Rule").
 */
import "./styles/tiles.css";

import { STALE_AFTER_MS, formatDuration, formatStamp } from "./format.js";
import { currentLayout } from "./layout.js";
import type { DashboardSnapshot, GapStatus } from "./snapshot.js";

/** How many of the four-column grid's columns a tile spans. */
export type TileSpan = 1 | 2 | 3 | 4;

/** One section of a tab, declared in that tab's registry. */
export interface TileSection {
  /** A stable id: the tile's `data-section`, for tests and styles. */
  id: string;
  /** The tile's accessible name. */
  label: string;
  /**
   * The tile's subject, sentence case, shown above its finding: fixed, or
   * read from the snapshot (the leading tile names what leads).
   */
  kicker: string | ((snapshot: DashboardSnapshot) => string);
  /**
   * Columns spanned in the four-column grid. Default 1. In the two-column
   * grid any span of 2 or more takes both columns; on a phone, one.
   */
  span?: TileSpan;
  /**
   * Whether the section belongs to its tab's fixed top on a phone (DESIGN.md
   * "The Phone First-Screen Rule"): shown open, never a finding row.
   */
  phoneTop?: boolean;
  /**
   * Whether the section stays an open tile on a phone below the fixed top,
   * never a finding row: a section that is itself an accordion (the Rain
   * events list).
   */
  phoneOpen?: boolean;
  /**
   * The now tile: the one tile on `now-surface`. Its kicker is for screen
   * readers only, as the temperature beside its condition is its finding.
   */
  now?: boolean;
  /** The leading tile: its kicker carries the leading-tile `now` dot. */
  leading?: boolean;
  /**
   * Fill the tile from the snapshot, after its kicker: the finding first
   * (`tileFinding`), then standfirst, chart and folded lines.
   *
   * @param host - The tile, attached to the panel, holding its kicker.
   * @param snapshot - The dashboard snapshot.
   */
  render(host: HTMLElement, snapshot: DashboardSnapshot): void;
}

/**
 * The kicker a section shows for a snapshot.
 *
 * @param section - The section.
 * @param snapshot - The dashboard snapshot.
 * @returns The kicker text.
 */
function kickerText(section: TileSection, snapshot: DashboardSnapshot): string {
  return typeof section.kicker === "function" ? section.kicker(snapshot) : section.kicker;
}

function renderKicker(
  section: TileSection,
  snapshot: DashboardSnapshot,
  tag: "p" | "span" = "p",
): HTMLElement {
  // The now tile's kicker is its heading for screen readers alone.
  const node = document.createElement(section.now ? "h2" : tag);
  node.className = section.now ? "tile-kicker ov-sr" : "tile-kicker";
  node.dataset.part = "kicker";
  if (section.leading) {
    const dot = document.createElement("span");
    dot.className = "tile-lead-dot";
    dot.setAttribute("aria-hidden", "true");
    node.append(dot);
  }
  node.append(kickerText(section, snapshot));
  return node;
}

/** How `renderTiles` sets out a tab. */
export interface TileOptions {
  /**
   * On a phone, the heading over the tab's finding rows ("In context" on
   * the Overview); none by default.
   */
  rowsHeading?: string;
}

/** The section's `data-` attributes, the same on a tile and a row. */
function mark(host: HTMLElement, section: TileSection): void {
  host.dataset.section = section.id;
  host.dataset.span = String(section.span ?? 1);
  if (section.phoneTop) host.dataset.phoneTop = "true";
  if (section.now) host.dataset.surface = "now";
}

/**
 * A section as a finding row: a closed `details` in the tab's group. Its
 * summary holds the kicker, then (once the section has rendered) the
 * finding and the chevron; `finishRow` moves them there.
 */
function startRow(section: TileSection, snapshot: DashboardSnapshot, group: string): HTMLElement {
  const row = document.createElement("details");
  row.className = "tile-row";
  row.setAttribute("name", group);
  row.dataset.row = "true";
  mark(row, section);
  const line = document.createElement("summary");
  line.className = "tile-row-line";
  line.append(renderKicker(section, snapshot, "span"));
  row.append(line);
  return row;
}

/** Finish a rendered row: the finding into its summary, the rest in a body. */
function finishRow(row: HTMLElement): void {
  const line = row.querySelector<HTMLElement>(":scope > summary")!;
  const finding = row.querySelector<HTMLElement>(":scope > [data-part=finding]");
  if (finding) line.append(finding);
  const chevron = document.createElement("span");
  chevron.className = "tile-row-chevron";
  chevron.setAttribute("aria-hidden", "true");
  line.append(chevron);
  const body = document.createElement("div");
  body.className = "tile-row-body";
  body.append(...[...row.children].filter((child) => child !== line));
  row.append(body);
}

/**
 * Keep one finding row open in a grid: opening one closes the others, for
 * browsers (and happy-dom) that do not honour `details name=`.
 */
function exclusiveRows(grid: HTMLElement): void {
  grid.addEventListener(
    "toggle",
    (event) => {
      const opened = event.target;
      if (!(opened instanceof HTMLElement) || !opened.matches("details.tile-row[open]")) return;
      for (const other of grid.querySelectorAll<HTMLDetailsElement>("details.tile-row[open]")) {
        if (other !== opened) other.open = false;
      }
    },
    // `toggle` does not bubble; the grid hears it on the way down.
    true,
  );
}

/**
 * Render a tab's sections into its panel as a grid of tiles, replacing what
 * was there. Each tile is attached before its section renders, so a chart
 * can measure its width when it draws. On a phone, each section that is not
 * in the fixed top (nor kept open) renders as a closed finding row; a run of
 * rows shares one tile, and every row in the panel one group name.
 *
 * @param panel - The tab's panel.
 * @param snapshot - The dashboard snapshot.
 * @param sections - The tab's registry, in order.
 * @param options - The rows' heading, on a phone.
 * @returns The grid.
 */
export function renderTiles(
  panel: HTMLElement,
  snapshot: DashboardSnapshot,
  sections: readonly TileSection[],
  options: TileOptions = {},
): HTMLElement {
  const grid = document.createElement("div");
  grid.className = "tile-grid";
  panel.replaceChildren(grid);
  const phone = currentLayout() === "phone";
  const group = `${panel.id || "tab"}-rows`;
  let rows: HTMLElement | null = null;
  let headed = false;
  for (const section of sections) {
    if (phone && !section.phoneTop && !section.phoneOpen) {
      if (!rows) {
        rows = document.createElement("section");
        rows.className = "tile tile-rows";
        rows.dataset.part = "rows";
        if (options.rowsHeading && !headed) {
          const heading = document.createElement("h2");
          heading.className = "tile-kicker tile-rows-heading";
          heading.textContent = options.rowsHeading;
          rows.setAttribute("aria-label", options.rowsHeading);
          rows.append(heading);
          headed = true;
        }
        grid.append(rows);
      }
      const row = startRow(section, snapshot, group);
      rows.append(row);
      section.render(row, snapshot);
      finishRow(row);
      continue;
    }
    rows = null;
    const host = document.createElement("section");
    host.className = "tile";
    mark(host, section);
    host.setAttribute("aria-label", section.label);
    host.append(renderKicker(section, snapshot));
    grid.append(host);
    section.render(host, snapshot);
  }
  if (phone) exclusiveRows(grid);
  return grid;
}

/**
 * A tile's finding: what the data shows, with the number in it.
 *
 * @param text - The finding the section composes.
 * @param options - `lead`: the leading tile's or a tab's top finding, in
 *   Finding, lead type.
 * @returns The `h2`.
 */
export function tileFinding(text: string, options: { lead?: boolean } = {}): HTMLElement {
  const node = document.createElement("h2");
  node.className = options.lead ? "tile-finding tile-finding-lead" : "tile-finding";
  node.dataset.part = "finding";
  node.textContent = text;
  return node;
}

/**
 * A closed disclosure at a tile's foot.
 *
 * @param summary - The closed line: text, or nodes (a source line and its chip).
 * @param body - What opening it shows.
 * @param part - Its `data-part`.
 * @returns The `details`, closed.
 */
export function fold(
  summary: string | readonly Node[],
  body: readonly Node[],
  part: string,
): HTMLDetailsElement {
  const details = document.createElement("details");
  details.className = "fold";
  details.dataset.part = part;
  const line = document.createElement("summary");
  line.className = "fold-line";
  if (typeof summary === "string") line.textContent = summary;
  else line.append(...summary);
  const inside = document.createElement("div");
  inside.className = "fold-body";
  inside.append(...body);
  details.append(line, inside);
  return details;
}

/**
 * How far the archive lags the live reading, from the snapshot's own times.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns Milliseconds; 0 when either time is missing or the archive is level.
 */
export function archiveLagMs(snapshot: DashboardSnapshot): number {
  const { archive_newest_ms: archived, observed_at_ms: observed } = snapshot;
  if (archived === null || observed === null) return 0;
  return Math.max(0, observed - archived);
}

/**
 * The chip's short reason for the archive lag, past 15 minutes.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns E.g. "archive 3 h 0 m behind"; null within 15 minutes.
 */
export function lagCaveat(snapshot: DashboardSnapshot): string | null {
  const lag = archiveLagMs(snapshot);
  return lag > STALE_AFTER_MS ? `archive ${formatDuration(lag)} behind` : null;
}

/** What a tile's source note says. */
export interface SourceNoteSpec {
  /** What the numbers come from, short: "five-minute readings". */
  what: string;
  /** The full method paragraph, shown when the note is opened. */
  method: string;
  /**
   * The short reason a gap status hides part of the chart, from the
   * snapshot's gap statuses ("5 windows not compared"); null when none does.
   */
  gaps?: string | null;
}

/**
 * The hatched chip: a 12px hatched swatch, then its reason, never on the
 * stripes.
 *
 * @param reason - The short reason.
 * @returns The chip.
 */
export function hatchChip(reason: string): HTMLElement {
  const chip = document.createElement("span");
  chip.className = "hatch-chip";
  chip.dataset.part = "caveat";
  const swatch = document.createElement("span");
  swatch.className = "hatch-swatch";
  swatch.setAttribute("aria-hidden", "true");
  const words = document.createElement("span");
  words.textContent = reason;
  chip.append(swatch, words);
  return chip;
}

/**
 * The closed source line's words.
 *
 * @param snapshot - The dashboard snapshot.
 * @param what - What the numbers come from.
 * @returns "Source · <what>, archived <time>".
 */
export function sourceLine(snapshot: DashboardSnapshot, what: string): string {
  const archived = snapshot.archive_newest_ms;
  // A time never breaks between its digits and "am" or "pm".
  const stamp = archived === null ? null : formatStamp(archived).replace(/ ([ap]m)$/, "\u00a0$1");
  const when = stamp === null ? "nothing archived yet" : `archived ${stamp}`;
  return `Source · ${what}, ${when}`;
}

/**
 * A tile's folded source note. The chip shows when the archive lags the
 * live reading past 15 minutes, or a gap hides part of the chart, and
 * nowhere else; the in-chart hatch stays, and the note explains it.
 *
 * @param snapshot - The dashboard snapshot.
 * @param spec - What the note says.
 * @returns The closed `details`; its body is the method paragraph,
 *   `data-part="source"`.
 */
export function sourceNote(snapshot: DashboardSnapshot, spec: SourceNoteSpec): HTMLDetailsElement {
  const words = document.createElement("span");
  words.className = "fold-words";
  words.textContent = sourceLine(snapshot, spec.what);
  const reasons = [lagCaveat(snapshot), spec.gaps ?? null].filter((r): r is string => r !== null);
  // One chip per reason, each a whole pill, wrapping as a unit.
  const line: Node[] = [words, ...reasons.map(hatchChip)];
  const method = document.createElement("p");
  method.className = "ov-source";
  method.dataset.part = "source";
  method.textContent = spec.method;
  return fold(line, [method], "source-note");
}

/**
 * A count of things a gap hides, as the chip's short reason.
 *
 * @param count - How many are not compared.
 * @param one - The singular noun ("window").
 * @param many - The plural noun ("windows").
 * @returns E.g. "5 windows not compared"; null for none.
 */
export function notCompared(count: number, one: string, many: string): string | null {
  if (count <= 0) return null;
  return `${count} ${count === 1 ? one : many} not compared`;
}

/**
 * How many of a set of gap statuses are hatched as not compared: invalid,
 * and not merely under way (the current window or month, compared once over).
 *
 * @param statuses - The windows' or months' gap statuses.
 * @returns The count.
 */
export function hatchedCount(statuses: readonly GapStatus[]): number {
  return statuses.filter((s) => s.status === "invalid" && s.reason !== "in_progress").length;
}

/**
 * Counts of several kinds of thing a gap hides, as one chip's short reason.
 *
 * @param counts - Each kind's count, singular and plural noun, in order.
 * @returns E.g. "5 windows, 2 days not compared", naming only the kinds
 *   with a count; null when none has one.
 */
export function gapReasons(counts: readonly [number, string, string][]): string | null {
  const parts = counts
    .filter(([count]) => count > 0)
    .map(([count, one, many]) => `${count} ${count === 1 ? one : many}`);
  return parts.length ? `${parts.join(", ")} not compared` : null;
}

/**
 * The folded lines at a tile's foot, in a group pushed to the tile's
 * bottom edge so tiles in a row line their source notes up.
 *
 * @param folds - "Why this leads", then "Source".
 * @returns The group.
 */
export function tileFolds(...folds: readonly HTMLElement[]): HTMLElement {
  const group = document.createElement("div");
  group.className = "tile-folds";
  group.append(...folds);
  return group;
}
