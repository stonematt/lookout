/**
 * The note a tab shows when the refresh run could not update its catalog
 * (#118): the run published with the last good catalog, so the tab's
 * figures are older than the rest of the page.
 */
import { formatStamp } from "./format.js";
import type { CatalogName, DashboardSnapshot } from "./snapshot.js";
import { staleCatalog } from "./snapshot.js";

const SUBJECT: Record<CatalogName, { catalog: string; figures: string }> = {
  events: { catalog: "The Event Catalog", figures: "these events" },
  energy: { catalog: "The energy catalog", figures: "the solar energy figures" },
};

/**
 * The stale-catalog note for one tab.
 *
 * @param snapshot - The dashboard snapshot.
 * @param catalog - The tab's catalog.
 * @returns The note, or null when the run updated the catalog.
 */
export function staleCatalogNote(
  snapshot: DashboardSnapshot,
  catalog: CatalogName,
): string | null {
  const stale = staleCatalog(snapshot, catalog);
  if (!stale) return null;
  const { catalog: name, figures } = SUBJECT[catalog];
  const when =
    stale.as_of_ms === null
      ? "may be out of date"
      : `are as of ${formatStamp(stale.as_of_ms)}`;
  return `${name} could not be updated on the latest refresh, so ${figures} ${when}.`;
}
