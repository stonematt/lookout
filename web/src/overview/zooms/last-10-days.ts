/**
 * The Last 10 days zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): under
 * the tile's number and delta, the ten-day high-low list the Overview drew
 * before the mosaic, at the zoom's full width: each day's low to high over
 * the usual range for its date, newest first. No source fold (prototype
 * decision 22).
 *
 * The content is `temperature.ts`'s; this file only names it.
 *
 * Offers `tenDaysZoom`.
 */
import { tenDaysTile } from "../mosaic-tiles.js";
import { renderTenDaysZoom } from "../temperature.js";
import type { ZoomDefinition } from "../zoom.js";

/** The Last 10 days zoom. */
export const tenDaysZoom: ZoomDefinition = {
  slug: tenDaysTile.slug,
  title: tenDaysTile.title,
  lead: (snapshot) => tenDaysTile.read(snapshot),
  panels: [{ id: "ten-days", title: "Day by day", wide: true, render: renderTenDaysZoom }],
};
