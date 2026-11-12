/**
 * The zoom registry (#474): every zoom the Overview can open, by slug.
 *
 * To add a zoom: write `zooms/<slug>.ts` exporting one `ZoomDefinition`
 * (`../zoom.ts`) whose `slug` is its mosaic tile's, then add it to `ZOOMS`
 * below, one line, in the mosaic's order. The shell (`../zoom.ts`) does the
 * rest: the route `#overview/<slug>`, the tile's `aria-expanded` and
 * `aria-controls`, the heading, Close and Back, focus, Escape and motion.
 * A tile whose slug is not here keeps its link, and its route falls back to
 * the Overview until its zoom lands.
 *
 * Offers `ZOOMS`.
 */
import type { ZoomDefinition } from "../zoom.js";
import { temperatureZoom } from "./temperature.js";

/** The registered zooms, in the mosaic's order. */
export const ZOOMS: readonly ZoomDefinition[] = [
  temperatureZoom, // #474
];
