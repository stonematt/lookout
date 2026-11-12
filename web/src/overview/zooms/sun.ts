/**
 * The Sun zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): under the Sun
 * tile's number and delta, the Solar tab's top section, Sunlight now (the
 * last 24 hours against the usual range), drawn by the Solar tab's own
 * `renderSunlightNow`, then "More in Solar". No source note (prototype
 * decision 22): the Overview's sources footer carries the tile's line.
 *
 * Offers `sunZoom`.
 */
import { renderSunlightNow } from "../../solar/index.js";
import { sunTile } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";
import { onwardLink } from "./parts.js";

/** The Sun zoom. */
export const sunZoom: ZoomDefinition = {
  slug: sunTile.slug,
  title: sunTile.title,
  lead: (snapshot) => sunTile.read(snapshot),
  panels: [
    {
      id: "sunlight-now",
      title: "Sunlight now",
      wide: true,
      render(host, snapshot) {
        renderSunlightNow(host, snapshot, { note: false });
        onwardLink(host, "#solar", "More in Solar");
      },
    },
  ],
};
