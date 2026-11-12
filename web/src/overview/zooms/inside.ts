/**
 * The Inside zoom (#477, spec #460; boards `Zoom`, `ZoomDark`): the Inside
 * tile's rows at full size, one per indoor sensor still reporting
 * (`indoor`, #466): its name, its temperature and humidity in the tile's
 * own words (`sensorRow`), then its last 24 hours, hourly, as a line
 * the panel's width, its newest hour the now dot. No source note (prototype
 * decision 22).
 *
 * Offers `insideZoom`.
 */
import { scale } from "../../charts/svg.js";
import { spoken, svgEl } from "../../dom.js";
import { degrees } from "../../format.js";
import type { DashboardSnapshot, IndoorSensor } from "../../snapshot.js";
import { insideTile, lastIndex, sensorRow, traceMarks } from "../mosaic-conditions.js";
import { miniChart } from "../mosaic.js";
import type { ZoomDefinition } from "../zoom.js";
import { factValue, sizedChart } from "./parts.js";

/** The line's height: the mini chart's, and room for its labels. */
const HEIGHT = 112;

/** The strip under the line that holds its end labels. */
const AXIS = 18;

/** The space at the left for the low and high labels. */
const LEFT = 36;

/**
 * A sensor's last 24 hours at a width: the line, its low and high at the
 * left, "24 h ago" and "now" under it, the newest hour as the now dot.
 *
 * @param sensor - The indoor sensor.
 * @param width - The width to draw at.
 * @returns The chart, or null with fewer than two hours.
 */
function sensorChart(sensor: IndoorSensor, width: number): SVGSVGElement | null {
  const values = sensor.trace.temperature_f;
  if (values.length < 2 || lastIndex(values) < 0) return null;
  const numbers = values.filter((v): v is number => v !== null);
  const low = Math.min(...numbers);
  const high = Math.max(...numbers);
  const bottom = HEIGHT - AXIS;
  const x = scale(0, values.length - 1, LEFT, width - 6);
  const y = scale(low - 1, high + 1, bottom - 6, 8);
  const svg = miniChart(width, HEIGHT);
  svg.append(
    svgEl("line", { class: "mosaic-base", x1: LEFT, x2: width, y1: bottom, y2: bottom }),
    svgEl("text", { class: "mosaic-label", x: 0, y: y(high) + 4 }, degrees(high, 0)),
    svgEl("text", { class: "mosaic-label", x: 0, y: y(low) + 4 }, degrees(low, 0)),
    svgEl("text", { class: "mosaic-label", x: LEFT, y: HEIGHT - 2 }, "24 h ago"),
    svgEl("text", { class: "mosaic-label", x: width, y: HEIGHT - 2, "text-anchor": "end" }, "now"),
  );
  svg.append(...traceMarks(values, x, y, 4));
  return svg;
}

/** Each sensor: its name, its reading, then its 24 hours. */
function renderSensors(host: HTMLElement, snapshot: DashboardSnapshot): void {
  const list = document.createElement("div");
  list.className = "zoom-sensors";
  host.append(list);
  for (const sensor of snapshot.indoor ?? []) {
    const row = sensorRow(sensor);
    const item = document.createElement("section");
    item.className = "zoom-sensor";
    item.dataset.part = "sensor";
    const name = document.createElement("h4");
    name.className = "zoom-sensor-name";
    name.textContent = row.name;
    const reading = document.createElement("p");
    reading.className = "zoom-sensor-reading";
    factValue(reading, row.value, "sensor-value");
    // The aside as the tile says it: "humidity 45%".
    if (row.aside) reading.append(spoken(` ${row.aside.spoken} `), row.aside.text);
    item.append(name, reading);
    list.append(item);
    sizedChart(item, "sensor-line", (width) => sensorChart(sensor, width));
  }
}

/** The Inside zoom. */
export const insideZoom: ZoomDefinition = {
  slug: insideTile.slug,
  title: insideTile.title,
  panels: [
    {
      id: "sensors",
      title: "The last 24 hours",
      wide: true,
      shown: (snapshot) => (snapshot.indoor?.length ?? 0) > 0,
      render: renderSensors,
    },
  ],
};
