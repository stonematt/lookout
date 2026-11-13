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
import { spoken } from "../../dom.js";
import { degrees } from "../../format.js";
import type { DashboardSnapshot, IndoorSensor } from "../../snapshot.js";
import { insideTile, sensorRow } from "../mosaic-conditions.js";
import type { ZoomDefinition } from "../zoom.js";
import { dayTrace, factValue, sizedChart } from "./parts.js";

/** A sensor's last 24 hours at a width (`dayTrace`), its low and high in whole degrees. */
function sensorChart(sensor: IndoorSensor, width: number): SVGSVGElement | null {
  return dayTrace(sensor.trace.temperature_f, width, (v) => degrees(v, 0), 1, {
    value: sensor.temperature_f,
    text: degrees,
  });
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
