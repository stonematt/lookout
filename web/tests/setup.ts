/**
 * The tests format times through `format.ts`, which takes its zone from a
 * snapshot. The goldens are a Pacific station, so tests that format without
 * rendering a snapshot start in its zone; a test of another zone sets its own.
 */
import { beforeEach } from "vitest";

import { setStationZone } from "../src/format.js";

beforeEach(() => {
  setStationZone("America/Los_Angeles");
});
