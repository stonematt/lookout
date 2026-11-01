/**
 * The lane server's read functions: `/dashboard` and `/current` served by
 * the real handlers from the golden snapshot documents in
 * `tests/golden/snapshots/`, with no secrets and no network.
 *
 * - `?fixture=dry|raining` (or `LOOKOUT_FIXTURE`) picks the golden; `dry` by
 *   default.
 * - `?age=<minutes>` (or `LOOKOUT_AGE_MIN`) shifts every `_ms` time by one
 *   offset so the newest observation is that many minutes old, and every
 *   ISO day (a `_day` or `day` field, a day `key` or a `first..last` span)
 *   by the Pacific days that offset crosses, so day labels agree with the
 *   times. Month keys ("YYYY-MM"), water years and day indexes are not
 *   moved. Without `age` the goldens are served as recorded, which is long
 *   stale.
 * - `/current` carries no live reading (`last_data: null`): Ambient is never
 *   called in dev.
 * - `/event?start=<ms>` serves the golden curve files in
 *   `tests/golden/snapshots/events/<scenario>.<start_ms>.json`, shifted by
 *   the same offset (so a start read from a shifted `/dashboard` finds its
 *   file, matched to the nearest recorded start).
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Plugin, PreviewServer, ViteDevServer } from "vite";

import { DAY_MS, addDays, localDay } from "../src/format.js";
import type { Env } from "./env.js";
import { handleEvent } from "./event.js";
import type { ReadSources } from "./handlers.js";
import { errorResponse, handleCurrent, handleDashboard } from "./handlers.js";
import type { SnapshotKind } from "./storj.js";

/** Where the goldens live, relative to this file. */
const GOLDEN_DIR = resolve(import.meta.dirname, "../../tests/golden/snapshots");

const SCENARIOS = ["dry", "raining"] as const;
type Scenario = (typeof SCENARIOS)[number];

/** One golden document, parsed. */
function goldenDoc(scenario: Scenario, kind: SnapshotKind): Record<string, unknown> {
  const text = readFileSync(`${GOLDEN_DIR}/${scenario}.${kind}.json`, "utf8");
  return JSON.parse(text) as Record<string, unknown>;
}

/**
 * Each `age` request shifts by its own clock, so a start read from an
 * earlier `/dashboard` is off by the time since; match within this.
 */
const CURVE_MATCH_MS = 6 * 60 * 60_000;

/** The golden curve file nearest a recorded start, or null when none is near. */
function goldenCurve(scenario: Scenario, startMs: number): string | null {
  const dir = `${GOLDEN_DIR}/events`;
  if (!existsSync(dir)) return null;
  const starts = readdirSync(dir)
    .map((name) => new RegExp(`^${scenario}\\.(\\d+)\\.json$`).exec(name)?.[1])
    .filter((start): start is string => start !== undefined)
    .map(Number);
  const near = starts.find((start) => Math.abs(start - startMs) <= CURVE_MATCH_MS);
  return near === undefined ? null : readFileSync(`${dir}/${scenario}.${near}.json`, "utf8");
}

/** How far a served golden moves: milliseconds, and the Pacific days that crosses. */
interface Shift {
  ms: number;
  days: number;
}

/** The Pacific days between two times' calendar days. */
function daysBetween(fromMs: number, toMs: number): number {
  const day = (ms: number) => Date.parse(localDay(ms));
  return Math.round((day(toMs) - day(fromMs)) / DAY_MS);
}

/** A time, or each time in a list of times, moved. */
function shiftTime(item: unknown, shift: Shift): unknown {
  if (typeof item === "number") return item + shift.ms;
  if (Array.isArray(item) && item.every((t) => typeof t === "number")) {
    return item.map((t: number) => t + shift.ms);
  }
  return shiftTimes(item, shift);
}

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;
const DAY_SPAN = /^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$/;

/** An ISO day, or a `first..last` span of them, moved; anything else as is. */
function shiftDay(item: unknown, shift: Shift): unknown {
  if (typeof item !== "string") return shiftTimes(item, shift);
  if (ISO_DAY.test(item)) return addDays(item, shift.days);
  const span = DAY_SPAN.exec(item);
  return span ? `${addDays(span[1]!, shift.days)}..${addDays(span[2]!, shift.days)}` : item;
}

/** Move every `_ms` time and every ISO day, at any depth. */
function shiftTimes(value: unknown, shift: Shift): unknown {
  if (Array.isArray(value)) return value.map((item) => shiftTimes(item, shift));
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      key,
      key.endsWith("_ms")
        ? shiftTime(item, shift)
        : key.endsWith("_day") || key === "day" || key === "key"
          ? shiftDay(item, shift)
          : shiftTimes(item, shift),
    ]),
  );
}

/** Read sources over one golden, shifted so it is `ageMinutes` old. */
function fixtureSources(
  scenario: Scenario,
  ageMinutes: number | null,
  nowMs: number,
): ReadSources {
  const head = goldenDoc(scenario, "dashboard");
  const mac = (head.station as { mac: string }).mac;
  const observed = head.observed_at_ms as number | null;
  const delta =
    ageMinutes === null || observed === null ? 0 : nowMs - ageMinutes * 60_000 - observed;
  const shift: Shift = {
    ms: delta,
    days: observed === null ? 0 : daysBetween(observed, observed + delta),
  };
  return {
    defaultMac: mac,
    readSnapshot: async (asked, kind) =>
      asked === mac ? JSON.stringify(shiftTimes(goldenDoc(scenario, kind), shift)) : null,
    readLive: async () => null,
    readCurve: async (asked, startMs) => {
      const text = asked === mac ? goldenCurve(scenario, startMs - delta) : null;
      return text === null ? null : JSON.stringify(shiftTimes(JSON.parse(text), shift));
    },
  };
}

/**
 * Answer a lane-server request for `/dashboard`, `/current` or `/event`.
 *
 * @param url - The request path and query, e.g. `/dashboard?fixture=raining`.
 * @param env - The environment (`LOOKOUT_FIXTURE`, `LOOKOUT_AGE_MIN`).
 * @param nowMs - The clock, for `age`.
 * @returns The response, or null for any other path.
 */
export async function devResponse(
  url: string,
  env: Env,
  nowMs: number,
): Promise<Response | null> {
  const parsed = new URL(url, "http://lane.local");
  const handler = {
    "/dashboard": handleDashboard,
    "/current": handleCurrent,
    "/event": handleEvent,
  }[parsed.pathname];
  if (!handler) return null;

  const scenario = parsed.searchParams.get("fixture") ?? env.LOOKOUT_FIXTURE ?? "dry";
  if (!SCENARIOS.includes(scenario as Scenario)) {
    return errorResponse(400, `fixture must be one of ${SCENARIOS.join(", ")}`);
  }
  const ageText = parsed.searchParams.get("age") ?? env.LOOKOUT_AGE_MIN ?? "";
  const age = ageText === "" ? null : Number(ageText);
  if (age !== null && !Number.isFinite(age)) {
    return errorResponse(400, "age must be a number of minutes");
  }
  const sources = fixtureSources(scenario as Scenario, age, nowMs);
  // `fixture` and `age` are the lane server's own: the real read functions
  // refuse them, so they are taken off before the handler sees the request,
  // and put back on any redirect it answers.
  const own = new URLSearchParams();
  for (const name of DEV_PARAMS) {
    const value = parsed.searchParams.get(name);
    if (value !== null) own.set(name, value);
    parsed.searchParams.delete(name);
  }
  const response = await handler(new Request(parsed), sources);
  const location = response.headers.get("location");
  if (location === null || !own.size) return response;
  const headers = new Headers(response.headers);
  headers.set("location", `${location}${location.includes("?") ? "&" : "?"}${own}`);
  return new Response(response.body, { status: response.status, headers });
}

/** The lane server's own query parameters. */
const DEV_PARAMS = ["fixture", "age"] as const;

/**
 * A Vite plugin serving `/dashboard`, `/current` and `/event` from the
 * goldens on the dev server, and on `vite preview` over the built page.
 *
 * @returns The plugin.
 */
export function goldenReadFunctions(): Plugin {
  const serve = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use((req, res, next) => {
      devResponse(req.url ?? "/", process.env, Date.now())
        .then(async (response) => {
          if (!response) return next();
          res.statusCode = response.status;
          response.headers.forEach((value, name) => res.setHeader(name, value));
          res.end(await response.text());
        })
        .catch(next);
    });
  };
  return {
    name: "lookout-golden-read-functions",
    configureServer: serve,
    configurePreviewServer: serve,
  };
}
