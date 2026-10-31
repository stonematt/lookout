/**
 * The lane server's read functions: `/dashboard` and `/current` served by
 * the real handlers from the golden snapshot documents in
 * `tests/golden/snapshots/`, with no secrets and no network.
 *
 * - `?fixture=dry|raining` (or `LOOKOUT_FIXTURE`) picks the golden; `dry` by
 *   default.
 * - `?age=<minutes>` (or `LOOKOUT_AGE_MIN`) shifts every `_ms` time by one
 *   offset so the newest observation is that many minutes old. Without it
 *   the goldens are served as recorded, which is long stale.
 * - `/current` carries no live reading (`last_data: null`): Ambient is never
 *   called in dev.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Plugin } from "vite";

import type { Env } from "./env.js";
import type { ReadSources } from "./handlers.js";
import { errorResponse, handleCurrent, handleDashboard } from "./handlers.js";
import type { SnapshotKind } from "./storj.js";

/** Where the goldens live, relative to this file. */
const GOLDEN_DIR = resolve(import.meta.dirname, "../../tests/golden/snapshots");

const SCENARIOS = ["dry", "raining"] as const;
type Scenario = (typeof SCENARIOS)[number];

function goldenDoc(scenario: Scenario, kind: SnapshotKind): Record<string, unknown> {
  const text = readFileSync(`${GOLDEN_DIR}/${scenario}.${kind}.json`, "utf8");
  return JSON.parse(text) as Record<string, unknown>;
}

/** Add `delta` to every number in a field ending `_ms`, at any depth. */
function shiftTimes(value: unknown, delta: number): unknown {
  if (Array.isArray(value)) return value.map((item) => shiftTimes(item, delta));
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      key,
      key.endsWith("_ms") && typeof item === "number"
        ? item + delta
        : shiftTimes(item, delta),
    ]),
  );
}

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
  return {
    defaultMac: mac,
    readSnapshot: async (asked, kind) =>
      asked === mac ? JSON.stringify(shiftTimes(goldenDoc(scenario, kind), delta)) : null,
    readLive: async () => null,
  };
}

/**
 * Answer a lane-server request for `/dashboard` or `/current`.
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
  const handler = { "/dashboard": handleDashboard, "/current": handleCurrent }[
    parsed.pathname
  ];
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
  return handler(new Request(parsed), sources);
}

/**
 * A Vite plugin serving `/dashboard` and `/current` from the goldens on the
 * dev server.
 *
 * @returns The plugin.
 */
export function goldenReadFunctions(): Plugin {
  return {
    name: "lookout-golden-read-functions",
    configureServer(server) {
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
    },
  };
}
