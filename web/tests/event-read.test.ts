/**
 * `/event?start=<ms>`, the curve-file read function, called the way Vercel
 * calls it (`GET(request)`, `process.env` set, `fetch` faked so Storj
 * serves the golden curve files), and on the lane server. Plus the curve
 * contract and the chart helpers that read a curve. No network.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { devResponse } from "../server/dev-fixtures.js";
import { CURVE_CACHE } from "../server/event.js";
import { accumulationLine, gapSpans, stationWallClock } from "../src/events/charts.js";
import { readUrl } from "../src/load.js";
import type { EventCurve } from "../src/snapshot.js";
import { SnapshotContractError, parseDashboard, parseEventCurve } from "../src/snapshot.js";
import { golden, goldenCurve, goldenCurveText, goldenText } from "./golden.js";

const MAC = "00:11:22:33:44:55";
const STORJ = "https://gateway.storj.test";
const EVENT_2 = Date.UTC(2026, 0, 7, 4);
const EVENT_3 = Date.UTC(2026, 0, 8, 22);
const NOW = Date.UTC(2026, 9, 30, 12, 0);
const MIN = 60_000;

const ENV = {
  STORJ_ACCESS_KEY_ID: "test-access",
  STORJ_SECRET_ACCESS_KEY: "test-secret",
  STORJ_ENDPOINT_URL: STORJ,
  STORJ_BUCKET: "lookout",
};

/** The station index the refresh run publishes: this station only. */
const INDEX_KEY = "snapshot/stations.json";
const INDEX = {
  schema_version: 1,
  kind: "station_index",
  built_at_ms: 1_767_918_000_000,
  stations: [
    {
      mac: MAC,
      name: "Fixture Station",
      tz: "America/Los_Angeles",
      newest_reading_ms: 1_767_917_700_000,
      archive_start_ms: 1_651_683_600_000,
    },
  ],
};

const DASHBOARD_KEY = `snapshot/${MAC}.dashboard.json`;
/** A station in the index with no snapshot yet. */
const NEW = "00:11:22:33:44:77";

/** The Storj requests, the station index's left out. */
let calls: Request[];
let storjBody: (key: string) => Response;
let eventGET: (request: Request) => Promise<Response>;

/** Only the curve-file reads: `/event` reads the dashboard to know the starts. */
const curveCalls = () =>
  calls.filter((request) => !new URL(request.url).pathname.endsWith(".dashboard.json"));

beforeEach(async () => {
  calls = [];
  storjBody = (key) => {
    if (key === DASHBOARD_KEY) return new Response(goldenText("raining", "dashboard"));
    const match = /^snapshot\/(.+)\.events\/(\d+)\.json$/.exec(key);
    if (!match || match[1] !== MAC) return new Response("NoSuchKey", { status: 404 });
    try {
      return new Response(goldenCurveText("raining", Number(match[2])));
    } catch {
      return new Response("NoSuchKey", { status: 404 });
    }
  };
  for (const [name, value] of Object.entries(ENV)) vi.stubEnv(name, value);
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const key = decodeURIComponent(new URL(request.url).pathname).replace("/lookout/", "");
      if (key === INDEX_KEY) return Promise.resolve(Response.json(INDEX));
      calls.push(request);
      return Promise.resolve(storjBody(key));
    }),
  );
  // A fresh module per test: the starts a function instance knows start empty.
  vi.resetModules();
  eventGET = (await import("../api/event.js")).GET;
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

const get = (path: string) => new Request(`https://lookout.test${path}`);

describe("GET /event", () => {
  it("returns the curve file as Storj holds it, cached as immutable", async () => {
    const response = await eventGET(get(`/event?start=${EVENT_2}`));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe(CURVE_CACHE);
    expect(await response.json()).toEqual(goldenCurve("raining", EVENT_2));
  });

  it("reads the event's key under snapshot/ with a signed request", async () => {
    await eventGET(get(`/event?start=${EVENT_2}`));

    const curves = curveCalls();
    expect(curves).toHaveLength(1);
    expect(decodeURIComponent(new URL(curves[0]!.url).pathname)).toBe(
      `/lookout/snapshot/${MAC}.events/${EVENT_2}.json`,
    );
    expect(curves[0]!.headers.get("authorization")).toMatch(/^AWS4-HMAC-SHA256 /);
  });

  it("answers 404, briefly cached, for the live event, whose curve is not written yet", async () => {
    const response = await eventGET(get(`/event?start=${EVENT_3}`));

    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("s-maxage=60");
    expect(curveCalls()).toHaveLength(0);
  });

  it.each(["", "?start=", "?start=abc", "?start=../dashboard", "?start=12"])(
    "refuses a malformed start (%s) without reading Storj",
    async (query) => {
      const response = await eventGET(get(`/event${query}`));

      expect(response.status).toBe(400);
      expect(calls).toHaveLength(0);
    },
  );

  it.each([`?start=${EVENT_2}&x=1`, `?start=${EVENT_2}&start=${EVENT_2}`, "?x=1"])(
    "refuses an unknown or repeated parameter (%s) without reading Storj",
    async (query) => {
      const response = await eventGET(get(`/event${query}`));

      expect(response.status).toBe(400);
      expect(calls).toHaveLength(0);
    },
  );

  it.each([`?start=0${EVENT_2}`])(
    "sends another spelling of the event (%s) to its one cached URL",
    async (query) => {
      const response = await eventGET(get(`/event${query}`));

      expect(response.status).toBe(308);
      expect(response.headers.get("location")).toBe(`event?start=${EVENT_2}`);
      expect(calls).toHaveLength(0);
    },
  );

  it("sends a station in another spelling to the event's one URL for it", async () => {
    const response = await eventGET(get(`/event?start=${EVENT_2}&station=${MAC.toLowerCase()}`));
    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      `event?start=${EVENT_2}&station=${encodeURIComponent(MAC)}`,
    );
    expect(calls).toHaveLength(0);
  });

  it("serves the station named outright, as the index lists it", async () => {
    const response = await eventGET(
      get(`/event?start=${EVENT_2}&station=${encodeURIComponent(MAC)}`),
    );
    expect(response.status).toBe(200);
  });

  it("answers 404, briefly cached, for a station in the index with no snapshot yet", async () => {
    INDEX.stations.push({ ...INDEX.stations[0]!, mac: NEW });
    try {
      const response = await eventGET(
        get(`/event?start=${EVENT_2}&station=${encodeURIComponent(NEW)}`),
      );
      expect(response.status).toBe(404);
      expect(response.headers.get("cache-control")).toMatch(/^s-maxage=\d+$/);
      expect(curveCalls()).toHaveLength(0);
    } finally {
      INDEX.stations.pop();
    }
  });

  it("answers 404, uncached, for a station the index does not list, reading no curve", async () => {
    const other = encodeURIComponent("AA:BB:CC:DD:EE:FF");
    const response = await eventGET(get(`/event?start=${EVENT_2}&station=${other}`));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(calls).toHaveLength(0);
  });

  it("answers 502, briefly cached, when Storj fails", async () => {
    storjBody = () => new Response("slow down", { status: 503 });

    const response = await eventGET(get(`/event?start=${EVENT_2}`));

    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it("answers 502 for a file that breaks the contract", async () => {
    storjBody = (key) =>
      key === DASHBOARD_KEY
        ? new Response(goldenText("raining", "dashboard"))
        : Response.json({ kind: "event_curve", schema_version: 99 });

    const response = await eventGET(get(`/event?start=${EVENT_2}`));

    expect(response.status).toBe(502);
  });

  it("answers 500 naming the unset variables", async () => {
    vi.stubEnv("STORJ_BUCKET", "");

    const response = await eventGET(get(`/event?start=${EVENT_2}`));

    expect(response.status).toBe(500);
    expect(await response.text()).toContain("STORJ_BUCKET");
  });
});

describe("the lane server's /event", () => {
  it("serves a golden curve file by its start", async () => {
    const response = await devResponse(`/event?fixture=raining&start=${EVENT_2}`, {}, NOW);

    expect(response!.status).toBe(200);
    expect(await response!.json()).toEqual(goldenCurve("raining", EVENT_2));
  });

  it("finds the file by a start read from an age-shifted /dashboard", async () => {
    const shifted = await devResponse("/dashboard?fixture=raining&age=3", {}, NOW);
    const dashboard = parseDashboard(await shifted!.json());
    const start = dashboard.rain_events.events[1]!.start_ms;
    const recorded = golden("raining", "dashboard") as { observed_at_ms: number };
    const delta = NOW - 3 * MIN - recorded.observed_at_ms;
    expect(start).toBe(EVENT_2 + delta);

    const response = await devResponse(`/event?fixture=raining&age=3&start=${start}`, {}, NOW);
    const curve = parseEventCurve(await response!.json());

    expect(curve.start_ms).toBe(start);
    expect(curve.t_ms[0]).toBe(start);
  });

  it("still finds the file when the event is opened minutes after the page", async () => {
    const shifted = await devResponse("/dashboard?fixture=raining&age=3", {}, NOW);
    const start = parseDashboard(await shifted!.json()).rain_events.events[1]!.start_ms;

    const later = NOW + 7 * MIN;
    const response = await devResponse(`/event?fixture=raining&age=3&start=${start}`, {}, later);

    expect(response!.status).toBe(200);
    expect(parseEventCurve(await response!.json()).t_ms).toHaveLength(
      (goldenCurve("raining", EVENT_2) as EventCurve).t_ms.length,
    );
  });

  it("answers 404 on the lane server for a start with no golden near it", async () => {
    const response = await devResponse("/event?fixture=raining&start=1600000000000", {}, NOW);

    expect(response!.status).toBe(404);
  });

  it("shifts the ongoing curve's times with the rest of the snapshot", async () => {
    const response = await devResponse("/dashboard?fixture=raining&age=3", {}, NOW);
    const dashboard = parseDashboard(await response!.json());
    const curve = dashboard.rain_events.open_curve!;

    expect(curve.t_ms.at(-1)).toBe(NOW - 3 * MIN);
  });
});

describe("the curve contract", () => {
  it("reads every golden curve file", () => {
    expect(parseEventCurve(goldenCurve("raining", EVENT_2)).start_ms).toBe(EVENT_2);
  });

  it("still reads a curve file written at schema version 2, never rewritten (#344)", () => {
    // A curve file as the refresh run wrote it at version 2, frozen outside the
    // regenerated goldens: the stored curves on Storj look like this forever.
    const frozen = resolve(import.meta.dirname, "../../tests/golden/frozen");
    const text = readFileSync(`${frozen}/curve-v2.raining.${EVENT_2}.json`, "utf8");
    const doc: unknown = JSON.parse(text);
    expect(doc).toHaveProperty("schema_version", 2);

    const curve = parseEventCurve(doc);
    expect(curve.start_ms).toBe(EVENT_2);
    expect(curve.t_ms.length).toBeGreaterThan(0);
  });

  it("refuses a curve file of a version it does not know", () => {
    const doc = { ...(goldenCurve("raining", EVENT_2) as object), schema_version: 1 };

    expect(() => parseEventCurve(doc)).toThrow("schema_version 1");
  });

  it("refuses uneven arrays", () => {
    const doc = goldenCurve("raining", EVENT_2) as EventCurve;
    doc.rate_in_hr = doc.rate_in_hr.slice(1);

    expect(() => parseEventCurve(doc)).toThrow(SnapshotContractError);
  });

  it("refuses a dashboard without its rain events", () => {
    const doc = golden("raining", "dashboard") as Record<string, unknown>;
    delete doc.rain_events;

    expect(() => parseDashboard(doc)).toThrow(SnapshotContractError);
  });

  it("asks a lane server with fixture and age as well as the start", () => {
    expect(readUrl("/event", "?fixture=raining&age=3&theme=dark", { start: "17" }, true)).toBe(
      "/event?start=17&fixture=raining&age=3",
    );
  });

  it("sends the deployed functions only station, never fixture or age", () => {
    expect(readUrl("/event", "?fixture=raining&age=3&station=aa:bb", { start: "17" }, false)).toBe(
      "/event?start=17&station=aa%3Abb",
    );
    expect(readUrl("/dashboard", "?fixture=dry&age=3&theme=dark", {}, false)).toBe("/dashboard");
  });
});

describe("the chart helpers", () => {
  const ongoing = () => parseDashboard(golden("raining", "dashboard")).rain_events.open_curve!;

  it("plots in the station zone", () => {
    expect(stationWallClock(EVENT_3)).toBe("2026-01-08 14:00:00");
  });

  it("marks the live reading past the archive as the live tail", () => {
    const curve = ongoing();

    expect(gapSpans(curve, true)).toEqual([
      { fromMs: curve.t_ms.at(-2), toMs: curve.t_ms.at(-1), liveTail: true },
    ]);
    expect(gapSpans(curve, false)[0]!.liveTail).toBe(false);
  });

  it("leaves the live tail off the bold line, to be drawn dotted", () => {
    const curve = ongoing();

    const line = accumulationLine(curve, gapSpans(curve, true));

    expect(line.x).toHaveLength(curve.t_ms.length - 1);
    expect(line.y.at(-1)).toBe(curve.event_in.at(-2));
  });

  it("breaks the line at a gap in the record", () => {
    const curve: EventCurve = {
      start_ms: 0,
      end_ms: 3 * 300_000 + 1_800_000,
      t_ms: [0, 300_000, 2_100_000, 2_400_000],
      event_in: [0.01, 0.02, 0.05, 0.06],
      rate_in_hr: [0, 0.12, null, 0.12],
      gap_minutes: [null, null, 30, null],
    };

    const line = accumulationLine(curve, gapSpans(curve, false));

    expect(line.y).toEqual([0.01, 0.02, null, 0.05, 0.06]);
  });
});
