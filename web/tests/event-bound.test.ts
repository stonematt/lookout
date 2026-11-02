/**
 * `/event` cannot be made to read Storj without bound (#231). A curve file is
 * read only for a start the Event Catalog lists, as the dashboard snapshot
 * has it, and the dashboard is read at most once per refresh interval per
 * function instance, however many starts are asked. Called the way Vercel
 * calls it, with `fetch` faked as Storj. No network.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { goldenCurveText, goldenText } from "./golden.js";

const MAC = "00:11:22:33:44:55";
const NOW = Date.UTC(2026, 9, 30, 12, 0);
/** Completed events in the raining golden, each with a curve file. */
const EVENT_1 = 1767549600000;
const EVENT_2 = 1767758400000;
/** The raining golden's live event: its curve file follows its close. */
const LIVE = 1767909600000;

const ENV = {
  STORJ_ACCESS_KEY_ID: "test-access",
  STORJ_SECRET_ACCESS_KEY: "test-secret",
  STORJ_ENDPOINT_URL: "https://gateway.storj.test",
  STORJ_BUCKET: "lookout",
  STATION_MAC: MAC,
};

const DASHBOARD_KEY = `snapshot/${MAC}.dashboard.json`;

/** The Storj keys read, in order. */
let keys: string[];
/** What Storj answers for a key. */
let storj: (key: string) => Response | Promise<Response>;
let eventGET: (request: Request) => Promise<Response>;
let refreshMs: number;

/** Storj as the refresh run leaves it: the raining dashboard and its curves. */
function raining(key: string): Response {
  if (key === DASHBOARD_KEY) return new Response(goldenText("raining", "dashboard"));
  const match = /^snapshot\/.+\.events\/(\d+)\.json$/.exec(key);
  try {
    return new Response(goldenCurveText("raining", Number(match?.[1])));
  } catch {
    return new Response("NoSuchKey", { status: 404 });
  }
}

const curveReads = () => keys.filter((key) => key !== DASHBOARD_KEY);
const dashboardReads = () => keys.filter((key) => key === DASHBOARD_KEY);
const ask = (start: number) =>
  eventGET(new Request(`https://lookout.test/event?start=${start}`));

beforeEach(async () => {
  keys = [];
  storj = raining;
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  for (const [name, value] of Object.entries(ENV)) vi.stubEnv(name, value);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const key = decodeURIComponent(new URL(request.url).pathname).replace("/lookout/", "");
      keys.push(key);
      return storj(key);
    }),
  );
  // A fresh module per test: what a function instance knows starts empty.
  vi.resetModules();
  eventGET = (await import("../api/event.js")).GET;
  refreshMs = (await import("../server/storj.js")).EVENT_STARTS_REFRESH_MS;
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("/event with a start the Event Catalog does not hold", () => {
  it("answers 404 without reading a curve file", async () => {
    const response = await ask(1600000000000);

    expect(response.status).toBe(404);
    expect(curveReads()).toEqual([]);
  });

  it("reads Storj once for any number of unknown starts", async () => {
    for (let i = 0; i < 50; i++) await ask(1600000000000 + i);

    expect(keys).toEqual([DASHBOARD_KEY]);
  });

  it("reads nothing at all for an unknown start once the catalog is known", async () => {
    await ask(EVENT_2);
    keys = [];

    const response = await ask(1600000000000);

    expect(response.status).toBe(404);
    expect(keys).toEqual([]);
  });

  it("asks the dashboard again at most once per refresh interval", async () => {
    await ask(1600000000000);
    vi.setSystemTime(NOW + refreshMs - 1);
    await ask(1600000000001);
    expect(dashboardReads()).toHaveLength(1);

    vi.setSystemTime(NOW + refreshMs);
    await ask(1600000000002);
    await ask(1600000000003);

    expect(dashboardReads()).toHaveLength(2);
    expect(curveReads()).toEqual([]);
  });

  it("shares one dashboard read between requests that arrive together", async () => {
    await Promise.all([1, 2, 3, 4, 5].map((i) => ask(1600000000000 + i)));

    expect(keys).toEqual([DASHBOARD_KEY]);
  });

  it("does not read the live event's curve file, which is not written yet", async () => {
    const response = await ask(LIVE);

    expect(response.status).toBe(404);
    expect(curveReads()).toEqual([]);
  });

  it("answers 404 without reading a curve when there is no dashboard", async () => {
    storj = () => new Response("NoSuchKey", { status: 404 });

    const response = await ask(EVENT_2);

    expect(response.status).toBe(404);
    expect(keys).toEqual([DASHBOARD_KEY]);
  });
});

describe("/event with a start the Event Catalog holds", () => {
  it("serves its curve file, reading the dashboard only the first time", async () => {
    expect((await ask(EVENT_2)).status).toBe(200);
    expect((await ask(EVENT_1)).status).toBe(200);

    expect(keys).toEqual([
      DASHBOARD_KEY,
      `snapshot/${MAC}.events/${EVENT_2}.json`,
      `snapshot/${MAC}.events/${EVENT_1}.json`,
    ]);
  });

  it("answers 404, briefly cached, when the backfill has not written its file", async () => {
    storj = (key) =>
      key === DASHBOARD_KEY ? raining(key) : new Response("NoSuchKey", { status: 404 });

    const response = await ask(EVENT_2);

    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("s-maxage=300");
    expect(curveReads()).toEqual([`snapshot/${MAC}.events/${EVENT_2}.json`]);
  });

  it("finds an event the catalog gained, once the refresh interval has passed", async () => {
    const without = JSON.parse(goldenText("raining", "dashboard")) as {
      rain_events: { events: { start_ms: number }[] };
    };
    without.rain_events.events = without.rain_events.events.filter(
      (e) => e.start_ms !== EVENT_2,
    );
    storj = (key) =>
      key === DASHBOARD_KEY ? Response.json(without) : raining(key);
    expect((await ask(EVENT_2)).status).toBe(404);

    storj = raining;
    vi.setSystemTime(NOW + refreshMs);

    expect((await ask(EVENT_2)).status).toBe(200);
  });
});

describe("/event while the dashboard cannot be read", () => {
  it("answers 502, briefly cached, and asks Storj again only after the interval", async () => {
    storj = () => new Response("slow down", { status: 503 });

    const first = await ask(EVENT_2);
    await ask(EVENT_1);
    await ask(1600000000000);

    expect(first.status).toBe(502);
    expect(first.headers.get("cache-control")).toBe("s-maxage=15");
    expect(keys).toEqual([DASHBOARD_KEY]);

    storj = raining;
    vi.setSystemTime(NOW + refreshMs);
    expect((await ask(EVENT_2)).status).toBe(200);
  });

  it("keeps serving the starts it knows when a later dashboard read fails", async () => {
    await ask(EVENT_2);
    storj = (key) =>
      key === DASHBOARD_KEY ? new Response("slow down", { status: 503 }) : raining(key);
    vi.setSystemTime(NOW + refreshMs);

    expect((await ask(1600000000000)).status).toBe(502);
    expect((await ask(EVENT_1)).status).toBe(200);
  });
});
