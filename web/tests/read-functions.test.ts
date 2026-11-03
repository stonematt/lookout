/**
 * The read functions `/dashboard`, `/current`, `/status` and `/stations`,
 * called the way Vercel calls them: `GET(request)` with `process.env` set
 * and `fetch` faked. Storj serves a station index and the golden documents
 * for the stations it has snapshots of; Ambient serves a recorded-shape
 * `/devices` body. No network.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { describeError } from "../server/errors.js";
import { SCHEMA_VERSION } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { golden, goldenText } from "./golden.js";

const MAC = "00:11:22:33:44:55";
/** A second station in the index, with the dry golden as its snapshot. */
const OTHER = "00:11:22:33:44:66";
/** A station in the index with no snapshot yet: added minutes ago. */
const NEW = "00:11:22:33:44:77";
/** A MAC the index does not list. */
const STRANGER = "AA:BB:CC:DD:EE:FF";
const STORJ = "https://gateway.storj.test";
const AMBIENT = "https://ambient.test/v1";
const INDEX_KEY = "snapshot/stations.json";

const ENV = {
  STORJ_ACCESS_KEY_ID: "test-access",
  STORJ_SECRET_ACCESS_KEY: "test-secret",
  STORJ_ENDPOINT_URL: STORJ,
  STORJ_BUCKET: "lookout",
  AMBIENT_API_KEY: "test-api",
  AMBIENT_APPLICATION_KEY: "test-app",
  AMBIENT_ENDPOINT: AMBIENT,
};

interface FakeWorld {
  scenario: Scenario;
  /** The station index's stations, in order; null: no index stored. */
  index: string[] | null;
  /** Storj's answer to the index read; null: serve `index`. */
  indexStatus: number | null;
  /** Storj's answer to a snapshot document read; null: serve the golden. */
  storjStatus: number | null;
  ambientStatus: number | null; // null: serve `devices`
  devices: unknown[];
  /** A key's text, served in place of the usual answer. */
  bodies: Record<string, string>;
  calls: Request[];
}

let world: FakeWorld;

/** A station index document listing `macs`, as the refresh run writes it. */
function indexDocument(macs: string[]) {
  return {
    schema_version: 1,
    kind: "station_index",
    built_at_ms: 1_767_918_000_000,
    stations: macs.map((mac) => ({
      mac,
      name: `Station ${mac.slice(-2)}`,
      tz: "America/Los_Angeles",
      newest_reading_ms: 1_767_917_700_000,
      archive_start_ms: 1_651_683_600_000,
    })),
  };
}

/** Which golden each station's snapshot documents are; `NEW` has none. */
const SNAPSHOTS: Record<string, () => Scenario> = {
  [MAC]: () => world.scenario,
  [OTHER]: () => "dry",
};

function fakeFetch(input: RequestInfo | URL, init?: RequestInit) {
  const request = new Request(input, init);
  world.calls.push(request);
  const url = new URL(request.url);
  if (url.origin === STORJ) {
    const key = decodeURIComponent(url.pathname).replace("/lookout/", "");
    if (Object.hasOwn(world.bodies, key)) {
      return Promise.resolve(new Response(world.bodies[key]));
    }
    if (key === INDEX_KEY) {
      if (world.indexStatus !== null) {
        return Promise.resolve(new Response("nope", { status: world.indexStatus }));
      }
      return Promise.resolve(
        world.index === null
          ? new Response("NoSuchKey", { status: 404 })
          : Response.json(indexDocument(world.index)),
      );
    }
    if (world.storjStatus !== null) {
      return Promise.resolve(new Response("nope", { status: world.storjStatus }));
    }
    const match = /^snapshot\/(.+)\.(dashboard|context|status)\.json$/.exec(key);
    const scenario = match ? SNAPSHOTS[match[1]!]?.() : undefined;
    if (!match || scenario === undefined) {
      return Promise.resolve(new Response("NoSuchKey", { status: 404 }));
    }
    const kind = match[2] as "dashboard" | "context" | "status";
    return Promise.resolve(new Response(goldenText(scenario, kind)));
  }
  if (url.origin + "/v1" === AMBIENT) {
    if (world.ambientStatus !== null) {
      return Promise.resolve(new Response("{}", { status: world.ambientStatus }));
    }
    return Promise.resolve(Response.json(world.devices));
  }
  return Promise.reject(new Error(`unexpected fetch ${request.url}`));
}

function context(scenario: Scenario) {
  return golden(scenario, "context") as {
    observed_at_ms: number;
    built_at_ms: number;
  };
}

function ambientDevice(dateutc: number, mac = MAC) {
  return {
    macAddress: mac,
    info: { name: "Ambient Name" },
    lastData: { dateutc, tempf: 59.5, eventrainin: 0.31 },
  };
}

type Get = (request: Request) => Promise<Response>;
let currentGET: Get;
let dashboardGET: Get;
let statusGET: Get;
let stationsGET: Get;

beforeEach(async () => {
  world = {
    scenario: "raining",
    index: [MAC, OTHER, NEW],
    indexStatus: null,
    storjStatus: null,
    ambientStatus: null,
    devices: [],
    bodies: {},
    calls: [],
  };
  for (const [name, value] of Object.entries(ENV)) vi.stubEnv(name, value);
  vi.stubGlobal("fetch", vi.fn(fakeFetch));
  // A fresh module per test: what a function instance knows of the station
  // index starts empty.
  vi.resetModules();
  currentGET = (await import("../api/current.js")).GET;
  dashboardGET = (await import("../api/dashboard.js")).GET;
  statusGET = (await import("../api/status.js")).GET;
  stationsGET = (await import("../api/stations.js")).GET;
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

const get = (path: string) => new Request(`https://lookout.test${path}`);

/** A station's one URL spelling: `?station=` with the MAC upper case, encoded. */
const station = (mac: string) => `station=${encodeURIComponent(mac)}`;

/** The Storj keys read, in order. */
const storjKeys = () =>
  world.calls
    .filter((c) => c.url.startsWith(STORJ))
    .map((c) => decodeURIComponent(new URL(c.url).pathname).replace("/lookout/", ""));

describe("GET /dashboard", () => {
  it("returns the snapshot document as Storj holds it, CDN-cached", async () => {
    const response = await dashboardGET(get("/dashboard"));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe(
      "s-maxage=300, stale-while-revalidate=300, stale-if-error=300",
    );
    expect(response.headers.get("content-type")).toMatch(/application\/json/);
    expect(await response.json()).toEqual(golden("raining", "dashboard"));
  });

  it("reads the station index, then the default station's key, with signed requests", async () => {
    world.scenario = "dry";
    await dashboardGET(get("/dashboard"));

    expect(storjKeys()).toEqual([INDEX_KEY, `snapshot/${MAC}.dashboard.json`]);
    for (const call of world.calls) {
      expect(call.headers.get("authorization")).toMatch(
        /^AWS4-HMAC-SHA256 Credential=test-access\//,
      );
    }
  });

  it("serves the index's first station when none is given", async () => {
    world.index = [OTHER, MAC];
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(golden("dry", "dashboard"));
    expect(storjKeys()).toEqual([INDEX_KEY, `snapshot/${OTHER}.dashboard.json`]);
  });

  it("serves any station in the index by ?station=", async () => {
    const response = await dashboardGET(get(`/dashboard?${station(OTHER)}`));
    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe(
      "s-maxage=300, stale-while-revalidate=300, stale-if-error=300",
    );
    expect(await response.json()).toEqual(golden("dry", "dashboard"));
    expect(storjKeys()).toEqual([INDEX_KEY, `snapshot/${OTHER}.dashboard.json`]);
  });

  it("serves the default station named outright at its own URL", async () => {
    const response = await dashboardGET(get(`/dashboard?${station(MAC)}`));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(golden("raining", "dashboard"));
  });

  it("redirects a trailing-slash path to a location that is never empty", async () => {
    const response = await dashboardGET(get(`/dashboard/?station=${MAC}`));
    expect(response.status).toBe(308);
    const location = response.headers.get("location");
    expect(location).toBe(`./?${station(MAC)}`);
    expect(new URL(location!, "https://x.test/dashboard/?station=a").href).toBe(
      `https://x.test/dashboard/?${station(MAC)}`,
    );
  });

  it("never calls Ambient", async () => {
    await dashboardGET(get("/dashboard"));
    expect(world.calls.every((c) => c.url.startsWith(STORJ))).toBe(true);
  });

  it("sends a station in any other spelling to its one URL, unread", async () => {
    const response = await dashboardGET(get(`/dashboard?station=${OTHER.toLowerCase()}`));
    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(`dashboard?${station(OTHER)}`);
    expect(world.calls).toHaveLength(0);
  });

  it("answers 404, uncached, for a station the index does not list, reading only the index", async () => {
    const response = await dashboardGET(get(`/dashboard?${station(STRANGER)}`));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(storjKeys()).toEqual([INDEX_KEY]);
  });

  it("refuses a station that is not a MAC address, reading nothing", async () => {
    const response = await dashboardGET(get("/dashboard?station=../archive"));
    expect(response.status).toBe(400);
    expect(world.calls).toHaveLength(0);
  });

  it("answers 404, briefly cached, for a station in the index with no snapshot yet", async () => {
    const response = await dashboardGET(get(`/dashboard?${station(NEW)}`));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it("answers 404, briefly cached, before the refresh run has published an index", async () => {
    world.index = null;
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
    expect(storjKeys()).toEqual([INDEX_KEY]);
  });

  it.each([[500], [403]])(
    "answers 502, briefly cached, when the station index read gets %s",
    async (status) => {
      world.indexStatus = status;
      const logged = vi.spyOn(console, "error").mockImplementation(() => {});
      const response = await dashboardGET(get("/dashboard"));
      logged.mockRestore();
      expect(response.status).toBe(502);
      expect(response.headers.get("cache-control")).toBe("s-maxage=15");
      expect(storjKeys()).toEqual([INDEX_KEY]);
    },
  );

  it("answers 502, briefly cached, when the station index breaks its contract", async () => {
    world.bodies[INDEX_KEY] = JSON.stringify({ ...indexDocument([MAC]), schema_version: 2 });
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const response = await dashboardGET(get("/dashboard"));
    logged.mockRestore();
    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it("answers 502, briefly cached, when Storj fails", async () => {
    world.storjStatus = 500;
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it("answers 502 when the document breaks the contract", async () => {
    world.bodies[`snapshot/${MAC}.dashboard.json`] = JSON.stringify({ schema_version: 99 });
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(502);
  });

  it("names the env vars it is missing, not their values", async () => {
    vi.stubEnv("STORJ_SECRET_ACCESS_KEY", "");
    vi.stubEnv("STORJ_BUCKET", "");
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(500);
    const body = (await response.json()) as { error: string };
    expect(body.error).toContain("STORJ_SECRET_ACCESS_KEY");
    expect(body.error).toContain("STORJ_BUCKET");
    expect(world.calls).toHaveLength(0);
  });

  it("needs no STATION_MAC: the station index names the stations", async () => {
    vi.stubEnv("STATION_MAC", STRANGER);
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(200);
    expect(storjKeys()).toEqual([INDEX_KEY, `snapshot/${MAC}.dashboard.json`]);
  });
});

describe("GET /current", () => {
  it("returns the live reading with the context block", async () => {
    const ctx = context("raining");
    const live = ctx.observed_at_ms + 60_000;
    world.devices = [ambientDevice(live, "FF:FF:FF:FF:FF:FF"), ambientDevice(live)];

    const response = await currentGET(get("/current"));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe(
      "s-maxage=60, stale-while-revalidate=60, stale-if-error=300",
    );
    const body = await response.json();
    expect(body).toEqual({
      schema_version: SCHEMA_VERSION,
      kind: "current",
      station: { mac: MAC, name: "Fixture Station" },
      observed_at_ms: live,
      built_at_ms: ctx.built_at_ms,
      last_data: { dateutc: live, tempf: 59.5, eventrainin: 0.31 },
      context: golden("raining", "context"),
    });
  });

  it("asks Ambient with both keys", async () => {
    world.devices = [ambientDevice(1)];
    await currentGET(get("/current"));
    const ambient = world.calls.find((c) => c.url.startsWith(AMBIENT))!;
    const url = new URL(ambient.url);
    expect(url.pathname).toBe("/v1/devices");
    expect(url.searchParams.get("apiKey")).toBe("test-api");
    expect(url.searchParams.get("applicationKey")).toBe("test-app");
  });

  it("keeps the context's observation time when it is newer", async () => {
    const ctx = context("dry");
    world.scenario = "dry";
    world.devices = [ambientDevice(ctx.observed_at_ms - 600_000)];

    const body = await (await currentGET(get("/current"))).json();
    expect(body.observed_at_ms).toBe(ctx.observed_at_ms);
  });

  it.each([429, 500])(
    "still serves the context when Ambient answers %s, briefly cached",
    async (status) => {
      world.ambientStatus = status;
      const response = await currentGET(get("/current"));
      expect(response.status).toBe(200);
      expect(response.headers.get("cache-control")).toBe("s-maxage=15");
    },
  );

  it("still serves the context when Ambient fails", async () => {
    world.ambientStatus = 429;
    const response = await currentGET(get("/current"));
    const body = await response.json();
    expect(body.last_data).toBeNull();
    expect(body.context).toEqual(golden("raining", "context"));
    expect(body.observed_at_ms).toBe(context("raining").observed_at_ms);
  });

  it("serves the live reading alone when Storj fails", async () => {
    world.storjStatus = 403;
    world.devices = [ambientDevice(1_767_918_000_000)];
    const response = await currentGET(get("/current"));
    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
    const body = await response.json();
    expect(body.context).toBeNull();
    expect(body.built_at_ms).toBeNull();
    expect(body.observed_at_ms).toBe(1_767_918_000_000);
    expect(body.station).toEqual({ mac: MAC, name: "Ambient Name" });
  });

  it("never logs the Ambient keys when the fetch rejects", async () => {
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = new Request(input, init).url;
        if (url.startsWith(STORJ)) return fakeFetch(input, init);
        // What a network failure can look like: the URL, keys and all, in
        // the message, the stack and the cause.
        const error = new TypeError(`fetch failed: ${ENV.AMBIENT_API_KEY} ${url}`, {
          cause: new Error(`connect ECONNREFUSED ${url}`),
        });
        return Promise.reject(error);
      }),
    );

    const response = await currentGET(get("/current"));

    expect(response.status).toBe(200);
    expect(logged).toHaveBeenCalled();
    for (const args of logged.mock.calls) {
      for (const arg of args) {
        expect(typeof arg).toBe("string");
        expect(arg).not.toContain(ENV.AMBIENT_API_KEY);
        expect(arg).not.toContain(ENV.AMBIENT_APPLICATION_KEY);
      }
    }
    expect(logged.mock.calls.flat().join(" ")).toContain("fetch failed");
    logged.mockRestore();
  });

  it("answers 502, briefly cached, when neither source can be read", async () => {
    world.storjStatus = 500;
    world.ambientStatus = 429;
    const response = await currentGET(get("/current"));
    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it("serves no live reading for a station Ambient does not list", async () => {
    world.devices = [ambientDevice(1, "FF:FF:FF:FF:FF:FF")];
    const body = await (await currentGET(get("/current"))).json();
    expect(body.last_data).toBeNull();
  });

  it("refuses a station that is not a MAC address", async () => {
    const response = await currentGET(get("/current?station=x"));
    expect(response.status).toBe(400);
  });

  it("reads Ambient and the context for the chosen station only", async () => {
    world.devices = [ambientDevice(111, MAC), ambientDevice(222, OTHER)];

    const response = await currentGET(get(`/current?${station(OTHER)}`));

    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.station).toEqual({ mac: OTHER, name: "Fixture Station" });
    expect(body.last_data.dateutc).toBe(222);
    expect(body.context).toEqual(golden("dry", "context"));
    expect(storjKeys()).toEqual([INDEX_KEY, `snapshot/${OTHER}.context.json`]);
  });

  it("answers 404, uncached, for a station the index does not list, never asking Ambient", async () => {
    world.devices = [ambientDevice(1, STRANGER)];
    const response = await currentGET(get(`/current?${station(STRANGER)}`));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(storjKeys()).toEqual([INDEX_KEY]);
    expect(world.calls.some((c) => c.url.startsWith(AMBIENT))).toBe(false);
  });

  it("answers 502, briefly cached, when the station index cannot be read", async () => {
    world.indexStatus = 500;
    world.devices = [ambientDevice(1)];
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const response = await currentGET(get("/current"));
    logged.mockRestore();
    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
    expect(world.calls.some((c) => c.url.startsWith(AMBIENT))).toBe(false);
  });
});

describe("GET /status", () => {
  it("returns the refresh run's status document as Storj holds it", async () => {
    const response = await statusGET(get("/status"));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe("s-maxage=60, stale-if-error=300");
    expect(response.headers.get("content-type")).toMatch(/application\/json/);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body).toEqual(golden("raining", "status"));
    expect(body.outcome).toBe("ok");
    expect(body.sha).toMatch(/^[0-9a-f]{40}$/);
  });

  it("reads only the index and the station's status key from Storj, never Ambient", async () => {
    await statusGET(get("/status"));

    expect(world.calls.every((c) => c.url.startsWith(STORJ))).toBe(true);
    expect(storjKeys()).toEqual([INDEX_KEY, `snapshot/${MAC}.status.json`]);
  });

  it("serves another station's status by ?station=", async () => {
    const response = await statusGET(get(`/status?${station(OTHER)}`));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(golden("dry", "status"));
  });

  it("answers 404, briefly cached, when no run has written one", async () => {
    const response = await statusGET(get(`/status?${station(NEW)}`));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it("answers 502, briefly cached, when Storj fails", async () => {
    world.storjStatus = 500;
    const response = await statusGET(get("/status"));
    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it.each([JSON.stringify({ kind: "dashboard" }), "not json"])(
    "answers 502 when Storj holds something else: %s",
    async (body) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(() => Promise.resolve(new Response(body))),
      );
      const response = await statusGET(get("/status"));
      expect(response.status).toBe(502);
    },
  );

  it("answers 404, uncached, for a station the index does not list", async () => {
    const response = await statusGET(get(`/status?${station(STRANGER)}`));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
  });
});

describe("GET /stations", () => {
  it("returns the station index, CDN-cached like the dashboard", async () => {
    const response = await stationsGET(get("/stations"));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe(
      "s-maxage=300, stale-while-revalidate=300, stale-if-error=300",
    );
    expect(response.headers.get("content-type")).toMatch(/application\/json/);
    expect(await response.json()).toEqual(indexDocument([MAC, OTHER, NEW]));
    expect(storjKeys()).toEqual([INDEX_KEY]);
  });

  it.each(["/stations?x=1", `/stations?${station(MAC)}`, "/stations?fixture=dry"])(
    "refuses %s without reading anything",
    async (path) => {
      const response = await stationsGET(get(path));
      expect(response.status).toBe(400);
      expect(world.calls).toHaveLength(0);
    },
  );

  it("answers 404, briefly cached, before the refresh run has published one", async () => {
    world.index = null;
    const response = await stationsGET(get("/stations"));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("s-maxage=15");
  });

  it.each([JSON.stringify(indexDocument([])), "not json"])(
    "answers 502, briefly cached, for an index that breaks the contract: %s",
    async (body) => {
      world.bodies[INDEX_KEY] = body;
      const logged = vi.spyOn(console, "error").mockImplementation(() => {});
      const response = await stationsGET(get("/stations"));
      logged.mockRestore();
      expect(response.status).toBe(502);
      expect(response.headers.get("cache-control")).toBe("s-maxage=15");
    },
  );
});

describe("the station index, as one function instance knows it", () => {
  const NOW = Date.UTC(2026, 10, 3, 12, 0);
  const indexReads = () => storjKeys().filter((key) => key === INDEX_KEY).length;
  let logged: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    logged = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    logged.mockRestore();
    vi.useRealTimers();
  });

  it("is read once for any number of requests and unknown stations", async () => {
    await dashboardGET(get("/dashboard"));
    for (const mac of [STRANGER, "AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02"]) {
      expect((await currentGET(get(`/current?${station(mac)}`))).status).toBe(404);
    }
    await stationsGET(get("/stations"));
    expect(indexReads()).toBe(1);
  });

  it("is read again once the refresh interval has passed", async () => {
    expect((await dashboardGET(get(`/dashboard?${station(OTHER)}`))).status).toBe(200);
    world.index = [MAC];
    vi.setSystemTime(NOW + 59_000);
    expect((await dashboardGET(get(`/dashboard?${station(OTHER)}`))).status).toBe(200);
    vi.setSystemTime(NOW + 60_000);
    expect((await dashboardGET(get(`/dashboard?${station(OTHER)}`))).status).toBe(404);
    expect(indexReads()).toBe(2);
  });

  it("keeps the last good index when a later read fails", async () => {
    await dashboardGET(get("/dashboard"));
    world.indexStatus = 500;
    vi.setSystemTime(NOW + 60_000);
    const response = await dashboardGET(get(`/dashboard?${station(OTHER)}`));
    expect(response.status).toBe(200);
    expect(indexReads()).toBe(2);
  });

  it("is not read again within the interval after a failed read", async () => {
    world.indexStatus = 500;
    expect((await dashboardGET(get("/dashboard"))).status).toBe(502);
    world.indexStatus = null;
    expect((await dashboardGET(get("/dashboard"))).status).toBe(502);
    vi.setSystemTime(NOW + 60_000);
    expect((await dashboardGET(get("/dashboard"))).status).toBe(200);
    expect(indexReads()).toBe(2);
  });
});

describe("a query string", () => {
  // The CDN keys on the whole URL, so only the bare path may read a source:
  // anything else is refused or redirected, and no variant reaches Storj or
  // Ambient.
  const read = (path: string) =>
    path.startsWith("/current")
      ? currentGET(get(path))
      : path.startsWith("/status")
        ? statusGET(get(path))
        : path.startsWith("/stations")
          ? stationsGET(get(path))
          : dashboardGET(get(path));

  it.each([
    "/current?x=1",
    "/current?x=2",
    "/dashboard?x=1",
    "/status?x=1",
    "/status?verbose",
    "/current?fixture=dry",
    "/dashboard?age=3",
    `/current?station=${MAC}&station=${MAC}`,
  ])("refuses %s without reading anything", async (path) => {
    const response = await read(path);
    expect(response.status).toBe(400);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(world.calls).toHaveLength(0);
  });

  it.each([
    `/current?station=${MAC}`,
    `/current?station=${MAC.toLowerCase()}`,
    `/dashboard?station=${encodeURIComponent(MAC).replaceAll("%3A", "%3a")}`,
    `/status?station=%20${encodeURIComponent(MAC)}`,
  ])("sends %s to the station's one cached URL", async (path) => {
    const response = await read(path);
    expect(response.status).toBe(308);
    const fn = path.slice(1, path.indexOf("?"));
    expect(response.headers.get("location")).toBe(`${fn}?${station(MAC)}`);
    expect(world.calls).toHaveLength(0);
  });
});

describe("an error as a log line", () => {
  it("is its name and message, key-shaped parameters redacted", () => {
    const error = new TypeError("fetch failed: https://x.test/v1?apiKey=abc&applicationKey=def", {
      cause: new Error("socket hang up"),
    });
    expect(describeError(error)).toBe(
      "TypeError: fetch failed: https://x.test/v1?apiKey=[redacted]&applicationKey=[redacted]",
    );
  });

  it("takes out the given secrets, plain or URL-encoded", () => {
    const error = new Error("key a/b+c rejected; again a%2Fb%2Bc");
    expect(describeError(error, ["a/b+c"])).toBe(
      "Error: key [redacted] rejected; again [redacted]",
    );
  });

  it("names a rejection that is not an Error", () => {
    expect(describeError("boom")).toBe("non-Error: boom");
  });
});
