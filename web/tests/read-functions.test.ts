/**
 * The two read functions, called the way Vercel calls them: `GET(request)`
 * with `process.env` set and `fetch` faked. Storj serves the golden
 * documents; Ambient serves a recorded-shape `/devices` body. No network.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GET as currentGET } from "../api/current.js";
import { GET as dashboardGET } from "../api/dashboard.js";
import { describeError } from "../server/errors.js";
import { SCHEMA_VERSION } from "../src/snapshot.js";
import type { Scenario } from "./golden.js";
import { golden, goldenText } from "./golden.js";

const MAC = "00:11:22:33:44:55";
const STORJ = "https://gateway.storj.test";
const AMBIENT = "https://ambient.test/v1";

const ENV = {
  STORJ_ACCESS_KEY_ID: "test-access",
  STORJ_SECRET_ACCESS_KEY: "test-secret",
  STORJ_ENDPOINT_URL: STORJ,
  STORJ_BUCKET: "lookout",
  AMBIENT_API_KEY: "test-api",
  AMBIENT_APPLICATION_KEY: "test-app",
  AMBIENT_ENDPOINT: AMBIENT,
  STATION_MAC: MAC,
};

interface FakeWorld {
  scenario: Scenario;
  storjStatus: number | null; // null: serve the golden
  ambientStatus: number | null; // null: serve `devices`
  devices: unknown[];
  calls: Request[];
}

let world: FakeWorld;

function fakeFetch(input: RequestInfo | URL, init?: RequestInit) {
  const request = new Request(input, init);
  world.calls.push(request);
  const url = new URL(request.url);
  if (url.origin === STORJ) {
    if (world.storjStatus !== null) {
      return Promise.resolve(new Response("nope", { status: world.storjStatus }));
    }
    const key = decodeURIComponent(url.pathname).replace("/lookout/", "");
    const match = /^snapshot\/(.+)\.(dashboard|context)\.json$/.exec(key);
    if (!match || match[1] !== MAC) {
      return Promise.resolve(new Response("NoSuchKey", { status: 404 }));
    }
    const kind = match[2] as "dashboard" | "context";
    return Promise.resolve(new Response(goldenText(world.scenario, kind)));
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

beforeEach(() => {
  world = {
    scenario: "raining",
    storjStatus: null,
    ambientStatus: null,
    devices: [],
    calls: [],
  };
  for (const [name, value] of Object.entries(ENV)) vi.stubEnv(name, value);
  vi.stubGlobal("fetch", vi.fn(fakeFetch));
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

const get = (path: string) => new Request(`https://lookout.test${path}`);

describe("GET /dashboard", () => {
  it("returns the snapshot document as Storj holds it, CDN-cached", async () => {
    const response = await dashboardGET(get("/dashboard"));

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toBe(
      "s-maxage=300, stale-while-revalidate=300",
    );
    expect(response.headers.get("content-type")).toMatch(/application\/json/);
    expect(await response.json()).toEqual(golden("raining", "dashboard"));
  });

  it("reads the default station's key with a signed request", async () => {
    world.scenario = "dry";
    await dashboardGET(get("/dashboard"));

    expect(world.calls).toHaveLength(1);
    const call = world.calls[0]!;
    expect(decodeURIComponent(new URL(call.url).pathname)).toBe(
      `/lookout/snapshot/${MAC}.dashboard.json`,
    );
    expect(call.headers.get("authorization")).toMatch(
      /^AWS4-HMAC-SHA256 Credential=test-access\//,
    );
  });

  it("never calls Ambient", async () => {
    await dashboardGET(get("/dashboard"));
    expect(world.calls.every((c) => c.url.startsWith(STORJ))).toBe(true);
  });

  it("takes the pinned station as a parameter, in either case", async () => {
    const response = await dashboardGET(
      get(`/dashboard?station=${encodeURIComponent(MAC.toLowerCase())}`),
    );
    expect(response.status).toBe(200);
  });

  it("answers 404, uncached, for any other station, without reading Storj", async () => {
    const response = await dashboardGET(get("/dashboard?station=AA:BB:CC:DD:EE:FF"));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(world.calls).toHaveLength(0);
  });

  it("refuses a station that is not a MAC address", async () => {
    const response = await dashboardGET(get("/dashboard?station=../archive"));
    expect(response.status).toBe(400);
    expect(world.calls).toHaveLength(0);
  });

  it("answers 404 when the station has no snapshot", async () => {
    vi.stubEnv("STATION_MAC", "AA:BB:CC:DD:EE:FF");
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
  });

  it("answers 502 when Storj fails", async () => {
    world.storjStatus = 500;
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(502);
    expect(response.headers.get("cache-control")).toBe("no-store");
  });

  it("answers 502 when the document breaks the contract", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(Response.json({ schema_version: 99 }))),
    );
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(502);
  });

  it("names the env vars it is missing, not their values", async () => {
    vi.stubEnv("STORJ_SECRET_ACCESS_KEY", "");
    vi.stubEnv("STATION_MAC", "");
    const response = await dashboardGET(get("/dashboard"));
    expect(response.status).toBe(500);
    const body = (await response.json()) as { error: string };
    expect(body.error).toContain("STORJ_SECRET_ACCESS_KEY");
    expect(body.error).toContain("STATION_MAC");
    expect(world.calls).toHaveLength(0);
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
      "s-maxage=60, stale-while-revalidate=60",
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

  it("still serves the context when Ambient fails, uncached", async () => {
    world.ambientStatus = 429;
    const response = await currentGET(get("/current"));
    expect(response.headers.get("cache-control")).toBe("no-store");
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
    expect(response.headers.get("cache-control")).toBe("no-store");
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

  it("answers 502 when neither source can be read", async () => {
    world.storjStatus = 500;
    world.ambientStatus = 500;
    const response = await currentGET(get("/current"));
    expect(response.status).toBe(502);
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

  it("answers 404, uncached, for any other station, reading nothing", async () => {
    world.devices = [ambientDevice(1, "AA:BB:CC:DD:EE:FF")];
    const response = await currentGET(get("/current?station=AA:BB:CC:DD:EE:FF"));
    expect(response.status).toBe(404);
    expect(response.headers.get("cache-control")).toBe("no-store");
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
