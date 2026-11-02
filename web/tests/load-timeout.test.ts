/**
 * The browser's reads fail when a read function hangs, so the page's refresh
 * cycle never waits forever on one. No network: `fetch` is faked.
 */
import { describe, expect, it } from "vitest";

import {
  READ_TIMEOUT_MS,
  fetchCurrent,
  fetchDashboard,
  fetchEventCurve,
} from "../src/load.js";

/** A fetch that never answers, and rejects only when its signal aborts. */
function hungFetch(seen: { signal?: AbortSignal | null }): typeof fetch {
  return ((_url: unknown, init?: RequestInit) =>
    new Promise((_resolve, reject) => {
      seen.signal = init?.signal;
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
    })) as typeof fetch;
}

describe("client read timeout", () => {
  it("allows a read longer than the server's own limits and shorter than a hang", () => {
    expect(READ_TIMEOUT_MS).toBeGreaterThan(10_000);
    expect(READ_TIMEOUT_MS).toBeLessThanOrEqual(30_000);
  });

  it("fails a hung /dashboard within the timeout", async () => {
    const seen: { signal?: AbortSignal | null } = {};
    const started = Date.now();
    await expect(fetchDashboard("", hungFetch(seen), 30)).rejects.toMatchObject({
      name: "TimeoutError",
    });
    expect(Date.now() - started).toBeLessThan(2_000);
    expect(seen.signal).toBeDefined();
  });

  it("fails a hung /current", async () => {
    await expect(fetchCurrent("", hungFetch({}), 30)).rejects.toMatchObject({
      name: "TimeoutError",
    });
  });

  it("fails a hung /event", async () => {
    await expect(fetchEventCurve(1, "", hungFetch({}), 30)).rejects.toMatchObject({
      name: "TimeoutError",
    });
  });

  it("fails a body that stalls after the headers arrive", async () => {
    const stalled = ((_url: unknown, init?: RequestInit) =>
      Promise.resolve({
        ok: true,
        json: () =>
          new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
          }),
      })) as unknown as typeof fetch;
    await expect(fetchDashboard("", stalled, 30)).rejects.toMatchObject({
      name: "TimeoutError",
    });
  });

  it("gives every read the default timeout signal", async () => {
    const seen: { signal?: AbortSignal | null } = {};
    const capture = ((_url: unknown, init?: RequestInit) => {
      seen.signal = init?.signal;
      return Promise.resolve({ ok: false, status: 500 });
    }) as unknown as typeof fetch;
    await expect(fetchDashboard("", capture)).rejects.toThrow("500");
    expect(seen.signal).toBeInstanceOf(AbortSignal);
    expect(seen.signal?.aborted).toBe(false);
  });
});
