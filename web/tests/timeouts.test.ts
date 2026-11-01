/**
 * A hung upstream: Ambient or Storj accepts the connection and never
 * answers. Each read gives up after its timeout, so `/current` still serves
 * the half that did answer. The fetch double never settles unless the
 * request's signal aborts it. No network.
 */
import { describe, expect, it } from "vitest";

import { ambientReader } from "../server/ambient.js";
import { handleCurrent } from "../server/handlers.js";
import { storjCurveReader, storjReader } from "../server/storj.js";
import { goldenText } from "./golden.js";

const MAC = "00:11:22:33:44:55";
const TIMEOUT_MS = 20;
const AMBIENT = { apiKey: "k", applicationKey: "a", endpoint: "https://ambient.test/v1" };
const STORJ = {
  accessKeyId: "id",
  secretAccessKey: "secret",
  endpointUrl: "https://gateway.storj.test",
  bucket: "lookout",
};

/** A fetch that never answers; it rejects only when its signal aborts. */
function hungFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const signal = init?.signal ?? (input instanceof Request ? input.signal : undefined);
  return new Promise((_resolve, reject) => {
    signal?.addEventListener("abort", () => reject(signal.reason));
  });
}

/** Settles with how the promise settled, or "hung" after `ms`. */
function within<T>(promise: Promise<T>, ms: number): Promise<string> {
  return Promise.race([
    promise.then(
      () => "resolved",
      () => "rejected",
    ),
    new Promise<string>((resolve) => setTimeout(() => resolve("hung"), ms)),
  ]);
}

describe("a hung upstream", () => {
  it("fails the Ambient read once its timeout passes", async () => {
    const read = ambientReader(AMBIENT, hungFetch, TIMEOUT_MS);
    expect(await within(read(MAC), 1000)).toBe("rejected");
  });

  it("fails a snapshot read from Storj once its timeout passes", async () => {
    const read = storjReader(STORJ, hungFetch, TIMEOUT_MS);
    expect(await within(read(MAC, "dashboard"), 1000)).toBe("rejected");
  });

  it("fails a curve read from Storj once its timeout passes", async () => {
    const read = storjCurveReader(STORJ, hungFetch, TIMEOUT_MS);
    expect(await within(read(MAC, 1767758400000), 1000)).toBe("rejected");
  });

  it("leaves /current serving the context when Ambient hangs", async () => {
    const response = await handleCurrent(new Request("https://lookout.test/current"), {
      defaultMac: MAC,
      readSnapshot: async () => goldenText("raining", "context"),
      readLive: ambientReader(AMBIENT, hungFetch, TIMEOUT_MS),
    });
    expect(response.status).toBe(200);
    const body = (await response.json()) as { last_data: unknown; context: unknown };
    expect(body.last_data).toBeNull();
    expect(body.context).not.toBeNull();
  });
});
