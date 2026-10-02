import { describe, expect, it } from "vitest";

import { ApiError, PassageWatchApi, detailOf } from "./api";
import { ObservationCache } from "./player";

interface Call {
  url: string;
  init?: RequestInit;
}

function fakeFetch(responses: Array<{ status: number; body: unknown }>): { fetch: typeof fetch; calls: Call[] } {
  const calls: Call[] = [];
  const queue = [...responses];
  const impl = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    calls.push({ url: String(input), init });
    const next = queue.shift() ?? { status: 500, body: { detail: "no more responses" } };
    return new Response(JSON.stringify(next.body), { status: next.status, headers: { "Content-Type": "application/json" } });
  };
  return { fetch: impl as typeof fetch, calls };
}

describe("PassageWatchApi", () => {
  it("creates jobs with an idempotency key and JSON counting", async () => {
    const { fetch, calls } = fakeFetch([{ status: 202, body: { job_id: "job_1", status: "queued" } }]);
    const api = new PassageWatchApi("", fetch);

    const job = await api.createJob("clip_1", { line_x_normalized: 0.5, upstream_direction: "right" }, "ui-1");

    expect(job.job_id).toBe("job_1");
    expect(calls[0]?.url).toBe("/v1/jobs");
    expect((calls[0]?.init?.headers as Record<string, string>)["Idempotency-Key"]).toBe("ui-1");
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual({
      clip_id: "clip_1",
      counting: { line_x_normalized: 0.5, upstream_direction: "right" },
    });
  });

  it("reads every page of tracks", async () => {
    const page = (ids: number[], total: number) => ({
      status: 200,
      body: { revision: 2, total, tracks: ids.map((track_id) => ({ track_id })) },
    });
    const { fetch, calls } = fakeFetch([page([1, 2], 3), page([3], 3)]);

    const result = await new PassageWatchApi("", fetch).getAllTracks("job_1", 2);

    expect(result.revision).toBe(2);
    expect(result.tracks.map((t) => t.track_id)).toEqual([1, 2, 3]);
    expect(calls.map((c) => c.url)).toEqual([
      "/v1/jobs/job_1/tracks?offset=0&limit=2",
      "/v1/jobs/job_1/tracks?offset=2&limit=2",
    ]);
  });

  it("turns a stale review into a recognizable conflict", async () => {
    const { fetch } = fakeFetch([{ status: 409, body: { detail: "review is based on revision 0, but the latest is 1" } }]);

    const error = await new PassageWatchApi("", fetch)
      .submitReview("job_1", { base_revision: 0, action: "accept", track_id: 1 })
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).isConflict).toBe(true);
    expect((error as ApiError).detail).toContain("latest is 1");
  });

  it("explains validation errors field by field", () => {
    expect(detailOf({ detail: [{ loc: ["body", "counting", "line_x_normalized"], msg: "too large" }] })).toBe(
      "counting.line_x_normalized: too large",
    );
    expect(detailOf("<html>")).toBe("unexpected response");
  });

  it("builds frame and export URLs", () => {
    const api = new PassageWatchApi("");

    expect(api.frameUrl("clip_1", 7)).toBe("/v1/clips/clip_1/frames/7");
    expect(api.exportUrl("job_1", "csv")).toBe("/v1/jobs/job_1/export?format=csv");
  });
});

describe("ObservationCache", () => {
  it("fetches each window once and filters by frame", async () => {
    const boxes = [
      { track_id: 1, frame_index: 3, x_min: 0, y_min: 0, x_max: 1, y_max: 1, score: 0.9 },
      { track_id: 2, frame_index: 4, x_min: 0, y_min: 0, x_max: 1, y_max: 1, score: 0.9 },
    ];
    const { fetch, calls } = fakeFetch([{ status: 200, body: { boxes } }]);
    const cache = new ObservationCache(new PassageWatchApi("", fetch), "job_1", 50, 10);

    expect((await cache.at(3)).map((b) => b.track_id)).toEqual([1]);
    expect((await cache.at(4)).map((b) => b.track_id)).toEqual([2]);
    expect(calls.map((c) => c.url)).toEqual(["/v1/jobs/job_1/observations?start=0&stop=10"]);
  });
});
