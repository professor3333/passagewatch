import { describe, expect, it } from "vitest";

import {
  auditWindowAt,
  countRows,
  directionLabel,
  formatTime,
  newIdempotencyKey,
  nextUnreviewed,
  reasonText,
  reviewOrder,
  tracksAt,
} from "./review";
import type { AuditWindow, Counts, Track } from "./types";

function track(id: number, start: number, state: Track["review_state"] = "automatic"): Track {
  return {
    track_id: id,
    start_frame: start,
    end_frame: start + 10,
    start_time_s: start / 10,
    end_time_s: (start + 10) / 10,
    observations: 11,
    start_u: 0.2,
    end_u: 0.8,
    displacement: 0.6,
    outcome: "passage",
    direction: "right",
    mean_score: 0.8,
    min_score: 0.5,
    review_state: state,
    final_direction: state === "rejected" ? null : "right",
    review_score: null,
    triage: null,
    review_reasons: null,
  };
}

describe("review queue order", () => {
  const scored = (id: number, triage: Track["triage"], score: number, passage = true): Track => ({
    ...track(id, id),
    triage,
    review_score: score,
    review_reasons: [],
    direction: passage ? "right" : null,
  });

  it("goes by triage, then passages first, then lowest review score", () => {
    const tracks = [
      scored(1, "suggested", 0.9),
      scored(2, "needs_review", 0.5),
      scored(3, "suggested", 0.3, false),
      scored(4, "unresolved", 0.8),
      scored(5, "needs_review", 0.2),
      scored(6, "suggested", 0.4),
    ];
    expect(reviewOrder(tracks).map((t) => t.track_id)).toEqual([4, 5, 2, 6, 1, 3]);
  });

  it("explains triage in plain language", () => {
    expect(reasonText(["duration", "possible_missed_passage"])).toBe("short track, may be part of a missed passage");
    expect(reasonText(null)).toBe("");
  });

  it("finds the audit window that contains a frame", () => {
    const window = (index: number, start: number, stop: number): AuditWindow => ({
      index,
      start_frame: start,
      stop_frame: stop,
      start_time_s: start / 10,
      stop_time_s: stop / 10,
      state: "pending",
      passages_added: 0,
    });
    const windows = [window(0, 10, 60), window(1, 100, 150)];
    expect(auditWindowAt(windows, 10)?.index).toBe(0);
    expect(auditWindowAt(windows, 60)).toBeNull();
    expect(auditWindowAt(windows, 149)?.index).toBe(1);
  });
});

describe("review order", () => {
  const tracks = [track(3, 50), track(1, 50), track(2, 5, "accepted"), track(4, 90)];

  it("orders by start time, then track ID", () => {
    expect(reviewOrder(tracks).map((t) => t.track_id)).toEqual([2, 1, 3, 4]);
  });

  it("finds the next unreviewed track, wrapping around", () => {
    expect(nextUnreviewed(tracks, null)?.track_id).toBe(1);
    expect(nextUnreviewed(tracks, 3)?.track_id).toBe(4);
    expect(nextUnreviewed(tracks, 4)?.track_id).toBe(1);
    expect(nextUnreviewed([track(1, 0, "accepted")], null)).toBeNull();
  });

  it("lists tracks visible at a frame", () => {
    expect(tracksAt(tracks, 55).map((t) => t.track_id).sort()).toEqual([1, 3]);
  });
});

describe("labels", () => {
  it("uses river directions only when orientation is configured", () => {
    expect(directionLabel("right", null)).toBe("→ right");
    expect(directionLabel("right", "right")).toBe("→ upstream");
    expect(directionLabel("left", "right")).toBe("← downstream");
    expect(directionLabel(null, "right")).toBe("no passage");
  });

  it("formats times as minutes:seconds", () => {
    expect(formatTime(0)).toBe("0:00.0");
    expect(formatTime(75.34)).toBe("1:15.3");
  });
});

describe("counts panel", () => {
  const plain: Counts = { right: 4, left: 1, upstream: null, downstream: null, net_upstream: null };
  const oriented: Counts = { right: 4, left: 1, upstream: 4, downstream: 1, net_upstream: 3 };

  it("shows image directions, and a dash for reviewed counts before review", () => {
    expect(countRows(plain, null)).toEqual([
      { label: "→ right", automatic: "4", reviewed: "—" },
      { label: "← left", automatic: "1", reviewed: "—" },
    ]);
  });

  it("adds river directions when orientation is configured", () => {
    const rows = countRows(oriented, { ...oriented, right: 3, upstream: 3, net_upstream: 2 });

    expect(rows.map((r) => r.label)).toEqual(["→ right", "← left", "upstream", "downstream", "net upstream"]);
    expect(rows[4]).toEqual({ label: "net upstream", automatic: "3", reviewed: "2" });
  });
});

describe("newIdempotencyKey", () => {
  it("prefixes a random value", () => {
    expect(newIdempotencyKey(() => "abc")).toBe("ui-abc");
  });
});
