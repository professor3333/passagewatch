import { describe, expect, it } from "vitest";

import { ActiveClock, adjust, blocks, nextStep, susScore, type StudyPlan } from "./study";

const plan: StudyPlan = {
  release: "pw-test",
  clips: {},
  practice: ["p01", "p02"],
  sets: { S1: ["c03", "c01"], S2: ["c02", "c04"] },
  participants: {
    D1: [
      { condition: "manual", set: "S1" },
      { condition: "assisted", set: "S2" },
    ],
  },
  jobs: { D1: { p02: "job_p", c02: "job_2", c04: "job_4" } },
};

const progress = (clips: string[], forms: ("manual" | "assisted")[] = []) => ({
  participant: "D1",
  clips_done: clips,
  questionnaires_done: forms,
});

describe("study sequence", () => {
  it("puts each block's practice clip first", () => {
    expect(blocks(plan, "D1")).toEqual([
      { condition: "manual", clips: ["p01", "c03", "c01"] },
      { condition: "assisted", clips: ["p02", "c02", "c04"] },
    ]);
  });

  it("walks clips in order, then the block's questionnaires", () => {
    expect(nextStep(plan, "D1", progress([]))).toMatchObject({ kind: "trial", block: 1, clip: "p01", practice: true, jobId: null });
    expect(nextStep(plan, "D1", progress(["p01"]))).toMatchObject({ clip: "c03", practice: false, position: 2, total: 3 });
    expect(nextStep(plan, "D1", progress(["p01", "c03", "c01"]))).toEqual({ kind: "questionnaire", block: 1, condition: "manual" });
  });

  it("gives assisted trials the participant's own job", () => {
    const step = nextStep(plan, "D1", progress(["p01", "c03", "c01", "p02"], ["manual"]));
    expect(step).toMatchObject({ kind: "trial", block: 2, condition: "assisted", clip: "c02", jobId: "job_2" });
  });

  it("ends after both questionnaires", () => {
    const all = ["p01", "c03", "c01", "p02", "c02", "c04"];
    expect(nextStep(plan, "D1", progress(all, ["manual"]))).toMatchObject({ kind: "questionnaire", block: 2 });
    expect(nextStep(plan, "D1", progress(all, ["manual", "assisted"]))).toEqual({ kind: "done" });
  });

  it("refuses an unknown participant", () => {
    expect(() => nextStep(plan, "P9", progress([]))).toThrow("unknown participant");
  });
});

describe("active clock", () => {
  it("excludes paused time and counts pauses", () => {
    let t = 1000;
    const clock = new ActiveClock(() => t);
    t += 5000;
    clock.pause();
    t += 60_000; // an interruption
    clock.pause(); // no effect while paused
    clock.resume();
    t += 2500;
    expect(clock.elapsedMs()).toBe(7500);
    expect(clock.pauses).toBe(1);
    expect(clock.paused).toBe(false);
  });
});

describe("manual tally and SUS", () => {
  it("never goes below zero", () => {
    const counts = { right: 0, left: 1 };
    adjust(counts, "right", -1);
    adjust(counts, "left", 1);
    expect(counts).toEqual({ right: 0, left: 2 });
  });

  it("scores SUS the standard way", () => {
    expect(susScore([5, 1, 5, 1, 5, 1, 5, 1, 5, 1])).toBe(100);
    expect(susScore([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])).toBe(50);
    expect(susScore([1, 5, 1, 5, 1, 5, 1, 5, 1, 5])).toBe(0);
  });
});
