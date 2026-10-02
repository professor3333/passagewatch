import { describe, expect, it } from "vitest";

import { boxToCanvas, countingLineX, fitFrame, frameAtTimeline } from "./geometry";

describe("fitFrame", () => {
  it("fits a tall sonar frame by height and centers it horizontally", () => {
    const fit = fitFrame(288, 624, 1000, 624 * 2);

    expect(fit.scale).toBe(2);
    expect(fit.width).toBe(576);
    expect(fit.offsetX).toBe((1000 - 576) / 2);
    expect(fit.offsetY).toBe(0);
  });

  it("fits a wide frame by width", () => {
    const fit = fitFrame(800, 200, 400, 400);

    expect(fit.scale).toBe(0.5);
    expect(fit.offsetY).toBe((400 - 100) / 2);
  });

  it("rejects an empty frame", () => {
    expect(() => fitFrame(0, 10, 100, 100)).toThrow();
  });
});

describe("overlays", () => {
  const fit = fitFrame(100, 200, 300, 200); // scale 1, offsetX 100

  it("maps internal-convention boxes onto the canvas", () => {
    expect(boxToCanvas({ x_min: 10, y_min: 20, x_max: 30, y_max: 25 }, fit)).toEqual({ x: 110, y: 20, w: 20, h: 5 });
  });

  it("places the counting line at its fraction of the frame width", () => {
    expect(countingLineX(0.5, fit)).toBe(150);
    expect(countingLineX(0.25, fit)).toBe(125);
  });
});

describe("frameAtTimeline", () => {
  it("maps positions to frames and clamps the ends", () => {
    expect(frameAtTimeline(0, 200, 101)).toBe(0);
    expect(frameAtTimeline(100, 200, 101)).toBe(50);
    expect(frameAtTimeline(500, 200, 101)).toBe(100);
    expect(frameAtTimeline(-5, 200, 101)).toBe(0);
    expect(frameAtTimeline(10, 200, 1)).toBe(0);
  });
});
