// Mapping between frame pixels and the canvas the frame is drawn on.
// Boxes use the internal convention: continuous 0-based pixel coordinates,
// [x_min, x_max) x [y_min, y_max), in the original frame.

export interface Fit {
  scale: number; // canvas pixels per frame pixel
  offsetX: number;
  offsetY: number;
  width: number; // drawn frame size in canvas pixels
  height: number;
}

/** Fit a frame inside a canvas without distortion, centered. */
export function fitFrame(frameWidth: number, frameHeight: number, canvasWidth: number, canvasHeight: number): Fit {
  if (frameWidth <= 0 || frameHeight <= 0) throw new Error("frame size must be positive");
  const scale = Math.min(canvasWidth / frameWidth, canvasHeight / frameHeight);
  const width = frameWidth * scale;
  const height = frameHeight * scale;
  return { scale, offsetX: (canvasWidth - width) / 2, offsetY: (canvasHeight - height) / 2, width, height };
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function boxToCanvas(box: { x_min: number; y_min: number; x_max: number; y_max: number }, fit: Fit): Rect {
  return {
    x: fit.offsetX + box.x_min * fit.scale,
    y: fit.offsetY + box.y_min * fit.scale,
    w: (box.x_max - box.x_min) * fit.scale,
    h: (box.y_max - box.y_min) * fit.scale,
  };
}

/** Canvas x of the counting line, which is defined as a fraction of the frame width. */
export function countingLineX(lineXNormalized: number, fit: Fit): number {
  return fit.offsetX + lineXNormalized * fit.width;
}

/** The frame index under a canvas x position on a timeline of `width` pixels. */
export function frameAtTimeline(x: number, width: number, numFrames: number): number {
  if (numFrames <= 1 || width <= 0) return 0;
  const fraction = Math.min(1, Math.max(0, x / width));
  return Math.round(fraction * (numFrames - 1));
}
