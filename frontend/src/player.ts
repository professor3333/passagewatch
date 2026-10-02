// Frame and overlay caches for smooth playback over the network.

import type { PassageWatchApi } from "./api";
import type { Box } from "./types";

/** Keeps up to `limit` decoded frames, evicting the oldest; prefetches ahead of playback. */
export class FrameCache {
  private readonly frames = new Map<number, Promise<HTMLImageElement>>();

  constructor(
    private readonly api: PassageWatchApi,
    private readonly clipId: string,
    private readonly numFrames: number,
    private readonly limit = 160,
  ) {}

  get(index: number): Promise<HTMLImageElement> {
    let frame = this.frames.get(index);
    if (frame === undefined) {
      frame = new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`frame ${index} could not be loaded`));
        image.src = this.api.frameUrl(this.clipId, index);
      });
      this.frames.set(index, frame);
      frame.catch(() => this.frames.delete(index));
      while (this.frames.size > this.limit) {
        const oldest = this.frames.keys().next().value;
        if (oldest === undefined) break;
        this.frames.delete(oldest);
      }
    }
    return frame;
  }

  prefetch(from: number, count: number): void {
    for (let i = from; i < Math.min(this.numFrames, from + count); i++) {
      void this.get(i).catch(() => undefined);
    }
  }
}

/** Overlay boxes, fetched in windows of `window` frames and cached per window. */
export class ObservationCache {
  private readonly windows = new Map<number, Promise<Box[]>>();

  constructor(
    private readonly api: PassageWatchApi,
    private readonly jobId: string,
    private readonly numFrames: number,
    private readonly window = 200,
  ) {}

  async at(frame: number): Promise<Box[]> {
    const key = Math.floor(frame / this.window);
    let boxes = this.windows.get(key);
    if (boxes === undefined) {
      const start = key * this.window;
      const stop = Math.min(this.numFrames, start + this.window);
      boxes = this.api.getObservations(this.jobId, start, stop);
      this.windows.set(key, boxes);
      boxes.catch(() => this.windows.delete(key));
    }
    return (await boxes).filter((b) => b.frame_index === frame);
  }
}
