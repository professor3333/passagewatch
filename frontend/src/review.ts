// Review logic that does not touch the DOM: ordering, labels, colors, and counts.

import type { Counts, ImageDirection, ReviewState, Track } from "./types";

/** Tracks in the order a reviewer goes through them: by start time, then ID. */
export function reviewOrder(tracks: Track[]): Track[] {
  return [...tracks].sort((a, b) => a.start_frame - b.start_frame || a.track_id - b.track_id);
}

/** The first track at or after `afterTrackId` (in review order) that nobody reviewed yet. */
export function nextUnreviewed(tracks: Track[], afterTrackId: number | null): Track | null {
  const ordered = reviewOrder(tracks);
  const start = afterTrackId === null ? 0 : ordered.findIndex((t) => t.track_id === afterTrackId) + 1;
  for (let i = 0; i < ordered.length; i++) {
    const track = ordered[(start + i) % ordered.length];
    if (track && track.review_state === "automatic") return track;
  }
  return null;
}

/** Tracks visible at a frame. */
export function tracksAt(tracks: Track[], frame: number): Track[] {
  return tracks.filter((t) => t.start_frame <= frame && frame <= t.end_frame);
}

export const STATE_LABELS: Record<ReviewState, string> = {
  automatic: "not reviewed",
  accepted: "accepted",
  rejected: "rejected",
  corrected: "corrected",
  unresolved: "unresolved",
};

// Colors chosen to stay distinguishable for common color-vision deficiencies,
// and every state also differs in line style or label text.
export const STATE_COLORS: Record<ReviewState, string> = {
  automatic: "#4ea1ff",
  accepted: "#2ec27e",
  rejected: "#9a9a9a",
  corrected: "#ff9f1c",
  unresolved: "#d16dff",
};

export function directionArrow(direction: ImageDirection | null): string {
  if (direction === "right") return "→";
  if (direction === "left") return "←";
  return "·";
}

/** "→ upstream" when orientation is configured, otherwise just the image direction. */
export function directionLabel(direction: ImageDirection | null, upstream: ImageDirection | null): string {
  if (direction === null) return "no passage";
  const arrow = directionArrow(direction);
  if (upstream === null) return `${arrow} ${direction}`;
  return `${arrow} ${direction === upstream ? "upstream" : "downstream"}`;
}

export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

export interface CountRow {
  label: string;
  automatic: string;
  reviewed: string;
}

/** Rows for the counts panel; river directions only when orientation is configured. */
export function countRows(automatic: Counts, reviewed: Counts | null): CountRow[] {
  const value = (counts: Counts | null, key: keyof Counts): string => {
    if (counts === null) return "—";
    const v = counts[key];
    return v === null ? "—" : String(v);
  };
  const rows: CountRow[] = [
    { label: "→ right", automatic: value(automatic, "right"), reviewed: value(reviewed, "right") },
    { label: "← left", automatic: value(automatic, "left"), reviewed: value(reviewed, "left") },
  ];
  if (automatic.upstream !== null) {
    rows.push(
      { label: "upstream", automatic: value(automatic, "upstream"), reviewed: value(reviewed, "upstream") },
      { label: "downstream", automatic: value(automatic, "downstream"), reviewed: value(reviewed, "downstream") },
      { label: "net upstream", automatic: value(automatic, "net_upstream"), reviewed: value(reviewed, "net_upstream") },
    );
  }
  return rows;
}

/** Random idempotency key for job creation, so a retried submit cannot start two jobs. */
export function newIdempotencyKey(random: () => string = () => crypto.randomUUID()): string {
  return `ui-${random()}`;
}
