// Types of the PassageWatch v1 API responses used by the review interface.
// They mirror passagewatch.service.api.schemas.

export type ImageDirection = "right" | "left";
export type ReviewState = "automatic" | "accepted" | "rejected" | "corrected" | "unresolved";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Counting {
  policy: string;
  line_x_normalized: number;
  upstream_direction: ImageDirection | null;
}

export interface Clip {
  clip_id: string;
  sha256: string;
  media_kind: "frames" | "video";
  num_frames: number;
  width: number;
  height: number;
  framerate: number;
  duration_seconds: number;
  meters: { x_start: number; x_stop: number; y_start: number; y_stop: number };
  created_at: string;
  expires_at: string | null;
}

export interface JobAccepted {
  job_id: string;
  status: JobStatus;
  pipeline_version: string;
  status_url: string;
}

export interface Job {
  job_id: string;
  clip_id: string;
  status: JobStatus;
  progress: number;
  attempts: number;
  max_attempts: number;
  error: string | null;
  pipeline_version: string;
  counting: Counting;
  cached_from: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  results_url: string | null;
}

export interface Counts {
  right: number;
  left: number;
  upstream: number | null;
  downstream: number | null;
  net_upstream: number | null;
}

export interface Results {
  job_id: string;
  clip_id: string;
  recording_sha256: string;
  status: JobStatus;
  cached: boolean;
  counting: Counting;
  pipeline_version: string;
  pipeline_config_sha256: string;
  automatic: Counts;
  reviewed: Counts | null;
  review: { revision: number; kind: "automatic" | "reviewed"; state: "pending" | "reviewed" };
  tracks: number;
  tracks_url: string;
  created_at: string;
}

export interface Track {
  track_id: number;
  start_frame: number;
  end_frame: number;
  start_time_s: number;
  end_time_s: number;
  observations: number;
  start_u: number;
  end_u: number;
  displacement: number;
  outcome: "passage" | "stationary" | "no_crossing";
  direction: ImageDirection | null;
  mean_score: number;
  min_score: number;
  review_state: ReviewState;
  final_direction: ImageDirection | null;
}

export interface TracksPage {
  job_id: string;
  revision: number;
  total: number;
  offset: number;
  limit: number;
  tracks: Track[];
}

export interface Box {
  track_id: number;
  frame_index: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  score: number;
}

export type ReviewAction = "accept" | "reject" | "set_direction" | "mark_unresolved" | "add_passage";

export interface ReviewRequest {
  base_revision: number;
  action: ReviewAction;
  track_id?: number;
  passage_id?: string;
  direction?: ImageDirection;
  frame_index?: number;
  reason?: string;
}

export interface ReviewResponse {
  job_id: string;
  revision: number;
  reviewed: Counts;
  unresolved: number;
  added_passages: number;
  results_url: string;
}

export interface AddedPassage {
  passage_id: string;
  state: "added" | "rejected";
  direction: ImageDirection;
  river_direction: "upstream" | "downstream" | null;
  frame_index: number;
  time_s: number;
}
