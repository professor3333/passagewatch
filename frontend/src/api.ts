// A small typed client for the PassageWatch v1 API (same origin as the UI).

import type {
  AddedPassage,
  Audit,
  Box,
  Clip,
  Counting,
  DemoListing,
  Job,
  JobAccepted,
  Results,
  ReviewRequest,
  ReviewResponse,
  Track,
  TracksPage,
} from "./types";
import type { QuestionnaireIn, StudyPlan, StudyProgress, TrialIn, TrialOut } from "./study";

/** An HTTP error with the API's explanation (FastAPI's `detail`). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`${status}: ${detail}`);
  }

  /** The review was based on an outdated revision: someone else saved first. */
  get isConflict(): boolean {
    return this.status === 409;
  }
}

export type Fetch = typeof fetch;

export function detailOf(body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // Validation errors: "field: message; ..."
      return detail
        .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`)
        .join("; ");
    }
  }
  return "unexpected response";
}

export class PassageWatchApi {
  constructor(
    private readonly base = "",
    private readonly fetchImpl: Fetch = (...args) => fetch(...args),
  ) {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImpl(this.base + path, init);
    if (!response.ok) {
      let body: unknown = null;
      try {
        body = await response.json();
      } catch {
        // Not JSON (e.g. a proxy error page).
      }
      throw new ApiError(response.status, detailOf(body));
    }
    return (await response.json()) as T;
  }

  uploadClip(file: Blob, fields: Record<string, string>): Promise<Clip> {
    const form = new FormData();
    form.append("file", file, file instanceof File ? file.name : "recording");
    for (const [key, value] of Object.entries(fields)) form.append(key, value);
    return this.request<Clip>("/v1/clips", { method: "POST", body: form });
  }

  getClip(clipId: string): Promise<Clip> {
    return this.request<Clip>(`/v1/clips/${encodeURIComponent(clipId)}`);
  }

  createJob(clipId: string, counting: Partial<Counting>, idempotencyKey: string): Promise<JobAccepted> {
    return this.request<JobAccepted>("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": idempotencyKey },
      body: JSON.stringify({ clip_id: clipId, counting }),
    });
  }

  getDemos(): Promise<DemoListing> {
    return this.request<DemoListing>("/v1/demos");
  }

  getJob(jobId: string): Promise<Job> {
    return this.request<Job>(`/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  getResults(jobId: string): Promise<Results> {
    return this.request<Results>(`/v1/jobs/${encodeURIComponent(jobId)}/results`);
  }

  /** All tracks of the latest revision (the API pages them). */
  async getAllTracks(jobId: string, pageSize = 200): Promise<{ revision: number; tracks: Track[] }> {
    const tracks: Track[] = [];
    let revision = 0;
    for (let offset = 0; ; offset += pageSize) {
      const page = await this.request<TracksPage>(
        `/v1/jobs/${encodeURIComponent(jobId)}/tracks?offset=${offset}&limit=${pageSize}`,
      );
      revision = page.revision;
      tracks.push(...page.tracks);
      if (tracks.length >= page.total || page.tracks.length === 0) break;
    }
    return { revision, tracks };
  }

  async getObservations(jobId: string, start: number, stop: number): Promise<Box[]> {
    const body = await this.request<{ boxes: Box[] }>(
      `/v1/jobs/${encodeURIComponent(jobId)}/observations?start=${start}&stop=${stop}`,
    );
    return body.boxes;
  }

  getAudit(jobId: string): Promise<Audit> {
    return this.request<Audit>(`/v1/jobs/${encodeURIComponent(jobId)}/audit`);
  }

  submitReview(jobId: string, review: ReviewRequest): Promise<ReviewResponse> {
    return this.request<ReviewResponse>(`/v1/jobs/${encodeURIComponent(jobId)}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    });
  }

  async getAddedPassages(jobId: string): Promise<AddedPassage[]> {
    const report = await this.request<{ added_passages: AddedPassage[] }>(
      `/v1/jobs/${encodeURIComponent(jobId)}/export?format=json`,
    );
    return report.added_passages;
  }

  // Usability study (study mode only) ------------------------------------------------

  getStudyPlan(): Promise<StudyPlan> {
    return this.request<StudyPlan>("/v1/study/plan");
  }

  getStudyProgress(participant: string): Promise<StudyProgress> {
    return this.request<StudyProgress>(`/v1/study/progress/${encodeURIComponent(participant)}`);
  }

  submitTrial(trial: TrialIn): Promise<TrialOut> {
    return this.request<TrialOut>("/v1/study/trials", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(trial),
    });
  }

  submitQuestionnaire(form: QuestionnaireIn): Promise<{ status: string }> {
    return this.request<{ status: string }>("/v1/study/questionnaires", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
  }

  frameUrl(clipId: string, index: number): string {
    return `${this.base}/v1/clips/${encodeURIComponent(clipId)}/frames/${index}`;
  }

  exportUrl(jobId: string, format: "json" | "csv"): string {
    return `${this.base}/v1/jobs/${encodeURIComponent(jobId)}/export?format=${format}`;
  }
}
