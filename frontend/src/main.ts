// PassageWatch review interface: upload a recording, follow the job, review, export.
// Routes: #/ (upload), #/jobs/<job_id> (progress, then review), and in study mode
// #/study[/<participant>] (docs/usability_study.md).

import { ApiError, PassageWatchApi } from "./api";
import { el, errorText, showMessage } from "./dom";
import { studyView } from "./studyPages";
import { boxToCanvas, countingLineX, fitFrame } from "./geometry";
import { FrameCache, ObservationCache } from "./player";
import {
  STATE_COLORS,
  STATE_LABELS,
  TRIAGE_COLORS,
  TRIAGE_LABELS,
  auditWindowAt,
  countRows,
  directionLabel,
  formatTime,
  newIdempotencyKey,
  nextUnreviewed,
  reasonText,
  reviewOrder,
} from "./review";
import type { AddedPassage, Audit, Clip, ImageDirection, Job, Results, ReviewRequest, Track } from "./types";

const api = new PassageWatchApi();
const app = document.getElementById("app") as HTMLElement;
let cleanups: (() => void)[] = [];

// -- routing ----------------------------------------------------------------------------

function route(): void {
  for (const cleanup of cleanups) cleanup();
  cleanups = [];
  app.replaceChildren();
  const match = /^#\/jobs\/([A-Za-z0-9_-]+)$/.exec(location.hash);
  const study = /^#\/study(?:\/([A-Z][0-9]{1,3}))?$/.exec(location.hash);
  if (match?.[1]) {
    void jobView(match[1]);
  } else if (study) {
    void studyView(
      {
        app,
        api,
        review: reviewView,
        onLeave: (cleanup) => cleanups.push(cleanup),
        refresh: route,
      },
      study[1] ?? null,
    );
  } else {
    uploadView();
  }
}

window.addEventListener("hashchange", route);
route();

// -- upload -----------------------------------------------------------------------------

function field(label: string, input: HTMLElement, help?: string): HTMLLabelElement {
  return el("label", {}, label, input, help ? el("small", {}, help) : null);
}

function uploadView(): void {
  const file = el("input", { type: "file", name: "file", accept: ".zip,video/*", required: "" });
  const framerate = el("input", { type: "number", name: "framerate", step: "any", min: "0.1", placeholder: "e.g. 10" });
  const meters = ["x_meter_start", "x_meter_stop", "y_meter_start", "y_meter_stop"].map((name) =>
    el("input", { type: "number", name, step: "any", required: "" }),
  );
  const upstream = el(
    "select",
    { name: "upstream" },
    el("option", { value: "" }, "not set (report left/right only)"),
    el("option", { value: "right" }, "right"),
    el("option", { value: "left" }, "left"),
  );
  const line = el("input", { type: "number", name: "line", min: "0.05", max: "0.95", step: "0.01", value: "0.5" });
  const submit = el("button", { type: "submit", class: "primary" }, "Upload and analyze");
  const message = el("p", { class: "message" });
  const [x0, x1, y0, y1] = meters as [HTMLInputElement, HTMLInputElement, HTMLInputElement, HTMLInputElement];

  const form = el(
    "form",
    { class: "grid" },
    field("Recording", file, "A ZIP of frames 0.jpg … N-1.jpg, or a video file."),
    field("Frame rate (frames/s)", framerate, "Required for a ZIP of frames; videos carry their own."),
    field("Sonar window x start (m)", x0, "Horizontal extent of the image, in meters."),
    field("Sonar window x stop (m)", x1),
    field("Sonar window y start (m)", y0, "Range at the top of the image."),
    field("Sonar window y stop (m)", y1, "Range at the bottom of the image."),
    field("Upstream is to the", upstream, "Confirm on the recording; counts map to upstream/downstream only if set."),
    field("Counting line (fraction of width)", line, "0.5 = image center (the CFC benchmark convention)."),
    el("div", {}, submit),
  );
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const chosen = file.files?.[0];
    if (!chosen) return;
    submit.disabled = true;
    showMessage(message, "Uploading…");
    const fields: Record<string, string> = {
      x_meter_start: x0.value,
      x_meter_stop: x1.value,
      y_meter_start: y0.value,
      y_meter_stop: y1.value,
    };
    if (framerate.value) fields.framerate = framerate.value;
    const key = newIdempotencyKey();
    api
      .uploadClip(chosen, fields)
      .then((clip) => {
        showMessage(message, `Uploaded ${clip.num_frames} frames. Starting the analysis…`);
        return api.createJob(
          clip.clip_id,
          {
            line_x_normalized: Number(line.value),
            upstream_direction: (upstream.value || null) as ImageDirection | null,
          },
          key,
        );
      })
      .then((job) => {
        location.hash = `#/jobs/${job.job_id}`;
      })
      .catch((error: unknown) => {
        showMessage(message, errorText(error), "error");
        submit.disabled = false;
      });
  });

  const jobId = el("input", { type: "text", placeholder: "job_…", "aria-label": "Job ID" });
  const open = el("button", { type: "button" }, "Open");
  open.addEventListener("click", () => {
    if (jobId.value.trim()) location.hash = `#/jobs/${jobId.value.trim()}`;
  });

  app.append(
    el("section", { class: "panel" }, el("h2", {}, "Analyze a recording"), form, message),
    el("section", { class: "panel" }, el("h2", {}, "Open an existing job"), el("div", { class: "actions" }, jobId, open)),
  );
}

// -- job progress -------------------------------------------------------------------------

async function jobView(jobId: string): Promise<void> {
  const status = el("p", { class: "message" }, "Loading…");
  const bar = el("div", { style: "width: 0%" });
  app.append(el("section", { class: "panel" }, el("h2", {}, `Job ${jobId}`), el("div", { class: "progress" }, bar), status));
  let stopped = false;
  cleanups.push(() => {
    stopped = true;
  });
  while (!stopped) {
    let job: Job;
    try {
      job = await api.getJob(jobId);
    } catch (error) {
      showMessage(status, errorText(error), "error");
      return;
    }
    bar.style.width = `${Math.round(job.progress * 100)}%`;
    if (job.status === "succeeded") {
      if (!stopped) {
        app.replaceChildren();
        await reviewView(job);
      }
      return;
    }
    if (job.status === "failed") {
      showMessage(status, `The analysis failed after ${job.attempts} attempt(s): ${job.error ?? "unknown error"}`, "error");
      return;
    }
    showMessage(status, job.status === "queued" ? "Waiting for a worker…" : `Analyzing… ${Math.round(job.progress * 100)}%`);
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

// -- review -------------------------------------------------------------------------------

interface ReviewState {
  job: Job;
  clip: Clip;
  results: Results;
  tracks: Track[];
  passages: AddedPassage[];
  audit: Audit;
  revision: number;
  frame: number;
  selected: number | null;
  playing: boolean;
  speed: number;
}

/** The review view; in the usability study, `studyBar` replaces the export links. */
async function reviewView(job: Job, studyBar?: HTMLElement): Promise<void> {
  const [clip, results, { revision, tracks }, passages, audit] = await Promise.all([
    api.getClip(job.clip_id),
    api.getResults(job.job_id),
    api.getAllTracks(job.job_id),
    api.getAddedPassages(job.job_id),
    api.getAudit(job.job_id),
  ]);
  const state: ReviewState = {
    job,
    clip,
    results,
    tracks,
    passages,
    audit,
    revision,
    frame: 0,
    selected: null,
    playing: false,
    speed: 1,
  };
  const frames = new FrameCache(api, clip.clip_id, clip.num_frames);
  const overlays = new ObservationCache(api, job.job_id, clip.num_frames);
  const upstream = job.counting.upstream_direction;

  // Layout ----------------------------------------------------------------------------
  const canvas = el("canvas", { "aria-label": "Recording with tracked fish" });
  const timeline = el("input", { type: "range", min: "0", max: String(clip.num_frames - 1), value: "0", "aria-label": "Frame" });
  const time = el("span", { class: "time" });
  const play = el("button", { type: "button", title: "Play / pause (space)" }, "▶");
  const back = el("button", { type: "button", title: "Previous frame (←)" }, "◀︎");
  const forward = el("button", { type: "button", title: "Next frame (→)" }, "▶︎|");
  const speed = el(
    "select",
    { "aria-label": "Playback speed" },
    ...["0.5", "1", "2", "4"].map((s) => el("option", s === "1" ? { value: s, selected: "" } : { value: s }, `${s}×`)),
  );
  const message = el("p", { class: "message" });
  const countsBody = el("tbody");
  const revisionLabel = el("span", { class: "badge" });
  const selectedPanel = el("div");
  const trackBody = el("tbody");
  const auditPanel = el("div");

  const header = el(
    "section",
    { class: "panel" },
    el("h2", {}, `Review — job ${job.job_id}`),
    el(
      "p",
      { class: "muted" },
      `Recording ${clip.clip_id} (sha256 ${clip.sha256.slice(0, 12)}…), ${clip.num_frames} frames at ${Number(clip.framerate.toFixed(3))} fps · `,
      `pipeline ${job.pipeline_version} · counting line at ${job.counting.line_x_normalized} · `,
      upstream ? `upstream is to the ${upstream}` : "orientation not set",
      " ",
      results.cached ? el("span", { class: "badge", title: "This result was reused from an identical earlier analysis." }, "cached result") : null,
    ),
    el(
      "div",
      { class: "actions" },
      revisionLabel,
      studyBar ? null : el("a", { href: api.exportUrl(job.job_id, "csv"), download: "" }, "Export CSV"),
      studyBar ? null : el("a", { href: api.exportUrl(job.job_id, "json"), download: "" }, "Export JSON"),
    ),
  );
  const stage = el(
    "section",
    { class: "panel stage" },
    canvas,
    el("div", { class: "controls" }, back, play, forward, speed, timeline, time),
    el(
      "p",
      { class: "muted" },
      "Keys: ",
      el("kbd", {}, "space"), " play · ",
      el("kbd", {}, "←"), el("kbd", {}, "→"), " frame · ",
      el("kbd", {}, "j"), el("kbd", {}, "k"), " track · ",
      el("kbd", {}, "n"), " next unreviewed · ",
      el("kbd", {}, "a"), " accept · ",
      el("kbd", {}, "x"), " reject · ",
      el("kbd", {}, "r"), el("kbd", {}, "l"), " direction · ",
      el("kbd", {}, "u"), " unresolved",
    ),
  );
  const side = el(
    "div",
    {},
    el(
      "section",
      { class: "panel" },
      el("h2", {}, "Counts"),
      el(
        "table",
        {},
        el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", { class: "num" }, "automatic"), el("th", { class: "num" }, "reviewed"))),
        countsBody,
      ),
      message,
    ),
    el("section", { class: "panel" }, el("h2", {}, "Selected track"), selectedPanel),
    el(
      "section",
      { class: "panel" },
      el("h2", {}, "Tracks"),
      el(
        "div",
        { class: "track-list" },
        el(
          "table",
          {},
          el(
            "thead",
            {},
            el(
              "tr",
              {},
              el("th", {}, "#"),
              el("th", {}, "time"),
              el("th", {}, "automatic"),
              el("th", { title: "Automatic triage and heuristic review score (not a probability)" }, "triage"),
              el("th", {}, "review"),
              el("th", {}, "final"),
            ),
          ),
          trackBody,
        ),
      ),
    ),
    el("section", { class: "panel" }, el("h2", {}, "Random audit"), auditPanel),
  );
  app.append(...(studyBar ? [studyBar] : []), header, el("div", { class: "review" }, stage, side));

  // Rendering -------------------------------------------------------------------------
  const context = canvas.getContext("2d") as CanvasRenderingContext2D;
  const trackById = (): Map<number, Track> => new Map(state.tracks.map((t) => [t.track_id, t]));

  async function draw(): Promise<void> {
    const frameIndex = state.frame;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    let image: HTMLImageElement;
    let boxes;
    try {
      [image, boxes] = await Promise.all([frames.get(frameIndex), overlays.at(frameIndex)]);
    } catch (error) {
      showMessage(message, errorText(error), "error");
      return;
    }
    if (frameIndex !== state.frame) return; // a newer frame was requested meanwhile
    const fit = fitFrame(clip.width, clip.height, canvas.width, canvas.height);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, fit.offsetX, fit.offsetY, fit.width, fit.height);

    const lineX = countingLineX(job.counting.line_x_normalized, fit);
    context.setLineDash([6 * ratio, 6 * ratio]);
    context.strokeStyle = "rgba(255,255,255,0.8)";
    context.lineWidth = ratio;
    context.beginPath();
    context.moveTo(lineX, fit.offsetY);
    context.lineTo(lineX, fit.offsetY + fit.height);
    context.stroke();

    const byId = trackById();
    context.font = `${12 * ratio}px system-ui`;
    for (const box of boxes) {
      const track = byId.get(box.track_id);
      const reviewState = track?.review_state ?? "automatic";
      const rect = boxToCanvas(box, fit);
      const selected = box.track_id === state.selected;
      context.setLineDash(reviewState === "rejected" ? [4 * ratio, 4 * ratio] : []);
      context.strokeStyle = STATE_COLORS[reviewState];
      context.lineWidth = (selected ? 3 : 1.5) * ratio;
      context.strokeRect(rect.x, rect.y, rect.w, rect.h);
      context.fillStyle = STATE_COLORS[reviewState];
      context.fillText(`#${box.track_id}${reviewState === "automatic" ? "" : ` ${STATE_LABELS[reviewState]}`}`, rect.x, rect.y - 3 * ratio);
    }
    context.setLineDash([]);
    for (const passage of state.passages) {
      if (passage.state === "added" && passage.frame_index === frameIndex) {
        context.fillStyle = STATE_COLORS.corrected;
        context.fillText(`added passage ${directionLabel(passage.direction, upstream)}`, fit.offsetX + 6 * ratio, fit.offsetY + 18 * ratio);
      }
    }
    time.textContent = `${formatTime(frameIndex / clip.framerate)} · frame ${frameIndex}/${clip.num_frames - 1}`;
    timeline.value = String(frameIndex);
    frames.prefetch(frameIndex + 1, 24);
  }

  function renderCounts(): void {
    const reviewed = state.revision > 0 ? state.results.reviewed : null;
    countsBody.replaceChildren(
      ...countRows(state.results.automatic, reviewed).map((row) =>
        el("tr", {}, el("td", {}, row.label), el("td", { class: "num" }, row.automatic), el("td", { class: "num" }, row.reviewed)),
      ),
    );
    revisionLabel.textContent = state.revision === 0 ? "automatic result (revision 0)" : `reviewed revision ${state.revision}`;
  }

  function stateBadge(track: Track): HTMLElement {
    return el("span", { class: "state", style: `background:${STATE_COLORS[track.review_state]}` }, STATE_LABELS[track.review_state]);
  }

  function triageBadge(track: Track): HTMLElement | string {
    if (!track.triage) return "—";
    const score = track.review_score === null ? "" : ` ${track.review_score.toFixed(2)}`;
    return el(
      "span",
      {
        class: "state",
        style: `background:${TRIAGE_COLORS[track.triage]}`,
        title: reasonText(track.review_reasons) || "no weak evidence",
      },
      `${TRIAGE_LABELS[track.triage]}${score}`,
    );
  }

  function renderTracks(): void {
    const rows = reviewOrder(state.tracks).map((track) => {
      const row = el(
        "tr",
        track.track_id === state.selected ? { class: "selected" } : {},
        el("td", {}, String(track.track_id)),
        el("td", {}, `${formatTime(track.start_time_s)}–${formatTime(track.end_time_s)}`),
        el("td", {}, directionLabel(track.direction, upstream)),
        el("td", {}, triageBadge(track)),
        el("td", {}, stateBadge(track)),
        el("td", {}, directionLabel(track.final_direction, upstream)),
      );
      row.addEventListener("click", () => select(track.track_id));
      return row;
    });
    const passageRows = state.passages.map((passage) => {
      const reject = el("button", { type: "button" }, "Remove");
      reject.disabled = passage.state !== "added";
      reject.addEventListener("click", () => void act({ action: "reject", passage_id: passage.passage_id, reason: "removed added passage" }));
      return el(
        "tr",
        {},
        el("td", {}, passage.passage_id),
        el("td", {}, formatTime(passage.time_s)),
        el("td", {}, "missed by the model"),
        el("td", {}, ""),
        el("td", {}, passage.state),
        el("td", {}, passage.state === "added" ? directionLabel(passage.direction, upstream) : "—", " ", reject),
      );
    });
    trackBody.replaceChildren(...rows, ...passageRows);
  }

  function renderSelected(): void {
    const track = state.tracks.find((t) => t.track_id === state.selected);
    const addRight = el("button", { type: "button" }, `Add missed fish here ${directionLabel("right", upstream)}`);
    const addLeft = el("button", { type: "button" }, `Add missed fish here ${directionLabel("left", upstream)}`);
    addRight.addEventListener("click", () => void addPassage("right"));
    addLeft.addEventListener("click", () => void addPassage("left"));
    const addMissing = el("div", { class: "actions" }, addRight, addLeft);
    if (!track) {
      selectedPanel.replaceChildren(el("p", { class: "muted" }, "Select a track in the list, or press n for the next unreviewed one."), addMissing);
      return;
    }
    const button = (label: string, request: Omit<ReviewRequest, "base_revision">): HTMLButtonElement => {
      const b = el("button", { type: "button" }, label);
      b.addEventListener("click", () => void act(request));
      return b;
    };
    selectedPanel.replaceChildren(
      el(
        "p",
        {},
        `Track #${track.track_id}: ${formatTime(track.start_time_s)}–${formatTime(track.end_time_s)}, ${track.observations} boxes. `,
        `Automatic: ${directionLabel(track.direction, upstream)}. Review: `,
        stateBadge(track),
      ),
      ...(track.triage
        ? [
            el(
              "p",
              { class: "muted" },
              `Triage: ${TRIAGE_LABELS[track.triage]}, review score ${track.review_score?.toFixed(2) ?? "—"} `,
              "(a heuristic ranking, not a probability)",
              track.review_reasons && track.review_reasons.length ? `. Why: ${reasonText(track.review_reasons)}.` : ".",
            ),
          ]
        : []),
      el(
        "div",
        { class: "actions" },
        button("Accept (a)", { action: "accept", track_id: track.track_id }),
        button("Reject (x)", { action: "reject", track_id: track.track_id }),
        button(`Set ${directionLabel("right", upstream)} (r)`, { action: "set_direction", track_id: track.track_id, direction: "right" }),
        button(`Set ${directionLabel("left", upstream)} (l)`, { action: "set_direction", track_id: track.track_id, direction: "left" }),
        button("Unresolved (u)", { action: "mark_unresolved", track_id: track.track_id }),
      ),
      addMissing,
    );
  }

  function renderAudit(): void {
    const windows = state.audit.windows;
    if (state.audit.calibration_version === null) {
      auditPanel.replaceChildren(el("p", { class: "muted" }, "This release has no audit windows (no calibration version)."));
      return;
    }
    if (windows.length === 0) {
      auditPanel.replaceChildren(el("p", { class: "muted" }, "No unflagged footage to audit in this recording."));
      return;
    }
    const checked = windows.filter((w) => w.state === "checked").length;
    const rows = windows.map((w) => {
      const go = el("button", { type: "button" }, "Go");
      go.addEventListener("click", () => seek(w.start_frame));
      const mark = el("button", { type: "button" }, w.state === "checked" ? "Checked" : "Mark checked");
      mark.disabled = w.state === "checked";
      mark.addEventListener("click", () => void act({ action: "mark_audited", audit_window: w.index }));
      return el(
        "tr",
        {},
        el("td", {}, String(w.index + 1)),
        el("td", {}, `${formatTime(w.start_time_s)}–${formatTime(w.stop_time_s)}`),
        el("td", {}, w.passages_added ? `${w.passages_added} added` : ""),
        el("td", {}, go, " ", mark),
      );
    });
    auditPanel.replaceChildren(
      el(
        "p",
        { class: "muted" },
        "Random stretches of footage the system did not flag. Watch each one for fish it missed entirely; ",
        "add any you find, then mark the window checked. ",
        `${checked} of ${windows.length} checked.`,
      ),
      el("table", {}, el("tbody", {}, ...rows)),
    );
  }

  function renderAll(): void {
    renderCounts();
    renderTracks();
    renderSelected();
    renderAudit();
    void draw();
  }

  /** Adds a missed fish at the current frame, tied to the audit window it is in (if any). */
  async function addPassage(direction: ImageDirection): Promise<void> {
    const audited = auditWindowAt(state.audit.windows, state.frame);
    await act({
      action: "add_passage",
      direction,
      frame_index: state.frame,
      ...(audited ? { audit_window: audited.index } : {}),
    });
  }

  // Interaction -----------------------------------------------------------------------
  function seek(frame: number): void {
    state.frame = Math.max(0, Math.min(clip.num_frames - 1, frame));
    void draw();
  }

  function select(trackId: number | null): void {
    state.selected = trackId;
    const track = state.tracks.find((t) => t.track_id === trackId);
    if (track) seek(track.start_frame);
    renderTracks();
    renderSelected();
  }

  function selectRelative(step: number): void {
    const ordered = reviewOrder(state.tracks);
    if (ordered.length === 0) return;
    const index = ordered.findIndex((t) => t.track_id === state.selected);
    const next = ordered[(index + step + ordered.length) % ordered.length];
    if (next) select(next.track_id);
  }

  async function reload(): Promise<void> {
    const [results, page, passages, audit] = await Promise.all([
      api.getResults(job.job_id),
      api.getAllTracks(job.job_id),
      api.getAddedPassages(job.job_id),
      api.getAudit(job.job_id),
    ]);
    state.results = results;
    state.tracks = page.tracks;
    state.revision = page.revision;
    state.passages = passages;
    state.audit = audit;
    renderAll();
  }

  async function act(request: Omit<ReviewRequest, "base_revision">): Promise<void> {
    try {
      const response = await api.submitReview(job.job_id, { ...request, base_revision: state.revision });
      await reload();
      showMessage(message, `Saved as revision ${response.revision}.`, "ok");
    } catch (error) {
      if (error instanceof ApiError && error.isConflict) {
        await reload();
        showMessage(message, "Someone else saved a review first. The latest revision is now shown; please check it and repeat your change if still needed.", "error");
      } else {
        showMessage(message, errorText(error), "error");
      }
    }
  }

  let timer: number | undefined;
  function tick(): void {
    if (!state.playing) return;
    if (state.frame >= clip.num_frames - 1) {
      togglePlay();
      return;
    }
    seek(state.frame + 1);
    timer = window.setTimeout(tick, 1000 / (clip.framerate * state.speed));
  }
  function togglePlay(): void {
    state.playing = !state.playing;
    play.textContent = state.playing ? "⏸" : "▶";
    window.clearTimeout(timer);
    if (state.playing) tick();
  }

  play.addEventListener("click", togglePlay);
  back.addEventListener("click", () => seek(state.frame - 1));
  forward.addEventListener("click", () => seek(state.frame + 1));
  speed.addEventListener("change", () => {
    state.speed = Number(speed.value);
  });
  timeline.addEventListener("input", () => seek(Number(timeline.value)));

  const onKey = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    const track = state.tracks.find((t) => t.track_id === state.selected);
    const id = track?.track_id;
    const keys: Record<string, () => void> = {
      " ": togglePlay,
      ArrowLeft: () => seek(state.frame - 1),
      ArrowRight: () => seek(state.frame + 1),
      j: () => selectRelative(-1),
      k: () => selectRelative(1),
      n: () => select(nextUnreviewed(state.tracks, state.selected)?.track_id ?? null),
      a: () => id !== undefined && void act({ action: "accept", track_id: id }),
      x: () => id !== undefined && void act({ action: "reject", track_id: id }),
      r: () => id !== undefined && void act({ action: "set_direction", track_id: id, direction: "right" }),
      l: () => id !== undefined && void act({ action: "set_direction", track_id: id, direction: "left" }),
      u: () => id !== undefined && void act({ action: "mark_unresolved", track_id: id }),
    };
    const handler = keys[event.key];
    if (handler) {
      event.preventDefault();
      handler();
    }
  };
  document.addEventListener("keydown", onKey);
  const onResize = (): void => void draw();
  window.addEventListener("resize", onResize);
  cleanups.push(() => {
    state.playing = false;
    window.clearTimeout(timer);
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", onResize);
  });

  renderAll();
}
