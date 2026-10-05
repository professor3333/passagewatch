// Usability study pages (docs/usability_study.md), active only when the service runs in
// study mode. Routes: #/study (participant choice) and #/study/<code> (the next step).

import { ApiError, type PassageWatchApi } from "./api";
import { el, errorText, showMessage } from "./dom";
import { countingLineX, fitFrame } from "./geometry";
import { FrameCache } from "./player";
import { formatTime } from "./review";
import {
  ActiveClock,
  INSTRUCTIONS,
  SUS_ITEMS,
  TLX_SCALES,
  adjust,
  blocks,
  nextStep,
  type Condition,
  type Step,
  type StudyPlan,
} from "./study";
import type { Clip, ImageDirection, Job } from "./types";

export interface StudyHooks {
  app: HTMLElement;
  api: PassageWatchApi;
  /** Renders the shipped review view for `job` with `bar` above it. */
  review(job: Job, bar: HTMLElement): Promise<void>;
  /** Registers cleanup for when the page is left. */
  onLeave(cleanup: () => void): void;
  /** Shows the participant's next step. */
  refresh(): void;
}

const CONDITION_NAMES: Record<Condition, string> = { manual: "manual counting", assisted: "assisted review" };

export async function studyView(hooks: StudyHooks, participant: string | null): Promise<void> {
  const { app, api } = hooks;
  let plan: StudyPlan;
  try {
    plan = await api.getStudyPlan();
  } catch (error) {
    const text = error instanceof ApiError && error.status === 404 ? "Study mode is not enabled on this service." : errorText(error);
    app.append(el("section", { class: "panel" }, el("h2", {}, "Usability study"), el("p", { class: "message error" }, text)));
    return;
  }
  if (participant === null || !(participant in plan.participants)) {
    participantChoice(app, plan);
    return;
  }
  const step = nextStep(plan, participant, await api.getStudyProgress(participant));
  if (step.kind === "done") {
    app.append(
      el(
        "section",
        { class: "panel" },
        el("h2", {}, `Participant ${participant}: finished`),
        el("p", {}, "Both blocks and questionnaires are recorded. Thank you!"),
        el("p", { class: "muted" }, "The facilitator will now ask which condition you preferred, and why."),
      ),
    );
    return;
  }
  if (step.kind === "questionnaire") {
    questionnaire(hooks, participant, step.block, step.condition);
    return;
  }
  stepIntro(hooks, plan, participant, step);
}

function participantChoice(app: HTMLElement, plan: StudyPlan): void {
  const buttons = Object.keys(plan.participants).map((code) => el("a", { href: `#/study/${code}`, class: "badge" }, code));
  app.append(
    el(
      "section",
      { class: "panel" },
      el("h2", {}, "Usability study"),
      el("p", {}, "The facilitator chooses the participant's code. Codes are pseudonymous; no names are recorded."),
      el("div", { class: "actions" }, ...buttons),
    ),
  );
}

function stepIntro(hooks: StudyHooks, plan: StudyPlan, participant: string, step: Extract<Step, { kind: "trial" }>): void {
  const start = el("button", { type: "button", class: "primary" }, step.practice ? "Start the practice clip" : "Start the clip");
  const message = el("p", { class: "message" });
  const blockStart = step.position === 1;
  const order = blocks(plan, participant)
    .map((b, i) => `block ${i + 1}: ${CONDITION_NAMES[b.condition]}`)
    .join(" · ");
  hooks.app.append(
    el(
      "section",
      { class: "panel study-intro" },
      el("h2", {}, `Participant ${participant} · block ${step.block} of 2: ${CONDITION_NAMES[step.condition]}`),
      el("p", { class: "muted" }, order),
      el(
        "p",
        {},
        step.practice
          ? "Practice clip (not scored): use it to get familiar with this condition. "
          : `Clip ${step.position - 1} of ${step.total - 1}. `,
        "The clock starts when the clip opens.",
      ),
      blockStart ? el("ul", {}, ...INSTRUCTIONS[step.condition].map((line) => el("li", {}, line))) : null,
      el("div", { class: "actions" }, start),
      message,
    ),
  );
  start.addEventListener("click", () => {
    start.disabled = true;
    hooks.app.replaceChildren();
    const run = step.condition === "manual" ? manualTrial(hooks, plan, participant, step) : assistedTrial(hooks, participant, step);
    run.catch((error: unknown) => {
      hooks.app.append(el("p", { class: "message error" }, errorText(error)));
    });
  });
}

/** The bar above a trial: what is running, Pause (with a covering overlay) and Done. */
function studyBar(
  hooks: StudyHooks,
  participant: string,
  step: Extract<Step, { kind: "trial" }>,
  clock: ActiveClock,
  done: (button: HTMLButtonElement, message: HTMLElement) => void,
  onPause: () => void = () => undefined,
): HTMLElement {
  const pause = el("button", { type: "button" }, "Pause");
  const finish = el("button", { type: "button", class: "primary" }, "Done");
  const message = el("span", { class: "message" });
  const resume = el("button", { type: "button", class: "primary" }, "Resume");
  const overlay = el(
    "div",
    { class: "study-overlay", hidden: "" },
    el("div", { class: "panel" }, el("h2", {}, "Paused"), el("p", {}, "The clock is stopped."), resume),
  );
  // While paused, keys must not reach the page underneath.
  const blockKeys = (event: KeyboardEvent): void => {
    if (clock.paused) event.stopImmediatePropagation();
  };
  window.addEventListener("keydown", blockKeys, true);
  hooks.onLeave(() => {
    window.removeEventListener("keydown", blockKeys, true);
    overlay.remove();
  });
  pause.addEventListener("click", () => {
    clock.pause();
    onPause();
    overlay.hidden = false;
  });
  resume.addEventListener("click", () => {
    clock.resume();
    overlay.hidden = true;
  });
  finish.addEventListener("click", () => done(finish, message));
  document.body.append(overlay);
  return el(
    "div",
    { class: "panel study-bar" },
    el(
      "strong",
      {},
      `${participant} · block ${step.block}: ${CONDITION_NAMES[step.condition]} · `,
      step.practice ? "practice clip" : `clip ${step.position - 1} of ${step.total - 1}`,
    ),
    el("span", { class: "spacer" }),
    message,
    pause,
    finish,
  );
}

async function submit(
  hooks: StudyHooks,
  participant: string,
  step: Extract<Step, { kind: "trial" }>,
  clock: ActiveClock,
  counts: { right: number; left: number } | null,
  button: HTMLButtonElement,
  message: HTMLElement,
): Promise<void> {
  button.disabled = true;
  const activeMs = clock.elapsedMs();
  try {
    await hooks.api.submitTrial({
      participant,
      block: step.block,
      condition: step.condition,
      clip: step.clip,
      started_at: clock.startedAt.toISOString(),
      finished_at: new Date().toISOString(),
      active_ms: activeMs,
      pauses: clock.pauses,
      right: counts?.right ?? null,
      left: counts?.left ?? null,
    });
    hooks.refresh();
  } catch (error) {
    showMessage(message, errorText(error), "error");
    button.disabled = false;
  }
}

async function assistedTrial(hooks: StudyHooks, participant: string, step: Extract<Step, { kind: "trial" }>): Promise<void> {
  if (step.jobId === null) throw new Error(`no job for ${participant} on clip ${step.clip}; check study/plan.json`);
  const job = await hooks.api.getJob(step.jobId);
  const clock = new ActiveClock();
  const bar = studyBar(hooks, participant, step, clock, (button, message) => {
    if (!window.confirm("Finish this clip? Your reviewed counts will be recorded.")) return;
    void submit(hooks, participant, step, clock, null, button, message);
  });
  await hooks.review(job, bar);
}

async function manualTrial(
  hooks: StudyHooks,
  plan: StudyPlan,
  participant: string,
  step: Extract<Step, { kind: "trial" }>,
): Promise<void> {
  const { app, api } = hooks;
  const entry = plan.clips[step.clip];
  if (!entry) throw new Error(`clip ${step.clip} is not in the plan`);
  const clip: Clip = await api.getClip(entry.clip_id);
  const frames = new FrameCache(api, clip.clip_id, clip.num_frames);
  const counts: Record<ImageDirection, number> = { right: 0, left: 0 };
  const state = { frame: 0, playing: false, speed: 1 };
  const clock = new ActiveClock();

  const canvas = el("canvas", { "aria-label": "Recording" });
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
  const values: Record<ImageDirection, HTMLElement> = { right: el("output", { class: "tally" }, "0"), left: el("output", { class: "tally" }, "0") };
  const counter = (direction: ImageDirection, label: string): HTMLElement => {
    const minus = el("button", { type: "button", "aria-label": `${label} minus one` }, "−1");
    const plus = el("button", { type: "button", class: "primary", "aria-label": `${label} plus one` }, "+1");
    const change = (by: 1 | -1) => () => {
      adjust(counts, direction, by);
      values[direction].textContent = String(counts[direction]);
    };
    minus.addEventListener("click", change(-1));
    plus.addEventListener("click", change(1));
    return el("div", { class: "counter" }, el("span", {}, label), values[direction], minus, plus);
  };

  const bar = studyBar(
    hooks,
    participant,
    step,
    clock,
    (button, message) => {
      if (!window.confirm(`Record → right ${counts.right} and ← left ${counts.left} for this clip?`)) return;
      stopPlaying();
      void submit(hooks, participant, step, clock, { ...counts }, button, message);
    },
    () => stopPlaying(),
  );
  app.append(
    bar,
    el(
      "div",
      { class: "review" },
      el(
        "section",
        { class: "panel stage" },
        canvas,
        el("div", { class: "controls" }, back, play, forward, speed, timeline, time),
        el("p", { class: "muted" }, "Keys: ", el("kbd", {}, "space"), " play · ", el("kbd", {}, "←"), el("kbd", {}, "→"), " frame"),
      ),
      el(
        "section",
        { class: "panel" },
        el("h2", {}, "Your counts"),
        counter("right", "→ right"),
        counter("left", "← left"),
        el("p", { class: "muted" }, "Count fish that cross the dashed line and end on the other side."),
      ),
    ),
  );

  const context = canvas.getContext("2d") as CanvasRenderingContext2D;
  async function draw(): Promise<void> {
    const index = state.frame;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    const image = await frames.get(index);
    if (index !== state.frame) return;
    const fit = fitFrame(clip.width, clip.height, canvas.width, canvas.height);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, fit.offsetX, fit.offsetY, fit.width, fit.height);
    const lineX = countingLineX(0.5, fit);
    context.setLineDash([6 * ratio, 6 * ratio]);
    context.strokeStyle = "rgba(255,255,255,0.8)";
    context.lineWidth = ratio;
    context.beginPath();
    context.moveTo(lineX, fit.offsetY);
    context.lineTo(lineX, fit.offsetY + fit.height);
    context.stroke();
    context.setLineDash([]);
    time.textContent = `${formatTime(index / clip.framerate)} · frame ${index}/${clip.num_frames - 1}`;
    timeline.value = String(index);
    frames.prefetch(index + 1, 24);
  }
  function seek(frame: number): void {
    state.frame = Math.max(0, Math.min(clip.num_frames - 1, frame));
    void draw();
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
  function stopPlaying(): void {
    if (state.playing) togglePlay();
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
    const keys: Record<string, () => void> = {
      " ": togglePlay,
      ArrowLeft: () => seek(state.frame - 1),
      ArrowRight: () => seek(state.frame + 1),
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
  hooks.onLeave(() => {
    state.playing = false;
    window.clearTimeout(timer);
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", onResize);
  });
  await draw();
}

function questionnaire(hooks: StudyHooks, participant: string, block: number, condition: Condition): void {
  const sus: (number | null)[] = SUS_ITEMS.map(() => null);
  const tlx: (number | null)[] = TLX_SCALES.map(() => null);
  const message = el("p", { class: "message" });
  const send = el("button", { type: "submit", class: "primary" }, "Submit");

  const susRows = SUS_ITEMS.map((item, i) =>
    el(
      "fieldset",
      { class: "likert" },
      el("legend", {}, `${i + 1}. ${item}`),
      el("span", { class: "muted" }, "Strongly disagree"),
      ...[1, 2, 3, 4, 5].map((value) => {
        const radio = el("input", { type: "radio", name: `sus-${i}`, value: String(value), "aria-label": String(value) });
        radio.addEventListener("change", () => {
          sus[i] = value;
        });
        return el("label", { class: "inline" }, radio, String(value));
      }),
      el("span", { class: "muted" }, "Strongly agree"),
    ),
  );
  const tlxRows = TLX_SCALES.map((scale, i) => {
    const slider = el("input", { type: "range", min: "0", max: "100", step: "5", value: "50", "aria-label": scale.name });
    const shown = el("output", {}, "not rated");
    slider.addEventListener("input", () => {
      tlx[i] = Number(slider.value);
      shown.textContent = slider.value;
    });
    return el(
      "fieldset",
      { class: "tlx" },
      el("legend", {}, `${scale.name}: ${scale.question}`),
      el("span", { class: "muted" }, scale.low),
      slider,
      el("span", { class: "muted" }, scale.high),
      shown,
    );
  });

  const form = el(
    "form",
    {},
    el("h3", {}, "System Usability Scale"),
    el("p", { class: "muted" }, `"The system" means the ${CONDITION_NAMES[condition]} you just used.`),
    ...susRows,
    el("h3", {}, "Workload (NASA-TLX)"),
    el("p", { class: "muted" }, "Move each slider to your rating. Every scale must be rated."),
    ...tlxRows,
    el("div", { class: "actions" }, send),
    message,
  );
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const missingSus = sus.findIndex((v) => v === null);
    const missingTlx = tlx.findIndex((v) => v === null);
    if (missingSus >= 0) {
      showMessage(message, `Please answer statement ${missingSus + 1}.`, "error");
      return;
    }
    if (missingTlx >= 0) {
      showMessage(message, `Please rate "${TLX_SCALES[missingTlx]?.name}".`, "error");
      return;
    }
    send.disabled = true;
    hooks.api
      .submitQuestionnaire({ participant, condition, sus: sus as number[], tlx: tlx as number[] })
      .then(() => hooks.refresh())
      .catch((error: unknown) => {
        showMessage(message, errorText(error), "error");
        send.disabled = false;
      });
  });
  hooks.app.append(
    el(
      "section",
      { class: "panel" },
      el("h2", {}, `Participant ${participant} · after block ${block}: ${CONDITION_NAMES[condition]}`),
      form,
    ),
  );
}
