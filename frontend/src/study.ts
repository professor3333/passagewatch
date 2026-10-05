// Usability study (docs/usability_study.md): the participant's sequence, active time and the
// questionnaires. Only used when the service runs in study mode.

import type { ImageDirection } from "./types";

export type Condition = "manual" | "assisted";

/** The plan as the service gives it to the pages: no reference or automatic counts. */
export interface StudyPlan {
  release: string;
  clips: Record<string, { clip_id: string; role: "practice" | "S1" | "S2"; num_frames: number; framerate: number }>;
  practice: string[];
  sets: Record<string, string[]>;
  participants: Record<string, { condition: Condition; set: string }[]>;
  jobs: Record<string, Record<string, string>>;
}

export interface StudyProgress {
  participant: string;
  clips_done: string[];
  questionnaires_done: Condition[];
}

export interface TrialIn {
  participant: string;
  block: number;
  condition: Condition;
  clip: string;
  started_at: string;
  finished_at: string;
  active_ms: number;
  pauses: number;
  right: number | null;
  left: number | null;
}

export interface TrialOut {
  clip: string;
  final: [number, number];
  revision: number | null;
}

export interface QuestionnaireIn {
  participant: string;
  condition: Condition;
  sus: number[];
  tlx: number[];
}

export type Step =
  | {
      kind: "trial";
      block: number;
      condition: Condition;
      clip: string;
      practice: boolean;
      position: number; // 1-based within the block, practice first
      total: number;
      jobId: string | null; // the participant's own job, for an assisted trial
    }
  | { kind: "questionnaire"; block: number; condition: Condition }
  | { kind: "done" };

/** Each block's condition and clips, the practice clip first (as the service checks them). */
export function blocks(plan: StudyPlan, participant: string): { condition: Condition; clips: string[] }[] {
  const assigned = plan.participants[participant];
  if (!assigned) throw new Error(`unknown participant ${participant}`);
  return assigned.map((block, index) => {
    const practice = plan.practice[index];
    const set = plan.sets[block.set];
    if (practice === undefined || set === undefined) throw new Error("the study plan is incomplete");
    return { condition: block.condition, clips: [practice, ...set] };
  });
}

/** What the participant does next: the next unfinished clip, then that block's questionnaires. */
export function nextStep(plan: StudyPlan, participant: string, progress: StudyProgress): Step {
  const done = new Set(progress.clips_done);
  const forms = new Set(progress.questionnaires_done);
  const all = blocks(plan, participant);
  for (const [index, block] of all.entries()) {
    const position = block.clips.findIndex((clip) => !done.has(clip));
    if (position >= 0) {
      const clip = block.clips[position] as string;
      return {
        kind: "trial",
        block: index + 1,
        condition: block.condition,
        clip,
        practice: position === 0,
        position: position + 1,
        total: block.clips.length,
        jobId: block.condition === "assisted" ? (plan.jobs[participant]?.[clip] ?? null) : null,
      };
    }
    if (!forms.has(block.condition)) return { kind: "questionnaire", block: index + 1, condition: block.condition };
  }
  return { kind: "done" };
}

/** Active time on one clip: runs from when the clip's page opens until Done, except paused. */
export class ActiveClock {
  private activeMs = 0;
  private since: number | null;
  pauses = 0;
  readonly startedAt: Date;

  constructor(private readonly now: () => number = () => performance.now()) {
    this.since = now();
    this.startedAt = new Date();
  }

  get paused(): boolean {
    return this.since === null;
  }

  pause(): void {
    if (this.since === null) return;
    this.activeMs += this.now() - this.since;
    this.since = null;
    this.pauses += 1;
  }

  resume(): void {
    if (this.since === null) this.since = this.now();
  }

  elapsedMs(): number {
    return Math.round(this.activeMs + (this.since === null ? 0 : this.now() - this.since));
  }
}

/** Manual tally: never negative. */
export function adjust(counts: Record<ImageDirection, number>, direction: ImageDirection, step: 1 | -1): void {
  counts[direction] = Math.max(0, counts[direction] + step);
}

// System Usability Scale (Brooke, 1996), answered 1 (strongly disagree) to 5 (strongly agree).
export const SUS_ITEMS = [
  "I think that I would like to use this system frequently.",
  "I found the system unnecessarily complex.",
  "I thought the system was easy to use.",
  "I think that I would need the support of a technical person to be able to use this system.",
  "I found the various functions in this system were well integrated.",
  "I thought there was too much inconsistency in this system.",
  "I would imagine that most people would learn to use this system very quickly.",
  "I found the system very cumbersome to use.",
  "I felt very confident using the system.",
  "I needed to learn a lot of things before I could get going with this system.",
];

/** SUS score, 0-100: odd items score (answer - 1), even items (5 - answer); the sum × 2.5. */
export function susScore(answers: number[]): number {
  if (answers.length !== 10) throw new Error("SUS has 10 items");
  return answers.reduce((sum, a, i) => sum + (i % 2 === 0 ? a - 1 : 5 - a), 0) * 2.5;
}

// NASA Task Load Index, raw (unweighted) ratings, 0-100 in steps of 5.
export const TLX_SCALES: { name: string; question: string; low: string; high: string }[] = [
  { name: "Mental demand", question: "How mentally demanding was the task?", low: "Very low", high: "Very high" },
  { name: "Physical demand", question: "How physically demanding was the task?", low: "Very low", high: "Very high" },
  { name: "Temporal demand", question: "How hurried or rushed was the pace of the task?", low: "Very low", high: "Very high" },
  {
    name: "Performance",
    question: "How successful were you in accomplishing what you were asked to do?",
    low: "Perfect",
    high: "Failure",
  },
  { name: "Effort", question: "How hard did you have to work to accomplish your level of performance?", low: "Very low", high: "Very high" },
  {
    name: "Frustration",
    question: "How insecure, discouraged, irritated, stressed and annoyed were you?",
    low: "Very low",
    high: "Very high",
  },
];

export const INSTRUCTIONS: Record<Condition, string[]> = {
  manual: [
    "Watch the recording and count the fish that pass the dashed vertical line.",
    "A fish counts as a passage when it crosses the line and ends on the other side. A fish that crosses and comes back does not count.",
    "Keep two tallies: fish moving to the right (→) and fish moving to the left (←). Use +1 and −1 to correct a tally.",
    "Use play, pause, the frame steps, the speed and the slider as you like. Take the time you need to get both counts right.",
    "Press Pause if you are interrupted, and Done when you are satisfied with both counts.",
  ],
  assisted: [
    "The system has already counted the fish. Its tracks are drawn on the recording and listed on the right, with the ones it is least sure about first.",
    "Check its work: accept tracks that are right, reject ones that are not fish or not passages, fix wrong directions, and add any passing fish it missed.",
    "A passage is a fish that crosses the dashed line and ends on the other side; one that crosses and comes back does not count.",
    "The random audit windows are stretches the system did not flag: watch them for fish it missed entirely.",
    "Take the time you need to make both counts right. Press Pause if you are interrupted, and Done when you are satisfied: the reviewed counts are recorded.",
  ],
};
