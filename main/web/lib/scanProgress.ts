/**
 * Simulated scan-progress model.
 *
 * The detector API is a single request/response call — it doesn't stream
 * per-stage progress. To still show users *what* is happening (and give a
 * fast response enough screen time to be understood), the UI simulates the
 * backend's pipeline on the client:
 *
 *   1. Progress ramps from 0 to `SIMULATED_PROGRESS_CAP` (90%) over
 *      `RAMP_MS`, walking through the analysis stages in the same order the
 *      backend runs them (see app/main.py `_run_image_analysis` /
 *      `_run_pdf_analysis`).
 *   2. It then *holds at 90%* until the real response arrives — the last
 *      10% is never simulated, so the bar can't claim completion before
 *      the backend has actually finished.
 *   3. If the backend answers in under `FAST_RESPONSE_THRESHOLD_MS`, the
 *      UI pads the wait until `MIN_PIPELINE_VISIBLE_MS` has elapsed so the
 *      stage walkthrough is actually visible instead of flashing by.
 *   4. On completion progress jumps to 100%, every stage is marked done,
 *      and results are revealed after `COMPLETION_HOLD_MS`.
 *
 * Everything in this file is pure (no React, no DOM) so it can be unit
 * tested and reasoned about independently of the rendering.
 */

/** Responses faster than this get padded so the pipeline is legible. */
export const FAST_RESPONSE_THRESHOLD_MS = 1000;

/** Minimum time the progress view stays up when a response was "fast". */
export const MIN_PIPELINE_VISIBLE_MS = 2700;

/** Time for the simulated ramp to reach the 90% cap. Slightly shorter than
 *  MIN_PIPELINE_VISIBLE_MS so a fast scan visibly *holds* at 90% for a beat
 *  before completing — which is the honest picture: "we're waiting on the
 *  analyzer for the last bit". */
export const RAMP_MS = 2400;

/** Where the simulation stops and waits for the real response. */
export const SIMULATED_PROGRESS_CAP = 90;

/** How long the bar sits at 100% (all stages done) before results show. */
export const COMPLETION_HOLD_MS = 450;

export type ScanInputKind = "image" | "pdf" | "url";

export type StageStatus = "pending" | "running" | "done";

export interface ScanStage {
  /** Short mono tag, e.g. "FREQ" — matches SIGNAL_META tags where relevant. */
  tag: string;
  /** Human label shown in the stage list. */
  label: string;
  /** Relative share of the simulated ramp this stage occupies. Kept
   *  equal across stages so each per-stage visual gets even screen time. */
  weight: number;
}

export interface StageView extends ScanStage {
  status: StageStatus;
}

const READ_STAGE: ScanStage = { tag: "READ", label: "Read file & detect format", weight: 1 };
const FETCH_STAGE: ScanStage = { tag: "FETCH", label: "Fetch URL", weight: 1 };
const DECODE_STAGE: ScanStage = { tag: "DECODE", label: "Decode image", weight: 1 };
const RASTER_STAGE: ScanStage = { tag: "RASTER", label: "Render PDF pages & extract images", weight: 1 };

/** The four visual analyzers + metadata, in backend execution order. */
const SIGNAL_STAGES: ScanStage[] = [
  { tag: "FREQ", label: "Frequency spectrum", weight: 1 },
  { tag: "NOISE", label: "Sensor noise floor", weight: 1 },
  { tag: "GRAD", label: "Texture smoothness", weight: 1 },
  { tag: "LIGHT", label: "Lighting consistency", weight: 1 },
  { tag: "META", label: "Metadata signatures", weight: 1 },
];

const PDF_INTEGRITY_STAGE: ScanStage = {
  tag: "DOC",
  label: "Document integrity checks",
  weight: 1,
};

const COMBINE_STAGE: ScanStage = { tag: "SCORE", label: "Combine into verdict", weight: 1 };

/** Ordered pipeline stages for a given input type. */
export function stagesFor(kind: ScanInputKind): ScanStage[] {
  switch (kind) {
    case "pdf":
      return [READ_STAGE, RASTER_STAGE, ...SIGNAL_STAGES, PDF_INTEGRITY_STAGE, COMBINE_STAGE];
    case "url":
      // We can't know the type before the backend fetches it; show the
      // common image path, which is what the backend runs for most URLs.
      return [FETCH_STAGE, READ_STAGE, DECODE_STAGE, ...SIGNAL_STAGES, COMBINE_STAGE];
    case "image":
    default:
      return [READ_STAGE, DECODE_STAGE, ...SIGNAL_STAGES, COMBINE_STAGE];
  }
}

/** Clamp helper. */
function clamp01(t: number): number {
  return Math.min(1, Math.max(0, t));
}

/**
 * Simulated progress (0..SIMULATED_PROGRESS_CAP) for a given elapsed time.
 * Linear on purpose: with equal stage weights every stage gets the same
 * on-screen time, so each per-stage pixel visual is actually seen (an
 * ease-out curve made the first stages flash by in ~100ms). Never exceeds
 * the cap — the final 10% belongs to the real response.
 */
export function simulatedProgress(elapsedMs: number): number {
  const t = clamp01(elapsedMs / RAMP_MS);
  return Math.min(SIMULATED_PROGRESS_CAP, SIMULATED_PROGRESS_CAP * t);
}

/**
 * Maps a progress value onto per-stage statuses.
 *
 * Each stage owns a slice of [0, SIMULATED_PROGRESS_CAP] proportional to its
 * weight. A stage is `done` once progress has passed its slice, `running`
 * while progress is inside it, `pending` before. The final stage stays
 * `running` while the bar holds at the cap and only flips to `done` when
 * `complete` is true — i.e. when the real response has landed.
 */
export function stageViews(stages: ScanStage[], progress: number, complete: boolean): StageView[] {
  if (complete) return stages.map((s) => ({ ...s, status: "done" }));

  const totalWeight = stages.reduce((sum, s) => sum + s.weight, 0);
  let cursor = 0;
  let runningAssigned = false;

  return stages.map((stage, index) => {
    const start = cursor;
    const end = cursor + (stage.weight / totalWeight) * SIMULATED_PROGRESS_CAP;
    cursor = end;

    const isLast = index === stages.length - 1;
    let status: StageStatus;
    if (!isLast && progress >= end) {
      status = "done";
    } else if (!runningAssigned && progress >= start) {
      status = "running";
      runningAssigned = true;
    } else {
      status = "pending";
    }
    return { ...stage, status };
  });
}

/**
 * How much longer to keep the progress view up once the backend has
 * responded. Fast responses (< FAST_RESPONSE_THRESHOLD_MS) are padded up to
 * MIN_PIPELINE_VISIBLE_MS; anything slower has already given the user time
 * to watch the pipeline, so it completes immediately.
 */
export function extraDelayMs(elapsedMs: number): number {
  if (elapsedMs >= FAST_RESPONSE_THRESHOLD_MS) return 0;
  return Math.max(0, MIN_PIPELINE_VISIBLE_MS - elapsedMs);
}

/** Infers the pipeline variant from the submitted form data. */
export function inputKindFor(formData: FormData): ScanInputKind {
  const media = formData.get("media");
  if (media instanceof File) {
    const isPdf = media.type === "application/pdf" || media.name.toLowerCase().endsWith(".pdf");
    return isPdf ? "pdf" : "image";
  }
  return "url";
}
