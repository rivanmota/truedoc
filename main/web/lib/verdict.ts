export type Verdict = "real" | "inconclusive" | "ai";

/**
 * Verdict bands for the not-true-doc likelihood (0..1). Exported so the
 * gauge, the legend, and any future copy all read from one place.
 *
 *   score <  REAL_MAX            -> "real"          (likely a true doc)
 *   REAL_MAX <= score <= AI_MIN  -> "inconclusive"
 *   score >  AI_MIN              -> "ai"            (likely not a true doc)
 */
export const VERDICT_THRESHOLDS = {
  REAL_MAX: 0.35,
  AI_MIN: 0.65,
} as const;

export function verdictFor(score: number): Verdict {
  if (score < VERDICT_THRESHOLDS.REAL_MAX) return "real";
  if (score > VERDICT_THRESHOLDS.AI_MIN) return "ai";
  return "inconclusive";
}

/**
 * The score is the likelihood that the file is NOT a true document — i.e.
 * that its visual content was AI-generated or AI-manipulated. Labels are
 * phrased against that framing so a low number reads as "true doc" and a
 * high number reads as "not a true doc".
 */
export const VERDICT_LABEL: Record<Verdict, string> = {
  real: "Likely a true doc",
  inconclusive: "Inconclusive",
  ai: "Likely not a true doc",
};

/** Qualitative level of the not-true-doc likelihood — shown instead of a number. */
export const LIKELIHOOD_LEVEL: Record<Verdict, string> = {
  real: "Low",
  inconclusive: "Moderate",
  ai: "High",
};

/** Short labels for the three-segment verdict scale. */
export const VERDICT_SHORT: Record<Verdict, string> = {
  real: "True doc",
  inconclusive: "Inconclusive",
  ai: "Not a true doc",
};

/** Solid colour per verdict band (CSS variable references). */
export const VERDICT_COLOR: Record<Verdict, string> = {
  real: "var(--signal-real)",
  inconclusive: "var(--signal-mid)",
  ai: "var(--signal-ai)",
};

/** Per-signal verdict: the same bands, plus "na" for signals that didn't apply. */
export type SignalVerdict = Verdict | "na";

export function signalVerdictFor(score: number | null | undefined): SignalVerdict {
  if (typeof score !== "number" || !Number.isFinite(score)) return "na";
  return verdictFor(score);
}

/** How a single signal reads, phrased as evidence for/against a true doc. */
export const SIGNAL_VERDICT_LABEL: Record<SignalVerdict, string> = {
  real: "Consistent with a true doc",
  inconclusive: "Inconclusive on its own",
  ai: "Points away from a true doc",
  na: "Not applicable to this file",
};

export const SIGNAL_VERDICT_COLOR: Record<SignalVerdict, string> = {
  ...VERDICT_COLOR,
  na: "var(--dimmer)",
};
