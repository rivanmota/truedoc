export type Verdict = "real" | "inconclusive" | "ai";

export function verdictFor(score: number): Verdict {
  if (score < 0.35) return "real";
  if (score > 0.65) return "ai";
  return "inconclusive";
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  real: "Likely real",
  inconclusive: "Inconclusive",
  ai: "Likely AI-generated",
};

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16),
  ];
}

const REAL_RGB = hexToRgb("#4fd1a5");
const MID_RGB = hexToRgb("#e8c547");
const AI_RGB = hexToRgb("#f0954b");

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

/** Interpolates real -> mid -> ai across a 0-1 score for a bar/ring fill color. */
export function scoreColor(score: number): string {
  const clamped = Math.min(1, Math.max(0, score));
  const [from, to, t] =
    clamped < 0.5
      ? [REAL_RGB, MID_RGB, clamped / 0.5]
      : [MID_RGB, AI_RGB, (clamped - 0.5) / 0.5];
  const r = lerp(from[0], to[0], t);
  const g = lerp(from[1], to[1], t);
  const b = lerp(from[2], to[2], t);
  return `rgb(${r} ${g} ${b})`;
}
