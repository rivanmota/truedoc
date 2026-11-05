"use client";

import {
  scoreColor,
  VERDICT_DESCRIPTION,
  VERDICT_LABEL,
  VERDICT_THRESHOLDS,
  verdictFor,
  type Verdict,
} from "@/lib/verdict";

const REAL_MAX_PCT = Math.round(VERDICT_THRESHOLDS.REAL_MAX * 100);
const AI_MIN_PCT = Math.round(VERDICT_THRESHOLDS.AI_MIN * 100);

const BANDS: { verdict: Verdict; range: string; swatch: number }[] = [
  { verdict: "real", range: `0–${REAL_MAX_PCT - 1}%`, swatch: 0.1 },
  { verdict: "inconclusive", range: `${REAL_MAX_PCT}–${AI_MIN_PCT}%`, swatch: 0.5 },
  { verdict: "ai", range: `${AI_MIN_PCT + 1}–100%`, swatch: 0.9 },
];

/**
 * "How to read this score" — the three verdict bands of the not-true-doc
 * likelihood, with the one the current result falls in highlighted.
 */
export function ScoreLegend({ score }: { score?: number | null }) {
  const active = typeof score === "number" && Number.isFinite(score) ? verdictFor(score) : null;

  return (
    <div className="rounded-lg border border-line bg-panel px-5 py-4" data-testid="score-legend">
      <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.15em] text-dimmer">
        How to read the score
      </p>

      {/* Continuous scale strip with band boundaries */}
      <div className="relative mb-3 h-1.5 w-full overflow-hidden rounded-full" aria-hidden="true">
        <div
          className="absolute inset-0"
          style={{
            background: `linear-gradient(90deg, ${scoreColor(0)} 0%, ${scoreColor(0.5)} 50%, ${scoreColor(1)} 100%)`,
          }}
        />
        <span className="absolute inset-y-0 w-px bg-ink" style={{ left: `${REAL_MAX_PCT}%` }} />
        <span className="absolute inset-y-0 w-px bg-ink" style={{ left: `${AI_MIN_PCT}%` }} />
        {active !== null && typeof score === "number" && (
          <span
            className="absolute inset-y-0 w-0.5 bg-paper"
            style={{ left: `calc(${Math.round(Math.min(1, Math.max(0, score)) * 100)}% - 1px)` }}
          />
        )}
      </div>

      <ul className="grid gap-2 sm:grid-cols-3">
        {BANDS.map(({ verdict, range, swatch }) => {
          const isActive = active === verdict;
          const color = scoreColor(swatch);
          return (
            <li
              key={verdict}
              data-band={verdict}
              data-active={isActive || undefined}
              className={`rounded-md border px-3 py-2 transition-colors ${
                isActive ? "border-paper/60 bg-panel-raised" : "border-line/60"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
                <span className="font-mono text-[11px] tabular-nums text-dim">{range}</span>
              </div>
              <p className="mt-1 text-sm text-paper">{VERDICT_LABEL[verdict]}</p>
              <p className="mt-0.5 text-[11px] leading-snug text-dim">{VERDICT_DESCRIPTION[verdict]}</p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
