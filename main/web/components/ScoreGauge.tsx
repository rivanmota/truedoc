"use client";

import {
  LIKELIHOOD_LEVEL,
  VERDICT_COLOR,
  VERDICT_LABEL,
  VERDICT_SHORT,
  verdictFor,
  type Verdict,
} from "@/lib/verdict";

const RINGS = [38, 54, 70, 86];
const ARC_RADIUS = 78;
const CIRCUMFERENCE = 2 * Math.PI * ARC_RADIUS;
const VERDICT_ORDER: Verdict[] = ["real", "inconclusive", "ai"];

/** The one name for the headline result, used everywhere it's shown. */
export const SCORE_NAME = "Not-true-doc likelihood";

function VerdictIcon({ verdict }: { verdict: Verdict }) {
  const color = VERDICT_COLOR[verdict];
  if (verdict === "real") {
    return (
      <svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">
        <circle cx="18" cy="18" r="16" fill={color} opacity="0.15" />
        <path d="M11 18.5 15.5 23 25 13.5" stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (verdict === "ai") {
    return (
      <svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">
        <circle cx="18" cy="18" r="16" fill={color} opacity="0.15" />
        <path d="M18 10v9" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="18" cy="25" r="1.6" fill={color} />
      </svg>
    );
  }
  return (
    <svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">
      <circle cx="18" cy="18" r="16" fill={color} opacity="0.15" />
      <path d="M13.5 14.5a4.5 4.5 0 1 1 6.3 4.1c-1.2.6-1.8 1.4-1.8 2.6" stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round" />
      <circle cx="18" cy="25.5" r="1.6" fill={color} />
    </svg>
  );
}

/** Three-segment scale: which band the result landed in, no numbers. */
function VerdictScale({ active }: { active: Verdict | null }) {
  return (
    <div className="mt-3 w-full max-w-[16.5rem]" data-testid="verdict-scale">
      <div className="flex gap-1" aria-hidden="true">
        {VERDICT_ORDER.map((v) => (
          <span
            key={v}
            data-band={v}
            data-active={active === v || undefined}
            className="h-1.5 flex-1 rounded-full transition-opacity"
            style={{ background: VERDICT_COLOR[v], opacity: active === null ? 0.25 : active === v ? 1 : 0.18 }}
          />
        ))}
      </div>
      <div className="mt-1.5 flex justify-between gap-3 font-mono text-[9px] uppercase tracking-[0.08em]">
        {VERDICT_ORDER.map((v) => (
          <span
            key={v}
            className={`whitespace-nowrap ${
              active === v ? "text-paper" : "text-dimmer"
            }`}
          >
            {VERDICT_SHORT[v]}
          </span>
        ))}
      </div>
    </div>
  );
}

export function ScoreGauge({
  score,
  applicable = true,
  loading,
  stageTag,
}: {
  /** 0..1 not-true-doc likelihood from the backend. undefined = no scan yet. */
  score?: number | null;
  /** False when the backend could not score the file (e.g. a text-only PDF). */
  applicable?: boolean;
  loading?: boolean;
  /** Short tag of the pipeline stage currently running, shown while loading. */
  stageTag?: string;
}) {
  const numericScore: number | null =
    typeof score === "number" && Number.isFinite(score) && applicable ? score : null;
  const verdict: Verdict | null = numericScore !== null ? verdictFor(numericScore) : null;
  const color = verdict ? VERDICT_COLOR[verdict] : "var(--dim)";
  const scanned = !loading && score !== undefined;
  const notApplicable = scanned && verdict === null;

  return (
    <figure className="flex flex-col items-center" aria-label={SCORE_NAME}>
      <div className="relative mx-auto h-48 w-48 shrink-0">
        <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90" aria-hidden="true">
          {/* background spectrum rings — the signature element, echoing the
              backend's radial power-spectrum profile */}
          {RINGS.map((r) => (
            <circle key={r} cx="100" cy="100" r={r} fill="none" stroke="var(--line)" strokeWidth="1" opacity={0.6} />
          ))}

          {loading &&
            ["0s", "0.7s", "1.4s"].map((delay) => (
              <circle
                key={delay}
                cx="100"
                cy="100"
                r={ARC_RADIUS}
                fill="none"
                stroke="var(--dim)"
                strokeWidth="2"
                className="animate-ring-pulse"
                style={{ animationDelay: delay }}
              />
            ))}

          {/* Result: a solid ring in the verdict colour. Deliberately not a
              proportional arc — the band is the message, not a magnitude. */}
          {verdict && !loading && (
            <circle
              cx="100"
              cy="100"
              r={ARC_RADIUS}
              fill="none"
              stroke={color}
              strokeWidth="7"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={0}
              style={{ transition: "stroke 500ms ease-out" }}
            />
          )}
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center px-8 text-center">
          {loading ? (
            <>
              <span className="font-mono text-xs uppercase tracking-[0.2em] text-dim">Scanning</span>
              {stageTag && (
                <span className="mt-2 font-mono text-[11px] tracking-[0.15em] text-paper" data-testid="gauge-stage">
                  [{stageTag}]
                </span>
              )}
            </>
          ) : verdict ? (
            <>
              <VerdictIcon verdict={verdict} />
              <span
                className="mt-1.5 font-display text-base font-semibold leading-tight"
                style={{ color }}
                data-testid="gauge-verdict"
              >
                {VERDICT_LABEL[verdict]}
              </span>
              <span className="mt-1 font-mono text-[9px] uppercase leading-tight tracking-[0.1em] text-dimmer">
                not-true-doc: <span className="text-dim">{LIKELIHOOD_LEVEL[verdict]}</span>
              </span>
            </>
          ) : notApplicable ? (
            <>
              <span className="font-display text-base font-semibold leading-tight text-dim">Not scorable</span>
              <span className="mt-1 font-mono text-[9px] uppercase leading-tight tracking-[0.1em] text-dimmer">
                not-true-doc: <span className="text-dim">n/a</span>
              </span>
            </>
          ) : (
            <>
              <span className="font-mono text-xs uppercase tracking-[0.2em] text-dimmer">No scan yet</span>
              <span className="mt-2 max-w-[6.5rem] font-mono text-[10px] uppercase leading-tight tracking-[0.12em] text-dimmer">
                {SCORE_NAME}
              </span>
            </>
          )}
        </div>
      </div>

      <VerdictScale active={verdict} />
    </figure>
  );
}
