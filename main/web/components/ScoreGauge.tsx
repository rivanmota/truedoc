"use client";

import { scoreColor, verdictFor, VERDICT_LABEL } from "@/lib/verdict";

const RINGS = [38, 54, 70, 86];
const ARC_RADIUS = 78;
const CIRCUMFERENCE = 2 * Math.PI * ARC_RADIUS;

export function ScoreGauge({
  score,
  loading,
}: {
  score?: number;
  loading?: boolean;
}) {
  const pct = score !== undefined ? Math.round(score * 100) : null;
  const color = score !== undefined ? scoreColor(score) : "var(--dim)";
  const dashoffset =
    score !== undefined ? CIRCUMFERENCE * (1 - Math.min(1, Math.max(0, score))) : CIRCUMFERENCE;

  return (
    <div className="relative mx-auto h-56 w-56 shrink-0">
      <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
        {/* background spectrum rings — the signature element, echoing the
            backend's radial power-spectrum profile */}
        {RINGS.map((r) => (
          <circle
            key={r}
            cx="100"
            cy="100"
            r={r}
            fill="none"
            stroke="var(--line)"
            strokeWidth="1"
            opacity={0.6}
          />
        ))}

        {loading && (
          <>
            <circle
              cx="100"
              cy="100"
              r={ARC_RADIUS}
              fill="none"
              stroke="var(--dim)"
              strokeWidth="2"
              className="animate-ring-pulse"
              style={{ animationDelay: "0s" }}
            />
            <circle
              cx="100"
              cy="100"
              r={ARC_RADIUS}
              fill="none"
              stroke="var(--dim)"
              strokeWidth="2"
              className="animate-ring-pulse"
              style={{ animationDelay: "0.7s" }}
            />
            <circle
              cx="100"
              cy="100"
              r={ARC_RADIUS}
              fill="none"
              stroke="var(--dim)"
              strokeWidth="2"
              className="animate-ring-pulse"
              style={{ animationDelay: "1.4s" }}
            />
          </>
        )}

        {!loading && score !== undefined && (
          <circle
            cx="100"
            cy="100"
            r={ARC_RADIUS}
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={dashoffset}
            style={{ transition: "stroke-dashoffset 700ms ease-out, stroke 700ms ease-out" }}
          />
        )}
      </svg>

      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        {loading ? (
          <span className="font-mono text-xs uppercase tracking-[0.2em] text-dim">
            Scanning
          </span>
        ) : pct !== null ? (
          <>
            <span
              className="font-display text-5xl font-semibold leading-none"
              style={{ color }}
            >
              {pct}
              <span className="text-2xl align-top">%</span>
            </span>
            <span className="mt-2 font-mono text-[11px] uppercase tracking-[0.15em] text-dim">
              {VERDICT_LABEL[verdictFor(score!)]}
            </span>
          </>
        ) : (
          <span className="font-mono text-xs uppercase tracking-[0.2em] text-dimmer">
            No scan yet
          </span>
        )}
      </div>
    </div>
  );
}
