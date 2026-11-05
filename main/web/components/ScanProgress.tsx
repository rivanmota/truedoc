"use client";

import { AnalysisVisual, type MediaPreview } from "@/components/AnalysisVisual";
import { SIMULATED_PROGRESS_CAP } from "@/lib/scanProgress";
import type { StageView } from "@/lib/scanProgress";

function StageIcon({ status }: { status: StageView["status"] }) {
  if (status === "done") {
    return (
      <svg width="13" height="13" viewBox="0 0 14 14" aria-hidden="true" className="shrink-0">
        <circle cx="7" cy="7" r="6" fill="var(--signal-real)" opacity="0.2" />
        <path
          d="M4 7.2 6.2 9.4 10.2 5"
          stroke="var(--signal-real)"
          strokeWidth="1.5"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  if (status === "running") {
    return (
      <svg width="13" height="13" viewBox="0 0 14 14" aria-hidden="true" className="shrink-0">
        <circle cx="7" cy="7" r="6" fill="none" stroke="var(--paper)" strokeWidth="1" opacity="0.35" />
        <circle cx="7" cy="7" r="3" fill="var(--paper)" className="animate-stage-blink" />
      </svg>
    );
  }
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" aria-hidden="true" className="shrink-0">
      <circle cx="7" cy="7" r="6" fill="none" stroke="var(--line)" strokeWidth="1" />
    </svg>
  );
}

/**
 * In-flight scan: the media rendered with the running stage's pixel
 * treatment on the left, the simulated progress bar + stage list on the
 * right. The bar climbs to 90% and holds until the backend answers, then
 * snaps to 100% (see lib/scanProgress.ts).
 */
export function ScanProgress({
  progress,
  stages,
  complete,
  media,
}: {
  progress: number;
  stages: StageView[];
  complete: boolean;
  media: MediaPreview;
}) {
  const pct = Math.round(progress);
  const holding = !complete && progress >= SIMULATED_PROGRESS_CAP - 0.5;
  const current = stages.find((s) => s.status === "running");
  const doneCount = stages.filter((s) => s.status === "done").length;

  const statusLine = complete
    ? "Complete — preparing the verdict"
    : holding
      ? "Waiting for the analyzer…"
      : current
        ? current.label
        : "Starting…";

  return (
    <div
      className="grid gap-5 rounded-lg border border-line bg-panel p-4 sm:grid-cols-[10.5rem_1fr]"
      data-testid="scan-progress"
    >
      <AnalysisVisual media={media} tag={complete ? "SCORE" : current?.tag} />

      <div className="min-w-0">
        <div className="mb-2 flex items-baseline justify-between gap-4">
          <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-dimmer">
            Analysis pipeline
          </p>
          <p className={`font-mono text-[11px] tabular-nums ${complete ? "text-signal-real" : "text-dim"}`}>
            {doneCount}/{stages.length} stages
          </p>
        </div>

        <div
          role="progressbar"
          aria-label="Scan progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-valuetext={`${pct}% — ${statusLine}`}
          className="relative h-2 w-full overflow-hidden rounded-full bg-line"
        >
          <span
            className="absolute inset-y-0 left-0 rounded-full transition-[width,background-color] duration-200 ease-linear"
            style={{
              width: `${pct}%`,
              background: complete ? "var(--signal-real)" : "var(--paper)",
            }}
          />
          {/* Marker where the simulation stops and the real response takes over. */}
          {!complete && (
            <span
              aria-hidden="true"
              className="absolute inset-y-0 w-px bg-dimmer"
              style={{ left: `${SIMULATED_PROGRESS_CAP}%` }}
            />
          )}
        </div>

        <p className="mt-1.5 min-h-[1.1rem] truncate text-xs text-dim" aria-live="polite">
          {statusLine}
        </p>

        <ol className="mt-3 flex flex-col gap-1">
          {stages.map((stage) => (
            <li
              key={stage.tag}
              data-stage={stage.tag}
              data-status={stage.status}
              className={`flex items-center gap-2.5 text-[13px] transition-colors ${
                stage.status === "running"
                  ? "text-paper"
                  : stage.status === "done"
                    ? "text-dim"
                    : "text-dimmer"
              }`}
            >
              <StageIcon status={stage.status} />
              <span className="w-[4.2rem] shrink-0 font-mono text-[10px] tracking-[0.1em]">
                [{stage.tag}]
              </span>
              <span className="min-w-0 flex-1 truncate leading-snug">{stage.label}</span>
              <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.08em]">
                {stage.status === "done" ? "done" : stage.status === "running" ? "running" : "queued"}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
