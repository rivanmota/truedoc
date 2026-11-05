"use client";

import { useEffect, useRef, useState } from "react";
import { AnalysisVisual, type MediaPreview } from "@/components/AnalysisVisual";
import { SIGNAL_META, type SignalResult, type SignalSet } from "@/lib/types";
import {
  SIGNAL_VERDICT_COLOR,
  SIGNAL_VERDICT_LABEL,
  signalVerdictFor,
  type SignalVerdict,
} from "@/lib/verdict";

const SIGNAL_KEYS = Object.keys(SIGNAL_META) as (keyof SignalSet)[];

/** Confidence as 0..3 filled bars — a glance, not a label to parse. */
const CONFIDENCE_BARS: Record<string, number> = {
  high: 3,
  "medium-high": 3,
  medium: 2,
  "low-medium": 1,
  low: 1,
  unavailable: 0,
};

function formatKey(key: string): string {
  return key.replace(/_/g, " ");
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function VerdictMark({ verdict, size = 18 }: { verdict: SignalVerdict; size?: number }) {
  const color = SIGNAL_VERDICT_COLOR[verdict];
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" className="shrink-0">
      <circle cx="10" cy="10" r="9" fill={color} opacity={verdict === "na" ? 0.1 : 0.18} />
      {verdict === "real" && (
        <path d="M6 10.3 8.6 13 14 7.5" stroke={color} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      )}
      {verdict === "ai" && (
        <>
          <path d="M10 5.5v5.5" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
          <circle cx="10" cy="14" r="1.1" fill={color} />
        </>
      )}
      {verdict === "inconclusive" && (
        <>
          <path d="M7.6 8a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .8-1 1.4" stroke={color} strokeWidth="1.8" fill="none" strokeLinecap="round" />
          <circle cx="10" cy="14.2" r="1.1" fill={color} />
        </>
      )}
      {verdict === "na" && <path d="M6.5 10h7" stroke={color} strokeWidth="1.8" strokeLinecap="round" />}
    </svg>
  );
}

function ConfidenceBars({ confidence }: { confidence: string }) {
  const filled = CONFIDENCE_BARS[confidence] ?? 1;
  return (
    <span className="inline-flex items-center gap-2" title={`${confidence} confidence`}>
      <span className="inline-flex items-end gap-0.5" aria-hidden="true">
        {[1, 2, 3].map((i) => (
          <span
            key={i}
            className="w-1.5 rounded-sm"
            style={{
              height: `${4 + i * 3}px`,
              background: i <= filled ? "var(--paper)" : "var(--line)",
            }}
          />
        ))}
      </span>
      <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-dim">
        {confidence === "unavailable" ? "n/a" : `${confidence} confidence`}
      </span>
    </span>
  );
}

function StepPanel({
  tag,
  label,
  result,
  media,
}: {
  tag: string;
  label: string;
  result: SignalResult;
  media: MediaPreview;
}) {
  const { score, confidence, note, ...rest } = result;
  const verdict = signalVerdictFor(score);
  const color = SIGNAL_VERDICT_COLOR[verdict];
  const measurements = Object.entries(rest).filter(([, v]) => v !== null && typeof v !== "object");

  return (
    <div className="grid gap-5 sm:grid-cols-[10.5rem_1fr]">
      {/* The media as this analyzer sees it (real convolution where possible). */}
      <AnalysisVisual media={media} tag={tag} active={false} />

      <div className="flex min-w-0 flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[11px] tracking-[0.1em] text-dim">[{tag}]</p>
            <h3 className="mt-0.5 font-display text-lg font-semibold leading-tight text-paper">{label}</h3>
          </div>
          <ConfidenceBars confidence={confidence} />
        </div>

        <div
          className="flex items-center gap-3 rounded-md border px-3 py-2.5"
          style={{ borderColor: color, background: `color-mix(in srgb, ${color} 8%, transparent)` }}
          data-testid="step-verdict"
        >
          <VerdictMark verdict={verdict} size={24} />
          <span className="font-display text-[15px] font-semibold leading-tight" style={{ color }}>
            {SIGNAL_VERDICT_LABEL[verdict]}
          </span>
        </div>

        {note && (
          <div>
            <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.15em] text-dimmer">Evidence</p>
            <p className="text-sm leading-relaxed text-paper">{note}</p>
          </div>
        )}

        {measurements.length > 0 && (
          <div>
            <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.15em] text-dimmer">Measurements</p>
            <dl className="grid grid-cols-1 gap-x-8 gap-y-0.5 font-mono text-xs md:grid-cols-2">
              {measurements.map(([key, value]) => (
                <div key={key} className="flex justify-between gap-3 border-b border-line/60 py-1">
                  <dt className="text-dimmer">{formatKey(key)}</dt>
                  <dd className="shrink-0 text-paper">{formatValue(value)}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Step-through view of the evidence behind the verdict: one analysis
 * signal at a time, with prev/next, clickable step markers, and arrow-key
 * navigation. Each step shows how that signal reads (in words and colour),
 * its confidence, the analyzer's note, and the raw measurements.
 */
export function EvidenceSteps({ signals, media }: { signals: SignalSet; media: MediaPreview }) {
  const [index, setIndex] = useState(0);
  const panelRef = useRef<HTMLDivElement>(null);
  const total = SIGNAL_KEYS.length;
  const key = SIGNAL_KEYS[index];
  const result = signals[key];

  const go = (next: number) => setIndex(Math.min(total - 1, Math.max(0, next)));

  // Keep focus inside the stepper when navigating via keyboard.
  useEffect(() => {
    panelRef.current?.focus({ preventScroll: true });
  }, [index]);

  return (
    <section
      className="rounded-lg border border-line bg-panel"
      data-testid="evidence-steps"
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") { e.preventDefault(); go(index + 1); }
        if (e.key === "ArrowLeft") { e.preventDefault(); go(index - 1); }
        if (e.key === "Home") { e.preventDefault(); go(0); }
        if (e.key === "End") { e.preventDefault(); go(total - 1); }
      }}
    >
      {/* Step markers: one per signal, coloured by how it reads */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-line px-4 py-2.5">
        <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-dimmer">
          Evidence · step {index + 1} of {total}
        </p>
        <div role="tablist" aria-label="Evidence steps" className="flex items-center gap-1.5">
          {SIGNAL_KEYS.map((k, i) => {
            const v = signalVerdictFor(signals[k].score);
            const active = i === index;
            return (
              <button
                key={k}
                type="button"
                role="tab"
                aria-selected={active}
                aria-label={`${SIGNAL_META[k].label}: ${SIGNAL_VERDICT_LABEL[v]}`}
                data-step={SIGNAL_META[k].tag}
                data-verdict={v}
                onClick={() => go(i)}
                className={`flex items-center gap-1.5 rounded-full px-2 py-1 transition-colors ${
                  active ? "bg-panel-raised" : "hover:bg-panel-raised/60"
                }`}
              >
                <VerdictMark verdict={v} size={14} />
                <span className={`font-mono text-[10px] tracking-[0.1em] ${active ? "text-paper" : "text-dimmer"}`}>
                  {SIGNAL_META[k].tag}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div
        ref={panelRef}
        role="tabpanel"
        tabIndex={0}
        aria-label={`${SIGNAL_META[key].label} evidence`}
        className="px-4 py-4 outline-none"
        data-testid="evidence-panel"
      >
        <StepPanel key={key} tag={SIGNAL_META[key].tag} label={SIGNAL_META[key].label} result={result} media={media} />
      </div>

      <div className="flex items-center justify-between border-t border-line px-4 py-2.5">
        <button
          type="button"
          onClick={() => go(index - 1)}
          disabled={index === 0}
          className="rounded-md px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] text-dim transition-colors hover:text-paper disabled:cursor-not-allowed disabled:opacity-30"
          data-testid="step-prev"
        >
          ‹ Previous
        </button>
        <span className="font-mono text-[10px] text-dimmer">← → to navigate</span>
        <button
          type="button"
          onClick={() => go(index + 1)}
          disabled={index === total - 1}
          className="rounded-md bg-paper px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] text-ink transition-opacity disabled:cursor-not-allowed disabled:opacity-30"
          data-testid="step-next"
        >
          Next ›
        </button>
      </div>
    </section>
  );
}
