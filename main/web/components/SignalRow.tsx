"use client";

import { scoreColor } from "@/lib/verdict";
import type { SignalResult } from "@/lib/types";

const CONFIDENCE_LABEL: Record<string, string> = {
  high: "HIGH CONF.",
  medium: "MED CONF.",
  "low-medium": "LOW-MED CONF.",
  low: "LOW CONF.",
};

function formatKey(key: string): string {
  return key.replace(/_/g, " ");
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function SignalRow({
  tag,
  label,
  result,
}: {
  tag: string;
  label: string;
  result: SignalResult;
}) {
  const { score, confidence, applicable, note, ...rest } = result;
  const extras = Object.entries(rest).filter(
    ([, v]) => v !== null && typeof v !== "object",
  );
  // A not-applicable signal carries no weight; showing its placeholder
  // score as a bar would read as evidence.
  const notApplicable = applicable === false;
  const color = notApplicable ? "var(--color-dimmer, #6b7280)" : scoreColor(score);

  return (
    <details className="group rounded-md border border-line bg-panel open:bg-panel-raised">
      <summary className="flex cursor-pointer list-none items-center gap-4 px-4 py-3 marker:content-none">
        <span className="w-16 shrink-0 font-mono text-[11px] tracking-[0.1em] text-dim">
          [{tag}]
        </span>
        <span className="w-40 shrink-0 text-sm text-paper">{label}</span>

        <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-line">
          <span
            className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-out"
            style={{ width: notApplicable ? "0%" : `${Math.round(score * 100)}%`, background: color }}
          />
        </span>

        <span className="w-12 shrink-0 text-right font-mono text-sm" style={{ color }}>
          {notApplicable ? "N/A" : `${Math.round(score * 100)}%`}
        </span>
        <span className="hidden w-24 shrink-0 text-right font-mono text-[10px] tracking-[0.08em] text-dimmer sm:block">
          {notApplicable ? "NOT USED" : (CONFIDENCE_LABEL[confidence] ?? confidence)}
        </span>
        <span className="shrink-0 text-dimmer transition-transform group-open:rotate-90">
          ›
        </span>
      </summary>

      <div className="border-t border-line px-4 py-3 pl-[5.5rem]">
        {note && <p className="text-sm leading-relaxed text-dim">{note}</p>}
        {extras.length > 0 && (
          <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs sm:grid-cols-3">
            {extras.map(([key, value]) => (
              <div key={key} className="flex justify-between gap-2 border-b border-line/60 py-1">
                <dt className="text-dimmer">{formatKey(key)}</dt>
                <dd className="text-paper">{formatValue(value)}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </details>
  );
}
