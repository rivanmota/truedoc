"use client";

import { SIGNAL_VERDICT_COLOR, SIGNAL_VERDICT_LABEL, signalVerdictFor } from "@/lib/verdict";
import type { PageResult } from "@/lib/types";

export function PagePicker({
  pages,
  worstPage,
  selected,
  onSelect,
}: {
  pages: PageResult[];
  worstPage: number;
  selected: number;
  onSelect: (page: number) => void;
}) {
  return (
    <div>
      <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.15em] text-dimmer">
        {pages.length} pages · evidence for:
      </p>
      <div className="flex flex-wrap gap-2">
        {pages.map((p) => {
          const isSelected = p.page === selected;
          const verdict = signalVerdictFor(p.ai_generated);
          const color = SIGNAL_VERDICT_COLOR[verdict];
          return (
            <button
              key={p.page}
              type="button"
              onClick={() => onSelect(p.page)}
              className={`flex items-center gap-2 rounded-md border px-3 py-1.5 font-mono text-xs transition-colors ${
                isSelected
                  ? "border-paper bg-panel-raised text-paper"
                  : "border-line bg-panel text-dim hover:text-paper"
              }`}
            >
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: color }}
                aria-hidden="true"
              />
              <span>
                Page {p.page}
                {p.page === worstPage && <span className="text-dimmer"> · key page</span>}
              </span>
              <span className="sr-only">{SIGNAL_VERDICT_LABEL[verdict]}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
