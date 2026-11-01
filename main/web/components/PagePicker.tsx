"use client";

import { scoreColor } from "@/lib/verdict";
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
        {pages.length} pages scanned — showing page-level breakdown for:
      </p>
      <div className="flex flex-wrap gap-2">
        {pages.map((p) => {
          const isSelected = p.page === selected;
          const color = scoreColor(p.ai_generated);
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
              <span>
                Page {p.page}
                {p.page === worstPage && <span className="text-dimmer"> ·highest</span>}
              </span>
              <span style={{ color }}>{Math.round(p.ai_generated * 100)}%</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
