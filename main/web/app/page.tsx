"use client";

import { useMemo, useState } from "react";
import { PagePicker } from "@/components/PagePicker";
import { ScoreGauge } from "@/components/ScoreGauge";
import { SignalRow } from "@/components/SignalRow";
import { UploadPanel } from "@/components/UploadPanel";
import { SIGNAL_META, type CheckResponse, type CheckErrorResponse, type SignalSet } from "@/lib/types";

type State =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "done"; data: CheckResponse }
  | { phase: "error"; message: string };

export default function Home() {
  const [state, setState] = useState<State>({ phase: "idle" });
  const [selectedPage, setSelectedPage] = useState<number | null>(null);

  async function handleSubmit(formData: FormData) {
    setState({ phase: "loading" });
    try {
      const res = await fetch("/api/check", { method: "POST", body: formData });
      const data = (await res.json()) as CheckResponse | CheckErrorResponse;
      if (data.status === "success") {
        setState({ phase: "done", data });
        setSelectedPage(data.type.worst_page ?? null);
      } else {
        setState({ phase: "error", message: data.error.message });
      }
    } catch {
      setState({ phase: "error", message: "The scan failed unexpectedly. Try again." });
    }
  }

  const loading = state.phase === "loading";
  const result = state.phase === "done" ? state.data.type : undefined;
  const pages = result?.pages;

  // For a PDF: show the selected page's own visual signals, but always
  // the document-level metadata (a single check covering the whole
  // file, not any one rasterized page — see app/main.py's
  // _run_pdf_analysis). For a plain image, just use its signals as-is.
  const displaySignals: SignalSet | undefined = useMemo(() => {
    if (!result) return undefined;
    if (!pages || pages.length === 0) return result.signals;
    const page = pages.find((p) => p.page === selectedPage) ?? pages[0];
    return { ...page.signals, metadata: result.signals.metadata };
  }, [result, pages, selectedPage]);

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col px-6 py-16">
      <header className="mb-10">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-dimmer">
          Local scan · no data leaves this machine
        </p>
        <h1 className="mt-2 font-display text-3xl font-semibold text-paper">
          AI Image Detector
        </h1>
        <p className="mt-2 max-w-lg text-sm leading-relaxed text-dim">
          Runs four signal analyzers — frequency spectrum, sensor noise, texture
          smoothness, and lighting consistency — plus a metadata check, and combines
          them into one estimate. PDFs are scanned page by page. Heuristic, not a
          trained classifier: read a result as a lead, not a verdict.
        </p>
      </header>

      <section className="mb-10 flex justify-center">
        <ScoreGauge score={result?.ai_generated} loading={loading} />
      </section>

      <section className="mb-10">
        <UploadPanel loading={loading} onSubmit={handleSubmit} />
      </section>

      {state.phase === "error" && (
        <div className="mb-10 rounded-lg border border-signal-ai/40 bg-panel px-4 py-3 text-sm text-signal-ai">
          {state.message}
        </div>
      )}

      {state.phase === "done" && result && displaySignals && (
        <section className="flex flex-col gap-4">
          {pages && pages.length > 1 && (
            <PagePicker
              pages={pages}
              worstPage={result.worst_page ?? pages[0].page}
              selected={selectedPage ?? pages[0].page}
              onSelect={setSelectedPage}
            />
          )}

          <div className="flex flex-col gap-2">
            <p className="mb-1 font-mono text-[11px] uppercase tracking-[0.15em] text-dimmer">
              Signal breakdown
            </p>
            {(Object.keys(SIGNAL_META) as (keyof typeof SIGNAL_META)[]).map((key) => (
              <SignalRow
                key={key}
                tag={SIGNAL_META[key].tag}
                label={SIGNAL_META[key].label}
                result={displaySignals[key]}
              />
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
