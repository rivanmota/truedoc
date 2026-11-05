"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnalysisFilterDefs, type MediaPreview } from "@/components/AnalysisVisual";
import { EvidenceSteps } from "@/components/EvidenceSteps";
import { PagePicker } from "@/components/PagePicker";
import { ScanProgress } from "@/components/ScanProgress";
import { ScoreGauge } from "@/components/ScoreGauge";
import { ThemeToggle } from "@/components/ThemeToggle";
import { UploadPanel } from "@/components/UploadPanel";
import { useScanProgress } from "@/hooks/useScanProgress";
import { COMPLETION_HOLD_MS, extraDelayMs, inputKindFor } from "@/lib/scanProgress";
import type { CheckResponse, CheckErrorResponse, SignalSet } from "@/lib/types";

type State =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "done"; data: CheckResponse }
  | { phase: "error"; message: string };

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function Home() {
  const [state, setState] = useState<State>({ phase: "idle" });
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const [media, setMedia] = useState<MediaPreview>({ src: "", kind: "image" });
  const progress = useScanProgress();
  // Guards against a stale request finishing after a newer one started
  // (or after unmount) and clobbering state.
  const runIdRef = useRef(0);
  // Object URL for the current preview, revoked when replaced/unmounted.
  const objectUrlRef = useRef<string | null>(null);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  /** Builds the per-stage visual's media source from the submitted form. */
  function previewFor(formData: FormData): MediaPreview {
    const kind = inputKindFor(formData);
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    const file = formData.get("media");
    if (kind === "image" && file instanceof File) {
      objectUrlRef.current = URL.createObjectURL(file);
      return { src: objectUrlRef.current, kind };
    }
    if (kind === "url") {
      const url = formData.get("url");
      return { src: typeof url === "string" ? url.trim() : "", kind };
    }
    return { src: "", kind }; // PDF: pages are rasterised server-side
  }

  async function handleSubmit(formData: FormData) {
    const runId = ++runIdRef.current;
    const isCurrent = () => runIdRef.current === runId;
    const startedAt = performance.now();

    setMedia(previewFor(formData));
    setState({ phase: "loading" });
    progress.start(inputKindFor(formData));

    let data: CheckResponse | CheckErrorResponse;
    try {
      const res = await fetch("/api/check", { method: "POST", body: formData });
      data = (await res.json()) as CheckResponse | CheckErrorResponse;
    } catch {
      if (!isCurrent()) return;
      progress.reset();
      setState({ phase: "error", message: "The scan failed unexpectedly. Try again." });
      return;
    }
    if (!isCurrent()) return;

    if (data.status !== "success") {
      // Errors are shown immediately — no point making someone wait to
      // learn their file was rejected.
      progress.reset();
      setState({ phase: "error", message: data.error.message });
      return;
    }

    // Fast responses get padded so the pipeline walkthrough is legible
    // (see lib/scanProgress.ts for the thresholds).
    const elapsed = performance.now() - startedAt;
    const pad = extraDelayMs(elapsed);
    if (pad > 0) await sleep(pad);
    if (!isCurrent()) return;

    // Snap to 100%, let the "all stages done" state register, then reveal.
    progress.complete();
    await sleep(COMPLETION_HOLD_MS);
    if (!isCurrent()) return;

    setState({ phase: "done", data });
    setSelectedPage(data.type.worst_page ?? null);
  }

  const loading = state.phase === "loading";
  const result = state.phase === "done" ? state.data.type : undefined;
  const pages = result?.pages;
  const scoreApplicable = result ? result.score_applicable !== false : true;

  // For a PDF: show the selected page's own visual signals, but always
  // the document-level metadata (a single check covering the whole
  // file, not any one rasterized page — see app/main.py's
  // _run_pdf_analysis). For a plain image, just use its signals as-is.
  const displaySignals: SignalSet | undefined = useMemo(() => {
    if (!result || !result.signals) return undefined;
    if (!pages || pages.length === 0) return result.signals;
    const page = pages.find((p) => p.page === selectedPage) ?? pages[0];
    return { ...page.signals, metadata: result.signals.metadata };
  }, [result, pages, selectedPage]);

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-10">
      <AnalysisFilterDefs />

      <header className="mb-8">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <h1 className="font-display text-2xl font-semibold text-paper">AI Image Detector</h1>
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-dimmer">
            Local scan · no data leaves this machine
          </p>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
        <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-dim">
          Is this image or PDF a genuine capture, or AI-generated / manipulated? Heuristic — a
          lead, not a verdict.
        </p>
      </header>

      <div className="grid gap-8 md:grid-cols-[16.5rem_1fr] md:items-start">
        <section className="flex justify-center md:sticky md:top-8">
          <ScoreGauge
            score={result?.ai_generated}
            applicable={scoreApplicable}
            loading={loading}
            stageTag={progress.current?.tag}
          />
        </section>

        <div className="flex min-w-0 flex-col gap-5">
          {loading && (
            <section aria-busy="true">
              <ScanProgress
                progress={progress.progress}
                stages={progress.stages}
                complete={progress.status === "complete"}
                media={media}
              />
            </section>
          )}

          {/* Hidden (not unmounted) while scanning so the picked file survives. */}
          <section className={loading ? "hidden" : undefined}>
            <UploadPanel loading={loading} onSubmit={handleSubmit} />
          </section>

          {state.phase === "error" && (
            <div className="rounded-lg border border-signal-ai/40 bg-panel px-4 py-3 text-sm text-signal-ai">
              {state.message}
            </div>
          )}

          {state.phase === "done" && result && !scoreApplicable && (
            <div
              className="rounded-lg border border-line bg-panel px-4 py-3 text-sm leading-relaxed text-dim"
              data-testid="not-scorable-note"
            >
              {result.description ?? "Not enough image content in this file to reach a verdict."}
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

              <EvidenceSteps key={state.data.request.id} signals={displaySignals} media={media} />
            </section>
          )}
        </div>
      </div>
    </main>
  );
}
