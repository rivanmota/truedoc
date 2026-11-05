"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  simulatedProgress,
  stagesFor,
  stageViews,
  type ScanInputKind,
  type StageView,
} from "@/lib/scanProgress";

/** Update cadence for the simulated bar. Progress is derived from wall-clock
 *  elapsed time (not incremented per tick), so a throttled background tab
 *  simply catches up rather than drifting. */
const TICK_MS = 50;

export type ScanProgressStatus = "idle" | "running" | "complete";

export interface ScanProgressState {
  status: ScanProgressStatus;
  /** 0..100. Caps at SIMULATED_PROGRESS_CAP while running; 100 once complete. */
  progress: number;
  stages: StageView[];
  /** The stage currently marked running, if any. */
  current: StageView | undefined;
}

export interface ScanProgressControls {
  /** Resets the clock and starts simulating for the given input type. */
  start: (kind: ScanInputKind) => void;
  /** Snaps to 100% and marks every stage done. */
  complete: () => void;
  /** Back to idle (e.g. on error). */
  reset: () => void;
}

/**
 * Drives the simulated pipeline progress. Rendering is left to
 * components/ScanProgress.tsx; timing rules live in lib/scanProgress.ts.
 */
export function useScanProgress(): ScanProgressState & ScanProgressControls {
  const [status, setStatus] = useState<ScanProgressStatus>("idle");
  const [kind, setKind] = useState<ScanInputKind>("image");
  const [progress, setProgress] = useState(0);
  const startedAtRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const start = useCallback(
    (nextKind: ScanInputKind) => {
      stopTimer();
      startedAtRef.current = performance.now();
      setKind(nextKind);
      setProgress(0);
      setStatus("running");
      timerRef.current = setInterval(() => {
        const startedAt = startedAtRef.current;
        if (startedAt === null) return;
        setProgress(simulatedProgress(performance.now() - startedAt));
      }, TICK_MS);
    },
    [stopTimer],
  );

  const complete = useCallback(() => {
    stopTimer();
    setProgress(100);
    setStatus("complete");
  }, [stopTimer]);

  const reset = useCallback(() => {
    stopTimer();
    startedAtRef.current = null;
    setProgress(0);
    setStatus("idle");
  }, [stopTimer]);

  // Never leave an interval running after unmount.
  useEffect(() => stopTimer, [stopTimer]);

  const stages = useMemo(
    () => stageViews(stagesFor(kind), progress, status === "complete"),
    [kind, progress, status],
  );
  const current = useMemo(() => stages.find((s) => s.status === "running"), [stages]);

  return { status, progress, stages, current, start, complete, reset };
}
