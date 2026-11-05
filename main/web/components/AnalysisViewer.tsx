"use client";

import { useEffect, useRef, useState } from "react";
import { renderStage, STAGE_VISUALS, WORKING_MAX_SIDE } from "@/lib/pixelFilters";

export type ViewerSource =
  | { kind: "file"; file: File }
  | { kind: "url"; url: string }
  | null;

/** Result of asynchronously decoding an image source, tagged with the source it belongs to. */
interface Decoded {
  forSource: ViewerSource;
  status: "ready" | "unavailable";
  width?: number;
  height?: number;
  bytes?: number;
  type?: string;
  name?: string;
}

function isPdf(file: File): boolean {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function loadImage(src: string, crossOrigin: boolean): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    if (crossOrigin) img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not load image"));
    img.src = src;
  });
}

/**
 * Shows the file under the transform the active pipeline stage applies —
 * spectrum for FREQ, noise residual for NOISE, gradient map for GRAD,
 * shading-direction grid for LIGHT, and so on (lib/pixelFilters.ts).
 * Filters are computed once per stage on a ≤512px working copy and cached.
 */
export function AnalysisViewer({
  source,
  stageTag,
  scanning,
}: {
  source: ViewerSource;
  /** Tag of the stage to visualise; undefined shows the original. */
  stageTag?: string;
  scanning?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const srcRef = useRef<ImageData | null>(null);
  const cacheRef = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const [decoded, setDecoded] = useState<Decoded | null>(null);

  // Synchronous cases are derived, not stored.
  const pdfFile = source?.kind === "file" && isPdf(source.file) ? source.file : null;
  const needsDecode = !!source && !pdfFile;
  const load = !source
    ? { status: "empty" as const }
    : pdfFile
      ? { status: "pdf" as const, name: pdfFile.name, bytes: pdfFile.size }
      : decoded && decoded.forSource === source
        ? decoded
        : { status: "loading" as const };

  // Decode the source into a working ImageData (async; state set in callbacks).
  useEffect(() => {
    if (!needsDecode || !source) return;
    let cancelled = false;
    let objectUrl: string | null = null;

    (async () => {
      try {
        let img: HTMLImageElement;
        if (source.kind === "file") {
          objectUrl = URL.createObjectURL(source.file);
          img = await loadImage(objectUrl, false);
        } else {
          img = await loadImage(source.url, true);
        }
        if (cancelled) return;
        const scale = Math.min(1, WORKING_MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
        const w = Math.max(1, Math.round(img.naturalWidth * scale));
        const h = Math.max(1, Math.round(img.naturalHeight * scale));
        const work = document.createElement("canvas");
        work.width = w;
        work.height = h;
        const wctx = work.getContext("2d", { willReadFrequently: true })!;
        wctx.drawImage(img, 0, 0, w, h);
        // Throws on a tainted (non-CORS) cross-origin image.
        const imageData = wctx.getImageData(0, 0, w, h);
        if (cancelled) return;
        srcRef.current = imageData;
        cacheRef.current.clear();
        setDecoded({
          forSource: source,
          status: "ready",
          width: img.naturalWidth,
          height: img.naturalHeight,
          bytes: source.kind === "file" ? source.file.size : undefined,
          type: source.kind === "file" ? source.file.type || undefined : undefined,
          name: source.kind === "file" ? source.file.name : source.url,
        });
      } catch {
        if (!cancelled) setDecoded({ forSource: source, status: "unavailable" });
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [source, needsDecode]);

  const ready = load.status === "ready";

  // Render the active stage's transform (cached per tag).
  useEffect(() => {
    const canvas = canvasRef.current;
    const src = srcRef.current;
    if (!canvas || !src || !ready) return;
    const tag = stageTag ?? "READ";
    let cached = cacheRef.current.get(tag);
    if (!cached) {
      cached = document.createElement("canvas");
      cached.width = src.width;
      cached.height = src.height;
      renderStage(tag, src, cached.getContext("2d")!);
      cacheRef.current.set(tag, cached);
    }
    canvas.width = src.width;
    canvas.height = src.height;
    canvas.getContext("2d")!.drawImage(cached, 0, 0);
    if (frameRef.current) frameRef.current.dataset.stage = tag;
  }, [stageTag, ready]);

  const tag = stageTag ?? "READ";
  const caption = STAGE_VISUALS[tag]?.caption ?? "Original";
  const showCanvas = ready;

  return (
    <div
      ref={frameRef}
      className="relative flex h-[300px] items-center justify-center overflow-hidden rounded-lg border border-line bg-ink sm:h-[360px] lg:h-auto lg:min-h-[420px] lg:flex-1"
      data-testid="analysis-viewer"
    >
      {/* faint grid backdrop */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          backgroundImage:
            "linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      />

      <canvas
        ref={canvasRef}
        className={`relative max-h-full max-w-full object-contain ${showCanvas ? "" : "hidden"}`}
        aria-label={`${caption} view of the image`}
      />

      {load.status === "empty" && (
        <p className="relative font-mono text-xs uppercase tracking-[0.2em] text-dimmer">Choose a file to preview</p>
      )}
      {load.status === "loading" && (
        <p className="relative font-mono text-xs uppercase tracking-[0.2em] text-dim">Loading preview…</p>
      )}
      {load.status === "unavailable" && (
        <p className="relative font-mono text-xs uppercase tracking-[0.2em] text-dimmer">Preview unavailable for this source</p>
      )}
      {load.status === "pdf" && (
        <div className="relative flex flex-col items-center gap-2 text-center">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" className="text-dim">
            <path d="M6 2h9l5 5v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Z" stroke="currentColor" strokeWidth="1.5" />
            <path d="M14 2v6h6" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          <p className="font-mono text-[11px] text-dim">{load.name}</p>
          <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-dimmer">
            Pages are rendered and analysed server-side
          </p>
        </div>
      )}

      {/* META stage: the bytes-level facts the analyzer reads */}
      {showCanvas && tag === "META" && load.status === "ready" && (
        <dl className="absolute bottom-10 left-3 grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 rounded bg-ink/85 px-3 py-2 font-mono text-[10px] text-dim backdrop-blur">
          <dt className="text-dimmer">type</dt><dd className="text-paper">{load.type ?? "—"}</dd>
          <dt className="text-dimmer">size</dt><dd className="text-paper">{load.bytes !== undefined ? formatBytes(load.bytes) : "—"}</dd>
          <dt className="text-dimmer">pixels</dt><dd className="text-paper">{load.width} × {load.height}</dd>
        </dl>
      )}

      {/* scan sweep while the pipeline runs */}
      {scanning && showCanvas && (
        <div aria-hidden="true" className="animate-scanline pointer-events-none absolute inset-x-0 h-16" />
      )}

      {/* stage badge + caption */}
      {(showCanvas || load.status === "pdf") && (
        <div className="absolute inset-x-3 bottom-3 flex items-center justify-between gap-3">
          <span className="rounded bg-ink/85 px-2 py-1 font-mono text-[10px] tracking-[0.15em] text-paper backdrop-blur">
            [{tag}]
          </span>
          <span className="truncate rounded bg-ink/85 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-dim backdrop-blur">
            {caption}
          </span>
        </div>
      )}
    </div>
  );
}
