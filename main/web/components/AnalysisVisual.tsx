"use client";

import { useState } from "react";
import { effectFor, FILTER_IDS } from "@/lib/analysisEffects";

export interface MediaPreview {
  /** Object URL (file upload) or remote URL; "" when there's nothing to show (PDF). */
  src: string;
  kind: "image" | "pdf" | "url";
}

/**
 * SVG filter definitions backing the "real pixel analysis" effects.
 * Mount once per page (page.tsx) — AnalysisVisual references them by id.
 */
export function AnalysisFilterDefs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true" focusable="false">
      <defs>
        {/* Laplacian edge detection — the gradients/texture view. */}
        <filter id={FILTER_IDS.edge} colorInterpolationFilters="sRGB">
          <feColorMatrix type="saturate" values="0" />
          <feConvolveMatrix
            order="3"
            kernelMatrix="-1 -1 -1 -1 8 -1 -1 -1 -1"
            preserveAlpha="true"
          />
          <feComponentTransfer>
            <feFuncR type="linear" slope="2.2" />
            <feFuncG type="linear" slope="2.2" />
            <feFuncB type="linear" slope="2.2" />
          </feComponentTransfer>
        </filter>

        {/* High-pass residual (pixels minus their blur) — the sensor-noise view. */}
        <filter id={FILTER_IDS.noise} colorInterpolationFilters="sRGB">
          <feColorMatrix type="saturate" values="0" result="gray" />
          <feGaussianBlur in="gray" stdDeviation="1.2" result="blurred" />
          <feComposite in="gray" in2="blurred" operator="arithmetic" k1="0" k2="5" k3="-5" k4="0.5" />
        </filter>

        {/* Blurred luminance quantised into bands — the lighting view. */}
        <filter id={FILTER_IDS.light} colorInterpolationFilters="sRGB">
          <feColorMatrix type="saturate" values="0" />
          <feGaussianBlur stdDeviation="5" />
          <feComponentTransfer>
            <feFuncR type="discrete" tableValues="0.05 0.25 0.45 0.65 0.85" />
            <feFuncG type="discrete" tableValues="0.05 0.25 0.45 0.65 0.85" />
            <feFuncB type="discrete" tableValues="0.05 0.25 0.45 0.65 0.85" />
          </feComponentTransfer>
        </filter>
      </defs>
    </svg>
  );
}

function PlaceholderGlyph({ kind }: { kind: MediaPreview["kind"] }) {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" className="text-dimmer" aria-hidden="true">
      {kind === "pdf" ? (
        <>
          <path d="M6 2h9l5 5v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Z" stroke="currentColor" strokeWidth="1.5" />
          <path d="M14 2v6h6" stroke="currentColor" strokeWidth="1.5" />
        </>
      ) : (
        <>
          <rect x="3" y="5" width="18" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="8.5" cy="10" r="1.5" stroke="currentColor" strokeWidth="1.5" />
          <path d="m5 17 5-5 4 4 2-2 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        </>
      )}
    </svg>
  );
}

/**
 * The uploaded media rendered "as the analyzer sees it" for a given
 * pipeline stage: real convolutions for NOISE/GRAD/LIGHT, illustrative
 * overlays for the rest. Falls back to a glyph for PDFs (pages are
 * rasterised server-side) and for remote URLs that refuse to load.
 */
export function AnalysisVisual({
  media,
  tag,
  active = true,
}: {
  media: MediaPreview;
  /** Stage/signal tag selecting the effect, e.g. "GRAD". */
  tag: string | undefined;
  /** False mutes the scan animation (e.g. on a finished step view). */
  active?: boolean;
}) {
  const effect = effectFor(tag);
  // Remember which src failed to load; a new src resets this implicitly.
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const showImage = media.src !== "" && failedSrc !== media.src;

  return (
    <figure
      className="mx-auto w-full max-w-[13rem] select-none sm:mx-0 sm:max-w-none"
      data-testid="analysis-visual"
      data-effect={tag ?? "none"}
    >
      <div className="relative aspect-square w-full overflow-hidden rounded-md border border-line bg-ink">
        {showImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={media.src}
            alt=""
            onError={() => setFailedSrc(media.src)}
            className="absolute inset-0 h-full w-full object-contain"
            style={{ filter: effect.filter }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <PlaceholderGlyph kind={media.kind} />
          </div>
        )}

        {effect.overlay === "scan" && (
          <span
            aria-hidden="true"
            className={`absolute inset-x-0 h-10 ${active ? "animate-scan-sweep" : "top-1/3"}`}
            style={{
              background:
                "linear-gradient(180deg, transparent, color-mix(in srgb, var(--signal-real) 25%, transparent) 45%, color-mix(in srgb, var(--signal-real) 65%, transparent) 50%, color-mix(in srgb, var(--signal-real) 25%, transparent) 55%, transparent)",
            }}
          />
        )}
        {effect.overlay === "grid" && (
          <span
            aria-hidden="true"
            className="absolute inset-0 opacity-40"
            style={{
              background:
                "repeating-linear-gradient(0deg, transparent 0 11px, var(--line) 11px 12px), repeating-linear-gradient(90deg, transparent 0 11px, var(--line) 11px 12px)",
            }}
          />
        )}
        {effect.overlay === "rings" && (
          <span
            aria-hidden="true"
            className="absolute inset-0 opacity-50"
            style={{
              background:
                "repeating-radial-gradient(circle at 50% 50%, transparent 0 14px, color-mix(in srgb, var(--paper) 30%, transparent) 14px 15px)",
            }}
          />
        )}
        {effect.overlay === "brackets" && (
          <span aria-hidden="true" className="absolute inset-2">
            {(["-top-0 -left-0 border-t border-l", "-top-0 -right-0 border-t border-r", "-bottom-0 -left-0 border-b border-l", "-bottom-0 -right-0 border-b border-r"] as const).map(
              (pos) => (
                <span key={pos} className={`absolute h-4 w-4 border-paper/70 ${pos}`} />
              ),
            )}
          </span>
        )}
      </div>
      <figcaption className="mt-1.5 text-center font-mono text-[10px] uppercase tracking-[0.12em] text-dimmer">
        {effect.caption}
      </figcaption>
    </figure>
  );
}
