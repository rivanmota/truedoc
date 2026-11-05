/**
 * Per-stage pixel-analysis visuals.
 *
 * Each pipeline stage (and each evidence step) shows the uploaded media
 * "as the analyzer sees it". Where possible the effect is a *real*
 * transformation of the actual preview pixels, done by the browser:
 *
 *  - GRAD  -> Laplacian edge map (SVG feConvolveMatrix)
 *  - NOISE -> high-pass residual: pixels minus their blur (feComposite)
 *  - LIGHT -> blurred luminance quantised into bands (feComponentTransfer)
 *
 * The rest (scan sweep, sampling grid, spectrum rings, metadata brackets)
 * are illustrative overlays. Pure data here — the SVG filter definitions
 * and rendering live in components/AnalysisVisual.tsx.
 */

export type EffectOverlay = "scan" | "grid" | "rings" | "brackets" | null;

export interface AnalysisEffect {
  /** CSS filter applied to the media preview (may reference an SVG filter id). */
  filter: string;
  overlay: EffectOverlay;
  /** Two-or-three-word caption under the visual. */
  caption: string;
}

/** SVG filter ids — must match components/AnalysisVisual.tsx defs. */
export const FILTER_IDS = {
  edge: "td-fx-edge",
  noise: "td-fx-noise",
  light: "td-fx-light",
} as const;

/** Keyed by pipeline stage tag (lib/scanProgress.ts) — signal tags included. */
export const ANALYSIS_EFFECTS: Record<string, AnalysisEffect> = {
  READ: { filter: "grayscale(0.9) brightness(0.85)", overlay: "scan", caption: "reading bytes" },
  FETCH: { filter: "grayscale(0.9) brightness(0.85)", overlay: "scan", caption: "fetching source" },
  DECODE: { filter: "none", overlay: "grid", caption: "decoded bitmap" },
  RASTER: { filter: "none", overlay: "grid", caption: "page raster" },
  FREQ: { filter: "grayscale(1) invert(0.92) contrast(1.7)", overlay: "rings", caption: "frequency spectrum" },
  NOISE: { filter: `url(#${FILTER_IDS.noise})`, overlay: null, caption: "noise residual" },
  GRAD: { filter: `url(#${FILTER_IDS.edge})`, overlay: null, caption: "edge / texture map" },
  LIGHT: { filter: `url(#${FILTER_IDS.light})`, overlay: null, caption: "luminance bands" },
  META: { filter: "grayscale(1) brightness(0.55)", overlay: "brackets", caption: "metadata fields" },
  DOC: { filter: "grayscale(1) brightness(0.7)", overlay: "brackets", caption: "structure checks" },
  SCORE: { filter: "none", overlay: null, caption: "combined view" },
};

const FALLBACK: AnalysisEffect = { filter: "none", overlay: null, caption: "preview" };

export function effectFor(tag: string | undefined): AnalysisEffect {
  return (tag && ANALYSIS_EFFECTS[tag]) || FALLBACK;
}
