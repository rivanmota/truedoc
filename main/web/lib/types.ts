export type Confidence = "high" | "medium" | "low-medium" | "low";

export interface SignalResult {
  score: number;
  confidence: Confidence;
  /** false when the signal does not apply to this input (e.g. gradients on
   * a non-portrait, noise on a JPEG-reshaped floor). Its score is then a
   * placeholder that carried no weight in ai_generated. */
  applicable?: boolean;
  note?: string;
  [key: string]: unknown;
}

export interface SignalSet {
  frequency: SignalResult;
  noise: SignalResult;
  gradients: SignalResult;
  lighting: SignalResult;
  metadata: SignalResult;
}

export interface PageResult {
  page: number;
  ai_generated: number;
  signals: SignalSet;
}

export interface CheckResult {
  ai_generated: number;
  signals: SignalSet;
  /** false when the file had too little content to score (e.g. a PDF with
   * no raster content and no usable metadata verdict). */
  score_applicable?: boolean;
  /** Plain-language note from the engine, e.g. why a file wasn't scorable. */
  description?: string;
  /** Present only when the input was a PDF. */
  page_count?: number;
  worst_page?: number;
  pages?: PageResult[];
}

export interface CheckResponse {
  status: "success";
  request: {
    id: string;
    timestamp: number;
    operations: number;
  };
  type: CheckResult;
  media: {
    id: string;
    uri: string;
  };
}

export interface CheckErrorResponse {
  status: "failure";
  error: {
    type: string;
    message: string;
  };
}

export const SIGNAL_META: Record<keyof SignalSet, { tag: string; label: string }> = {
  frequency: { tag: "FREQ", label: "Frequency spectrum" },
  noise: { tag: "NOISE", label: "Sensor noise floor" },
  gradients: { tag: "GRAD", label: "Texture smoothness" },
  lighting: { tag: "LIGHT", label: "Lighting consistency" },
  metadata: { tag: "META", label: "Metadata" },
};
