/**
 * Client-side visualisations of what each pipeline stage looks at.
 *
 * These mirror the backend analyzers (app/analyzers/*.py) closely enough
 * to be honest illustrations — same transforms, same grid sizes — but they
 * are *illustrations*: the verdict always comes from the backend.
 *
 *   FREQ   fftshift(log|FFT2|) of the grayscale image, with the radial
 *          rings the backend averages over (frequency.py)
 *   NOISE  gray − medianBlur(gray, 5), amplified, masked to flat regions
 *          (noise.py)
 *   GRAD   Sobel gradient magnitude (gradients.py)
 *   LIGHT  6×6 grid: mean luminance per cell + dominant shading direction
 *          from Sobel(ksize 5) (lighting.py)
 *   DECODE grayscale working copy (main.py `_load_image`)
 *
 * Pure functions over ImageData / Float32Array so they're testable without
 * a DOM; only `renderStage` touches a canvas context.
 */

export interface StageVisual {
  /** Short caption shown under the viewer. */
  caption: string;
}

export const STAGE_VISUALS: Record<string, StageVisual> = {
  READ: { caption: "Original bytes" },
  FETCH: { caption: "Original bytes" },
  DECODE: { caption: "Grayscale working copy" },
  RASTER: { caption: "Pages rendered server-side" },
  FREQ: { caption: "2-D power spectrum · radial rings" },
  NOISE: { caption: "Sensor-noise residual in flat regions ×8" },
  GRAD: { caption: "Gradient magnitude (texture)" },
  LIGHT: { caption: "Shading direction per cell" },
  META: { caption: "Embedded metadata" },
  DOC: { caption: "Document structure" },
  SCORE: { caption: "Original" },
};

/** Longest side of the working copy — plenty for a visual, cheap to filter. */
export const WORKING_MAX_SIDE = 512;

/** Luminance (0..255) per pixel, Rec.601 like OpenCV's BGR2GRAY. */
export function luminance(src: ImageData): Float32Array {
  const { data, width, height } = src;
  const out = new Float32Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  return out;
}

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** Grayscale ImageData from a luminance plane. */
export function grayImage(lum: Float32Array, width: number, height: number): ImageData {
  const out = new ImageData(width, height);
  for (let i = 0, p = 0; i < lum.length; i++, p += 4) {
    const v = clamp255(lum[i]);
    out.data[p] = out.data[p + 1] = out.data[p + 2] = v;
    out.data[p + 3] = 255;
  }
  return out;
}

/** Median filter with a (2r+1)² window, edge-clamped. r=2 ⇒ 5×5 like cv2.medianBlur(…, 5). */
export function medianBlur(lum: Float32Array, width: number, height: number, r = 2): Float32Array {
  const out = new Float32Array(lum.length);
  const win = new Float32Array((2 * r + 1) * (2 * r + 1));
  const mid = win.length >> 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let n = 0;
      for (let dy = -r; dy <= r; dy++) {
        const yy = Math.min(height - 1, Math.max(0, y + dy));
        for (let dx = -r; dx <= r; dx++) {
          const xx = Math.min(width - 1, Math.max(0, x + dx));
          win[n++] = lum[yy * width + xx];
        }
      }
      win.sort();
      out[y * width + x] = win[mid];
    }
  }
  return out;
}

/** Sobel gradients (gx, gy) with a 3×3 kernel, edge-clamped. */
export function sobel(lum: Float32Array, width: number, height: number): { gx: Float32Array; gy: Float32Array } {
  const gx = new Float32Array(lum.length);
  const gy = new Float32Array(lum.length);
  const at = (x: number, y: number) =>
    lum[Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const tl = at(x - 1, y - 1), t = at(x, y - 1), tr = at(x + 1, y - 1);
      const l = at(x - 1, y), rr = at(x + 1, y);
      const bl = at(x - 1, y + 1), b = at(x, y + 1), br = at(x + 1, y + 1);
      const i = y * width + x;
      gx[i] = -tl - 2 * l - bl + tr + 2 * rr + br;
      gy[i] = -tl - 2 * t - tr + bl + 2 * b + br;
    }
  }
  return { gx, gy };
}

/** 3×3 Laplacian magnitude — the backend's "flatness" measure. */
export function laplacianAbs(lum: Float32Array, width: number, height: number): Float32Array {
  const out = new Float32Array(lum.length);
  const at = (x: number, y: number) =>
    lum[Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = Math.abs(at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * at(x, y));
    }
  }
  return out;
}

/** Value at a given percentile (0..100) of a plane. */
export function percentile(values: Float32Array, p: number): number {
  const sorted = Float32Array.from(values).sort();
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))));
  return sorted[idx];
}

/**
 * NOISE stage: residual = gray − median5(gray), shown ×gain around mid-gray,
 * and dimmed outside the flat-region mask the backend actually measures.
 */
export function noiseResidualImage(lum: Float32Array, width: number, height: number, gain = 8): ImageData {
  const denoised = medianBlur(lum, width, height, 2);
  const lap = laplacianAbs(lum, width, height);
  const flatThresh = percentile(lap, 60);
  const out = new ImageData(width, height);
  for (let i = 0, p = 0; i < lum.length; i++, p += 4) {
    const flat = lap[i] < flatThresh;
    const v = clamp255(128 + gain * (lum[i] - denoised[i]));
    const shade = flat ? v : 128 + (v - 128) * 0.25;
    out.data[p] = shade;
    out.data[p + 1] = shade;
    out.data[p + 2] = flat ? shade : shade * 0.8;
    out.data[p + 3] = 255;
  }
  return out;
}

/** Inferno-ish 4-stop colormap for 0..1. */
export function heat(t: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [13, 15, 19],
    [120, 28, 109],
    [237, 105, 37],
    [252, 255, 164],
  ];
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** GRAD stage: Sobel magnitude, normalised to its 99th percentile, heat-mapped. */
export function gradientMagnitudeImage(lum: Float32Array, width: number, height: number): ImageData {
  const { gx, gy } = sobel(lum, width, height);
  const mag = new Float32Array(lum.length);
  for (let i = 0; i < mag.length; i++) mag[i] = Math.hypot(gx[i], gy[i]);
  const scale = Math.max(1e-6, percentile(mag, 99));
  const out = new ImageData(width, height);
  for (let i = 0, p = 0; i < mag.length; i++, p += 4) {
    const [r, g, b] = heat(mag[i] / scale);
    out.data[p] = r;
    out.data[p + 1] = g;
    out.data[p + 2] = b;
    out.data[p + 3] = 255;
  }
  return out;
}

/** In-place iterative radix-2 FFT. Length must be a power of two. */
export function fft1d(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Area-style resample of a plane to n×n (n power of two). */
export function resampleSquare(lum: Float32Array, width: number, height: number, n: number): Float32Array {
  const out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    const y0 = Math.floor((y * height) / n), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / n));
    for (let x = 0; x < n; x++) {
      const x0 = Math.floor((x * width) / n), x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / n));
      let sum = 0, cnt = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) { sum += lum[yy * width + xx]; cnt++; }
      out[y * n + x] = sum / cnt;
    }
  }
  return out;
}

/**
 * FREQ stage: centred log-magnitude spectrum (fftshift(log(1+|F|))) of the
 * mean-removed, Hann-windowed grayscale image, heat-mapped.
 */
export function spectrumImage(lum: Float32Array, width: number, height: number, n = 256): ImageData {
  const sq = resampleSquare(lum, width, height, n);
  let mean = 0;
  for (let i = 0; i < sq.length; i++) mean += sq[i];
  mean /= sq.length;
  const re = new Float32Array(n * n), im = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    const wy = 0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (n - 1));
    for (let x = 0; x < n; x++) {
      const wx = 0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (n - 1));
      re[y * n + x] = (sq[y * n + x] - mean) * wx * wy;
    }
  }
  const rowR = new Float32Array(n), rowI = new Float32Array(n);
  for (let y = 0; y < n; y++) {
    rowR.set(re.subarray(y * n, y * n + n)); rowI.set(im.subarray(y * n, y * n + n));
    fft1d(rowR, rowI);
    re.set(rowR, y * n); im.set(rowI, y * n);
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { rowR[y] = re[y * n + x]; rowI[y] = im[y * n + x]; }
    fft1d(rowR, rowI);
    for (let y = 0; y < n; y++) { re[y * n + x] = rowR[y]; im[y * n + x] = rowI[y]; }
  }
  const mag = new Float32Array(n * n);
  let lo = Infinity, hi = -Infinity;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      // fftshift: DC to the centre
      const sy = (y + n / 2) % n, sx = (x + n / 2) % n;
      const v = Math.log1p(Math.hypot(re[y * n + x], im[y * n + x]));
      mag[sy * n + sx] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const out = new ImageData(n, n);
  const span = Math.max(1e-6, hi - lo);
  for (let i = 0, p = 0; i < mag.length; i++, p += 4) {
    const [r, g, b] = heat((mag[i] - lo) / span);
    out.data[p] = r; out.data[p + 1] = g; out.data[p + 2] = b; out.data[p + 3] = 255;
  }
  return out;
}

export interface LightingCell {
  col: number;
  row: number;
  /** Mean luminance 0..255. */
  mean: number;
  /** Dominant shading direction in radians (screen coords, y down), magnitude-weighted like lighting.py. */
  angle: number;
  /** Normalised gradient strength 0..1 across cells. */
  strength: number;
  /** False for flat cells the backend skips (no reliable direction). */
  textured: boolean;
}

/** Mean Sobel magnitude below which lighting.py treats a cell as flat. */
export const LIGHTING_FLAT_THRESHOLD = 2.0;

/** LIGHT stage: per-cell mean luminance + magnitude-weighted gradient direction on a grid×grid lattice. */
export function lightingCells(lum: Float32Array, width: number, height: number, grid = 6): LightingCell[] {
  const { gx, gy } = sobel(lum, width, height);
  const cellW = Math.floor(width / grid), cellH = Math.floor(height / grid);
  const cells: LightingCell[] = [];
  let maxMag = 1e-6;
  for (let row = 0; row < grid; row++) {
    for (let col = 0; col < grid; col++) {
      let wx = 0, wy = 0, wsum = 0, sl = 0, n = 0;
      for (let y = row * cellH; y < (row + 1) * cellH; y++) {
        for (let x = col * cellW; x < (col + 1) * cellW; x++) {
          const i = y * width + x;
          const mag = Math.hypot(gx[i], gy[i]);
          wx += gx[i] * mag; wy += gy[i] * mag; wsum += mag; sl += lum[i]; n++;
        }
      }
      const meanMag = wsum / n;
      const mx = wsum > 0 ? wx / wsum : 0, my = wsum > 0 ? wy / wsum : 0;
      if (meanMag > maxMag) maxMag = meanMag;
      cells.push({
        col, row,
        mean: sl / n,
        angle: Math.atan2(my, mx),
        strength: meanMag,
        textured: meanMag >= LIGHTING_FLAT_THRESHOLD,
      });
    }
  }
  for (const c of cells) c.strength = c.strength / maxMag;
  return cells;
}

/**
 * Draws the visual for a stage into `ctx` (a canvas the size of `src`).
 * Unknown tags fall back to the original image.
 */
export function renderStage(tag: string, src: ImageData, ctx: CanvasRenderingContext2D): void {
  const { width, height } = src;
  ctx.clearRect(0, 0, width, height);

  if (tag === "DECODE") {
    ctx.putImageData(grayImage(luminance(src), width, height), 0, 0);
    return;
  }
  if (tag === "NOISE") {
    ctx.putImageData(noiseResidualImage(luminance(src), width, height), 0, 0);
    return;
  }
  if (tag === "GRAD") {
    ctx.putImageData(gradientMagnitudeImage(luminance(src), width, height), 0, 0);
    return;
  }
  if (tag === "FREQ") {
    const n = 256;
    const spec = spectrumImage(luminance(src), width, height, n);
    const tmp = document.createElement("canvas");
    tmp.width = n; tmp.height = n;
    tmp.getContext("2d")!.putImageData(spec, 0, 0);
    ctx.fillStyle = "#0d0f13";
    ctx.fillRect(0, 0, width, height);
    const side = Math.min(width, height);
    const ox = (width - side) / 2, oy = (height - side) / 2;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, ox, oy, side, side);
    // The radial bands the backend averages over.
    ctx.strokeStyle = "rgba(237, 238, 242, 0.35)";
    ctx.lineWidth = 1;
    for (const f of [0.15, 0.3, 0.45]) {
      ctx.beginPath();
      ctx.arc(width / 2, height / 2, side * f, 0, Math.PI * 2);
      ctx.stroke();
    }
    return;
  }
  if (tag === "LIGHT") {
    const grid = 6;
    const lum = luminance(src);
    const cells = lightingCells(lum, width, height, grid);
    const cellW = width / grid, cellH = height / grid;
    for (const c of cells) {
      const v = clamp255(c.mean);
      ctx.fillStyle = `rgb(${v} ${v} ${v})`;
      ctx.fillRect(c.col * cellW, c.row * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
    ctx.strokeStyle = "rgba(13, 15, 19, 0.6)";
    ctx.lineWidth = 1;
    for (let i = 1; i < grid; i++) {
      ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(width, i * cellH); ctx.stroke();
    }
    const maxLen = Math.min(cellW, cellH) * 0.42;
    for (const c of cells) {
      const cx = (c.col + 0.5) * cellW, cy = (c.row + 0.5) * cellH;
      if (!c.textured) {
        // Flat cell — skipped by the analyzer. Mark it rather than invent a direction.
        ctx.strokeStyle = "rgba(134, 140, 156, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.stroke();
        continue;
      }
      const len = maxLen * (0.25 + 0.75 * c.strength);
      const dx = Math.cos(c.angle) * len, dy = Math.sin(c.angle) * len;
      const bright = c.mean > 128;
      ctx.strokeStyle = bright ? "#f0954b" : "#4fd1a5";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx - dx, cy - dy); ctx.lineTo(cx + dx, cy + dy); ctx.stroke();
      // arrow head
      const ang = c.angle, hs = 5;
      ctx.beginPath();
      ctx.moveTo(cx + dx, cy + dy);
      ctx.lineTo(cx + dx - hs * Math.cos(ang - 0.5), cy + dy - hs * Math.sin(ang - 0.5));
      ctx.lineTo(cx + dx - hs * Math.cos(ang + 0.5), cy + dy - hs * Math.sin(ang + 0.5));
      ctx.closePath(); ctx.fill();
    }
    return;
  }
  ctx.putImageData(src, 0, 0);
}
