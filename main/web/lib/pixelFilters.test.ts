// Run with: npm test   (node --test, no extra dependencies)
import assert from "node:assert/strict";
import { before, describe, it } from "node:test";

// Minimal ImageData polyfill so the pure filters can run outside a browser.
before(() => {
  if (typeof globalThis.ImageData === "undefined") {
    class NodeImageData {
      readonly width: number;
      readonly height: number;
      readonly data: Uint8ClampedArray;
      constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
        this.data = new Uint8ClampedArray(width * height * 4);
      }
    }
    (globalThis as unknown as { ImageData: unknown }).ImageData = NodeImageData;
  }
});

const filters = () => import("./pixelFilters.ts");

function rgbaImage(width: number, height: number, fn: (x: number, y: number) => number): ImageData {
  const img = new ImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = fn(x, y);
      const p = (y * width + x) * 4;
      img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
  }
  return img;
}

describe("luminance", () => {
  it("returns one value per pixel using Rec.601 weights", async () => {
    const { luminance } = await filters();
    const img = new ImageData(2, 1);
    img.data.set([255, 0, 0, 255, 0, 0, 255, 255]); // red, blue
    const lum = luminance(img);
    assert.equal(lum.length, 2);
    assert.ok(Math.abs(lum[0] - 0.299 * 255) < 1e-3);
    assert.ok(Math.abs(lum[1] - 0.114 * 255) < 1e-3);
  });
});

describe("medianBlur", () => {
  it("removes an isolated outlier pixel", async () => {
    const { medianBlur } = await filters();
    const w = 9, h = 9;
    const lum = new Float32Array(w * h).fill(100);
    lum[4 * w + 4] = 255;
    const out = medianBlur(lum, w, h, 2);
    assert.equal(out[4 * w + 4], 100);
    assert.ok(out.every((v) => v === 100));
  });
});

describe("sobel", () => {
  it("responds to a vertical edge with gx only", async () => {
    const { sobel } = await filters();
    const w = 8, h = 8;
    const lum = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) lum[y * w + x] = x < 4 ? 0 : 200;
    const { gx, gy } = sobel(lum, w, h);
    const i = 4 * w + 4;
    assert.ok(gx[i] > 0, "gx should be positive across a dark→bright edge");
    assert.equal(gy[i], 0);
  });
});

describe("percentile", () => {
  it("returns min at 0, max at 100, and the middle at 50", async () => {
    const { percentile } = await filters();
    const v = Float32Array.from([5, 1, 9, 3, 7]);
    assert.equal(percentile(v, 0), 1);
    assert.equal(percentile(v, 100), 9);
    assert.equal(percentile(v, 50), 5);
  });
});

describe("fft1d", () => {
  it("turns a DC signal into a single bin at index 0", async () => {
    const { fft1d } = await filters();
    const n = 8;
    const re = new Float32Array(n).fill(1), im = new Float32Array(n);
    fft1d(re, im);
    assert.ok(Math.abs(re[0] - n) < 1e-4);
    for (let k = 1; k < n; k++) assert.ok(Math.hypot(re[k], im[k]) < 1e-4, `bin ${k} should be ~0`);
  });

  it("gives a flat magnitude spectrum for an impulse", async () => {
    const { fft1d } = await filters();
    const n = 16;
    const re = new Float32Array(n), im = new Float32Array(n);
    re[0] = 1;
    fft1d(re, im);
    for (let k = 0; k < n; k++) assert.ok(Math.abs(Math.hypot(re[k], im[k]) - 1) < 1e-4);
  });
});

describe("spectrumImage", () => {
  it("produces an n×n opaque image and puts a pure vertical stripe's energy on the horizontal axis", async () => {
    const { spectrumImage, luminance } = await filters();
    const w = 64, h = 64, n = 32;
    // 8-pixel-period vertical stripes → energy along the horizontal frequency axis (centre row).
    const img = rgbaImage(w, h, (x) => (Math.floor(x / 4) % 2 ? 220 : 30));
    const spec = spectrumImage(luminance(img), w, h, n);
    assert.equal(spec.width, n);
    assert.equal(spec.height, n);
    const brightness = (x: number, y: number) => {
      const p = (y * n + x) * 4;
      return spec.data[p] + spec.data[p + 1] + spec.data[p + 2];
    };
    const centreRow = n / 2;
    let onAxis = 0, offAxis = 0;
    for (let x = 0; x < n; x++) {
      if (Math.abs(x - n / 2) < 2) continue; // skip DC neighbourhood
      onAxis += brightness(x, centreRow);
      offAxis += brightness(x, centreRow + 8);
    }
    assert.ok(onAxis > offAxis, "stripe energy should sit on the horizontal frequency axis");
    assert.ok(spec.data.every((v, i) => (i % 4 === 3 ? v === 255 : true)), "alpha must be opaque");
  });
});

describe("lightingCells", () => {
  it("returns grid² cells whose shading direction follows a left→right ramp", async () => {
    const { lightingCells } = await filters();
    const w = 96, h = 96, grid = 6;
    const lum = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) lum[y * w + x] = (x / (w - 1)) * 255;
    const cells = lightingCells(lum, w, h, grid);
    assert.equal(cells.length, grid * grid);
    for (const c of cells) {
      assert.ok(Math.abs(c.angle) < 0.05, `cell (${c.col},${c.row}) angle ${c.angle} should point +x`);
      assert.ok(c.strength >= 0 && c.strength <= 1);
    }
    // Left column is darker than the right column.
    const left = cells.filter((c) => c.col === 0).reduce((s, c) => s + c.mean, 0);
    const right = cells.filter((c) => c.col === grid - 1).reduce((s, c) => s + c.mean, 0);
    assert.ok(left < right);
  });
});

describe("noiseResidualImage / gradientMagnitudeImage", () => {
  it("renders a flat image as mid-gray residual and zero gradient", async () => {
    const { noiseResidualImage, gradientMagnitudeImage, luminance, heat } = await filters();
    const img = rgbaImage(16, 16, () => 90);
    const lum = luminance(img);
    const res = noiseResidualImage(lum, 16, 16);
    assert.equal(res.data[0], 128);
    const grad = gradientMagnitudeImage(lum, 16, 16);
    const [r, g, b] = heat(0).map(Math.round);
    assert.deepEqual([grad.data[0], grad.data[1], grad.data[2]], [r, g, b]);
  });
});

describe("STAGE_VISUALS", () => {
  it("has a caption for every pipeline stage tag", async () => {
    const { STAGE_VISUALS } = await filters();
    const { stagesFor } = await import("./scanProgress.ts");
    for (const kind of ["image", "pdf", "url"] as const) {
      for (const stage of stagesFor(kind)) {
        assert.ok(STAGE_VISUALS[stage.tag], `missing caption for ${stage.tag}`);
      }
    }
  });
});
