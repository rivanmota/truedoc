# Local AI Image Detector

A local, dependency-light, signal-based AI-generated image detector,
served behind a `/1.0/check.json` endpoint shaped like
[Sightengine's `genai` model](https://sightengine.com/docs/ai-generated-image-detection),
so existing client code written against their API mostly just needs
a base-URL swap to point here instead.

This is **not** a trained classifier — it's four independent forensic
signals combined into one score. See "How it works" for what each one
does and, more importantly, what it can't do.

A Next.js UI (`web/`) sits in front of it — drop a file or paste a
URL, get the score plus the per-signal breakdown that produced it.

**Supported formats: PNG, JPEG, PDF.** Format is detected from the
file's actual bytes, not trusted from the filename or upload
Content-Type. Anything else is rejected with a clear error.

## Run everything (API + UI) on Docker

```bash
./run.sh
```

That builds and starts both containers via `docker-compose.yml`:
the API on **http://localhost:8008** and the UI on
**http://localhost:3000**. The UI talks to the API over the compose
network, not over `localhost`, so nothing needs configuring.

- `./run.sh -d` — same, but detached (background)
- `./run.sh down` — stop and remove both containers

## Local development (without Docker)

Prerequisites: **Python 3.12+** and **Node 22+**.

```bash
./init.sh
```

This installs `libgl1`/`libglib2.0-0` via `apt-get` if missing and
`apt-get` is available (needed by `opencv-python-headless` at import
time, as in `Dockerfile`), sets up (or reuses) a `.venv`, installs
`requirements.txt`, installs `web/node_modules` if missing, then
starts the API (`uvicorn app.main:app --reload`) on
**http://localhost:8008** and the UI (`next dev`) on
**http://localhost:3000** — no containers, no image builds, with hot
reload on both sides. Press `Ctrl-C` to stop both.

`./run.sh` remains available as the Docker-based alternative.

## Run just the API

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8008
```

or with Docker:

```bash
docker build -t ai-image-detector .
docker run -p 8008:8008 ai-image-detector
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'cv2'` even though you
  activated a venv and installed `requirements.txt`** — this usually
  means the `uvicorn` command your shell is running isn't actually the
  one inside your venv (a stale shell PATH cache from before you
  activated it). Run it via `python3 -m uvicorn` instead, which always
  uses the active virtualenv's Python:

  ```bash
  source .venv/bin/activate
  python3 -m uvicorn app.main:app --port 8008
  ```

  If you prefer to keep using the bare `uvicorn` command, clear the
  stale cache with `hash -r` and confirm `which uvicorn` now points at
  `.venv/bin/uvicorn` before retrying.

## Run just the UI (outside Docker)

```bash
cd web
npm install
npm run dev
```

By default it points at `http://localhost:8008` for the API — set
`BACKEND_URL` to override (see `web/app/api/check/route.ts`).

## Use the API directly

By file upload (image or PDF):

```bash
curl -X POST 'http://localhost:8008/1.0/check.json' \
  -F 'media=@/path/to/file.jpg' \
  -F 'models=genai' \
  -F 'api_user=anything' \
  -F 'api_secret=anything'
```

By URL:

```bash
curl -G 'http://localhost:8008/1.0/check.json' \
  --data-urlencode 'url=https://example.com/image.jpg' \
  -d 'models=genai' -d 'api_user=anything' -d 'api_secret=anything'
```

`api_user` / `api_secret` are accepted but ignored — there's no auth
locally. `models` must include `genai`; nothing else is implemented.

### Response shape — image input

```json
{
  "status": "success",
  "request": {"id": "req_...", "timestamp": 1234567890.1, "operations": 1},
  "type": {
    "ai_generated": 0.73,
    "signals": {
      "frequency": {"score": 0.6, "confidence": "medium", "...": "..."},
      "noise": {"score": 0.8, "confidence": "medium", "...": "..."},
      "gradients": {"score": 0.7, "confidence": "medium", "...": "..."},
      "lighting": {"score": 0.4, "confidence": "low", "...": "..."},
      "metadata": {"score": 0.5, "confidence": "low", "...": "..."}
    }
  },
  "media": {"id": "med_...", "uri": "..."}
}
```

`ai_generated` matches Sightengine's top-level field name and range
(0-1). `signals` is an addition, not part of the real API — it's the
whole point locally: you can see *why* it scored something the way it
did, which a black-box model won't give you. There is no
`ai_generators` per-generator breakdown, because that genuinely
requires a model trained on labeled examples per generator — pure
signal analysis can't tell you *which* tool made an image, only
whether it looks synthetic. See "Extending this" below if you want to
add that later.

### Response shape — PDF input

Same envelope, but `type` carries two more fields:

```json
{
  "type": {
    "ai_generated": 0.50,
    "signals": { "...": "worst-scoring page's frequency/noise/gradients/lighting, plus the PDF's own document-level metadata check" },
    "page_count": 2,
    "worst_page": 2,
    "pages": [
      {"page": 1, "ai_generated": 0.20, "signals": {"...": "..."}},
      {"page": 2, "ai_generated": 0.50, "signals": {"...": "..."}}
    ]
  }
}
```

Each page is rasterized and run through the same four visual signals
as a standalone image. The top-level `ai_generated`/`signals` reflect
whichever page scored highest (the worst case across the document),
with its visual signals combined against the **PDF's own
document-level metadata** (Title/Author/Creator/Producer fields) —
not that page's own metadata, which is meaningless once a page has
been rasterized to a fresh PNG (see `app/analyzers/metadata.py`,
`analyze_pdf_document`). Each entry in `pages` still carries its own
full signal breakdown, including its own (mostly neutral) per-page
metadata check, for transparency.

PDFs are capped at 20 pages per request (`MAX_PAGES` in
`app/pdf_utils.py`) so one huge upload can't hang a request — anything
beyond that is silently skipped, not rejected.

## How it works

Four independent visual signals, weighted and combined (`app/main.py`,
`WEIGHTS` dict). Thresholds are in physical units (grey levels, grey
levels per pixel, log10 power) and each analyzer's docstring explains
the measured ranges they were set from.

| Signal | Base weight | What it measures |
|---|---|---|
| `noise` | 0.40 | Noise floor in flat 16x16 blocks (Immerkaer estimator). Generated images that carry grain carry a *synthetic* one: the same amplitude everywhere (stationary) and spatially white. Real floors are patchy and correlated by the camera pipeline. Not applicable when the floor was erased or an aligned JPEG 8x8 grid shows quantisation reshaped it, since compression hides synthetic grain and distorts real noise alike. |
| `frequency` | 0.25 | Hann-windowed, Welch-averaged radial power spectrum against a power-law fit. Scores a *knee*: the spectrum follows the natural power law, then collapses in the top band, which is what enlarging an image from a lower resolution leaves (generator "hires fix"/upscaler passes). Gradual optical roll-off is discounted. |
| `gradients` | 0.25 | **Portraits only.** Gradient variance (grey levels/pixel) of facial skin on a canonical 128px face crop, plus the backdrop's flat regions. Airbrushed generated skin measures ~7-10, real faces >= 23; rendered backdrops ~1, real ones >= 3. A face must be found by two independent Haar cascades; colour alone is not trusted (it matched fire, wood floors and beige walls as "skin"). |
| `lighting` | 0.10 | Weighted circular variance of local shading-gradient directions across a grid, as a cheap proxy for lighting consistency. **The weakest and noisiest signal.** Real multi-light or overcast scenes trigger it too. Its "low" confidence halves its effective weight; it's a tie-breaker. |

How the four are combined:

- Each signal reports `applicable`. A signal that doesn't apply carries
  **no weight**; it doesn't contribute a placeholder 0.5, which isn't
  neutral against the UI's 0.35 "likely real" cut-off.
- **Non-portrait photos:** when `gradients` doesn't apply, its weight is
  redistributed to `noise` and `frequency` (keeping their 0.40 : 0.25
  proportion), so those two carry the score. Before context factors
  (below), with a 0.10 lighting weight at "low" confidence, that's
  roughly noise 0.58 / frequency 0.36 / lighting 0.05 of the effective
  weight.
- Each weight is scaled by the signal's confidence (`medium` 1.0,
  `low-medium` 0.7, `low` 0.5).
- **Corroboration:** if only one of noise/frequency/gradients applies,
  the score is capped at 0.60 ("inconclusive"). One cue alone, such as
  a frequency knee that an upscaled real photo also shows, can't produce
  "likely AI".
- The response's `scoring` block (with `details=true`) shows which
  signals applied, where the gradients weight went and the effective
  weights used.
- Noise and frequency run on a native-resolution centre crop (up to
  2048px). Downscaling would average away the grain and hide the knee.
  Gradients and lighting run on a copy capped at 1536px.

### Context-dependent weighting

How far each signal can be trusted depends on what happened to the
image. JPEG quantisation attenuates the top of the spectrum and smooths
skin; upscaling smooths real skin too and correlates the noise floor. So
each signal's weight is also multiplied by a factor for the context the
image was measured in (`app/context.py`, `app/weighting.py`). The
context is read from the file, never from its name:

| Dimension | Buckets | Read from |
|---|---|---|
| `compression` | `lossless`, `jpeg_history` (PNG carrying a JPEG 8x8 grid), `jpeg_high` (q >= 90), `jpeg_mid` (75-89), `jpeg_low` (< 75) | magic bytes; JPEG quality estimated from the quantisation tables stored in the file; JPEG grid in the pixels |
| `scale` | `native`, `upscaled` | the frequency analyzer's high-band knee |
| `size` | `small` (< 0.35 MP), `medium`, `large` (>= 1 MP) | decoded dimensions |
| `color` | `color`, `grayscale` | decoded channels |

A factor below 1 is applied as an *absolute* discount: a signal
weighted at 0.5x in this context contributes half its usual pull, even
when it's the only signal left. It isn't renormalised back to full
strength. The share of evidence discounted this way is reported as
`scoring.context_discount`, and `scoring.context` /
`scoring.context_multipliers` show what was detected and applied
(`details=true`).

The factors are not hand-set. `eval/calibrate_weights.py` builds a
synthetic degradation batch from labeled sources: every source rescaled
0.5x-2x and saved as PNG / JPEG 95-60, plus crops, grayscale,
resize-after-JPEG and double JPEG. It then measures, per context
bucket, how often each signal's tell *survives* the degradation on
generated sources and how often it *false-alarms* on genuine ones. It
validates leave-one-source-out and only writes dynamic factors
(`app/weighting_calibration.json`) if they don't do worse than static
weights on held-out sources. The full numbers are in
[`eval/REPORT.md`](eval/REPORT.md). Current state, from 3 generated and
16 genuine sources (684 variants):

- **Adopted: compression + scale factors.** For example, frequency gets
  1.12x on lossless, 0.99x on JPEG q75-94 and 0.83x below q75; noise
  0.86x under JPEG; gradients 0.88x on upscaled images, where upscaling
  smooths genuine skin.
- **It is not a measured improvement.** On held-out sources, balanced
  accuracy is 69.0% vs 68.5% for static weights (95% CI of the
  difference -0.3 to +1.3 points), with the same 9.2% false-positive
  rate. It was adopted because it's non-inferior and the factors follow
  the physics, not because it wins.
- **Size and colour are not fitted.** In the available sources they're
  confounded with the label: the genuine library photos are small and
  partly grayscale, the generated ones ~1 MP colour, so megapixels alone
  "detect" AI at AUC 0.78. Factors fitted on them would learn that.
  Adding large colour genuine photos (`--real-dir`) fixes this
  automatically.
- **Which signal indicates AI is deliberately not learned.** Each of the
  three generated sources is caught by a different signal. Any model
  that learned it (tried: likelihood-ratio weighting, a logistic
  regression) inverted on the held-out generator; the regression fell
  from AUC 0.81 in-sample to 0.05 held-out. Only degradation behaviour,
  which transfers between generators, is fitted.
- Along JPEG quality the fit is constrained so a signal never becomes
  *more* trustworthy as quality drops. Without that, sampling noise
  produced inverted factors (0.74x at q >= 90 but 1.31x at q75-89).

To recalibrate after changing an analyzer, or with your own labeled
images (the most valuable improvement: more generated sources):

```bash
python eval/calibrate_weights.py --ai-dir path/to/generated --real-dir path/to/genuine
```

Deleting `app/weighting_calibration.json` restores static weights.

`metadata` (in `app/analyzers/metadata.py`) is handled separately: if
it finds an explicit generator fingerprint (Stable Diffusion WebUI's
`parameters` PNG chunk, ComfyUI's `workflow`/`prompt` chunks, an EXIF
`Software` tag, or — for PDFs — a generator name in the document's
Title/Author/Creator/Producer fields), that **overrides** the combined
score toward high confidence — it's about as close to ground truth as
this gets, when present. Its absence proves nothing (trivial to
strip), so it doesn't pull the score down. Plausible camera EXIF
(Make/Model/FNumber/etc., image input only) pulls the score down
moderately. No metadata either way is treated as neutral.

For a PDF, `app/pdf_utils.py` rasterizes each page to a PNG (via
`pypdfium2`, which bundles its own PDFium binary — no system
`poppler` install needed) and extracts every embedded raster image.
Page renders (when the page is mostly raster) and embedded images go
through the same signals *and the same weighting* as a standalone image.
An embedded image gets the score it would get as an upload, and the PDF
layer only decides what counts as reliable evidence (see Limitations).
Each element reports a confidence tier from how much of the evidence
stack was measured: the sum of base weight × confidence factor over the
usable signals. It's `medium-high` at >= 0.65 (e.g. noise + frequency),
`medium` at >= 0.40 (e.g. noise alone), `low-medium` at >= 0.20 (e.g. a
faceless JPEG photo: frequency + lighting) and `low` below that.
Elements below 0.15 are unusable, and only `low-medium` or better can
drive the document score.

Each analyzer lives in its own file under `app/analyzers/` with a
docstring explaining the reasoning — read those before tuning
anything.

## Limitations (read this before trusting a score)

- **This is heuristics, not a trained model.** Sightengine trained on
  millions of labeled real/AI images per generator. These signals are
  physically motivated but hand-tuned, not fit to data. Expect a
  meaningfully higher false-positive/false-negative rate, especially
  on borderline cases.
- **JPEG re-compression, resizing, and screenshots remove evidence.**
  They don't manufacture it: compression makes `noise` step aside
  rather than read "too clean" as AI. But once the evidence is gone,
  nothing can recover it. Measured on the baseline set: a generated
  portrait re-saved as JPEG still scores ~0.55 (skin/backdrop texture
  survives), and the fireplace scene ~0.60 (the frequency knee
  survives), but `ai-retouched.png` re-saved as JPEG drops to ~0.07 and
  the fireplace downscaled by half to ~0.02. Across the whole synthetic
  degradation batch (`eval/REPORT.md`) the generated portrait is flagged
  in 92% of its variants, but the fireplace and the retouched photo in
  only 22-25%; genuine photos are flagged in 9% of theirs, mostly
  upscaled ones.
- **Known false positives.** Film scans and some high-ISO photos carry
  stationary, white grain that `noise` can't tell from synthetic grain
  (skimage's `astronaut.png`, a scanned NASA portrait, scores ~0.50).
  Real photos that were *upscaled* trip `frequency` (a 1.41x bicubic
  upscale of `human.jpeg` scores ~0.44).
- **PDF pages inherit the same limitations.** The PDF layer avoids
  adding its own resampling:
  - Embedded images are extracted as their own decoded pixels (not drawn
    through their placement transform). A JPEG stored as DCTDecode is
    analyzed exactly as that JPEG file would be, including its EXIF.
  - Photo pages are rendered at the dominant image's native pixel density
    with image smoothing off. The render is only used as pixel evidence
    if it preserves the image's fine detail (within 3% along both axes).
  - Stretched, rotated or clamped placements, where no render can be 1:1,
    don't feed frequency/noise/gradients from the render. Those signals
    would measure the renderer: an earlier fixed 200 DPI smoothed render
    made a genuine photo page read as "upscaled" and flagged it.
  - The embedded image's own pixels are still scored in those cases.
- **PDF rasters are scored like uploads, with PDF-specific gates on
  top.** Page renders and embedded images go through the same combiner as
  a direct upload (`app/scoring.py`): same weights, confidence factors,
  no-portrait redistribution, single-signal cap and context factors. The
  JPEG quality of an embedded DCTDecode stream is read from its own
  quantisation tables. The same JPEG therefore scores identically either
  way (tested end to end in `tests/test_pdf_image_parity.py`). The PDF
  path only adds: which signals may run on a rendered page (content type,
  image-area fraction), that flat/gradient fills and icon-sized rasters
  are not evidence, that an element with no substantive signal is
  "insufficient" (score 0) rather than 0.5 "inconclusive", and the
  confidence tiers that decide what may drive the document verdict.
  Metadata follows the upload rule too:
  - The embedded JPEG's own EXIF and the document's Info dictionary are
    folded in with `scoring.combine`: a generator signature raises the
    score to at least its own value and never lowers stronger visual
    evidence.
  - Every provenance check (PNG chunks, EXIF Software, PDF Info dict)
    uses one term list and one whole-token matcher
    (`document.AI_GENERATOR_TERMS`). Ordinary words like "flux"
    (French: cash flow), "dalle", "firefly", "reflux" or "InfluxData"
    are not read as generator names.
- **No per-generator identification.** You get "does this look
  synthetic," not "was this MidJourney vs Flux."
- **`lighting` is a genuinely weak signal** on its own; it's there as
  a minor corroborating factor, not evidence.
- **Thresholds were set from a small set of measurements, not fit to a
  labeled dataset.** They came from the three generated baseline images,
  `human.jpeg`, fifteen genuine photos bundled with
  scikit-image/scikit-learn/matplotlib (most non-portrait, several
  lossless PNG so format can't stand in for authenticity), and
  re-encoded/resized/upscaled variants of each. On that set every
  generated baseline image scores 0.59-0.73 and every genuine photo
  0.00-0.18, except `astronaut.png` above. Moving any individual
  threshold by +/-20% changes none of those outcomes. That is far too
  few generated images to claim a general accuracy: three images from
  unknown generators can't represent the space. If you have real/AI
  labeled images, the highest-value next step is to run a batch through
  and adjust the constants at the top of each analyzer until the ROC
  curve looks right for your use case.
- **Adversarial robustness is not a goal here.** Anyone who knows
  these are the checks can add noise, blur, or re-compress to defeat
  them specifically.

## Extending this

- **Tune against real data**: collect a folder of real photos and a
  folder of known-AI images (SD WebUI output keeps its metadata by
  default, so that's an easy labeled source), run them through
  `/1.0/check.json`, and adjust the per-analyzer thresholds based on
  where real scores and AI scores actually land.
- **Add a learned layer**: the per-signal breakdown already gives you
  a small feature vector (5 scores + the raw diagnostics). A simple
  logistic regression or gradient-boosted tree trained on that vector
  over a labeled dataset would likely outperform the hand-set weights
  in `WEIGHTS` with very little data, and is a much smaller lift than
  training an image classifier from scratch.
- **Per-generator ID** genuinely needs a CNN trained on labeled
  examples per generator (fingerprints differ by architecture/decoder).
  Out of scope for a heuristics-only approach.
- **Embedded-image extraction for PDFs**: right now every page is
  rasterized whole. Extracting the original embedded raster images
  instead (e.g. via `pikepdf`) would preserve their original bytes —
  and therefore their original EXIF/PNG-chunk metadata — at the cost
  of missing vector-drawn or composited content that whole-page
  rendering catches.

## Known deviations from the real Sightengine API

- Missing `models` param returns FastAPI's default 422 validation
  error rather than Sightengine's error JSON shape.
- No `operations`/billing tracking, no auth, no rate limiting.
- No video/audio/text endpoints, no other moderation models.
- `pages`/`page_count`/`worst_page` (PDF input) are a local addition;
  the real API doesn't take PDFs at all.
