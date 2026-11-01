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
`WEIGHTS` dict):

| Signal | Weight | What it measures |
|---|---|---|
| `frequency` | 0.35 | FFT of the image, DC-centered. Looks at the radial power spectrum: real photos fall off smoothly (~power law) from center to edge; AI decoders often suppress high-frequency energy (no genuine sensor noise) or leave periodic bumps from upsampling. |
| `noise` | 0.30 | Denoise-and-subtract to estimate the sensor noise floor in flat regions. Real cameras always leave some; AI images are often suspiciously clean, or have noise that's too spatially structured to be real. |
| `gradients` | 0.25 | Sobel gradient variance inside a rough skin mask and flat-background mask. Real skin has fine texture (pores, grain); AI skin is frequently over-smoothed. |
| `lighting` | 0.10 | Weighted circular variance of local shading-gradient directions across a grid, as a cheap proxy for lighting consistency. **The weakest and noisiest signal** — real multi-light or overcast scenes trigger it too. Weighted low on purpose. |

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
`poppler` install needed) and every page goes through the same
pipeline as a standalone image.

Each analyzer lives in its own file under `app/analyzers/` with a
docstring explaining the reasoning — read those before tuning
anything.

## Limitations (read this before trusting a score)

- **This is heuristics, not a trained model.** Sightengine trained on
  millions of labeled real/AI images per generator. These signals are
  physically motivated but hand-tuned, not fit to data. Expect a
  meaningfully higher false-positive/false-negative rate, especially
  on borderline cases.
- **JPEG re-compression, resizing, and screenshots degrade all four
  signals**, especially `noise` and `frequency` — recompression adds
  its own artifacts that can look either "too clean" or "too noisy"
  depending on quality settings. If your images pass through a CDN or
  get re-saved before you see them, expect noisier scores.
- **PDF pages inherit the same limitations, plus rasterization
  itself is a re-compression step** — a real photo embedded in a PDF
  and re-rendered at 200 DPI will read somewhat "cleaner" than the
  original file would, nudging scores upward. Treat PDF-derived scores
  as slightly less reliable than a direct image upload of the same
  content.
- **No per-generator identification.** You get "does this look
  synthetic," not "was this MidJourney vs Flux."
- **`lighting` is a genuinely weak signal** on its own; it's there as
  a minor corroborating factor, not evidence.
- **Thresholds in each analyzer (e.g. the `1.2` noise-std cutoff, the
  `60.0` skin-gradient-variance cutoff) were set from first-principles
  reasoning about what "normal" looks like, not calibrated against a
  labeled dataset.** If you have real/AI-labeled images to test
  against, that's the highest-value next step — run a batch through
  and adjust the constants in each analyzer until the ROC curve looks
  right for your use case.
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
