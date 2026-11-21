# TrueDoc: local AI image and PDF detector

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

## Run everything (API + UI) on Docker (optional)

Docker is only one way to run TrueDoc and is not required: see
"Local development" and "Offline desktop" below for running without Docker
and without internet.

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

This sets up (or reuses) a `.venv`, installs `requirements.txt` and
`web/node_modules` **only if they are missing**, runs `check_env.py` as a
gate, then starts the API (`uvicorn app.main:app --reload`) on
**http://localhost:8008** and the UI (`next dev`) on
**http://localhost:3000** — no containers, no image builds, with hot
reload on both sides. On Debian/Ubuntu it also installs
`libgl1`/`libglib2.0-0` with `apt-get` if they are missing (needed by
`opencv-python-headless` at import time, as in `Dockerfile`). Press
`Ctrl-C` to stop both.

`./run.sh` remains available as the Docker-based alternative.

## Offline desktop (no internet, no Docker, no GPU)

Nothing in TrueDoc needs a network, a GPU, Docker, audio, a cloud service
or Windows. The scanners are plain NumPy / OpenCV / Pillow / pypdfium2 on
the CPU. The only things that touch the internet are installing
dependencies and the optional "scan by URL" input. `tests/test_offline.py`
enforces this by blocking every socket and DNS call and scanning all the
baseline files.

```bash
./init.sh --check          # read-only: is this machine ready? (no network, installs nothing)
./init.sh --api-only       # API only, no Node needed (use curl or the scan API)
./init.sh --offline        # full run; fails with a clear message instead of reaching for the network
```

`check_env.py` verifies the Python version, every backend import, that
OpenCV can load its face detector, and Node / `web/node_modules`.

To prepare a machine that has no internet, on a machine **with** internet
(same OS, CPU architecture and Python version as the target):

```bash
./init.sh --prepare-offline   # downloads ./wheelhouse and fills web/node_modules
```

then copy the whole `main/` directory over and run `./init.sh --offline`.
When `./wheelhouse` exists, `init.sh` installs from it with `--no-index`.

What does not work offline: "scan by URL" (it returns a normal
`url_fetch_failed` error; file upload is unaffected) and fetching new
benchmark images. `eval/calibrate_weights.py` also refuses to run (and
leaves `app/weighting_calibration.json` untouched) when it finds fewer than
5 genuine sources, which is what happens without scikit-image,
scikit-learn and matplotlib; supply your own images with `--real-dir` /
`--ai-dir`.

## Run just the API

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8008
```

or with Docker:

```bash
docker build -t truedoc-api .
docker run -p 8008:8008 truedoc-api
```

## Run the tests

```bash
python3 -m pip install -r requirements.txt   # includes pytest and httpx
python3 -m pytest -q
```

The suite needs no network access (`tests/test_offline.py` fails if any
code path tries). It also exercises the reference files in `../baseline/`.

## Troubleshooting

- **Portrait analysis reports "face detector could not be loaded"** —
  the OpenCV 5.x wheels have no `cv2.CascadeClassifier` at all, which the
  gradients signal needs. `requirements.txt` pins OpenCV 4.x for that
  reason; don't upgrade `opencv-python-headless` to 5.x. Without it the
  portrait signal is skipped and scores shift, and three baseline tests
  fail. `./init.sh --check` (or `python3 check_env.py`) reports this up
  front; `pip install -r requirements.txt` fixes it.
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

## Scan API (`POST /api/v1/scans`)

The same scan the web UI runs, returned as JSON. It is served by the same
API process (port 8008) and scores every file exactly as `/1.0/check.json`
and the UI do. It only reshapes the engine's output (`app/scan_api.py`).

Every response has the same two sections:

| Section | What's in it | |
|---|---|---|
| **`summary`** | HTTP status, thresholds, final score, classification, confidence, one-line explanation, scan ID, timestamp, engine version | always present, including on errors |
| **`details`** | file metadata, engine info, every forensic signal with its measurements and explanation, how the signals were combined, and per-page results for PDFs | omitted when the request sends `details=false` |

Clients that only need the verdict can read `summary` (or request just
that with `details=false`) and ignore the rest.

### Request

Multipart form (or url-encoded form for `url`). Send **exactly one** of
`file` and `url`:

| Field | | |
|---|---|---|
| `file` | multipart file | PNG, JPEG or PDF (format is sniffed from the bytes) |
| `url` | text | public `http://` or `https://` URL of a PNG, JPEG or PDF |
| `details` | `true` / `false` (optional, default `true`) | `false` returns only `summary` |

#### Validation

Every request is checked before any analysis runs, and a failed check
returns one specific error (see [Errors](#errors)). The rules live in
`app/scan_validation.py`.

- **Body:** must be `multipart/form-data`, or
  `application/x-www-form-urlencoded` for `url` only. JSON or raw bytes
  are refused with 415.
- **Fields:**
  - Only `file`, `url` and `details` are accepted, each at most once.
  - An unknown field is an error, not silently ignored. A typo such as
    `detail=false` would otherwise quietly return the full response; the
    error suggests the right name (`media` → `file`).
  - `file` must be a file part and `url`/`details` text parts.
  - Text parts are limited to 16 KB, and a form to 10 parts.
- **Size:**
  - The body is capped at the file limit (`TRUEDOC_MAX_FILE_MB`,
    default 50) plus 64 KB for the form fields.
  - A url-encoded body, which can't carry a file, is capped at 64 KB.
  - An oversized body is refused from its `Content-Length` before
    anything is read. Without a `Content-Length` (chunked upload),
    reading stops as soon as the limit is passed, so a huge upload is
    never written to disk.
- **URL:**
  - Absolute `http(s)://`, at most 2048 characters.
  - No whitespace or control characters, no `user:password@`, and a
    valid port.
  - The host must resolve to **public** addresses only. Loopback,
    private, link-local (incl. cloud metadata `169.254.169.254`),
    carrier-grade NAT, multicast and reserved ranges are refused, so the
    server can't be used to reach itself or its network.
  - Redirects (at most 5) are re-checked hop by hop.
  - Set `TRUEDOC_ALLOW_PRIVATE_URLS=true` to scan URLs on your own
    network.
- **File content** (uploads and downloads alike):
  - Must be non-empty, and a PNG, JPEG or PDF by its bytes.
  - Images must have a header that reads cleanly (PNG chunks and
    checksums are verified), and at most `TRUEDOC_MAX_IMAGE_MEGAPIXELS`
    pixels (default 100). This is checked from the header, before any
    pixel is decoded: a few hundred bytes can declare a 400-megapixel
    image.
  - PDFs must open without a password and have at least one page.

The scan runs synchronously and the response *is* the result. Nothing is
stored (there is no database), so save the response if you need it later.
The `scan_id` identifies that response, for logs and support requests.
It can't be used to fetch the scan again.

Interactive docs with the full schema: **http://localhost:8008/docs**.

### curl examples

Run these from the repo root (the `baseline/` paths are relative to it).

```bash
# Full response (summary + details) for an image
curl -s -X POST http://localhost:8008/api/v1/scans \
  -F 'file=@baseline/jpg/real-photo.jpeg' | python3 -m json.tool

# Summary only: the critical information, nothing else
curl -s -X POST http://localhost:8008/api/v1/scans \
  -F 'file=@baseline/png/ai-generated.png' \
  -F 'details=false' | python3 -m json.tool

# Upload a PDF and save the full JSON response
curl -s -X POST http://localhost:8008/api/v1/scans \
  -F 'file=@baseline/pdf/baseline_sample.pdf' | python3 -m json.tool | tee scan.json

# Scan an image by URL (multipart or url-encoded form both work)
curl -s -X POST http://localhost:8008/api/v1/scans \
  -F 'url=https://example.com/photo.jpg' -F 'details=false'
curl -s -X POST http://localhost:8008/api/v1/scans \
  --data-urlencode 'url=https://example.com/photo.jpg'

# One row per signal from the details: score, verdict and share of the score (needs jq)
curl -s -X POST http://localhost:8008/api/v1/scans \
  -F 'file=@baseline/png/ai-generated.png' \
  | jq -r '.details.evidence[] | [.signal, .score, .verdict, .weight] | @tsv'

# An error: the body's summary.http_status matches the real status code
curl -s -w '\nHTTP %{http_code}\n' -X POST http://localhost:8008/api/v1/scans \
  -F 'file=@README.md'
```

### Response

With `details=false`, this is the whole response:

```json
{
  "summary": {
    "http_status": 200,
    "status": "completed",
    "thresholds": { "likely_real_below": 0.35, "likely_ai_above": 0.65 },
    "final_score": 0.0652,
    "classification": "likely_real",
    "label": "Likely a true doc",
    "confidence": "low-medium",
    "explanation": "Likely a true doc (low-medium confidence). Signals used: frequency, lighting. Not applicable or not used: noise, gradients.",
    "scan_id": "scn_5d17ab06e3484ba8b7da09d0a664d9f7",
    "timestamp": "2026-11-21T01:35:43.167Z",
    "engine_version": "0.3.4"
  }
}
```

By default `details` follows `summary` (trimmed here to two of the five
signals):

```json
{
  "summary": { "…": "as above" },
  "details": {
    "duration_ms": 142,
    "engine": {
      "name": "truedoc-forensics", "version": "0.3.4",
      "weighting": "static",
      "calibration_generated": "2026-11-20T20:41:00"
    },
    "file": {
      "source": "upload", "filename": "real-photo.jpeg", "url": null,
      "size_bytes": 51200, "sha256": "6751cea1…",
      "format": "jpeg", "mime_type": "image/jpeg",
      "width": 600, "height": 400, "page_count": null
    },
    "evidence": {
      "frequency": {
        "signal": "frequency", "label": "Frequency spectrum",
        "applicable": true, "used_in_score": true,
        "score": 0.0, "verdict": "likely_real", "confidence": "medium",
        "weight": 0.9026, "effect": "weighted",
        "explanation": "a spectrum that follows the natural power law and then collapses in the top band (a knee) indicates the image was enlarged …",
        "measurements": {
          "spectral_slope": -2.867, "mid_band_deviation": -0.06,
          "high_band_deviation": -0.143, "high_band_knee_drop": 0.083,
          "spectrum_tiles": "4 x 400px"
        }
      },
      "noise": {
        "signal": "noise", "label": "Sensor noise floor",
        "applicable": false, "used_in_score": false,
        "score": null, "verdict": "not_applicable", "confidence": "low",
        "weight": 0.0, "effect": "not_used",
        "explanation": "noise floor unavailable as evidence: an aligned JPEG grid shows quantisation has reshaped the grain …",
        "measurements": {
          "floor_state": "reshaped_by_jpeg", "flat_block_count": 174,
          "jpeg_grid_detected": true, "noise_sigma_median": 0.297, "…": "…"
        }
      },
      "gradients": { "…": "…" },
      "lighting":  { "…": "…" },
      "metadata":  { "…": "…" }
    },
    "scoring": {
      "visual_score": 0.0652, "metadata_effect": "neutral",
      "applicable_signals": ["frequency", "lighting"],
      "insufficient_evidence": false, "single_signal_cap_applied": false,
      "gradients_weight_redistributed_to": ["frequency"],
      "context_multipliers": { "frequency": 1.0, "noise": 1.0, "gradients": 1.0, "lighting": 1.0 },
      "context_discount": 0.0,
      "context": { "format": "jpeg", "jpeg_quality_estimate": 82, "…": "…" }
    }
  }
}
```

For a PDF, `details` also has a `document` section (see below). For an
image that key is left out.

### `summary` fields

| Field | |
|---|---|
| `http_status` | Same as the HTTP status code of the response (200 on success). |
| `status` | `completed` or `failed`. |
| `thresholds` | The classification bands, so `final_score` can be read on its own. |
| `final_score` | The engine's `ai_generated` likelihood, 0-1 (higher = more likely AI-generated/manipulated). `null` for a PDF with no scorable raster content. |
| `classification` / `label` | The verdict and its wording in the UI (table below). |
| `confidence` | How much of the evidence stack could be measured (see below). |
| `explanation` | The verdict in one paragraph: which signals were used and what decided it. |
| `scan_id`, `timestamp`, `engine_version` | Identify the response: ID, time of the scan (UTC, ISO 8601) and engine version. |

`classification` uses the same bands as `web/lib/verdict.ts`:

| `classification` | When | `label` |
|---|---|---|
| `likely_real` | `final_score` < 0.35 | Likely a true doc |
| `inconclusive` | 0.35 ≤ `final_score` ≤ 0.65 | Inconclusive |
| `likely_ai_generated` | `final_score` > 0.65 | Likely not a true doc |
| `insufficient_evidence` | PDF with no scorable raster content | Not scorable |

`confidence` is `high` | `medium-high` | `medium` | `low-medium` | `low`
| `insufficient`:

- PDFs report the engine's own document confidence.
- Images get the tier the PDF path gives an embedded image: base weight ×
  confidence factor summed over the applicable visual signals,
  `medium-high` ≥ 0.65, `medium` ≥ 0.40, `low-medium` ≥ 0.20. So the same
  picture gets the same confidence either way it's submitted.
- Explicit generator metadata makes it `high`.

### `details` fields

- **`duration_ms`**, **`engine`** (name, version, weighting model and
  calibration date) and **`file`** (source, filename/URL, size, SHA-256,
  sniffed format and MIME type, pixel size for images, page count for
  PDFs).
- **`evidence.<signal>`**, for each of `frequency`, `noise`, `gradients`,
  `lighting` and `metadata`:
  - `applicable`: whether the analyzer could take the measurement on this
    file.
  - `used_in_score`: whether the signal influenced the final score.
  - `score`: `null` when the signal doesn't apply, instead of the engine's
    0.5 placeholder.
  - `verdict`: `likely_real` | `inconclusive` | `likely_ai` | `neutral` |
    `not_applicable`, using the same bands as the summary.
  - `weight`: the signal's share of the visual score.
  - `effect`: `weighted` or `not_used` for the visual signals. Metadata
    doesn't average in: its `effect` is `override` (generator signature
    found), `pull_toward_real` (camera EXIF) or `neutral`, and its
    `weight` is `null`.
  - `explanation`: the analyzer's note.
  - `measurements`: the analyzer's raw diagnostics.
- **`scoring`** shows how the signals were combined: the visual score
  before metadata, the metadata effect, the single-signal cap, gradients
  redistribution, and the context buckets and multipliers.
- **`document`** (PDF only):
  - which page and element (`page_render` or `embedded_image`) drove the
    score;
  - the PDF engine's own finer-grained `engine_classification`;
  - flagged pages and embedded images;
  - `document_anomalies`: structural/tamper findings, kept separate from
    the AI score;
  - `pages[]`: per-page score, classification, confidence, signal
    evidence and embedded-image summaries.
  For a PDF, `evidence` is the signals of the element that drove the
  score. `metadata` there is the document's Info dictionary.

### Errors

Errors use the same `summary` section, with `error` in place of the score.
There is no `details` section:

```json
{
  "summary": {
    "http_status": 415,
    "status": "failed",
    "error": {"code": "unsupported_format", "message": "unsupported file format; supported formats: png, jpg/jpeg, pdf"},
    "scan_id": "scn_…",
    "timestamp": "…",
    "engine_version": "0.3.4"
  }
}
```

| HTTP | `error.code` | When |
|---|---|---|
| 400 | `missing_input` | no `file` and no `url` (including an empty body) |
| 400 | `ambiguous_input` | both `file` and `url` |
| 400 | `unknown_field` | a field other than `file`, `url`, `details` |
| 400 | `duplicate_field` | a field sent more than once (e.g. two files) |
| 400 | `invalid_field_type` | `file` sent as text, or `url`/`details` sent as a file |
| 400 | `invalid_parameter` | `details` is not `true`/`false` (`1`/`0`, `yes`/`no` and `on`/`off` also accepted) |
| 400 | `malformed_body` | the form can't be parsed: bad multipart, no boundary, over 10 parts, a text part over 16 KB |
| 400 | `invalid_request` | `Content-Length` header is not a number |
| 400 | `invalid_url` | not absolute `http(s)://`, over 2048 characters, whitespace/control characters, credentials or an invalid port (also checked on each redirect) |
| 400 | `url_not_allowed` | the URL's host (or a redirect's) resolves to a non-public address; see `TRUEDOC_ALLOW_PRIVATE_URLS` |
| 400 | `empty_file` | zero-byte file |
| 413 | `file_too_large` | body, upload or download over the size limit (`TRUEDOC_MAX_FILE_MB`, default 50; 64 KB for url-encoded bodies) |
| 415 | `unsupported_media_type` | body is not `multipart/form-data` or `application/x-www-form-urlencoded` |
| 415 | `unsupported_format` | the file is not a PNG, JPEG or PDF |
| 422 | `unprocessable_file` | right format, but corrupt, truncated or (PDF) without pages |
| 422 | `image_too_large` | image over `TRUEDOC_MAX_IMAGE_MEGAPIXELS` (default 100) |
| 422 | `encrypted_pdf` | PDF is password-protected |
| 502 | `url_fetch_failed` | host can't be resolved, connection failed, HTTP error status, or more than 5 redirects |
| 504 | `url_fetch_timeout` | downloading took longer than `TRUEDOC_URL_FETCH_TIMEOUT_S` (default 15) |
| 500 | `analysis_failed` | unexpected engine error (details are logged under the `scan_id`) |

### Idempotency

The same input always returns the same output. Input means the request
fields and the file's bytes; output is the whole response except:

| Field | Why it differs |
|---|---|
| `summary.scan_id` | a new ID for every scan |
| `summary.timestamp` | when the scan ran |
| `details.duration_ms` | how long it took |
| `summary.engine_version`, `details.engine.version` | a new engine version may score differently; one version never does |
| the `future_metadata_date` anomaly (PDFs) | judged against the time of the scan (see below) |

Everything else is identical between calls, byte for byte:
`final_score`, classification, confidence, every signal and measurement,
PDF pages, anomaly flags and error messages. To compare two responses,
drop those fields and compare the rest.

Scope and caveats:

- **The filename is part of the input.** It is echoed in
  `details.file.filename`, so renaming a file changes that one field.
- **URLs:**
  - The input is the bytes that were downloaded, identified by
    `details.file.sha256`. If the remote file changes, so does the
    result.
  - Network failures (host down, HTTP errors) depend on the remote side,
    but the same failure always produces the same error.
- **Future-dated PDF metadata is a deliberate exception.** A PDF whose
  metadata is dated more than a day after the scan gets a
  `future_metadata_date` anomaly.
  - The check is worth more than strict idempotency: finding the anomaly
    now but not later is expected.
  - It is judged against `summary.timestamp` (the clock is read once per
    scan). So a PDF dated next month is flagged today and not once that
    date has passed.
  - The flag, the anomaly score and classification it feeds, and the
    summary explanation's anomaly count can change as a result. Nothing
    else depends on when a scan runs.
- **"Same engine version" means the same deployment:** the same code,
  the pinned versions in `requirements.txt`, and the same weighting
  calibration (`details.engine.weighting` / `calibration_generated`).
- **Concurrent requests don't affect each other's results.**
  - Each analysis thread has its own face detector (a shared OpenCV
    cascade returned different face boxes, or crashed, under concurrent
    use).
  - PDFium, which is not thread-safe, is used by one thread at a time
    (`app/pdfium_guard.py`).

`tests/test_scan_idempotency.py` holds the guarantee in place. It covers:

- repeated scans of every baseline file, on success and error paths;
- concurrent scans, compared with sequential ones;
- fresh processes with different hash seeds;
- a pinned clock;
- that every PDF is opened under the PDFium guard.

## Use the API directly (Sightengine-compatible)

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

Four independent visual signals, weighted and combined (the
`WEIGHTS` dict in `app/scoring.py`). Thresholds are in physical units (grey levels, grey
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
15 genuine sources (648 variants):

- **Result: static weights kept.** No dynamic candidate (compression,
  scale, or both) was non-inferior on held-out sources without raising
  the false-positive rate, so `app/weighting_calibration.json` records
  static weights (no multipliers). Earlier baseline images produced a
  dynamic compression + scale fit; it does not survive the current ones.
- **The detector barely separates the two sets.** On held-out sources
  AUC is 0.55 and balanced accuracy 59.0% at the 0.35 cut-off: 27.8% of
  generated variants are flagged against 9.8% of genuine ones. The
  previous baseline pictures gave 68.9%, but they were replaced because
  their provenance could not be documented. The current ones come from a
  current generator that delivers JPEG, and none of the three originals is
  flagged (scores 0.04-0.21); they are flagged only in some rescaled
  variants (mostly upscaled 1.5x).
- **Size and colour are not fitted.** In the available sources they're
  confounded with the label: the genuine library photos are small and
  partly grayscale, the generated ones ~1 MP colour, so megapixels alone
  "detect" AI at AUC 0.78. Factors fitted on them would learn that.
  Adding large colour genuine photos (`--real-dir`) fixes this
  automatically.
- **Which signal indicates AI is deliberately not learned.** With a
  handful of generators, each generator has its own tell. On the previous
  baseline set, models that learned it (likelihood-ratio weighting, a
  logistic regression) inverted on the held-out generator; the regression
  fell from AUC 0.81 in-sample to 0.05 held-out. That experiment was not
  repeated on the current set. Only degradation behaviour, which
  transfers between generators, is fitted.
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
- **Output from a current image generator is mostly missed.** The three
  generated or retouched pictures in `../baseline/` were made with a
  current generator, which delivers JPEG (so there is no lossless grain
  for `noise` to read) and no airbrushing the `gradients` signal catches.
  None of them is flagged: they score 0.04-0.21 against the 0.35 "likely
  real" cut-off. A low score means "no synthetic tell found", not "this is
  authentic". The tests pin these three as known misses, so the day one is
  flagged the suite points at the docs to update.
- **JPEG re-compression, resizing, and screenshots remove evidence.**
  They don't manufacture it: compression makes `noise` step aside
  rather than read "too clean" as AI. But once the evidence is gone,
  nothing can recover it. Across the synthetic degradation batch
  (`eval/REPORT.md`) the generated portrait is flagged in 64% of its
  variants (mostly rescaled ones; the untouched file is not flagged), the
  fireplace in 6% and the retouched photo in 14%, while genuine photos
  are flagged in 10% of theirs, mostly upscaled ones.
- **Known false positives.** Film scans and some high-ISO photos carry
  stationary, white grain that `noise` can't tell from synthetic grain
  (skimage's `astronaut.png`, a scanned NASA portrait, scores ~0.49).
  Real photos that were *upscaled* trip `frequency`: in the synthetic
  batch (`eval/REPORT.md`) genuine variants rescaled to 1.5x are flagged
  about a third of the time, and a 1.41x bicubic upscale of one genuine
  phone JPEG scored ~0.44 in an earlier measurement. The bundled
  `real-photo.jpeg` stays low (~0.12) when upscaled the same way.
- **Re-saved JPEGs can slip past the noise gate.** The `noise` signal
  steps aside when it detects an aligned JPEG 8x8 grid. On small images
  that were JPEG-compressed twice at different qualities the grid is not
  always detected, so `noise` stays active and can read the (smoothed)
  grain as synthetic. On the bundled `real-photo.jpeg` the final score
  stays low either way, but this is a known weak spot.
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
  labeled dataset.** They came from an earlier set of generated baseline
  images (since replaced, see `../NOTICE`), `real-photo.jpeg`, fifteen
  genuine photos bundled with scikit-image/scikit-learn/matplotlib (most
  non-portrait, several lossless PNG so format can't stand in for
  authenticity), and re-encoded/resized/upscaled variants of each. The
  thresholds therefore do not transfer to the current baseline images:
  none of the three is flagged, while genuine photos score 0.00-0.18
  except `astronaut.png` (0.49). Three images from unknown generators
  can't represent the space, and that number is now too small to claim any
  general accuracy. If you have real/AI labeled images, the highest-value
  next step is to run a batch through and adjust the constants at the top
  of each analyzer until the ROC curve looks right for your use case.
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
