# truedoc

Local, heuristics-based AI-image/PDF forensics detector.

## Project layout

```
truedoc/
├── main/   # the app: FastAPI backend + Next.js web UI
├── baseline/                  # sample files used to sanity-check the detector
├── LICENSE / NOTICE           # Apache License 2.0
└── README.md                  # this file
```

- **`main/`** — the detector itself. It exposes a
  Sightengine-compatible `/1.0/check.json` endpoint that accepts a
  **PNG, JPEG, or PDF** and scores how likely the visual content is
  AI-generated, based on frequency/noise/gradient/lighting/metadata
  signals (for PDFs it also checks document-level metadata for
  tampering). See
  [`main/README.md`](main/README.md)
  for the full write-up of how the scoring works, its limitations, and
  the API/response shapes.
- **`baseline/`** — a small set of reference files to exercise the
  detector with, grouped by type:
  - `baseline/jpg/real-photo.jpeg` — a real, unedited photo of a coffee
    cup (expected to score low on `ai_generated`). This is scikit-image's
    `coffee` sample, CC0 / public domain by Rachel Michetti, re-saved as
    a JPEG (quality 82, no metadata).
  - `baseline/png/ai-generated.png` — a fully AI-generated portrait of a
    fictional person. **A known miss:** it scores below the "likely real"
    cut-off (see "Limitations" in `main/README.md`).
  - `baseline/png/fireplace-generated.png` — a fully AI-generated
    non-portrait scene (no face, so it is scored by the noise and
    frequency signals). Also a known miss.
  - `baseline/png/ai-retouched.png` — the CC0 coffee photo above, edited
    with an AI image tool. Also a known miss.

  The three generated/retouched pictures were made on 2026-11-20 with
  Cursor's built-in image generation tool, from text prompts (and, for the
  retouched one, the CC0 photo as reference). They show no real person and
  were saved as PNG from the tool's JPEG output. Details are in
  [`NOTICE`](NOTICE).
  - `baseline/pdf/baseline_sample.pdf` — a plain, unremarkable PDF.
  - `baseline/pdf/anomaly_metadata_sample.pdf` — a PDF with suspicious
    document metadata, for exercising the document-anomaly checks.

## Testing it

1. **Install and run the API** (no Docker needed; `./init.sh --api-only`
   in `main/` does the same in one step).
   Use a virtual environment (required on macOS and recent Debian/Ubuntu,
   where system-wide `pip install` is blocked):

   ```bash
   cd main
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   python3 -m uvicorn app.main:app --port 8008
   ```

   Or run everything (API + web UI) via Docker instead:

   ```bash
   cd main
   ./run.sh
   ```

2. **Run the baseline files through it**, from the repo root, in a
   second terminal:

   ```bash
   curl -s -X POST 'http://localhost:8008/1.0/check.json' \
     -F 'media=@baseline/jpg/real-photo.jpeg' -F 'models=genai' | python3 -m json.tool

   curl -s -X POST 'http://localhost:8008/1.0/check.json' \
     -F 'media=@baseline/png/ai-generated.png' -F 'models=genai' | python3 -m json.tool

   curl -s -X POST 'http://localhost:8008/1.0/check.json' \
     -F 'media=@baseline/png/ai-retouched.png' -F 'models=genai' | python3 -m json.tool

   curl -s -X POST 'http://localhost:8008/1.0/check.json' \
     -F 'media=@baseline/pdf/baseline_sample.pdf' -F 'models=genai' | python3 -m json.tool

   curl -s -X POST 'http://localhost:8008/1.0/check.json' \
     -F 'media=@baseline/pdf/anomaly_metadata_sample.pdf' -F 'models=genai' | python3 -m json.tool
   ```

   Check the `type.ai_generated` score for the image files, and
   `type.document_anomalies` for the PDF files, against the
   expectations listed under `baseline/` above.

3. **Or get the scan as JSON** from the scan API. The response has a
   `summary` section (HTTP status, thresholds, final score,
   classification, confidence) and a `details` section (file metadata,
   every signal's measurements and explanation). Send `details=false`
   for the summary alone. See "Scan API" in
   [`main/README.md`](main/README.md) for the response fields:

   ```bash
   curl -s -X POST 'http://localhost:8008/api/v1/scans' \
     -F 'file=@baseline/png/ai-generated.png' | python3 -m json.tool

   curl -s -X POST 'http://localhost:8008/api/v1/scans' \
     -F 'file=@baseline/png/ai-generated.png' -F 'details=false' | python3 -m json.tool

   curl -s -X POST 'http://localhost:8008/api/v1/scans' \
     -F 'url=https://example.com/photo.jpg' | python3 -m json.tool
   ```

4. **Or use the web UI** (`http://localhost:3000` after `./run.sh`) to
   drag-and-drop any file from `baseline/` and see the score plus
   per-signal breakdown visually.

## Running the automated tests

```bash
cd main
source .venv/bin/activate            # created in step 1 above
python3 -m pytest -q                 # backend: analyzers, scoring, scan API

cd web
npm install
npm test                             # UI logic tests (Node test runner)
```

The backend suite includes checks against the files in `baseline/` (for
example, the real photo must score below the "likely real" cut-off, and the
generated baseline pictures are pinned as known misses). It runs fully offline and takes about a
minute.

## Offline use and network requirements

The detector runs entirely on your machine: no cloud services, accounts,
API keys or external AI/LLM APIs are involved, and there is no database.
Internet access is only needed (a) once, to install dependencies, and
(b) for the optional "scan by URL" input, which fetches the image you point
it at. Uploading a file works fully offline. There is no GPU, audio,
Docker or Windows-only requirement; Docker is an optional packaging.

```bash
cd main
./init.sh --check        # is this machine ready? read-only, no network
./init.sh --api-only     # run the API without Node
./init.sh --offline      # full run, never reaches for the network
```

To provision a machine with no internet, run `./init.sh --prepare-offline`
on a connected machine of the same OS/CPU/Python, copy `main/` across, then
use `./init.sh --offline`. Details: "Offline desktop" in
[`main/README.md`](main/README.md).

## Tech stack

- Backend: Python 3.12+, FastAPI, Uvicorn, NumPy, OpenCV (headless),
  Pillow, pypdfium2
- Web UI: Next.js, React, Tailwind CSS, TypeScript
- Packaging: Docker / Docker Compose
- Tests: pytest (backend), Node test runner (UI)

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

