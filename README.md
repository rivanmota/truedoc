# truedoc

Local, heuristics-based AI-image/PDF forensics detector.

## Project layout

```
truedoc/
├── main/   # the app: FastAPI backend + Next.js web UI
├── baseline/                  # sample files used to sanity-check the detector
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
  - `baseline/jpg/human.jpeg` — a real, unedited photo (expected to
    score low on `ai_generated`).
  - `baseline/png/ai-generated.png` — a fully AI-generated image
    (expected to score high).
  - `baseline/png/fireplace-generated.png` — a fully AI-generated
    non-portrait scene (expected to score high; no face, so it is
    scored by the noise and frequency signals).
  - `baseline/png/ai-retouched.png` — a real photo with AI retouching
    (expected to land in between).
  - `baseline/pdf/baseline_sample.pdf` — a plain, unremarkable PDF.
  - `baseline/pdf/anomaly_metadata_sample.pdf` — a PDF with suspicious
    document metadata, for exercising the document-anomaly checks.

## Testing it

1. **Install and run the API** (no Docker needed for a quick check):

   ```bash
   cd main
   pip install -r requirements.txt
   uvicorn app.main:app --port 8008
   ```

    ```
     -- MacOSFix: use a virtual environment                                                                                                                                              
     cd /main                                                                                                                                      
     python3 -m venv .venv                                                                                                                                                       
     source .venv/bin/activate                                                                                                                                                   
     python3 -m pip install -r requirements.txt                                                                                                                         
     uvicorn app.main:app --port 8008 
    ```

    ```
    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install -r requirements.txt
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
     -F 'media=@baseline/jpg/human.jpeg' -F 'models=genai' | python3 -m json.tool

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

