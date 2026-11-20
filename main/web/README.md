# TrueDoc web UI

Next.js front end for the TrueDoc detector. Drop a PNG, JPEG or PDF (or
paste a URL), and it shows the score, the verdict and the per-signal
breakdown returned by the API in `../`.

```bash
npm install
npm run dev        # http://localhost:3000
npm test           # UI logic tests
npm run typecheck
npm run lint
```

The UI proxies uploads through `app/api/check/route.ts` to the API. It
defaults to `http://localhost:8008`; set `BACKEND_URL` to override. See
[`../README.md`](../README.md) for running the API and the full stack.

Licensed under the Apache License 2.0 (see `../../LICENSE`).
