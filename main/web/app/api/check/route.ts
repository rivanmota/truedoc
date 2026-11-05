import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8008";

export async function POST(req: NextRequest) {
  let incoming: FormData;
  try {
    incoming = await req.formData();
  } catch {
    return NextResponse.json(
      {
        status: "failure",
        error: { type: "invalid_request", message: "expected form data with an image/PDF file or a url" },
      },
      { status: 400 },
    );
  }

  const media = incoming.get("media");
  const url = incoming.get("url");

  if (!media && !url) {
    return NextResponse.json(
      {
        status: "failure",
        error: { type: "invalid_request", message: "provide a file or a url" },
      },
      { status: 400 },
    );
  }

  const outgoing = new FormData();
  outgoing.set("models", "genai");
  outgoing.set("api_user", "local");
  outgoing.set("api_secret", "local");
  if (media instanceof File) {
    outgoing.set("media", media);
  } else if (typeof url === "string" && url.trim()) {
    outgoing.set("url", url.trim());
  }

  // `details=true` asks the backend for the per-signal / per-page breakdown
  // (its default response is the compact score only). The UI renders that
  // breakdown so users can see what fed the not-true-doc likelihood.
  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/1.0/check.json?details=true`, {
      method: "POST",
      body: outgoing,
    });
  } catch {
    return NextResponse.json(
      {
        status: "failure",
        error: {
          type: "backend_unreachable",
          message: `couldn't reach the detector API at ${BACKEND_URL} — is it running?`,
        },
      },
      { status: 502 },
    );
  }

  const data = await backendResponse.json();
  return NextResponse.json(data, { status: backendResponse.status });
}
