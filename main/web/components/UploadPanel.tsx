"use client";

import { useRef, useState } from "react";

type Tab = "file" | "url";

const ACCEPTED_TYPES = "image/png,image/jpeg,application/pdf";

function isPdf(file: File): boolean {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

export function UploadPanel({
  loading,
  onSubmit,
}: {
  loading: boolean;
  onSubmit: (formData: FormData) => void;
}) {
  const [tab, setTab] = useState<Tab>("file");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function pickFile(f: File | null) {
    setFile(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return f && !isPdf(f) ? URL.createObjectURL(f) : null;
    });
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) pickFile(dropped);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const formData = new FormData();
    if (tab === "file" && file) {
      formData.set("media", file);
    } else if (tab === "url" && url.trim()) {
      formData.set("url", url.trim());
    } else {
      return;
    }
    onSubmit(formData);
  }

  const canSubmit = (tab === "file" && !!file) || (tab === "url" && url.trim().length > 0);

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="mb-3 flex gap-1 font-mono text-xs uppercase tracking-[0.1em]">
        {(["file", "url"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`rounded px-3 py-1.5 transition-colors ${
              tab === t
                ? "bg-panel-raised text-paper"
                : "text-dimmer hover:text-dim"
            }`}
          >
            {t === "file" ? "File" : "URL"}
          </button>
        ))}
      </div>

      {tab === "file" ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          className={`flex min-h-40 cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-8 text-center transition-colors ${
            dragActive ? "border-signal-real bg-panel-raised" : "border-line bg-panel"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES}
            className="hidden"
            onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
          />
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Selected image preview"
              className="max-h-40 rounded object-contain"
            />
          ) : file && isPdf(file) ? (
            <div className="flex flex-col items-center gap-2">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-dim">
                <path
                  d="M6 2h9l5 5v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path d="M14 2v6h6" stroke="currentColor" strokeWidth="1.5" />
              </svg>
              <span className="text-sm text-paper">PDF selected — every page will be scanned</span>
            </div>
          ) : (
            <>
              <span className="text-sm text-paper">Drop a file here, or click to browse</span>
              <span className="font-mono text-[11px] text-dimmer">PNG · JPG · PDF</span>
            </>
          )}
          {file && (
            <span className="font-mono text-[11px] text-dim">{file.name}</span>
          )}
        </div>
      ) : (
        <input
          type="url"
          inputMode="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/image.jpg"
          className="w-full rounded-lg border border-line bg-panel px-4 py-3 text-sm text-paper placeholder:text-dimmer focus:border-signal-real"
        />
      )}

      <button
        type="submit"
        disabled={!canSubmit || loading}
        className="mt-4 w-full rounded-lg bg-paper py-3 font-mono text-xs uppercase tracking-[0.15em] text-ink transition-opacity disabled:cursor-not-allowed disabled:opacity-30"
      >
        {loading ? "Scanning…" : "Run scan"}
      </button>
    </form>
  );
}
