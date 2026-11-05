"use client";

import { useRef, useState } from "react";

type Tab = "file" | "url";

const ACCEPTED_TYPES = "image/png,image/jpeg,application/pdf";

function isPdf(file: File): boolean {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Single-row upload control: [File|URL] [drop strip / url input] [Run scan].
 * Deliberately no thumbnail — the per-stage analysis visuals show the
 * media once a scan starts, so a preview here would only cost space.
 */
export function UploadPanel({
  loading,
  onSubmit,
}: {
  loading: boolean;
  onSubmit: (formData: FormData) => void;
}) {
  const [tab, setTab] = useState<Tab>("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function pickFile(f: File | null) {
    setFile(f);
    // Reset the native input so re-picking the same file fires onChange.
    if (!f && inputRef.current) inputRef.current.value = "";
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
    <form onSubmit={handleSubmit} className="w-full" data-testid="upload-panel">
      <div className="flex flex-wrap items-stretch gap-2">
        {/* Mode toggle */}
        <div
          role="tablist"
          aria-label="Input type"
          className="flex shrink-0 rounded-lg border border-line bg-panel p-0.5 font-mono text-[11px] uppercase tracking-[0.1em]"
        >
          {(["file", "url"] as Tab[]).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`rounded-md px-2.5 transition-colors ${
                tab === t ? "bg-panel-raised text-paper" : "text-dimmer hover:text-dim"
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
            aria-label={file ? `Selected ${file.name}. Click to choose a different file` : "Choose a file"}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            className={`flex h-10 min-w-[14rem] flex-1 cursor-pointer items-center gap-2.5 rounded-lg border border-dashed px-3 text-sm transition-colors ${
              dragActive ? "border-signal-real bg-panel-raised" : "border-line bg-panel hover:border-dim"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_TYPES}
              className="hidden"
              onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
            />
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="shrink-0 text-dim" aria-hidden="true">
              <path d="M12 16V4m0 0 4 4m-4-4-4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>

            {file ? (
              <>
                <span className="min-w-0 flex-1 truncate text-paper" title={file.name}>
                  {file.name}
                </span>
                <span className="hidden shrink-0 font-mono text-[10px] uppercase tracking-[0.08em] text-dimmer sm:inline">
                  {isPdf(file) ? "PDF · all pages" : "Image"} · {formatSize(file.size)}
                </span>
                <button
                  type="button"
                  aria-label="Clear selected file"
                  onClick={(e) => {
                    e.stopPropagation();
                    pickFile(null);
                  }}
                  className="shrink-0 rounded px-1 font-mono text-xs text-dimmer transition-colors hover:text-paper"
                >
                  ✕
                </button>
              </>
            ) : (
              <>
                <span className="min-w-0 flex-1 truncate text-paper">Drop a file or click to browse</span>
                <span className="hidden shrink-0 font-mono text-[10px] tracking-[0.08em] text-dimmer sm:inline">PNG · JPG · PDF</span>
              </>
            )}
          </div>
        ) : (
          <input
            type="url"
            inputMode="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/image.jpg"
            aria-label="Image URL"
            className="h-10 min-w-[14rem] flex-1 rounded-lg border border-line bg-panel px-3 text-sm text-paper placeholder:text-dimmer focus:border-signal-real"
          />
        )}

        <button
          type="submit"
          disabled={!canSubmit || loading}
          className="h-10 flex-1 rounded-lg bg-paper px-4 font-mono text-xs uppercase tracking-[0.15em] text-ink transition-opacity disabled:cursor-not-allowed disabled:opacity-30 sm:flex-none"
        >
          {loading ? "Scanning…" : "Run scan"}
        </button>
      </div>
    </form>
  );
}
