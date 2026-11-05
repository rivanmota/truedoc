import type { Metadata } from "next";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/600.css";
import "@fontsource/space-grotesk/700.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./globals.css";
import { THEME_BOOTSTRAP_SCRIPT } from "@/lib/theme";

export const metadata: Metadata = {
  title: "AI Image Detector — local scan",
  description:
    "Local, signal-based AI-generated image detection: frequency spectrum, sensor noise, texture, lighting and metadata analysis. Supports PNG, JPEG and PDF.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    // suppressHydrationWarning: data-theme is set by the bootstrap script
    // before React hydrates, so the server-rendered <html> never matches.
    <html lang="en" className="h-full" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP_SCRIPT }} />
      </head>
      <body
        className="min-h-full flex flex-col bg-ink text-paper antialiased"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
