// Run with: npm test   (node --test, no extra dependencies)
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  COMPLETION_HOLD_MS,
  extraDelayMs,
  FAST_RESPONSE_THRESHOLD_MS,
  inputKindFor,
  MIN_PIPELINE_VISIBLE_MS,
  RAMP_MS,
  SIMULATED_PROGRESS_CAP,
  simulatedProgress,
  stagesFor,
  stageViews,
} from "./scanProgress.ts";
import { VERDICT_LABEL, VERDICT_THRESHOLDS, verdictFor } from "./verdict.ts";

describe("simulatedProgress", () => {
  it("starts at 0 and never exceeds the 90% cap", () => {
    assert.equal(simulatedProgress(0), 0);
    for (const ms of [10, 500, RAMP_MS / 2, RAMP_MS, RAMP_MS * 2, 60_000]) {
      const p = simulatedProgress(ms);
      assert.ok(p >= 0 && p <= SIMULATED_PROGRESS_CAP, `progress ${p} at ${ms}ms out of range`);
    }
    assert.equal(SIMULATED_PROGRESS_CAP, 90);
  });

  it("reaches the cap exactly at RAMP_MS and holds there", () => {
    assert.equal(simulatedProgress(RAMP_MS), SIMULATED_PROGRESS_CAP);
    assert.equal(simulatedProgress(RAMP_MS + 5_000), SIMULATED_PROGRESS_CAP);
  });

  it("is monotonically non-decreasing", () => {
    let prev = -1;
    for (let ms = 0; ms <= RAMP_MS + 500; ms += 25) {
      const p = simulatedProgress(ms);
      assert.ok(p >= prev, `progress dropped from ${prev} to ${p} at ${ms}ms`);
      prev = p;
    }
  });

  it("is linear, so equal-weight stages get equal on-screen time", () => {
    assert.equal(simulatedProgress(RAMP_MS / 2), SIMULATED_PROGRESS_CAP / 2);
    assert.equal(simulatedProgress(RAMP_MS / 4), SIMULATED_PROGRESS_CAP / 4);
  });
});

describe("extraDelayMs (the <1s padding rule)", () => {
  it("pads fast responses up to the minimum visible duration", () => {
    assert.equal(extraDelayMs(0), MIN_PIPELINE_VISIBLE_MS);
    assert.equal(extraDelayMs(300), MIN_PIPELINE_VISIBLE_MS - 300);
    assert.equal(extraDelayMs(FAST_RESPONSE_THRESHOLD_MS - 1), MIN_PIPELINE_VISIBLE_MS - (FAST_RESPONSE_THRESHOLD_MS - 1));
  });

  it("adds nothing once the response took at least 1 second", () => {
    assert.equal(FAST_RESPONSE_THRESHOLD_MS, 1000);
    assert.equal(extraDelayMs(FAST_RESPONSE_THRESHOLD_MS), 0);
    assert.equal(extraDelayMs(2_500), 0);
    assert.equal(extraDelayMs(30_000), 0);
  });

  it("keeps the constants in a sensible relationship", () => {
    // The walkthrough must have time to reach the 90% hold before a padded
    // fast scan completes, and the padding must be longer than the
    // threshold that triggers it.
    assert.ok(MIN_PIPELINE_VISIBLE_MS >= RAMP_MS);
    assert.ok(MIN_PIPELINE_VISIBLE_MS > FAST_RESPONSE_THRESHOLD_MS);
    assert.ok(COMPLETION_HOLD_MS > 0);
  });
});

describe("stagesFor", () => {
  it("mirrors the backend's analyzer order for images", () => {
    const tags = stagesFor("image").map((s) => s.tag);
    assert.deepEqual(tags, ["READ", "DECODE", "FREQ", "NOISE", "GRAD", "LIGHT", "META", "SCORE"]);
  });

  it("adds rasterisation and document-integrity stages for PDFs", () => {
    const tags = stagesFor("pdf").map((s) => s.tag);
    assert.ok(tags.includes("RASTER"));
    assert.ok(tags.includes("DOC"));
    assert.ok(!tags.includes("DECODE"));
    assert.equal(tags[tags.length - 1], "SCORE");
  });

  it("starts with a fetch stage for URL input", () => {
    const tags = stagesFor("url").map((s) => s.tag);
    assert.equal(tags[0], "FETCH");
    assert.equal(tags[tags.length - 1], "SCORE");
  });

  it("uses unique tags so they can be React keys", () => {
    for (const kind of ["image", "pdf", "url"] as const) {
      const tags = stagesFor(kind).map((s) => s.tag);
      assert.equal(new Set(tags).size, tags.length, `${kind} has duplicate stage tags`);
    }
  });
});

describe("stageViews", () => {
  const stages = stagesFor("image");

  it("marks only the first stage running at 0%", () => {
    const views = stageViews(stages, 0, false);
    assert.equal(views[0].status, "running");
    assert.ok(views.slice(1).every((v) => v.status === "pending"));
  });

  it("has exactly one running stage at every point of the ramp", () => {
    for (let p = 0; p <= SIMULATED_PROGRESS_CAP; p += 0.5) {
      const running = stageViews(stages, p, false).filter((v) => v.status === "running");
      assert.equal(running.length, 1, `expected one running stage at ${p}%, got ${running.length}`);
    }
  });

  it("never lets a later stage finish before an earlier one", () => {
    for (let p = 0; p <= SIMULATED_PROGRESS_CAP; p += 1) {
      const statuses = stageViews(stages, p, false).map((v) => v.status);
      const firstNotDone = statuses.findIndex((s) => s !== "done");
      const laterDone = statuses.slice(firstNotDone + 1).includes("done");
      assert.ok(!laterDone, `out-of-order completion at ${p}%: ${statuses.join(",")}`);
    }
  });

  it("keeps the final stage running while holding at the 90% cap", () => {
    const views = stageViews(stages, SIMULATED_PROGRESS_CAP, false);
    assert.ok(views.slice(0, -1).every((v) => v.status === "done"));
    assert.equal(views[views.length - 1].status, "running");
  });

  it("marks everything done once the real response has landed", () => {
    const views = stageViews(stages, 100, true);
    assert.ok(views.every((v) => v.status === "done"));
    // Even if progress were somehow still low, `complete` wins.
    assert.ok(stageViews(stages, 12, true).every((v) => v.status === "done"));
  });
});

describe("inputKindFor", () => {
  it("detects PDFs by MIME type or extension", () => {
    const byType = new FormData();
    byType.set("media", new File([new Uint8Array([0x25, 0x50, 0x44, 0x46])], "scan.bin", { type: "application/pdf" }));
    assert.equal(inputKindFor(byType), "pdf");

    const byExt = new FormData();
    byExt.set("media", new File([new Uint8Array([1])], "Report.PDF", { type: "" }));
    assert.equal(inputKindFor(byExt), "pdf");
  });

  it("treats other files as images and non-files as URL input", () => {
    const img = new FormData();
    img.set("media", new File([new Uint8Array([1])], "photo.jpg", { type: "image/jpeg" }));
    assert.equal(inputKindFor(img), "image");

    const url = new FormData();
    url.set("url", "https://example.com/a.png");
    assert.equal(inputKindFor(url), "url");
  });
});

describe("verdict bands", () => {
  it("phrases labels against the not-true-doc framing", () => {
    assert.match(VERDICT_LABEL.real, /true doc/i);
    assert.match(VERDICT_LABEL.ai, /not a true doc/i);
  });

  it("maps scores to bands at the exported thresholds", () => {
    assert.equal(verdictFor(0), "real");
    assert.equal(verdictFor(VERDICT_THRESHOLDS.REAL_MAX - 0.001), "real");
    assert.equal(verdictFor(VERDICT_THRESHOLDS.REAL_MAX), "inconclusive");
    assert.equal(verdictFor(VERDICT_THRESHOLDS.AI_MIN), "inconclusive");
    assert.equal(verdictFor(VERDICT_THRESHOLDS.AI_MIN + 0.001), "ai");
    assert.equal(verdictFor(1), "ai");
  });
});

describe("per-signal verdicts (words, not numbers)", () => {
  it("maps null/NaN signal scores to 'na' and numbers to the shared bands", async () => {
    const { signalVerdictFor, SIGNAL_VERDICT_LABEL, LIKELIHOOD_LEVEL } = await import("./verdict.ts");
    assert.equal(signalVerdictFor(null), "na");
    assert.equal(signalVerdictFor(undefined), "na");
    assert.equal(signalVerdictFor(Number.NaN), "na");
    assert.equal(signalVerdictFor(0.1), "real");
    assert.equal(signalVerdictFor(0.5), "inconclusive");
    assert.equal(signalVerdictFor(0.95), "ai");
    // Every verdict has a human label and the headline has a qualitative level.
    for (const v of ["real", "inconclusive", "ai", "na"] as const) assert.ok(SIGNAL_VERDICT_LABEL[v].length > 0);
    assert.deepEqual(Object.values(LIKELIHOOD_LEVEL), ["Low", "Moderate", "High"]);
  });
});

describe("analysis effects (per-step pixel visuals)", () => {
  it("defines an effect for every pipeline stage of every input kind", async () => {
    const { ANALYSIS_EFFECTS } = await import("./analysisEffects.ts");
    for (const kind of ["image", "pdf", "url"] as const) {
      for (const stage of stagesFor(kind)) {
        assert.ok(ANALYSIS_EFFECTS[stage.tag], `missing effect for stage ${stage.tag} (${kind})`);
        assert.ok(ANALYSIS_EFFECTS[stage.tag].caption.length > 0);
      }
    }
  });

  it("falls back gracefully for unknown tags", async () => {
    const { effectFor } = await import("./analysisEffects.ts");
    assert.equal(effectFor(undefined).filter, "none");
    assert.equal(effectFor("NOPE").overlay, null);
    assert.equal(effectFor("GRAD").filter.includes("url(#"), true);
  });
});
