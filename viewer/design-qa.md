# Viewer Design QA

**Final result: PASSED**

## Source of truth

- Selected Metrics and Settings concept: `C:\Users\Sam\.codex\attachments\9b537ab6-a8a2-42b3-a490-fce65b3a4743\image-1.png` (user-selected two-screen design)
- Existing viewer context: `C:\Users\Sam\.codex\generated_images\01a0d426-1ffc-7613-b4b2-12d34525d389\exec-f320fee0-65ce-4f00-8d8a-91f7b490e53f.png`

## Preview evidence

- Local renderer: http://localhost:5174/
- Desktop viewport: 1440 × 1024 CSS pixels, device scale factor 1.1.
- Captured and inspected the Metrics and Settings screens in the Codex in-app browser. The CUA screenshot output was inline and did not provide an exportable local PNG path.
- Metrics view included the full performance summary, capture-rate panel, timing and queue panels, detailed-metrics disclosure, and bottom runtime status row.
- Settings view included the left section rail, camera source and capture controls, camera management disclosure, pose settings, calibration actions, and lower recording/output/advanced sections in the scrollable page.
- Checked navigation between Metrics and Settings and used the Settings section rail to jump to Pose Estimation.
- Verified the Settings rail highlight tracks both the first and last sections after anchor scrolling.

## Visual and interaction findings

- The dark slate surface, restrained teal accent, compact header, thin separators, and low-noise labels follow the selected concept.
- Metrics uses the live API counters, histograms, gauges, and queue-depth series. With the standalone renderer disconnected from the IRIS runtime, it displays its waiting state and em dashes rather than fabricated values.
- Settings preserves the existing API-backed camera, pose, calibration, recording, preview, shared-memory, synchronizer, metrics, and runtime controls. Camera resolution and frame rate are grouped under Camera Source. The no-camera preview shows the 60 FPS default; capture controls remain disabled until a camera is configured.
- The detached browser preview has no Electron preload bridge. Runtime subscriptions now skip initialization when that bridge is absent, so the standalone preview can render its disconnected state without an `onLog` error.
- The Metrics document has no horizontal overflow at the captured viewport. The Settings section rail remains available while the form scrolls.
- The capture-rate plot now keeps timestamped samples for the last 60 seconds, advances the time window as new metric snapshots arrive, and shows a marker when there is only one sample. The chart clips SVG overflow at the capture-rate panel boundary.
- Rechecked the chart after the fix at a 319 × 1582 CSS-pixel responsive viewport: the capture-rate panel and SVG both report `overflow: hidden`, and the empty-state message remains inside the panel.
- No blocking layout or accessibility issue was visible at this viewport.

## Iteration history

- Replaced the earlier metrics card grid with the selected compact KPI strip, rolling capture-rate panel, pipeline timing, and queue/frame-age summaries.
- Reorganized Runtime Controls into the selected Settings rail and Camera Source form while keeping the existing runtime operations intact.
- Corrected the no-camera frame-rate display to 60 FPS and verified the selected option in the rendered page.
- Replaced the capture-rate chart's sample-count axis with a timestamped 60-second window, added the single-sample marker, and clipped graph overflow to the panel.

## Verification

- `npm run typecheck`: passed.
- `npm run build`: passed.
- `git diff --check`: passed; only line-ending normalization warnings were reported.

## Follow-up when IRIS is connected

- Confirm the capture-rate trend and timing/queue values populate from a live runtime. The standalone browser has no Electron preload connection, so this preview could only verify the empty-data state and page layout.
