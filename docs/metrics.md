# Metrics

Metrics are registered once and updated by lightweight handles. `Runtime` takes periodic snapshots on a separate thread and writes `iris_metrics.json`.

Capture exposes sample outcomes, queue drops/depth, pool exhaustion, decode/source failures, clock drift/residuals, inter-frame timing, queue wait, decode submission and capture-to-emit latency. Names are stable and labels are intentionally absent until a bounded multi-camera label model is introduced.

Channels own their transport metrics directly. Instrumented channels report sent, received, aggregate drops, drop policy outcomes, rejected sends to closed channels, current depth and peak depth under an `iris_channel_<name>_*` prefix. Stages do not duplicate queue depth or drop accounting.
