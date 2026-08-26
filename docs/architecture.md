# IRIS V2 architecture

`Runtime` owns shared infrastructure and `Pipeline`; `Pipeline` owns topology; stages own domain processing only.

```mermaid
flowchart LR
  R[Runtime] --> M[Metrics registry/exporter]
  R --> P[Pipeline]
  P --> C[CaptureStage]
  C --> MF[Media Foundation source]
  C --> CLK[Capture clock]
  C --> GPU[GPU decoder and frame pool]
  C --> CH[Channel of packets]
  CH --> POSE[PoseStage]
  POSE --> OUT[OutputStage]
```

Shared CUDA wrappers and metrics live in `infrastructure/`. Windows capture, timing and decoding remain private below `src/stages/capture/`.
