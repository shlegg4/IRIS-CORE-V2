# IRIS V2

IRIS V2 is a C++20/CUDA staged vision runtime. Runtime composition, pipeline transport, shared infrastructure and domain stages have explicit ownership boundaries.

```text
Runtime
├── metrics registry and exporter
└── Pipeline
    └── CaptureStage
        ├── Windows Media Foundation callback source
        ├── drift-estimating capture clock
        ├── bounded latest-sample handoff
        └── CUDA/nvJPEG decoder and frame pool
    ├── PoseStage (pass-through scaffold)
    └── OutputStage
        ├── latest-value Windows shared-memory publication
        └── no-drop GPU NVENC whole-file MP4 recording
```

## Requirements

- Windows 10/11
- Visual Studio 2022 C++ build tools
- CMake 3.24+
- Ninja
- CUDA Toolkit with nvJPEG (validated with CUDA 12.8)
- NVIDIA GPU/driver with NVENC
- Pinned prebuilt FFmpeg package (`IRIS_FFMPEG_ROOT`)

## Build and test

Run from a Visual Studio developer shell:

```powershell
cmake --preset default
cmake --build --preset default
ctest --preset unit
```

## Diagnostics

Ensure the selected CUDA Toolkit `bin` directory is on `PATH`, then run:

```powershell
./build/default/iris_list_cameras.exe
./build/default/iris_mf_probe.exe
./build/default/iris_capture_probe.exe 120
```

- `iris_list_cameras` prints friendly names and stable Media Foundation symbolic links.
- `iris_mf_probe` isolates Media Foundation negotiation and sample delivery.
- `iris_capture_probe` exercises capture, clock mapping, GPU decode, packet emission and metrics.

## Interactive control

Run `build/default/bin/iris_app.exe` or the `IRIS: run interactive` VS Code task. The engine runs
on worker threads while the main thread accepts commands:

```text
status
metrics [prefix]
pipeline start
pipeline stop
record start <file.mp4> [bitrate] [fps]
record stop
record status
shm enable <name>
shm disable
capture list
capture sync <tolerance-ms> <capacity> <drop|partial>
capture video [--cuda-device <n>] [--frame-pool <n>] [--realtime <true|false>] <camera-id> <file> [<camera-id> <file> ...]
capture live
capture add <camera-id> <device-index> [width height fps format]
capture remove <camera-id>
capture configure [--camera <id>] [--device-link <link> | --device-index <n>] [--width <n>] [--height <n>] [--fps <n|n/d>] [--format <mjpeg|yuy2|bgra8>] [--cuda-device <n>] [--sample-queue <n>] [--frame-pool <n>] [--overflow <block|drop-oldest|drop-newest>] [--rotation <none|cw90|180|ccw90>] [--allow-fallback <true|false>] [--reconnect <true|false>]
quit
```

Quote paths containing spaces. Pipeline configuration persists when a stopped graph is rebuilt.
Stopping the pipeline finalizes an active recording before shutting capture down.

`capture configure` updates only the options supplied. When capture is running, IRIS stops the
graph, rebuilds Media Foundation and GPU resources with the requested configuration, then restarts
it. If the requested camera mode cannot be opened, IRIS attempts to restore the previous pipeline.
When multiple cameras are configured, pass `--camera <id>` to target one camera. Frames are matched
using their estimated capture timestamps before PoseStage. Recording a multi-camera batch creates
one file per camera by adding `-camera-<id>` before the `.mp4` extension.

For offline reprocessing, `capture video 0 "cam 0.mp4" 1 "cam 1.mp4"` reads one presentation
frame from each file per batch and bypasses the live synchronizer. The files must contain the same
number of frames in matching order; IRIS reports an error if one reaches EOF before the others.
Video ingestion processes as fast as downstream stages allow by default. Pass `--realtime true` to
pace batches by the first file's presentation timestamps, or `capture live` to return to webcam
ingestion. Paths with spaces should be quoted. To start directly in video mode when no webcam is
available, launch `iris_app.exe --video 0 "cam 0.mp4" 1 "cam 1.mp4"`; `--api --video` and
`--non-interactive --video` accept the same camera/file pairs.

Runtime metrics are periodically written to `iris_metrics.json`. Detailed design and validation notes are under `docs/`.
They are also exposed in Prometheus format at `http://127.0.0.1:9464/metrics`. A provisioned
Prometheus and Grafana developer stack is available under `tools/observability/`.

Hardware validation is opt-in:

```powershell
cmake --preset hardware
cmake --build --preset hardware
ctest --preset hardware
ctest --preset validation
# Explicit overnight run only:
ctest --preset soak
```

## Layout

- `runtime/`: application lifecycle and shared-service ownership
- `pipeline/`: stages, packets, frames and bounded channels
- `infrastructure/gpu/`: reusable CUDA resource wrappers and frame pooling
- `infrastructure/metrics/`: process-wide metrics and export
- `stages/capture/`: Media Foundation webcam capture, synchronized FFmpeg video ingestion, clock and decoding implementation
- `stages/output/`: runtime-configurable shared-memory and whole-file disk output
- `tools/`: camera and pipeline diagnostics
- `tests/`: deterministic unit validation
