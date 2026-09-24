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

For a hosted RTMO 2-D / multiview deployment, use the lean Release preset:

```powershell
cmake --preset hosted
cmake --build --preset hosted
```

This preset disables LibTorch/PEAR monocular inference, tests, and diagnostic tools while retaining
the TensorRT engines and runtime. The resulting `build/hosted/bin` directory is the deployment
bundle; `pose monocular` is rejected by this build. It still requires a compatible NVIDIA driver.

### Per-host TensorRT engines

IRIS can use engines compiled for the target GPU. Keep the TensorRT SDK available on the machine
that prepares the deployment, then:

1. The engine builder requires the RTMO candidate ONNX and the DA3 four-view, 504x504 ONNX graph
   plus its `.onnx.data` sidecar. If these files are missing, it downloads them from the private
   `shlegg4/iris-models` Hub repository. Run `hf auth login` first. Source checkpoints and model
   provenance are documented in `models/source/README.md`; exporting the DA3 graph remains an
   external step.
2. Build and cache both engines for the current GPU and TensorRT version by running the VS Code
   task `IRIS: build TensorRT engines`, or directly:

   ```powershell
   ./tools/deployment/build-host-engines.ps1 `
     -TensorRtRoot C:/TensorRT-10.10.0.31 `
     -CacheRoot models/cache
   ```

The builder uses the Hub's `main` revision by default when downloading missing inputs. Pass
`-ModelRevision <commit-hash>` or set `IRIS_MODEL_REVISION` in the VS Code task environment to pin
a specific model upload for a reproducible engine build.

The helper keys its cache by GPU, compute capability, TensorRT version, and both ONNX hashes. It
smoke-runs each built engine under `models/cache/<GPU>_sm<capability>_TensorRT<version>/` and
updates `models/cache/current.txt`; CMake copies the selected cached engines into the app's
`assets` directory. The TensorRT builder SDK is required during generation; the final app folder
uses only TensorRT inference runtime DLLs.
RTMO uses a fixed three-image profile; DA3 uses four 504x504 views for rig calibration. Engine
generation on a new host takes several minutes and requires enough GPU memory for the DA3 build.

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
capture video [--cuda-device <n>] [--frame-pool <n>] [--realtime <true|false>] [--loop <true|false>] [--rotation <camera-id> <none|cw90|180|ccw90>] <camera-id> <file> [<camera-id> <file> ...]
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
Add `--rotation <camera-id> <none|cw90|180|ccw90>` to either video CLI form to rotate a specific
feed; the viewer also provides a rotation selector for each video feed.
Use `--loop true` to restart all files together after they reach EOF. The viewer exposes the same
option under Video processing options.

For unattended inference runs, direct `--video` startup accepts `--engine`, `--calibration`, and
`--output-dir`; it runs multiview pose to EOF and writes `poses.jsonl` plus `run_summary.json`.
The CMU Panoptic runner converts the native calibration file and evaluates predictions against the
sequence ground truth. See [docs/panoptic-evaluation.md](docs/panoptic-evaluation.md).

Runtime metrics are periodically written to `iris_metrics.json`. See the [local HTTP API reference](docs/api.md)
for REST endpoints and preview protocol links. Detailed design and validation notes are under `docs/`.
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
