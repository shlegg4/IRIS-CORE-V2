# Synchronized video ingestion

Video ingestion is a pipeline source for recordings that already contain synchronized camera
frames. `SynchronizedVideoStage` owns one FFmpeg `VideoFileReader` per camera and advances each
reader once per batch. It sends the resulting `FrameBatch` directly to the capture-to-pose channel,
so the live `FrameSynchronizerStage` is not used.

The source decodes video frames in presentation order, converts supported FFmpeg output formats to
BGR8, and uploads them to pooled CUDA buffers. Each frame in a batch receives the same capture
time, while the packet and frame sequence are the batch index. As-fast-as-possible processing is
the default; real-time pacing follows the first file's presentation timestamps.

The files must have equal frame counts and matching order. By default, all files ending at the same
step is normal completion. If only some files end, the source fails with the camera ID and batch
index.
The video path uses blocking pipeline channels to preserve every batch under downstream load. At
normal EOF, an active IRIS recording is drained and finalized automatically.

Configure through the interactive CLI:

```text
capture video 0 camera-0.mp4 1 camera-1.mp4 2 camera-2.mp4
capture video --realtime true 0 camera-0.mp4 1 camera-1.mp4 2 camera-2.mp4
capture video --rotation 1 cw90 0 camera-0.mp4 1 camera-1.mp4
capture live
```

Video feed rotation is per camera and accepts `none`, `cw90`, `180`, or `ccw90`. The CUDA BGR
rotation runs after decoding and color conversion; quarter turns swap the reported frame width and
height. In the viewer, choose the rotation on each video feed before applying the source.

Video feeds stop at EOF by default. Set `loop: true` in the video-source request, enable “Loop feeds
continuously” in the viewer, or pass `--loop true` in the CLI to restart all selected files together.
Every replay keeps the batch sequence increasing, while the feed files must still contain the same
number of frames so each emitted batch remains synchronized.

To start the application directly in video mode, pass camera/file pairs after `--video`:

```powershell
iris_app.exe --video 0 camera-0.mp4 1 camera-1.mp4
iris_app.exe --video --rotation 1 cw90 0 camera-0.mp4 1 camera-1.mp4
iris_app.exe --video --loop true 0 camera-0.mp4 1 camera-1.mp4
iris_app.exe --api --video 0 camera-0.mp4 1 camera-1.mp4
```

The equivalent API is `POST /api/v1/video-source` with a JSON body such as:

```json
{
  "cuda_device": 0,
  "frame_pool_capacity": 8,
  "realtime": false,
  "loop": true,
  "cameras": [
    { "camera_id": 0, "path": "camera-0.mp4" },
    { "camera_id": 1, "path": "camera-1.mp4", "rotation": "cw90" }
  ]
}
```

Use `DELETE /api/v1/video-source` to select live camera capture again.
