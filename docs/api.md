# Local HTTP API

IRIS exposes a loopback-only JSON API. The default base URL is
`http://127.0.0.1:8090/api/v1`; requests from outside the local machine are not accepted.
Unless stated otherwise, request and response bodies use JSON. Mutating commands return an
object with `status` (`applied`, `rejected`, or `failed`) and `message`; successful command
responses may also include a `snapshot` of the runtime state.

## Read runtime state

| Method and path | Purpose |
| --- | --- |
| `GET /status` | Full runtime snapshot, including capture, video input, recording, preview, metrics, calibration, and calibration-tool state. |
| `GET /metrics` | Counters, gauges, and histograms. Optional `?prefix=<name>` keeps metric names beginning with that prefix. |
| `GET /cameras` | Configured camera capture settings and live capture counters. |
| `GET /cameras/discover` | Enumerate capture devices. Available on Windows; other platforms return `501`. |
| `GET /video-source` | Current live/video input mode, video files, and decoder details. |
| `GET /recording` | Recording active state and destination path. |
| `GET /synchronizer` | Synchronization tolerance, queue capacity, and incomplete-batch policy. |
| `GET /outputs/preview` | Preview enablement, bind address, port, client counts, and errors. |
| `GET /calibration` | Current calibration extrinsics and calibration-tool state. |

The status snapshot's `calibration` is `null` when no effective calibration is configured.
Otherwise it has this shape:

```json
{
  "revision": 0,
  "source": "C:/data/rig-calibration.json",
  "cameras": [
    {
      "camera_id": 0,
      "R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
      "t_w2c": [0, 0, 0]
    }
  ]
}
```

`R_w2c` is a row-major 3×3 world-to-camera rotation matrix; `t_w2c` is the corresponding
three-element translation. The source identifies the active calibration source. A rig
calibration generated at runtime has a nonzero revision; calibration loaded from pose
configuration may use revision `0`.

`GET /calibration` returns `{ "status", "message", "calibration", "calibration_tool" }`.
`calibration_tool` contains `state`, `message`, and `source_sequence`. The same fields are
included in `GET /status`, which is the normal way for clients to poll current state.

## Runtime commands

| Method and path | Body / behavior |
| --- | --- |
| `POST /pipeline/start` | Start the pipeline. No body. |
| `POST /pipeline/stop` | Stop the pipeline. No body. |
| `POST /shutdown` | Request runtime shutdown. No body. |
| `POST /cameras` | Add a camera. Requires `camera_id`; accepts capture settings described below. |
| `PATCH /cameras/{camera_id}` | Update selected capture settings for a configured camera. |
| `DELETE /cameras/{camera_id}` | Remove a camera. |
| `PATCH /pose` | Set pose backend and related paths. |
| `POST /calibration/start` | Start rig calibration; optional body `{ "output_path": "rig-calibration.json" }`. |
| `POST /calibration/cancel` | Cancel rig calibration. No body. |
| `POST /calibration/clear` | Clear the live rig calibration. No body. |
| `POST /recording/start` | Start recording. Requires `destination`; optional `bitrate` and `frame_rate`. |
| `POST /recording/stop` | Stop recording. No body. |
| `POST /video-source` | Select prerecorded video input; see [Video ingestion](video-ingestion.md). |
| `DELETE /video-source` | Switch back to live capture. |
| `PATCH /synchronizer` | Update synchronization settings. |
| `PATCH /outputs/preview` | Update preview transport settings. |
| `PATCH /outputs/shared-memory` | Configure shared-memory output. |

Camera capture settings include `device_symbolic_link`, `device_index`, `width`, `height`,
`frame_rate`, `format`, `cuda_device`, `sample_queue_capacity`, `frame_pool_capacity`,
`overflow`, `rotation`, `allow_format_fallback`, and `reconnect`. Frame rate can be an integer
or `{ "numerator": 30, "denominator": 1 }`. Supported pixel formats are `mjpeg`, `yuy2`, and
`bgra8`; overflow policies are `block`, `drop-oldest`, and `drop-newest`; rotations are
`none`, `cw90`, `180`, and `ccw90`.

Pose configuration requires a `backend` of `off`, `monocular`, `2d`, or `multiview`. Optional
fields are `model_path`, `engine_path`, and `calibration_path`.

Synchronizer settings are `tolerance_ms`, `queue_capacity`, and
`incomplete_batch_policy` (`drop` or `partial`). Preview settings include `http_enabled`,
`mjpeg_enabled`, `h264_enabled`, `bind_address`, `port`, `max_fps`, `max_width`,
`jpeg_quality`, `bitrate`, and `queue_capacity`. Shared-memory settings include `enabled`,
`destination`, `capacity_bytes`, and `legacy_v1`.

## Capture snapshot batch

`POST /capture/snapshot` captures the next synchronized frame batch for exactly four cameras.
The pipeline must be running. The output directory must be absolute and must not contain
existing files.

```json
{
  "session_id": "session-01",
  "batch_id": "batch-001",
  "orientation_degrees": 90,
  "camera_ids": [0, 1, 2, 3],
  "output_dir": "C:/data/session-01/batch-001"
}
```

Supported orientations are `0`, `45`, `90`, and `135`. A successful response includes
`packet_sequence`, `synchronization_skew_ms`, `pose_path`, pose metadata, and one BMP file
record per camera. The output directory is committed atomically after all images and pose
metadata are written.

## Errors

Malformed or invalid requests return `400` with `{ "status": "rejected", "message": "…" }`.
Unknown routes return `404`. Runtime command rejection returns `422`; command failure returns
`500`. Snapshot capture can return `409` for inconsistent or already-used output data and
`503` if capture is stopped or the next synchronized batch times out.

## Preview transports

The image and event transports use the preview server on `127.0.0.1:8080`, separate from the
REST API. MJPEG, pose-event WebSocket, H.264 WebSocket, and compatibility behavior are
documented in [Preview sink protocol](preview-sink.md).
