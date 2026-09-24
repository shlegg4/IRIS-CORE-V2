# Preview sink protocol

`OutputStage` retains its recording path and forwards each completed packet to `PreviewSink`.
Preview publication uses a bounded `DropOldest` queue, so a preview consumer cannot delay
capture, pose processing, or disk recording. A `PreviewPacket` is a `shared_ptr<const Packet>`:
the packet and every GPU buffer owner therefore remain alive until the asynchronous transport
has completed its use of the frame.

## Shared memory

When legacy compatibility is enabled, the existing v1 mapping remains at the configured
destination. Version 2 is always published at `<destination>_v2`. Both mappings start with an
odd/even sequence lock followed by packet sequence and payload size. Readers must read the lock,
copy the payload, then re-read the lock; a packet is valid only if both values match and are even.

The v2 payload begins with little-endian magic `IRS2` (`0x32535249`) and protocol version `2`,
then payload size, packet sequence, frame count, and pose count. Each frame encodes camera ID,
frame sequence, dimensions, pixel format, stride, byte size, CUDA device ID, source timestamp,
and a CUDA IPC handle. Pose records currently contain `sourceSequence`. Fields are written
individually; readers must not interpret C++ object layouts as wire data.

Preview faults are recorded in transport health and are non-fatal. Start and stop are idempotent;
stop closes the queue and joins the consumer before releasing mappings.

## Browser transports

Enable the local preview server before starting the pipeline with `preview enable [port]`.
It binds only to loopback (default `127.0.0.1:8080`). Each available camera is exposed at
`/api/preview/<cameraId>.mjpeg` as a multipart MJPEG response. Frames are synchronized on their
CUDA readiness event, rate limited to 10 FPS, resized to 960 pixels maximum width, and encoded
with nvJPEG quality 75. `/api/preview/pose-events` upgrades to a WebSocket and sends version-1
JSON envelopes for pose events only. Runtime and calibration status are available from the
request-based REST API. `/api/events` remains a compatibility alias and also sends pose events
only.

The Electron H.264 preview uses `/api/preview/stream`. The client sends a version-1
JSON `hello` containing requested camera IDs. The server replies with `config`, then
sends binary little-endian `IRWS` access-unit envelopes containing flags, camera ID,
sequence, timestamp in microseconds, payload length, and an H.264 access unit. Bit 0
marks keyframes and bit 1 marks discontinuities. The renderer decodes each camera
independently with WebCodecs `VideoDecoder`; bounded queues drop stale preview data so
a slow renderer cannot block capture, inference, or recording.
