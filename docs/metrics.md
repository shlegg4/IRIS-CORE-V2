# Metrics

Metrics are registered once and updated by lightweight handles. `Runtime` takes periodic snapshots on a separate thread and writes `iris_metrics.json`.

Capture exposes sample outcomes, queue drops/depth, pool exhaustion, decode/source failures, clock drift/residuals, inter-frame timing, queue wait, decode submission and capture-to-emit latency. Names are stable and labels are intentionally absent until a bounded multi-camera label model is introduced.

Channels own their transport metrics directly. Instrumented channels report sent, received, aggregate drops, drop policy outcomes, rejected sends to closed channels, current depth and peak depth under an `iris_channel_<name>_*` prefix. Stages do not duplicate queue depth or drop accounting.

The multiview pose stage exports `iris_pose_*_stream_ms` CUDA-event measurements for preprocessing, TensorRT inference, association, selected-observation gathering, triangulation, and output copies. `iris_pose_download_stream_ms` is retained as the aggregate interval from the end of TensorRT inference through geometry uploads, postprocess kernels, and output copies; use `iris_pose_output_copy_stream_ms` for output copies alone. Host-side geometry calculation and enqueue time is reported by `iris_pose_geometry_setup_host_ms`. Stream intervals can include idle gaps while the host prepares or queues work; host and stream measurements overlap and should not be summed.

The H.264 preview path correlates pose events and encoded video access units using camera and frame
sequence. `iris_preview_pose_video_publish_abs_skew_ms` is a histogram of the absolute publication
time difference for matching frames. Its `_sum / _count` gives the mean; Prometheus
`histogram_quantile(0.5, ...)` and `histogram_quantile(0.9, ...)` give the median and p90.
`iris_preview_last_pose_video_publish_skew_ms` is the most recent signed difference: positive means
the pose event was published after its video frame, while negative means pose led video.
