# Pose timing breakdown

The displayed pose processing duration covers frame readiness, preprocessing,
TensorRT, output download/wait, and CPU selection/triangulation. It is not an
engine-only benchmark. All breakdown metrics have a histogram
`iris_pose_<name>_ms` and a latest-sample gauge `iris_pose_last_<name>_ms`.

| Name | Boundary |
| --- | --- |
| frame_ready_wait | Sum of CPU waits on camera readiness events |
| preprocess_host | CPU calibration calculation, metadata upload submission and preprocess launch |
| trt_setup_host | Input shape and tensor address setup |
| trt_enqueue_host | Host duration of enqueueV3 |
| download_host | Five output copy API calls and final timing-event submission; pageable copies can block |
| result_wait_host | Final stream synchronization; includes any remaining upstream work |
| engine_call_host | Entire engine wrapper call, including timing instrumentation |
| postprocess_cpu | Candidate selection, coordinate mapping, triangulation and packet result assignment |
| preprocess_stream | CUDA event interval around metadata uploads and preprocessing |
| engine_stream | CUDA event interval around TensorRT enqueue |
| download_stream | CUDA event interval around output downloads |

Host and stream measurements overlap: do not sum them. CUDA event intervals can
include scheduling delays and host submission gaps; they are not sums of kernel
execution times. Instrumentation uses five persistent events and reads them after
the already-existing final synchronization, adding no per-stage synchronization.
The total includes small additional bookkeeping not separately attributed.
`postprocess_cpu` includes triangulation rather than isolating each SVD call.

Compare warmed-up latest values or histogram deltas, using the same engine,
precision, batch size (three), GPU workload and build type as the reference test.
The default CMake preset is Debug. Compare Release before attributing CPU time
to an architectural bottleneck.

Optimization candidates based on the measured dominant component:

- High postprocess CPU: Release build first, then hoist repeated projection
  matrix construction out of the joint loop.
- High enqueue/setup: bind fixed addresses/shape once and benchmark CUDA graph
  replay of inference with the existing stable tensor buffers.
- High download API time: persistent pinned output buffers; assess combining
  copies. Result-wait time alone does not establish a transfer bottleneck.
- High preprocessing: keep unchanged calibration metadata on the device and
  reduce tiny upload calls; profile the kernel before changing image math.
- High frame-ready wait: express dependencies with stream waits. This avoids
  blocking the CPU but cannot eliminate unfinished upstream GPU work.
- High engine stream interval: compare an isolated batch-three engine run;
  check competing decode/preview work and graph replay before changing models.
