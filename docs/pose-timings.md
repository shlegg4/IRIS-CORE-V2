# Pose timing breakdown

The displayed pose processing duration covers frame readiness, preprocessing,
TensorRT, result copies, and host packet assembly. All breakdown metrics have a histogram
`iris_pose_<name>_ms` and a latest-sample gauge `iris_pose_last_<name>_ms`.

| Name | Boundary |
| --- | --- |
| frame_ready_wait | Sum of CPU waits on camera readiness events |
| preprocess_host | CPU calibration calculation, metadata upload submission and preprocess launch |
| trt_setup_host | Input shape and tensor address setup |
| trt_enqueue_host | Host duration of enqueueV3 |
| download_host | Compact 2-D and tracked-pose output copy submissions |
| result_wait_host | Final stream synchronization; includes any remaining upstream work |
| engine_call_host | Entire engine wrapper call, including timing instrumentation |
| postprocess_cpu | Host-side packet assembly after compact GPU results return |
| preprocess_stream | CUDA event interval around metadata uploads and preprocessing |
| engine_stream | CUDA event interval around TensorRT enqueue |
| download_stream | CUDA interval from TensorRT completion through postprocessing and output-copy completion |
| association_stream | CUDA event interval for periodic cross-view new-track seeding |
| temporal_stream | CUDA event interval for track projection-cost assignment, prediction and weighted DLT update |
| temporal_assignment_stream | CUDA event interval for the per-camera Hungarian track/detection assignments |
| mapping_stream | CUDA event interval for calibrated 2-D coordinate mapping |
| postprocess_gpu_stream | Combined CUDA interval from 2-D mapping through temporal update, including periodic seeding |

Host and stream measurements overlap: do not sum them. CUDA event intervals can
include scheduling delays and host submission gaps; they are not sums of kernel
execution times. Instrumentation uses persistent CUDA events and reads them after
the existing stream synchronization, adding no per-stage synchronization.
The total includes small additional bookkeeping not separately attributed.
`postprocess_gpu_stream` is the acceptance metric for the GPU postprocessing budget.
It excludes TensorRT and device-to-host result copies. It includes the full
cross-view seed matcher every tenth frame, so compare both its warmed-up median
and tail samples; `temporal_stream` shows the normal tracking path separately.

Compare warmed-up latest values or histogram deltas, using the same engine,
precision, batch size (three), GPU workload and build type as the reference test.
The default CMake preset is Debug. Compare Release before attributing CPU time
to an architectural bottleneck.

Optimization candidates based on the measured dominant component:

- High postprocess CPU: keep packet assembly limited to public output data;
  move any new per-joint math into the CUDA path.
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
