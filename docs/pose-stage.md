# Pose stage

`PoseStage` runs the integrated `mvssm-ray-refinement-onnx-v1` graph. The graph owns detector
inference, hip-ray association, weighted DLT, refinement, score filtering, and pose NMS.

Build with ONNX Runtime installed and point CMake at its root:

```powershell
cmake -S . -B build -DIRIS_ONNXRUNTIME_ROOT=C:\deps\onnxruntime
```

Configure the model and world-to-camera calibration when constructing `Pipeline` or `Runtime`:

```cpp
iris::PoseConfig pose;
pose.model_path = "artifacts/runs/pose/model.onnx";
pose.calibrations = {camera0, camera1, camera2, camera3, camera4};

iris::Runtime runtime(capture_config, 9464, pose);
```

Each `CameraCalibration` uses row-major `rotation` and `intrinsic` matrices. `translation_mm`
is a world-to-camera translation in millimetres. Frames are converted from CUDA `Bgr8` to RGB
`uint8`, resized to `960x512`, and the first two rows of the intrinsic matrix are scaled by the
same resize. Fewer than five synchronized frames are zero-padded and represented by `view_mask`.

Only slots with `active_mask=true` appear in `Packet::poses`. Each returned `Pose` carries the
Panoptic-19 world-space joints, final score, refined per-view 2-D points, and both per-view and
view-averaged joint confidence.
