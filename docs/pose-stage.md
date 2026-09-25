# Pose stage

`PoseStage` runs the PEAR EHM TorchScript artifact through LibTorch. It performs **monocular** HMR:
every input frame is an independent batch item; synchronized cameras are not triangulated.

Build with LibTorch installed and point CMake at its root:

```powershell
cmake -S . -B build -DIRIS_TORCH_ROOT=C:\libtorch
```

Configure the model and world-to-camera calibration when constructing `Pipeline` or `Runtime`:

```cpp
iris::PoseConfig pose;
pose.model_path = "pear_ehm_libtorch.pt";
pose.device = "cpu";

iris::Runtime runtime(capture_config, 9464, pose);
```

Frames are copied from CUDA `Bgr8`, converted to RGB floats in `[0,1]`, and resized to `256x256`.
The model owns ImageNet normalization and removes 32 pixels from the top and bottom internally.

The model returns one `Pose` per frame. Its `hmr` payload contains the complete PEAR parameter
tuple: camera transform, SMPL-X global/body/hand rotation matrices, hand/head scale, body
shape/expression, and FLAME pose/expression/shape. `source_camera` identifies the input frame.
This model does not produce world-space joint positions, 2-D keypoints, or confidence scores; the
legacy Panoptic joint fields are consequently not populated and `score` is set to `1.0`.

## Single-camera 2-D keypoints

The RTMO-S TensorRT backend can also run against one configured camera without rig calibration or
triangulation. From the interactive CLI, select the bundled engine with:

```text
pose 2d
```

Or provide an engine explicitly with `pose 2d <engine-path>`. The stage emits COCO-17 image-space
keypoints and confidence scores through the same preview pose events used by multiview mode. The
shipped engine has a fixed batch size of three, so the implementation duplicates the input frame
inside the inference batch and ignores the two duplicate results.

## Multiview tracking

In multiview mode, CUDA retains each 3-D skeleton and joint velocity between frames. It projects
predicted joints into each camera, solves one-to-one detection assignments per view, and updates joints
with weighted DLT. A joint may be predicted for one frame after its last successful triangulation;
the next failed update invalidates it. The `predicted` flag distinguishes that frame from a measured
joint. The track moves to a dormant state after two consecutive frames with fewer than two assigned views. Its 3D joint anchors and IDs are retained for up to 60 frames, and cross-view recovery seeds are projected against those anchors to reconnect returning people to the dormant IDs. Unmatched recovery seeds create new tracks. The cross-view epipolar matcher
seeds tracks for detections unmatched by live 3-D tracks every ten frames. Multiview outputs include persistent `track_id`
values; detection indices remain frame-local. GPU postprocessing duration is exposed as
`iris_pose_last_postprocess_gpu_stream_ms` and excludes TensorRT and output copies.
