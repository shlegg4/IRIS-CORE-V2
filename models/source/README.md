# Deployment model sources

These local model files are intentionally excluded from Git because the official
checkpoints are large. Downloaded model provenance:

| File | Upstream | License | Use |
| --- | --- | --- | --- |
| `rtmo-s/rtmo-s_coco17.pth` | [OpenMMLab RTMO-S COCO checkpoint](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth) | See the upstream MMPose/model terms | Source checkpoint for the accompanying ONNX graph |
| `rtmo-s/rtmo_s_candidates.onnx` | Exported from the RTMO-S checkpoint with the PARALLAX-HMR MMPose exporter | — | Build the host-specific RTMO TensorRT engine |
| `da3-base/model.safetensors` and `da3-base/config.json` | [ByteDance Seed DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | Apache-2.0 | Export the DA3 base ONNX graph used to build a local TensorRT engine |

The RTMO checkpoint is COCO-17 and matches IRIS's existing RTMO candidate engine.
The DA3 source must be exported with the four-view, 504x504 geometry contract
expected by IRIS calibration. A checkpoint alone is not a TensorRT engine.
