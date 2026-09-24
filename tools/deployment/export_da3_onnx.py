#!/usr/bin/env python
"""Export DA3-base with a dynamic number of jointly processed camera views.

The exported input is [1, views, 3, 504, 504], where views is dynamic from 1 to
10. The fixed batch dimension keeps camera views in DA3's multiview attention.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from torch import nn


def _cartesian_prod_export_compatible(*vectors: torch.Tensor) -> torch.Tensor:
    """Use meshgrid ops for DA3's fixed spatial RoPE grid.

    PyTorch's legacy ONNX exporter has no symbolic for aten::cartesian_prod.
    DA3 uses it only to construct the fixed 2D patch-coordinate grid.
    """
    if len(vectors) != 2:
        raise RuntimeError("DA3 RoPE export expects a 2D Cartesian product.")
    y_grid, x_grid = torch.meshgrid(*vectors, indexing="ij")
    return torch.stack((y_grid.reshape(-1), x_grid.reshape(-1)), dim=-1)


def _affine_inverse_export_compatible(matrix: torch.Tensor) -> torch.Tensor:
    """Equivalent affine inverse using transpose ops supported by ONNX opset 18."""
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3:]
    bottom_row = matrix[..., 3:, :]
    rotation_t = rotation.transpose(-2, -1)
    return torch.cat(
        (torch.cat((rotation_t, -rotation_t @ translation), dim=-1), bottom_row),
        dim=-2,
    )


class MultiViewExport(nn.Module):
    """Expose only the float tensors consumed by the IRIS DA3 runtime."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        prediction = self.network(
            images,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="first",
        )
        return (
            prediction["depth"],
            prediction["depth_conf"],
            prediction["intrinsics"],
            prediction["extrinsics"],
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root / "models" / "source" / "da3-base",
        help="Directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "models" / "source" / "da3-base" / "da3-base-mv4-504.onnx",
        help="ONNX output path (TensorRT uses a dynamic external-data file when needed).",
    )
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--example-views", type=int, default=4)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing ONNX graph without loading the DA3 checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def validate_graph(onnx_path: Path, resolution: int) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("ONNX graph validation needs the 'onnx' Python package.") from exc

    onnx.checker.check_model(str(onnx_path))
    graph = onnx.load(str(onnx_path), load_external_data=False)
    image_input = next((item for item in graph.graph.input if item.name == "images"), None)
    if image_input is None:
        raise RuntimeError("ONNX graph has no 'images' input.")
    dimensions = image_input.type.tensor_type.shape.dim
    if len(dimensions) != 5 or dimensions[0].dim_value != 1:
        raise RuntimeError("ONNX input must have shape [1, dynamic_views, 3, H, W].")
    if not dimensions[1].dim_param:
        raise RuntimeError("ONNX view dimension is not dynamic.")
    expected_tail = (3, resolution, resolution)
    actual_tail = tuple(dimension.dim_value for dimension in dimensions[2:])
    if actual_tail != expected_tail:
        raise RuntimeError(f"ONNX input dimensions after view axis must be {expected_tail}, got {actual_tail}.")
    for name in ("depth", "depth_conf", "intrinsics", "extrinsics"):
        output = next((item for item in graph.graph.output if item.name == name), None)
        if output is None or not output.type.tensor_type.shape.dim:
            raise RuntimeError(f"ONNX graph has no shaped '{name}' output.")
        if len(output.type.tensor_type.shape.dim) < 2:
            raise RuntimeError(f"ONNX output '{name}' has no view dimension.")
        if not output.type.tensor_type.shape.dim[1].dim_param:
            raise RuntimeError(f"ONNX output '{name}' does not follow the dynamic view count.")


def externalize_weights(onnx_path: Path) -> None:
    """Write weights beside the graph, matching the deployment builder's sidecar contract."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    graph = onnx.load(str(onnx_path), load_external_data=True)
    for tensor in _get_all_tensors(graph):
        tensor.ClearField("external_data")
        tensor.data_location = onnx.TensorProto.DEFAULT
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.save_model(
        graph,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar.name,
        size_threshold=0,
        convert_attribute=False,
    )


def main() -> None:
    args = parse_args()
    if args.resolution <= 0 or args.resolution % 14 != 0:
        raise SystemExit("--resolution must be a positive multiple of DA3's 14-pixel patch size.")
    if not 1 <= args.example_views <= 10:
        raise SystemExit("--example-views must be between 1 and 10.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this Python environment has no CUDA device.")

    if args.validate_only:
        validate_graph(args.output, args.resolution)
        print(f"Validated dynamic-view DA3 ONNX graph: {args.output}")
        return

    try:
        from depth_anything_3.api import DepthAnything3
        import depth_anything_3.model.da3 as da3_model
    except ImportError as exc:
        raise SystemExit(
            "Depth Anything 3 is not installed in this Python environment. Install the official "
            "depth-anything-3 package and its dependencies, then rerun this exporter."
        ) from exc

    for required in (args.model_dir / "config.json", args.model_dir / "model.safetensors"):
        if not required.is_file():
            raise SystemExit(f"Required DA3 checkpoint file is missing: {required}")

    model = DepthAnything3.from_pretrained(str(args.model_dir))
    # DA3's scripted affine_inverse contains aten::mT, unsupported by the legacy exporter.
    da3_model.affine_inverse = _affine_inverse_export_compatible
    network = model.model.to(device=args.device, dtype=torch.float32).eval()
    export_model = MultiViewExport(network).eval()
    # Replace DA3's unsupported aten::cartesian_prod in its fixed-size RoPE grid.
    torch.cartesian_prod = _cartesian_prod_export_compatible
    images = torch.zeros(
        (1, args.example_views, 3, args.resolution, args.resolution),
        dtype=torch.float32,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "images": {1: "num_views"},
        "depth": {1: "num_views"},
        "depth_conf": {1: "num_views"},
        "intrinsics": {1: "num_views"},
        "extrinsics": {1: "num_views"},
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with torch.inference_mode():
            torch.onnx.export(
                export_model,
                (images,),
                str(args.output),
                input_names=["images"],
                output_names=["depth", "depth_conf", "intrinsics", "extrinsics"],
                dynamic_axes=dynamic_axes,
                opset_version=args.opset,
                do_constant_folding=True,
                export_params=True,
                external_data=True,
                dynamo=False,
                verbose=False,
            )

    externalize_weights(args.output)
    validate_graph(args.output, args.resolution)
    print(f"Exported {args.output}")
    print(f"Input: [1, num_views, 3, {args.resolution}, {args.resolution}]")
    print(f"ONNX opset: {args.opset}; external weights: {args.output.with_suffix(args.output.suffix + '.data')}")


if __name__ == "__main__":
    main()
