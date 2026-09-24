#!/usr/bin/env python3
"""Run IRIS on three synchronized CMU Panoptic HD views and score the result."""

import argparse
import json
import pathlib
import subprocess
import sys


def flatten(values):
    if isinstance(values, list):
        return [item for value in values for item in flatten(value)]
    return [values]


def calibration_for(sequence, camera_ids):
    source = sequence / f"calibration_{sequence.name}.json"
    if not source.is_file():
        raise FileNotFoundError(f"CMU calibration file not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    by_name = {camera["name"]: camera for camera in data["cameras"]
               if camera.get("type") == "hd"}
    selected = []
    for camera_id in camera_ids:
        name = f"00_{camera_id:02d}"
        camera = by_name.get(name)
        if camera is None:
            raise ValueError(f"HD camera {name} is absent from {source}")
        selected.append({
            "camera_id": camera_id,
            "R_w2c": flatten(camera["R"]),
            "t_w2c": flatten(camera["t"]),
            "intrinsics": flatten(camera["K"]),
            "distortion": flatten(camera["distCoef"]),
        })
    return source, {"cameras": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=pathlib.Path,
                        help="CMU sequence directory, e.g. 171204_pose1_sample")
    parser.add_argument("--camera-ids", default="0,4,8,12,16",
                        help="two to ten HD camera suffixes (default: 0,4,8,12,16)")
    parser.add_argument("--engine", required=True, type=pathlib.Path)
    parser.add_argument("--iris", required=True, type=pathlib.Path,
                        help="path to iris_app.exe")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--metrics-port", type=int, default=9464,
                        help="local Prometheus exporter port (default: 9464)")
    parser.add_argument("--realtime", action="store_true",
                        help="pace ingestion using source presentation timestamps")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    sequence = args.sequence.resolve()
    camera_ids = [int(value) for value in args.camera_ids.split(",") if value.strip()]
    if len(camera_ids) < 2 or len(camera_ids) > 10 or len(set(camera_ids)) != len(camera_ids):
        parser.error("--camera-ids must contain between two and ten unique IDs")
    if any(value < 0 or value > 30 for value in camera_ids):
        parser.error("sample HD camera IDs must be from 0 through 30")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_source, iris_calibration = calibration_for(sequence, camera_ids)
    converted_calibration = args.output_dir / "calibration.iris.json"
    converted_calibration.write_text(json.dumps(iris_calibration, indent=2) + "\n",
                                     encoding="utf-8")

    command = [str(args.iris.resolve()), "--video", "--engine", str(args.engine.resolve()),
               "--calibration", str(converted_calibration), "--output-dir",
               str(args.output_dir.resolve()), "--realtime", str(args.realtime).lower(),
               "--cuda-device", str(args.cuda_device), "--metrics-port", str(args.metrics_port),
               "--loop", "false"]
    for camera_id in camera_ids:
        video = sequence / "hdVideos" / f"hd_00_{camera_id:02d}.mp4"
        if not video.is_file():
            parser.error(f"video file is missing: {video}")
        command.extend([str(camera_id), str(video)])
    run_config = {
        "sequence": str(sequence), "camera_ids": camera_ids,
        "calibration_source": str(calibration_source), "engine": str(args.engine.resolve()),
        "realtime": args.realtime, "command": command,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n",
                                                       encoding="utf-8")
    print("Running:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, check=False)
    print(f"IRIS exit code: {result.returncode}", flush=True)
    if result.returncode != 0:
        return result.returncode

    ground_truth = sequence / "hdPose3d_stage1_coco19.tar"
    print(f"Ground truth archive: {ground_truth} (present={ground_truth.is_file()})", flush=True)
    if not args.skip_evaluation and ground_truth.is_file():
        evaluator = pathlib.Path(__file__).with_name("evaluate_panoptic.py")
        evaluation_command = [sys.executable, str(evaluator),
                              "--predictions", str(args.output_dir / "poses.jsonl"),
                              "--ground-truth", str(ground_truth),
                              "--calibration", str(calibration_source),
                              "--camera-ids", ",".join(map(str, camera_ids)),
                              "--output", str(args.output_dir / "evaluation.json")]
        print("Scoring:", subprocess.list2cmdline(evaluation_command), flush=True)
        return subprocess.run(evaluation_command, check=False).returncode
    if not args.skip_evaluation:
        print(f"Required ground-truth archive not found: {ground_truth}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Panoptic run failed: {error}", file=sys.stderr)
        raise SystemExit(2)
