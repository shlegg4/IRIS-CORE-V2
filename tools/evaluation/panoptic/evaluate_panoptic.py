#!/usr/bin/env python3
"""Compare IRIS JSONL output with CMU Panoptic COCO-19 3-D annotations."""

import argparse
import json
import math
import pathlib
import re
import statistics
import sys
import tarfile


# COCO-17 indices -> Panoptic COCO-19 indices.
COCO_TO_PANOPTIC = {0: 1, 1: 15, 2: 17, 3: 16, 4: 18,
                    5: 3, 6: 9, 7: 4, 8: 10, 9: 5, 10: 11,
                    11: 6, 12: 12, 13: 7, 14: 13, 15: 8, 16: 14}
FRAME_RE = re.compile(r"body3DScene_(\d+)\.json$")


def flatten(values):
    if isinstance(values, list):
        return [item for value in values for item in flatten(value)]
    return [values]


def load_ground_truth(path):
    entries = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            match = FRAME_RE.search(member.name)
            if not match or not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            data = json.load(stream)
            people = []
            for body in data.get("bodies", []):
                raw = body.get("joints19", [])
                if len(raw) < 19 * 4:
                    continue
                joints = []
                for index in range(19):
                    x, y, z, confidence = raw[index * 4:index * 4 + 4]
                    joints.append((float(x), float(y), float(z), float(confidence)))
                people.append({"id": body.get("id"), "joints": joints})
            entries[int(match.group(1))] = {"time": data.get("univTime"), "people": people}
    return entries


def prediction_to_panoptic(person):
    source = person.get("joints", [])
    result = [None] * 19
    for coco_index, panoptic_index in COCO_TO_PANOPTIC.items():
        if coco_index < len(source) and source[coco_index].get("valid"):
            joint = source[coco_index]
            result[panoptic_index] = (float(joint["x"]), float(joint["y"]), float(joint["z"]), 1.0)
    for index, first, second in ((0, 3, 9), (2, 6, 12)):
        a, b = result[first], result[second]
        if a is not None and b is not None:
            result[index] = tuple((a[axis] + b[axis]) * 0.5 for axis in range(3)) + (1.0,)
    return result


def distance(a, b):
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in range(3)))


def pair_cost(pred, gt):
    errors = [distance(p[:3], g[:3]) for p, g in zip(pred, gt)
              if p is not None and g[3] > 0.0]
    return (sum(errors) / len(errors), len(errors)) if errors else (math.inf, 0)


def hungarian(cost):
    """Minimum rectangular assignment; returns (row, column) pairs."""
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if not rows or not cols:
        return []
    transposed = rows > cols
    matrix = [[value if math.isfinite(value) else 1e9 for value in row] for row in cost]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
    n, m = len(matrix), len(matrix[0])
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return [(j, i) for i, j in pairs] if transposed else pairs


def cmu_camera_map(path, camera_ids):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    by_name = {camera["name"]: camera for camera in data["cameras"]
               if camera.get("type") == "hd"}
    result = {}
    for camera_id in camera_ids:
        name = f"00_{camera_id:02d}"
        if name not in by_name:
            raise ValueError(f"calibration is missing HD camera {name}")
        result[camera_id] = by_name[name]
    return result


def project(point, camera):
    rotation = [float(x) for x in flatten(camera["R"])]
    translation = [float(x) for x in flatten(camera["t"])]
    p = [sum(rotation[row * 3 + col] * point[col] for col in range(3)) + translation[row]
         for row in range(3)]
    if p[2] <= 1e-6:
        return None
    x, y = p[0] / p[2], p[1] / p[2]
    coeff = [float(v) for v in flatten(camera["distCoef"])]
    r2 = x * x + y * y
    radial = 1 + coeff[0] * r2 + coeff[1] * r2 * r2 + coeff[4] * r2 * r2 * r2
    xd = x * radial + 2 * coeff[2] * x * y + coeff[3] * (r2 + 2 * x * x)
    yd = y * radial + coeff[2] * (r2 + 2 * y * y) + 2 * coeff[3] * x * y
    k = [float(v) for v in flatten(camera["K"])]
    return k[0] * xd + k[2], k[4] * yd + k[5]


def coco_point(gt_person, coco_index, camera):
    panoptic_index = COCO_TO_PANOPTIC[coco_index]
    joint = gt_person["joints"][panoptic_index]
    if joint[3] <= 0:
        return None
    return project(joint[:3], camera)


def detection_cost(detection, gt_person, camera):
    pred_joints = detection.get("joints", [])
    errors = []
    for coco_index in range(17):
        if coco_index >= len(pred_joints) or not pred_joints[coco_index].get("valid"):
            continue
        target = coco_point(gt_person, coco_index, camera)
        if target is None:
            continue
        point = pred_joints[coco_index]
        errors.append(math.hypot(float(point["x"]) - target[0], float(point["y"]) - target[1]))
    return sum(errors) / len(errors) if errors else math.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True, type=pathlib.Path,
                        help="hdPose3d_stage1_coco19.tar or extracted directory")
    parser.add_argument("--calibration", required=True, type=pathlib.Path)
    parser.add_argument("--camera-ids", required=True,
                        help="two to ten HD camera suffixes used for this run")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--pck-threshold-cm", type=float, default=10.0)
    parser.add_argument("--max-person-match-error-cm", type=float, default=100.0,
                        help="larger 3-D person matches are counted as unmatched")
    args = parser.parse_args()
    camera_ids = [int(value) for value in args.camera_ids.split(",")]
    if len(camera_ids) < 2 or len(camera_ids) > 10 or len(set(camera_ids)) != len(camera_ids):
        parser.error("--camera-ids must contain between two and ten unique IDs")
    cameras = cmu_camera_map(args.calibration, camera_ids)
    if args.ground_truth.is_dir():
        gt = {}
        for path in args.ground_truth.rglob("body3DScene_*.json"):
            match = FRAME_RE.search(path.name)
            if match:
                data = json.loads(path.read_text(encoding="utf-8"))
                gt[int(match.group(1))] = {"time": data.get("univTime"),
                    "people": [{"id": body.get("id"), "joints": [
                        tuple(map(float, body["joints19"][i * 4:i * 4 + 4]))
                        for i in range(19)]} for body in data.get("bodies", [])
                        if len(body.get("joints19", [])) >= 76]}
    else:
        gt = load_ground_truth(args.ground_truth)
    errors, frame_errors = [], []
    joint_hits = joint_total = 0
    matched_people = total_ground_truth_people = 0
    reprojection_errors = []
    assignment_correct = assignment_total = 0
    matched_frames = 0
    prediction_count = 0

    with args.predictions.open(encoding="utf-8") as predictions:
        for line in predictions:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_id = int(record["batch_sequence"])
            if frame_id not in gt:
                continue
            matched_frames += 1
            truth = gt[frame_id]["people"]
            total_ground_truth_people += len(truth)
            predicted = record.get("persons_3d", [])
            prediction_count += len(predicted)
            pred_joints = [prediction_to_panoptic(person) for person in predicted]
            costs = [[pair_cost(p, person["joints"])[0] for person in truth]
                     for p in pred_joints]
            unmatched_cost = args.max_person_match_error_cm
            assignment_costs = [
                [(value if math.isfinite(value) and value <= unmatched_cost
                  else unmatched_cost + 1.0) for value in row] +
                [unmatched_cost] * len(predicted)
                for row in costs
            ]
            pairs = hungarian(assignment_costs)
            for pi, gi in pairs:
                if (pi >= len(predicted) or gi >= len(truth) or
                        not math.isfinite(costs[pi][gi]) or costs[pi][gi] > unmatched_cost):
                    continue
                matched_people += 1
                frame_joint_errors = []
                for p, g in zip(pred_joints[pi], truth[gi]["joints"]):
                    if p is None or g[3] <= 0:
                        continue
                    error = distance(p[:3], g[:3])
                    errors.append(error)
                    frame_joint_errors.append(error)
                    joint_total += 1
                    joint_hits += error <= args.pck_threshold_cm
                if frame_joint_errors:
                    frame_errors.append(sum(frame_joint_errors) / len(frame_joint_errors))

                detection_map = {(int(d["camera_id"]), int(d["detection_index"])): d
                                 for d in record.get("detections_2d", [])}
                for view_index, camera_id in enumerate(predicted[pi].get("camera_ids", camera_ids)):
                    if view_index >= len(predicted[pi].get("selected_detection_indices", [])):
                        continue
                    detection_index = int(predicted[pi]["selected_detection_indices"][view_index])
                    if detection_index < 0:
                        continue
                    detection = detection_map.get((int(camera_id), detection_index))
                    if detection is None:
                        continue
                    reproj = detection_cost(detection, truth[gi], cameras[int(camera_id)])
                    if math.isfinite(reproj):
                        reprojection_errors.append(reproj)
                    candidate_errors = [detection_cost(detection, person, cameras[int(camera_id)])
                                        for person in truth]
                    if candidate_errors and math.isfinite(min(candidate_errors)):
                        assignment_total += 1
                        assignment_correct += min(range(len(candidate_errors)),
                                                  key=candidate_errors.__getitem__) == gi

    if matched_frames == 0:
        raise ValueError("no prediction frame indices overlap the Panoptic ground truth")
    result = {
        "schema_version": 1,
        "coordinate_units": "Panoptic calibration units (centimeters for the supplied dataset)",
        "matched_frames": matched_frames,
        "ground_truth_frames": len(gt),
        "predicted_people": prediction_count,
        "matched_people": matched_people,
        "unmatched_predictions": prediction_count - matched_people,
        "ground_truth_people": total_ground_truth_people,
        "unmatched_ground_truth_people": total_ground_truth_people - matched_people,
        "max_person_match_error_cm": args.max_person_match_error_cm,
        "valid_3d_joints": joint_total,
        "mpjpe_cm": statistics.mean(errors) if errors else None,
        "median_joint_error_cm": statistics.median(errors) if errors else None,
        "mean_frame_mpjpe_cm": statistics.mean(frame_errors) if frame_errors else None,
        "pck_threshold_cm": args.pck_threshold_cm,
        "pck": joint_hits / joint_total if joint_total else None,
        "mean_2d_reprojection_error_px": statistics.mean(reprojection_errors) if reprojection_errors else None,
        "matched_detection_assignments": assignment_total,
        "correct_detection_assignments": assignment_correct,
        "detection_assignment_accuracy": assignment_correct / assignment_total if assignment_total else None,
        "notes": ["Frame association uses the decoded batch index and the Panoptic body3DScene frame suffix.",
                  "People are matched independently within each frame by minimum mean 3-D joint error.",
                  "Neck and BodyCenter are derived as the midpoints of shoulders and hips when both are valid."]
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"Panoptic evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
