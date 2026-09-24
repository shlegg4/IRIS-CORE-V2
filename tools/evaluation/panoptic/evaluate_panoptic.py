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
EPIPOLAR_GATE_THRESHOLDS_PX = (8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0)
EPIPOLAR_DIAGNOSTIC_FRAME_STRIDE = 20
EPIPOLAR_DIAGNOSTIC_LABEL_ERROR_PX = 150.0


def percentile(values, percentage):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentage / 100.0 * len(ordered)) - 1))
    return ordered[index]


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


def detection_cost(detection, gt_person, camera, minimum_score=0.0,
                  minimum_joints=1, require_in_frame=False):
    pred_joints = detection.get("joints", [])
    errors = []
    for coco_index in range(17):
        if coco_index >= len(pred_joints) or not pred_joints[coco_index].get("valid"):
            continue
        if float(pred_joints[coco_index].get("score", 0.0)) < minimum_score:
            continue
        target = coco_point(gt_person, coco_index, camera)
        if target is None:
            continue
        if require_in_frame:
            width, height = camera["resolution"]
            if not (0.0 <= target[0] < width and 0.0 <= target[1] < height):
                continue
        point = pred_joints[coco_index]
        errors.append(math.hypot(float(point["x"]) - target[0], float(point["y"]) - target[1]))
    return sum(errors) / len(errors) if len(errors) >= minimum_joints else math.inf


def visible_ground_truth_indices(people, camera, minimum_joints):
    width, height = camera["resolution"]
    visible = []
    for person_index, person in enumerate(people):
        count = 0
        for coco_index in range(17):
            point = coco_point(person, coco_index, camera)
            if point is not None and 0.0 <= point[0] < width and 0.0 <= point[1] < height:
                count += 1
        if count >= minimum_joints:
            visible.append(person_index)
    return visible


def frame_detection_matches(record, truth, cameras, camera_ids,
                            minimum_score, minimum_joints):
    detections_by_camera = {camera_id: [] for camera_id in camera_ids}
    for detection in record.get("detections_2d", []):
        camera_id = int(detection["camera_id"])
        if camera_id in detections_by_camera:
            detections_by_camera[camera_id].append(detection)

    matches_by_camera = {}
    for camera_id in camera_ids:
        camera = cameras[camera_id]
        detections = detections_by_camera[camera_id]
        visible = visible_ground_truth_indices(truth, camera, minimum_joints)
        costs = [[detection_cost(detection, truth[person_index], camera,
                                 minimum_score, minimum_joints, True)
                  for person_index in visible]
                 for detection in detections]
        matches = {}
        for detection_index, visible_index in hungarian(costs):
            if detection_index >= len(detections) or visible_index >= len(visible):
                continue
            error = costs[detection_index][visible_index]
            if math.isfinite(error):
                matches[int(detections[detection_index]["detection_index"])] = (
                    visible[visible_index], error)
        matches_by_camera[camera_id] = {
            "detections": detections,
            "visible_ground_truth": visible,
            "matches": matches,
        }
    return matches_by_camera


def new_detection_metrics(thresholds):
    return {
        "score_threshold": 0.35,
        "detection_label_minimum_joints": 3,
        "visible_ground_truth_minimum_joints": 5,
        "thresholds_px": {
            str(threshold): {"true_positives": 0, "false_positives": 0,
                             "false_negatives": 0}
            for threshold in thresholds
        },
        "multi_view_tracks_with_two_or_more_labeled_views": 0,
        "multi_view_identity_consistent_tracks": 0,
        "multi_view_identity_inconsistent_tracks": 0,
        "identity_label_threshold_px": 150.0,
    }


def accumulate_detection_metrics(metrics, record, frame_matches, thresholds):
    for camera_data in frame_matches.values():
        detections = camera_data["detections"]
        visible_count = len(camera_data["visible_ground_truth"])
        errors = [cost for _, cost in camera_data["matches"].values()]
        for threshold in thresholds:
            true_positives = sum(error <= threshold for error in errors)
            entry = metrics["thresholds_px"][str(threshold)]
            entry["true_positives"] += true_positives
            entry["false_positives"] += len(detections) - true_positives
            entry["false_negatives"] += visible_count - true_positives

    labeled_tracks = 0
    for pose in record.get("persons_3d", []):
        selected = pose.get("selected_detection_indices", [])
        pose_camera_ids = pose.get("camera_ids", [])
        identities = []
        for view_index, camera_id in enumerate(pose_camera_ids):
            if view_index >= len(selected) or int(selected[view_index]) < 0:
                continue
            camera_data = frame_matches.get(int(camera_id))
            if not camera_data:
                continue
            matched = camera_data["matches"].get(int(selected[view_index]))
            if matched and matched[1] <= metrics["identity_label_threshold_px"]:
                identities.append(matched[0])
        if len(identities) < 2:
            continue
        labeled_tracks += 1
        metrics["multi_view_tracks_with_two_or_more_labeled_views"] += 1
        if len(set(identities)) == 1:
            metrics["multi_view_identity_consistent_tracks"] += 1
        else:
            metrics["multi_view_identity_inconsistent_tracks"] += 1


def finalize_detection_metrics(metrics):
    for entry in metrics["thresholds_px"].values():
        tp, fp, fn = (entry["true_positives"], entry["false_positives"],
                      entry["false_negatives"])
        entry["precision"] = tp / (tp + fp) if tp + fp else None
        entry["recall"] = tp / (tp + fn) if tp + fn else None
        entry["f1"] = (2 * tp / (2 * tp + fp + fn)
                       if 2 * tp + fp + fn else None)
    consistent = metrics["multi_view_identity_consistent_tracks"]
    total = metrics["multi_view_tracks_with_two_or_more_labeled_views"]
    metrics["multi_view_identity_consistency"] = consistent / total if total else None


def mat3_multiply(a, b):
    return [[sum(a[row][k] * b[k][column] for k in range(3))
             for column in range(3)] for row in range(3)]


def mat3_transpose(matrix):
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def model_intrinsics(camera):
    width, height = map(float, camera["resolution"])
    scale = min(640.0 / width, 640.0 / height)
    resized_width = max(1, int(width * scale + 0.5))
    resized_height = max(1, int(height * scale + 0.5))
    pad_x, pad_y = (640.0 - resized_width) * 0.5, (640.0 - resized_height) * 0.5
    k = [float(value) for value in flatten(camera["K"])]
    return [[k[0] * scale, 0.0, k[2] * scale + pad_x],
            [0.0, k[4] * scale, k[5] * scale + pad_y],
            [0.0, 0.0, 1.0]]


def inverse_intrinsics(k):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    return [[1.0 / fx, 0.0, -cx / fx],
            [0.0, 1.0 / fy, -cy / fy],
            [0.0, 0.0, 1.0]]


def fundamental_for_cameras(first, second):
    first_r = [list(map(float, row)) for row in first["R"]]
    second_r = [list(map(float, row)) for row in second["R"]]
    first_r_t = mat3_transpose(first_r)
    relative_r = mat3_multiply(second_r, first_r_t)
    first_t = [float(value) for value in flatten(first["t"])]
    second_t = [float(value) for value in flatten(second["t"])]
    relative_t = [second_t[row] - sum(relative_r[row][column] * first_t[column]
                                      for column in range(3)) for row in range(3)]
    tx, ty, tz = relative_t
    cross_t = [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]]
    first_k_inverse = inverse_intrinsics(model_intrinsics(first))
    second_k_inverse = inverse_intrinsics(model_intrinsics(second))
    return mat3_multiply(mat3_multiply(mat3_transpose(second_k_inverse), cross_t),
                         mat3_multiply(relative_r, first_k_inverse))


def undistort_detection_to_model(point, camera):
    if point is None or len(point) < 2:
        return None
    try:
        point_x, point_y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(point_x) or not math.isfinite(point_y):
        return None
    k = [float(value) for value in flatten(camera["K"])]
    distortion = [float(value) for value in flatten(camera["distCoef"])]
    observed_x = (point_x - k[2]) / k[0]
    observed_y = (point_y - k[5]) / k[4]
    x, y = observed_x, observed_y
    k1, k2, p1, p2, k3 = distortion
    for _ in range(12):
        r2 = x * x + y * y
        r4 = r2 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
        slope = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
        radial_x, radial_y = 2.0 * x * slope, 2.0 * y * slope
        predicted_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        predicted_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        error_x, error_y = predicted_x - observed_x, predicted_y - observed_y
        if max(abs(error_x), abs(error_y)) < 1.0e-7:
            break
        j00 = radial + x * radial_x + 2.0 * p1 * y + 6.0 * p2 * x
        j01 = x * radial_y + 2.0 * p1 * x + 2.0 * p2 * y
        j10 = y * radial_x + 2.0 * p1 * x + 2.0 * p2 * y
        j11 = radial + y * radial_y + 6.0 * p1 * y + 2.0 * p2 * x
        determinant = j00 * j11 - j01 * j10
        if not math.isfinite(determinant) or abs(determinant) < 1.0e-9:
            return None
        x -= (j11 * error_x - j01 * error_y) / determinant
        y -= (-j10 * error_x + j00 * error_y) / determinant
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    k640 = model_intrinsics(camera)
    return k640[0][0] * x + k640[0][2], k640[1][1] * y + k640[1][2]


def symmetric_epipolar_distance(fundamental, first, second):
    x1, y1 = first
    x2, y2 = second
    line_second = [sum(fundamental[row][column] * (x1, y1, 1.0)[column]
                       for column in range(3)) for row in range(3)]
    line_first = [sum(fundamental[row][column] * (x2, y2, 1.0)[column]
                      for column in range(3)) for row in range(3)]
    residual = abs(x2 * line_second[0] + y2 * line_second[1] + line_second[2])
    norm_second = math.hypot(line_second[0], line_second[1])
    norm_first = math.hypot(line_first[0], line_first[1])
    return 0.5 * (residual / max(1.0e-6, norm_second) +
                  residual / max(1.0e-6, norm_first))


def detection_geometry(detection, camera):
    geometry = []
    for joint in detection.get("joints", [])[:17]:
        score = float(joint.get("score", 0.0))
        point = (undistort_detection_to_model((joint.get("x"), joint.get("y")), camera)
                 if joint.get("valid") and score >= 0.35 else None)
        geometry.append((point, score))
    geometry.extend([(None, 0.0)] * (17 - len(geometry)))
    return geometry


def robust_epipolar_cost(first_geometry, second_geometry, fundamental):
    residuals = []
    for joint_index in range(17):
        first_point, first_score = first_geometry[joint_index]
        second_point, second_score = second_geometry[joint_index]
        if first_point is None or second_point is None:
            continue
        distance = symmetric_epipolar_distance(fundamental, first_point, second_point)
        weight = math.sqrt(min(1.0, max(0.0, first_score)) *
                           min(1.0, max(0.0, second_score)))
        if math.isfinite(distance) and weight > 0.0:
            residuals.append((distance, weight))
    if len(residuals) < 5:
        return None
    residuals.sort(key=lambda value: value[0])
    retained_weight = sum(weight for _, weight in residuals) * 0.8
    used_weight = weighted_cost = 0.0
    for distance, weight in residuals:
        take = min(weight, retained_weight - used_weight)
        weighted_cost += distance * take
        used_weight += take
        if used_weight >= retained_weight:
            break
    return weighted_cost / used_weight if used_weight > 0.0 else None


def new_epipolar_diagnostics():
    return {"sampled_frames": 0, "same_person_labeled_pairs": 0,
            "different_person_labeled_pairs": 0,
            "unsupported_same_person_pairs": 0,
            "unsupported_different_person_pairs": 0,
            "same_person_costs_px": [], "different_person_costs_px": []}


def accumulate_epipolar_diagnostics(diagnostics, frame_matches, cameras,
                                    fundamentals):
    diagnostics["sampled_frames"] += 1
    camera_ids = list(frame_matches)
    geometries = {}
    for camera_id in camera_ids:
        geometries[camera_id] = {
            int(detection["detection_index"]): detection_geometry(detection, cameras[camera_id])
            for detection in frame_matches[camera_id]["detections"]
        }
    for first_index, first_camera_id in enumerate(camera_ids):
        first_data = frame_matches[first_camera_id]
        first_labeled = [detection for detection in first_data["detections"]
                             if (match := first_data["matches"].get(int(detection["detection_index"])))
                             and match[1] <= EPIPOLAR_DIAGNOSTIC_LABEL_ERROR_PX]
        for second_camera_id in camera_ids[first_index + 1:]:
            second_data = frame_matches[second_camera_id]
            fundamental = fundamentals[(first_camera_id, second_camera_id)]
            second_labeled = [detection for detection in second_data["detections"]
                              if (match := second_data["matches"].get(int(detection["detection_index"])))
                              and match[1] <= EPIPOLAR_DIAGNOSTIC_LABEL_ERROR_PX]
            for first_detection in first_labeled:
                first_match = first_data["matches"][int(first_detection["detection_index"])]
                for second_detection in second_labeled:
                    second_match = second_data["matches"][int(second_detection["detection_index"])]
                    same_person = first_match[0] == second_match[0]
                    label_key = "same_person" if same_person else "different_person"
                    diagnostics[f"{label_key}_labeled_pairs"] += 1
                    cost = robust_epipolar_cost(
                        geometries[first_camera_id][int(first_detection["detection_index"])],
                        geometries[second_camera_id][int(second_detection["detection_index"])],
                        fundamental)
                    if cost is None:
                        diagnostics[f"unsupported_{label_key}_pairs"] += 1
                    else:
                        diagnostics[f"{label_key}_costs_px"].append(cost)


def summarize_epipolar_diagnostics(diagnostics):
    same_costs = diagnostics.pop("same_person_costs_px")
    different_costs = diagnostics.pop("different_person_costs_px")
    same_costs.sort()
    different_costs.sort()
    def summary(values):
        if not values:
            return {"supported_pairs": 0, "mean_px": None, "median_px": None,
                    "p90_px": None, "p95_px": None}
        quantile = lambda fraction: values[min(len(values) - 1,
                                               int(math.ceil(fraction * len(values))) - 1)]
        return {"supported_pairs": len(values), "mean_px": statistics.mean(values),
                "median_px": statistics.median(values), "p90_px": quantile(0.90),
                "p95_px": quantile(0.95)}
    diagnostics["same_person_cost_distribution"] = summary(same_costs)
    diagnostics["different_person_cost_distribution"] = summary(different_costs)
    diagnostics["gate_sweep"] = {}
    for gate in EPIPOLAR_GATE_THRESHOLDS_PX:
        true_positive = sum(cost < gate for cost in same_costs)
        false_positive = sum(cost < gate for cost in different_costs)
        false_negative = diagnostics["same_person_labeled_pairs"] - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / diagnostics["same_person_labeled_pairs"] if diagnostics["same_person_labeled_pairs"] else None
        diagnostics["gate_sweep"][str(gate)] = {
            "true_positive_pairs": true_positive,
            "false_positive_pairs": false_positive,
            "false_negative_pairs": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision is not None and recall is not None and precision + recall else None),
        }
    diagnostics["frame_stride"] = EPIPOLAR_DIAGNOSTIC_FRAME_STRIDE
    diagnostics["label_error_threshold_px"] = EPIPOLAR_DIAGNOSTIC_LABEL_ERROR_PX
    diagnostics["minimum_shared_joints"] = 5
    diagnostics["joint_score_threshold"] = 0.35
    diagnostics["trim_fraction"] = 0.2
    return diagnostics


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
    parser.add_argument("--detection-match-thresholds-px", default="50,100,150,200",
                        help="comma-separated pixel thresholds for direct 2-D detection PR")
    args = parser.parse_args()
    camera_ids = [int(value) for value in args.camera_ids.split(",")]
    if len(camera_ids) < 2 or len(camera_ids) > 10 or len(set(camera_ids)) != len(camera_ids):
        parser.error("--camera-ids must contain between two and ten unique IDs")
    cameras = cmu_camera_map(args.calibration, camera_ids)
    epipolar_fundamentals = {
        (first_id, second_id): fundamental_for_cameras(cameras[first_id], cameras[second_id])
        for index, first_id in enumerate(camera_ids)
        for second_id in camera_ids[index + 1:]
    }
    try:
        detection_thresholds = sorted({float(value) for value in
                                       args.detection_match_thresholds_px.split(",")})
    except ValueError:
        parser.error("--detection-match-thresholds-px must be comma-separated numbers")
    if not detection_thresholds or any(value <= 0.0 for value in detection_thresholds):
        parser.error("detection match thresholds must be positive")
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
    frame_error_records = []
    last_track_by_person = {}
    track_assignments_by_person = {}
    person_assignments_by_track = {}
    temporal_identity_matches = identity_switches = 0
    direct_detection_metrics = new_detection_metrics(detection_thresholds)
    epipolar_diagnostics = new_epipolar_diagnostics()

    with args.predictions.open(encoding="utf-8") as predictions:
        for line in predictions:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_id = int(record["batch_sequence"])
            if frame_id not in gt:
                continue
            matched_frames += 1
            per_frame_joint_errors = []
            frame_person_matches = 0
            truth = gt[frame_id]["people"]
            direct_matches = frame_detection_matches(
                record, truth, cameras, camera_ids, minimum_score=0.35,
                minimum_joints=3)
            accumulate_detection_metrics(direct_detection_metrics, record,
                                         direct_matches, detection_thresholds)
            if frame_id % EPIPOLAR_DIAGNOSTIC_FRAME_STRIDE == 0:
                accumulate_epipolar_diagnostics(epipolar_diagnostics, direct_matches,
                                                cameras, epipolar_fundamentals)
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
                per_person_joint_errors = []
                for p, g in zip(pred_joints[pi], truth[gi]["joints"]):
                    if p is None or g[3] <= 0:
                        continue
                    error = distance(p[:3], g[:3])
                    errors.append(error)
                    per_person_joint_errors.append(error)
                    joint_total += 1
                    joint_hits += error <= args.pck_threshold_cm
                if per_person_joint_errors:
                    frame_errors.append(sum(per_person_joint_errors) / len(per_person_joint_errors))

                per_frame_joint_errors.extend(per_person_joint_errors)
                frame_person_matches += 1
                track_id = int(predicted[pi].get("track_id", 0) or 0)
                person_id = truth[gi].get("id")
                if track_id > 0 and person_id is not None:
                    person_key = str(person_id)
                    track_key = str(track_id)
                    previous_track = last_track_by_person.get(person_key)
                    if previous_track is not None and previous_track != track_id:
                        identity_switches += 1
                    last_track_by_person[person_key] = track_id
                    track_assignments_by_person.setdefault(person_key, set()).add(track_id)
                    person_assignments_by_track.setdefault(track_key, {}).setdefault(person_key, 0)
                    person_assignments_by_track[track_key][person_key] += 1
                    temporal_identity_matches += 1

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
            if per_frame_joint_errors:
                frame_error_records.append({
                    "frame_id": frame_id,
                    "mpjpe_cm": sum(per_frame_joint_errors) / len(per_frame_joint_errors),
                    "joint_error_sum_cm": sum(per_frame_joint_errors),
                    "valid_joints": len(per_frame_joint_errors),
                    "matched_people": frame_person_matches,
                })

    if matched_frames == 0:
        raise ValueError("no prediction frame indices overlap the Panoptic ground truth")
    finalize_detection_metrics(direct_detection_metrics)
    epipolar_diagnostics = summarize_epipolar_diagnostics(epipolar_diagnostics)
    frame_mpjpe_values = [frame["mpjpe_cm"] for frame in frame_error_records]
    total_frame_joint_error = sum(frame["joint_error_sum_cm"] for frame in frame_error_records)
    tail_frames = sorted(frame_error_records, key=lambda frame: frame["mpjpe_cm"], reverse=True)
    tail_frame_count = max(1, math.ceil(len(tail_frames) * 0.01)) if tail_frames else 0
    tail_joint_error = sum(frame["joint_error_sum_cm"] for frame in tail_frames[:tail_frame_count])
    matched_person_keys = list(track_assignments_by_person.values())
    track_purity_correct = sum(max(person_counts.values())
                               for person_counts in person_assignments_by_track.values())
    track_purity_total = sum(sum(person_counts.values())
                             for person_counts in person_assignments_by_track.values())
    temporal_track_identity = {
        "matched_track_person_observations": temporal_identity_matches,
        "unique_track_ids": len(person_assignments_by_track),
        "ground_truth_people_with_track_ids": len(matched_person_keys),
        "ground_truth_people_with_multiple_track_ids": sum(len(track_ids) > 1
                                                            for track_ids in matched_person_keys),
        "track_id_switches": identity_switches,
        "track_id_purity": track_purity_correct / track_purity_total if track_purity_total else None,
        "notes": ["Per-frame person matches use the existing 3-D assignment and 100 cm cutoff.",
                  "Switches count a change in assigned track ID for a ground-truth person across matched frames."],
    }
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
        "frame_error_distribution": {
            "frames_with_valid_matches": len(frame_mpjpe_values),
            "frame_mpjpe_cm_p50": percentile(frame_mpjpe_values, 50),
            "frame_mpjpe_cm_p90": percentile(frame_mpjpe_values, 90),
            "frame_mpjpe_cm_p95": percentile(frame_mpjpe_values, 95),
            "frame_mpjpe_cm_p99": percentile(frame_mpjpe_values, 99),
            "worst_frames": tail_frames[:10],
            "top_1_percent_frames_joint_error_share": tail_joint_error / total_frame_joint_error
                if total_frame_joint_error else None,
        },
        "pck_threshold_cm": args.pck_threshold_cm,
        "pck": joint_hits / joint_total if joint_total else None,
        "mean_2d_reprojection_error_px": statistics.mean(reprojection_errors) if reprojection_errors else None,
        "matched_detection_assignments": assignment_total,
        "correct_detection_assignments": assignment_correct,
        "detection_assignment_accuracy": assignment_correct / assignment_total if assignment_total else None,
        "direct_detection_matching": direct_detection_metrics,
        "temporal_track_identity": temporal_track_identity,
        "epipolar_pair_diagnostics": epipolar_diagnostics,
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
