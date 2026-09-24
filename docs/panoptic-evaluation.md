# CMU Panoptic offline evaluation

The offline video mode runs synchronized files through the configured multiview pose engine,
writes one JSON record per batch, drains the pipeline at EOF, and saves run-scoped metrics. The
TensorRT RTMO engine accepts dynamic batches of one to ten images. Multiview pose requires between
two and ten calibrated views. `tools/evaluation/panoptic/` contains a launcher that selects HD
cameras, converts their CMU calibration records, runs IRIS, and compares its output with Panoptic's
COCO-19 ground truth.

## Run the haggling sequence

Build the `hosted` preset and use its `iris_app.exe` with a TensorRT engine built for the current
GPU. The sample directory must contain its `hdVideos`, `calibration_<sequence>.json`, and
`hdPose3d_stage1_coco19.tar` files.

```powershell
python tools/evaluation/panoptic/run_panoptic.py `
  --sequence C:/Users/Sam/Documents/CMU/panoptic-toolbox/170221_haggling_b1 `
  --iris build/hosted/bin/iris_app.exe `
  --engine build/hosted/bin/assets/rtmo_s.engine `
  --camera-ids 0,4,8,12,16 `
  --output-dir results/170221_haggling_b1
```

This sequence contains five HD views and 14,086 annotated frames. The example uses all five.

The default run processes as fast as downstream stages allow. Add `--realtime` to pace batches by
source presentation timestamps. Select between two and ten distinct available HD camera suffixes
from 0 to 30. The selected files must have matching frame counts and synchronized frame order, as
required by IRIS's synchronized video source.

`iris_app.exe --video` also supports direct batch use:

```powershell
iris_app.exe --video --engine rtmo_s.engine --calibration calibration.iris.json `
  --output-dir results/run --realtime false --loop false `
  0 camera-00.mp4 4 camera-04.mp4 8 camera-08.mp4 `
  12 camera-12.mp4 16 camera-16.mp4
```

Batch mode requires `--engine`, `--calibration`, and `--output-dir`. The calibration file contains
one calibrated camera entry per input view. The Panoptic launcher writes it as `calibration.iris.json`
in the output directory.

## Outputs

- `poses.jsonl`: batch index and per-camera source frame sequence/time, all 2-D detections, selected
  detection indices for each 3-D person, 3-D joints, validity, and per-view scores.
- `run_summary.json`: run status, configured inputs, processed batch/frame counts, elapsed time,
  effective batches per second, and the final counters, gauges, and histograms.
- `run_config.json`: dataset, camera selection, engine, and exact command used by the launcher.
- `calibration.iris.json`: selected calibration entries translated from the CMU camera file.
- `evaluation.json`: 3-D joint error, PCK, 2-D reprojection error, and cross-view selected-detection
  assignment accuracy when the ground-truth archive is present. It also reports direct per-camera
  detection precision/recall at several reprojection thresholds and within-frame multi-view identity
  consistency.

## Metrics and interpretation

The evaluator matches predicted and annotated people independently for every overlapping frame by
minimum mean 3-D joint error. It maps COCO-17 detections into Panoptic COCO-19 ordering and derives
Neck and BodyCenter from the shoulder and hip midpoints when both source joints are valid. It
compares the world-space coordinates directly in the Panoptic calibration units (centimeters for
the supplied Panoptic calibration), and reports MPJPE and median joint error in those units, plus
PCK at a configurable 10 cm threshold by default. Person matches above 100 cm are left unmatched;
the report includes unmatched prediction and ground-truth counts.

The 2-D reprojection metric compares the selected detection with the matched ground-truth person's
calibrated projection. Assignment accuracy checks whether each selected detection's reprojection
error identifies the same ground-truth person as the 3-D match. These scores are only meaningful
when the selected videos are synchronized, calibration and image dimensions match, and video batch
indices correspond to `body3DScene_<index>.json`. The report includes the number of overlapping
frames; zero overlap is an error rather than a successful empty evaluation.

`direct_detection_matching` labels each camera's detections independently by one-to-one assignment
to projected ground-truth people. It includes ground-truth people with at least five COCO joints
projecting inside the camera image, compares detections using at least three joints with score >=0.35,
and reports precision/recall/F1 at configurable pixel thresholds (50, 100, 150, and 200 by default).
These per-camera scores evaluate detector localization and coverage, not cross-camera association.
`multi_view_identity_consistency` then checks the selected detections from each 3-D pose against
those per-camera ground-truth labels; it counts only poses with at least two selected detections
whose reprojection error is <=150 px. This is a within-frame association metric, not a tracker metric.

The current epipolar matcher compares all cross-camera candidate pairs, uses a confidence-weighted
trimmed mean of joint residuals (trimming the highest 20% of residual weight), requires at least five
shared joints above 0.35 confidence, and gates pairs at 8 px in the 640 x 640 inference canvas.
Compatible detections are joined through a cost-sorted pair graph; a track is retained only when it
contains at least two cameras. The graph ordering uses camera IDs as deterministic tie-breakers, so
the selected groups do not depend on the camera list order.

`epipolar_pair_diagnostics` samples every twentieth overlapping frame. It labels candidate pairs
using independent per-camera ground-truth assignments (accepted at <=150 px), reports same-person
and different-person robust epipolar cost distributions, and sweeps gates from 8 to 48 px. The
current 8 px gate was selected because it gave the best pair-classification F1 in the sampled
Panoptic data; the full-sequence result is still the acceptance check for coverage and 3-D accuracy.

For before/after comparisons, keep the sequence, camera selection, engine, CUDA device, realtime
setting, and hardware constant. Compare `evaluation.json` for accuracy and
`run_summary.json`'s `batches_per_second` and pose timing histograms for throughput. Run each trial
into a fresh output directory so files from different changes cannot be mixed.
