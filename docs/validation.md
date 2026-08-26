# Validation suites

IRIS separates deterministic tests from tests that own physical cameras.

| Suite | Command | Intended use |
|---|---|---|
| Unit | `ctest --preset unit` | Every build and CI run |
| Hardware smoke | `ctest --preset hardware` | Capture-development workstation |
| Validation | `ctest --preset validation` | Release candidates and capture changes |
| Soak | `ctest --preset soak` | Overnight stability validation |

Configure and build the hardware tree before running a hardware suite:

```powershell
cmake --preset hardware
cmake --build --preset hardware
ctest --preset hardware
```

The hardware suite is built only when `IRIS_BUILD_HARDWARE_TESTS=ON`. Camera tests use CTest resource locks and one test job, preventing concurrent access to camera 0.

Use the VS Code tasks `IRIS: test hardware smoke verbose` or `IRIS: validate verbose` to print metrics for passing tests. Standard CTest output shows full metrics only for failures.

## Suite contents

- `unit`: channels, clock estimation and metrics snapshots.
- `hardware`: device enumeration, Media Foundation delivery and a short GPU pipeline run.
- `validation`: unit and smoke tests plus threshold validation, artificial consumer backpressure and a five-minute drift run.
- `soak`: a one-hour capture, decode, latency and drift run.

## Validation contract

`iris_capture_validation` accepts capture settings and quality thresholds:

```powershell
iris_capture_validation.exe `
    --device-index 0 `
    --width 1920 --height 1080 --fps 30 `
    --frames 300 `
    --max-drop-rate 0.01 `
    --max-decode-failure-rate 0.0 `
    --max-p95-latency-ms 40 `
    --max-clock-drift-ppm 100 `
    --min-drift-samples 120 `
    --output capture-validation.json
```

Exit codes are `0` for pass, `2` for a threshold failure and `1` for a configuration or runtime error. The JSON report records sample reconciliation, failures, drop rate, latency percentiles and final clock drift.

Every validation test writes a unique report under `build/hardware/tests/hardware/validation-results/`. Reports contain the complete counter, gauge and histogram snapshot and are registered as CTest failure attachments for CI/CDash collection.

Decode failures have zero tolerance by default. A single rejected or failed frame causes smoke, validation and soak to fail; thresholds must not be relaxed to conceal unexplained frame corruption.

Drift is reported immediately but is only used as a pass/fail threshold after `--min-drift-samples` observations. This prevents short smoke runs from treating a one-second clock fit as a stable drift measurement.
