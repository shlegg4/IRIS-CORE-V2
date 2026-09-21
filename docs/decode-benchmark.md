# JPEG decoder benchmark

Build `iris_decode_benchmark` using the existing CMake build. Prefer a Release
build for performance comparisons. Stop other camera applications before capture.

```powershell
cmake --build build/default --target iris_decode_benchmark
.\build\default\bin\iris_decode_benchmark.exe --input build/jpeg-corpus --capture --device-index 0 --width 1280 --height 720 --frames 32
```

Capture requires a new output directory, saves the original compressed camera
JPEGs, closes the camera, and then benchmarks them. To repeat without the camera:

```powershell
.\build\default\bin\iris_decode_benchmark.exe --input build/jpeg-corpus --iterations 500 --warmup 50 --lanes 4
```

The corpus must have uniform dimensions. JPEGs from multiple cameras can be
collected into one directory when dimensions match. All paths replay the same
sorted corpus. Four lanes simulate four concurrent decode workers, each with
its own persistent decoder, stream, and output storage; this is not synchronized
camera capture or true four-image batch decoding.

Paths: simple nvJPEG (including header check), simple with pinned input copy,
decoupled GPU hybrid, hardware batched API with batch size one, and Windows WIC CPU
JPEG decode plus upload into pitched GPU BGR storage. Hardware failures are
reported as unavailable, never silently replaced with another backend.

Host wall-clock timings include parsing, copying/staging, decode, and waiting
for GPU output readiness. File I/O, setup, output validation, and warmup are
excluded. No CUDA event intervals are labelled as kernel execution time.
Median and p95 are per-image completion latencies. Throughput is total images
divided by elapsed concurrent worker time, not the reciprocal of the median.
Validation checks every image against CPU BGR output (mean absolute byte error
<=8 allows JPEG chroma upsampling differences); it is not pixel-exact equivalence.
The CPU backend is Windows Imaging Component's JPEG decoder, writing BGR into
pinned host memory before upload. It is not a libjpeg-turbo benchmark.

```powershell
.\build\default\bin\iris_decode_benchmark.exe --self-test --iterations 20 --warmup 5 --lanes 2
```

This synthetic smoke test exercises all available paths without a camera. Its
timings are not representative of real camera JPEGs. A sub-2ms goal should be
evaluated against real-frame p95 with all desired lanes and representative GPU
load. This tool intentionally omits pose/preview load; compare an isolated run
with a run while those workloads are active.
