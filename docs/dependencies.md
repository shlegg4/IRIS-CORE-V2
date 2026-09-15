# Dependencies

IRIS_V2 uses vcpkg manifest mode. Its dependency declaration is `vcpkg.json` and its local
installed tree is `vcpkg_installed/`; neither relies on the sibling `Iris` checkout.

Install the dependencies from VS Code with **IRIS: install dependencies**, or run:

```powershell
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
C:\vcpkg\vcpkg.exe install --triplet x64-windows --overlay-triplets vcpkg-triplets --x-manifest-root . --x-install-root vcpkg_installed
```

The configured packages are Boost.System, FFmpeg's `avcodec`, `avformat`, and NV codec support,
and
ONNX Runtime. The current pose runner uses ONNX Runtime's CPU execution provider; CUDA remains
enabled for the project's capture and image-processing code. The task deliberately uses the
installed VS 2022 Build Tools (MSVC 14.44), rather than VS 18/MSVC 14.51, which CUDA 12.8 cannot
use reliably. `vcpkg-triplets/x64-windows.cmake` keeps the CUDA compatibility flag for the
project build.
The default CMake preset assumes vcpkg is installed at `C:\vcpkg`.

## Future CUDA ONNX Runtime

The standard install deliberately uses the CPU execution provider. To request CUDA-enabled ONNX
Runtime later, install the opt-in manifest feature and configure the GPU preset:

```powershell
C:\vcpkg\vcpkg.exe install --triplet x64-windows --overlay-triplets vcpkg-triplets --x-feature=onnxruntime-cuda --x-manifest-root . --x-install-root vcpkg_installed
cmake --preset gpu-inference
```

This currently requires a CUDA/MSVC combination that can build CUTLASS successfully. It must also
be paired with a runner update that explicitly enables ONNX Runtime's CUDA execution provider;
installing the feature alone does not move inference to the GPU.
If it is elsewhere, update `CMAKE_TOOLCHAIN_FILE` in `CMakePresets.json`.
