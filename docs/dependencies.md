# Dependencies

IRIS_V2 uses vcpkg manifest mode. Its dependency declaration is `vcpkg.json` and its local
installed tree is `vcpkg_installed/`; neither relies on the sibling `Iris` checkout.

Install the dependencies from VS Code with **IRIS: install dependencies**, or run:

```powershell
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
C:\vcpkg\vcpkg.exe install --triplet x64-windows --overlay-triplets vcpkg-triplets --x-manifest-root . --x-install-root vcpkg_installed
```

The configured packages are Boost.System and FFmpeg's `avcodec`, `avformat`, and NV codec support.
Pose inference uses the separately installed LibTorch bundle at `C:\libtorch` by default (override
`IRIS_TORCH_ROOT` at configure time). CUDA remains enabled for the project's capture and
image-processing code. The task deliberately uses the
installed VS 2022 Build Tools (MSVC 14.44), rather than VS 18/MSVC 14.51, which CUDA 12.8 cannot
use reliably. `vcpkg-triplets/x64-windows.cmake` keeps the CUDA compatibility flag for the
project build.
The default CMake preset assumes vcpkg is installed at `C:\vcpkg`.
