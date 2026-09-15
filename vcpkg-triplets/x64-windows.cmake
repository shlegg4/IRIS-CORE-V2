set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

# Keep dependency builds aligned with the project's CUDA 12.8 compatibility setting.
set(VCPKG_CMAKE_CONFIGURE_OPTIONS
    "-DCMAKE_CUDA_FLAGS_INIT=--allow-unsupported-compiler")
