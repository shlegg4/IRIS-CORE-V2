cmake_minimum_required(VERSION 3.24)
# CUDA 12.8 predates the installed VS 18 toolset; nvcc otherwise rejects it during compiler ID.
set(CMAKE_CUDA_FLAGS_INIT "--allow-unsupported-compiler")
project(IRIS VERSION 0.2.0 LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(IRIS_BUILD_ASSET_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
option(IRIS_BUILD_TESTS "Build IRIS tests" ON)
option(IRIS_BUILD_TOOLS "Build IRIS diagnostic tools" ON)
option(IRIS_BUILD_HARDWARE_TESTS "Build tests requiring a camera and CUDA GPU" OFF)
option(IRIS_ENABLE_TENSORRT "Enable TensorRT multiview inference" ON)
set(IRIS_TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root")
set(IRIS_VCPKG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows"
    CACHE PATH "IRIS_V2 local vcpkg x64-windows install root")
set(IRIS_FFMPEG_ROOT "${IRIS_VCPKG_ROOT}"
    CACHE PATH "FFmpeg x64-windows package root")
set(IRIS_TORCH_ROOT "C:/libtorch" CACHE PATH "LibTorch installation root")
find_package(CUDAToolkit REQUIRED)
list(PREPEND CMAKE_PREFIX_PATH "${IRIS_TORCH_ROOT}" "${IRIS_VCPKG_ROOT}" "${IRIS_FFMPEG_ROOT}")
find_package(Boost CONFIG REQUIRED COMPONENTS system)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(IRIS_ENABLE_TENSORRT AND IRIS_TENSORRT_ROOT)
  find_path(IRIS_TENSORRT_INCLUDE NvInfer.h PATHS "${IRIS_TENSORRT_ROOT}/include" NO_DEFAULT_PATH)
  find_library(IRIS_TENSORRT_LIBRARY nvinfer PATHS "${IRIS_TENSORRT_ROOT}/lib" "${IRIS_TENSORRT_ROOT}/lib/x64" NO_DEFAULT_PATH)
  if(IRIS_TENSORRT_INCLUDE AND IRIS_TENSORRT_LIBRARY)
    set(IRIS_HAS_TENSORRT TRUE)
    message(STATUS "TensorRT enabled: ${IRIS_TENSORRT_INCLUDE}")
  else()
    message(WARNING "IRIS_TENSORRT_ROOT was supplied but TensorRT was not found; multiview inference disabled")
  endif()
endif()

foreach(component avformat avcodec avutil)
  if(NOT EXISTS "${IRIS_FFMPEG_ROOT}/include/lib${component}/${component}.h")
    message(FATAL_ERROR "Pinned FFmpeg headers not found under IRIS_FFMPEG_ROOT=${IRIS_FFMPEG_ROOT}")
  endif()
  add_library(FFmpeg::${component} SHARED IMPORTED)
  set_target_properties(FFmpeg::${component} PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${IRIS_FFMPEG_ROOT}/debug/lib/${component}.lib"
    IMPORTED_IMPLIB_RELEASE "${IRIS_FFMPEG_ROOT}/lib/${component}.lib"
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    INTERFACE_INCLUDE_DIRECTORIES "${IRIS_FFMPEG_ROOT}/include")
endforeach()
add_library(iris
  src/pipeline/Stage.cpp src/runtime/Pipeline.cpp src/runtime/Runtime.cpp
  src/cli/InteractiveCli.cpp
  src/infrastructure/gpu/CudaResources.cpp src/infrastructure/gpu/FramePool.cpp
  src/infrastructure/metrics/MetricRegistry.cpp src/infrastructure/metrics/MetricsExporter.cpp
  src/infrastructure/metrics/PrometheusExporter.cpp
  src/stages/capture/CaptureStage.cpp src/stages/capture/timing/CaptureClock.cpp
  src/stages/capture/FrameSynchronizerStage.cpp
  src/stages/capture/media_foundation/MediaFoundationSource.cpp
  src/stages/capture/decoding/GpuDecoder.cpp src/stages/capture/decoding/Yuy2CudaConverter.cu
  src/stages/capture/decoding/BgrRotate.cu src/stages/pose/MultiviewPreprocess.cu
  src/stages/pose/PoseStage.cpp src/stages/pose/MultiviewPoseStage.cpp src/stages/pose/TensorRtMultiviewEngine.cpp src/stages/output/OutputStage.cpp src/stages/output/PreviewSink.cpp
  src/stages/output/PreviewHttpServer.cpp
  src/stages/output/MjpegResize.cu
  src/stages/output/disk/GpuMp4Writer.cpp src/stages/output/disk/BgrToNv12.cu)
add_library(iris::iris ALIAS iris)
target_compile_features(iris PUBLIC cxx_std_20)
target_compile_definitions(iris PRIVATE IRIS_BUILD_ASSET_DIRECTORY="${IRIS_BUILD_ASSET_DIRECTORY}")
set_target_properties(iris PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(iris PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src "${IRIS_VCPKG_ROOT}/include")
if(IRIS_HAS_TENSORRT)
  target_include_directories(iris PRIVATE "${IRIS_TENSORRT_INCLUDE}")
  target_link_libraries(iris PRIVATE "${IRIS_TENSORRT_LIBRARY}")
  target_compile_definitions(iris PRIVATE IRIS_HAS_TENSORRT=1)
endif()
target_link_libraries(iris PUBLIC CUDA::cudart PRIVATE CUDA::nvjpeg FFmpeg::avformat
  FFmpeg::avcodec FFmpeg::avutil CUDA::cuda_driver Boost::system)
target_link_libraries(iris PRIVATE "${TORCH_LIBRARIES}")
if(WIN32)
  target_compile_definitions(iris PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(iris PRIVATE mf mfplat mfreadwrite mfuuid ole32 strmiids ws2_32)
endif()
if(MSVC)
  target_compile_options(iris PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
endif()
add_executable(iris_app src/main.cpp)
target_link_libraries(iris_app PRIVATE iris::iris)

add_custom_command(TARGET iris_app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:iris_app>/assets"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/pear_ehm_libtorch.pt"
    "$<TARGET_FILE_DIR:iris_app>/assets/pear_ehm_libtorch.pt"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/rtmo_s_full_epipolar_fp16.engine"
    "$<TARGET_FILE_DIR:iris_app>/assets/rtmo_s_full_epipolar_fp16.engine"
  COMMENT "Provisioning IRIS pose assets")

add_custom_command(TARGET iris_app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${IRIS_FFMPEG_ROOT}/$<IF:$<CONFIG:Debug>,debug/bin,bin>"
    "$<TARGET_FILE_DIR:iris_app>"
  COMMENT "Copying pinned FFmpeg runtime bundle")
if(WIN32 AND TORCH_INSTALL_PREFIX)
  file(GLOB _iris_torch_dlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET iris_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${_iris_torch_dlls} "$<TARGET_FILE_DIR:iris_app>"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying LibTorch runtime bundle")
endif()
if(WIN32 AND IRIS_HAS_TENSORRT)
  file(GLOB _iris_tensorrt_dlls "${IRIS_TENSORRT_ROOT}/bin/nvinfer*.dll" "${IRIS_TENSORRT_ROOT}/lib/nvinfer*.dll")
  if(_iris_tensorrt_dlls)
    add_custom_command(TARGET iris_app POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_iris_tensorrt_dlls} "$<TARGET_FILE_DIR:iris_app>"
      COMMAND_EXPAND_LISTS COMMENT "Copying TensorRT runtime")
  endif()
endif()
if(IRIS_BUILD_TOOLS)
  add_executable(iris_list_cameras tools/list_cameras.cpp)
  target_link_libraries(iris_list_cameras PRIVATE iris::iris)
  target_include_directories(iris_list_cameras PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_executable(iris_capture_probe tools/capture_probe.cpp)
  target_link_libraries(iris_capture_probe PRIVATE iris::iris)
  add_executable(iris_capture_validation tools/capture_validation.cpp)
  target_link_libraries(iris_capture_validation PRIVATE iris::iris)
  add_executable(iris_mf_probe tools/mf_probe.cpp)
  target_link_libraries(iris_mf_probe PRIVATE iris::iris)
  target_include_directories(iris_mf_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
if(IRIS_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
