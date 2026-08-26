cmake_minimum_required(VERSION 3.24)
project(IRIS VERSION 0.2.0 LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
option(IRIS_BUILD_TESTS "Build IRIS tests" ON)
option(IRIS_BUILD_TOOLS "Build IRIS diagnostic tools" ON)
option(IRIS_BUILD_HARDWARE_TESTS "Build tests requiring a camera and CUDA GPU" OFF)
set(IRIS_FFMPEG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Iris/build/vcpkg_installed/x64-windows"
    CACHE PATH "Pinned prebuilt FFmpeg x64-windows package root")
find_package(CUDAToolkit REQUIRED)

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
  src/stages/capture/decoding/BgrRotate.cu
  src/stages/pose/PoseStage.cpp src/stages/output/OutputStage.cpp src/stages/output/PreviewSink.cpp
  src/stages/output/disk/GpuMp4Writer.cpp src/stages/output/disk/BgrToNv12.cu)
add_library(iris::iris ALIAS iris)
target_compile_features(iris PUBLIC cxx_std_20)
set_target_properties(iris PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(iris PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(iris PUBLIC CUDA::cudart PRIVATE CUDA::nvjpeg FFmpeg::avformat
  FFmpeg::avcodec FFmpeg::avutil CUDA::cuda_driver)
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
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${IRIS_FFMPEG_ROOT}/$<IF:$<CONFIG:Debug>,debug/bin,bin>"
    "$<TARGET_FILE_DIR:iris_app>"
  COMMENT "Copying pinned FFmpeg runtime bundle")
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
