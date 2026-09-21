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
option(IRIS_ENABLE_WEBRTC "Enable GStreamer WebRTC preview transport" ON)
set(IRIS_TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root")
set(IRIS_VCPKG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/x64-windows"
    CACHE PATH "IRIS_V2 local vcpkg x64-windows install root")
set(IRIS_FFMPEG_ROOT "${IRIS_VCPKG_ROOT}"
    CACHE PATH "FFmpeg x64-windows package root")
set(IRIS_TORCH_ROOT "C:/libtorch" CACHE PATH "LibTorch installation root")
set(IRIS_GSTREAMER_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/gstreamer-1.26.5-official/PFiles64/gstreamer/1.0/msvc_x86_64"
    CACHE PATH "Official GStreamer 1.26.5 MSVC x64 runtime/development root")
set(IRIS_GSTREAMER_DEV_ROOT "${IRIS_GSTREAMER_RUNTIME_ROOT}"
    CACHE PATH "Official GStreamer 1.26.5 MSVC x64 development package root")
find_package(CUDAToolkit REQUIRED)
list(PREPEND CMAKE_PREFIX_PATH "${IRIS_TORCH_ROOT}" "${IRIS_VCPKG_ROOT}" "${IRIS_FFMPEG_ROOT}")
find_package(Boost CONFIG REQUIRED COMPONENTS system)
find_package(Torch REQUIRED)
if(IRIS_ENABLE_WEBRTC)
  unset(IRIS_GSTREAMER_INCLUDE_DIR CACHE)
  unset(IRIS_GLIB_INCLUDE_DIR CACHE)
  unset(IRIS_GLIB_CONFIG_INCLUDE_DIR CACHE)
  foreach(component gstreamer-1.0 gstapp-1.0 gstvideo-1.0 gstwebrtc-1.0 gstsdp-1.0
                    glib-2.0 gobject-2.0 gio-2.0 nice srtp2)
    unset(IRIS_GSTREAMER_${component}_LIBRARY CACHE)
  endforeach()
  foreach(_gst_required IN ITEMS bin lib/gstreamer-1.0)
    if(NOT EXISTS "${IRIS_GSTREAMER_RUNTIME_ROOT}/${_gst_required}")
      message(FATAL_ERROR "Official GStreamer runtime is incomplete: missing ${IRIS_GSTREAMER_RUNTIME_ROOT}/${_gst_required}")
    endif()
  endforeach()
  find_path(IRIS_GSTREAMER_INCLUDE_DIR gst/gst.h
    PATHS "${IRIS_GSTREAMER_DEV_ROOT}/include/gstreamer-1.0" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  find_path(IRIS_GLIB_INCLUDE_DIR glib.h
    PATHS "${IRIS_GSTREAMER_DEV_ROOT}/include/glib-2.0" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  find_path(IRIS_GLIB_CONFIG_INCLUDE_DIR glibconfig.h
    PATHS "${IRIS_GSTREAMER_DEV_ROOT}/lib/glib-2.0/include" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  foreach(component gstreamer-1.0 gstapp-1.0 gstvideo-1.0 gstwebrtc-1.0 gstsdp-1.0
                    glib-2.0 gobject-2.0 gio-2.0 nice srtp2)
    find_library(IRIS_GSTREAMER_${component}_LIBRARY NAMES ${component}
      PATHS "${IRIS_GSTREAMER_DEV_ROOT}/lib" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    list(APPEND IRIS_GSTREAMER_LIBRARIES "${IRIS_GSTREAMER_${component}_LIBRARY}")
  endforeach()
  message(STATUS "GStreamer WebRTC enabled: ${IRIS_GSTREAMER_INCLUDE_DIR}")
  message(STATUS "Using official GStreamer headers, libraries, runtime DLLs, and plugins")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(IRIS_ENABLE_TENSORRT AND IRIS_TENSORRT_ROOT)
  find_path(IRIS_TENSORRT_INCLUDE NvInfer.h PATHS "${IRIS_TENSORRT_ROOT}/include" NO_DEFAULT_PATH)
  find_library(IRIS_TENSORRT_LIBRARY NAMES nvinfer_10 nvinfer PATHS "${IRIS_TENSORRT_ROOT}/lib" "${IRIS_TENSORRT_ROOT}/lib/x64" NO_DEFAULT_PATH)
  if(IRIS_TENSORRT_INCLUDE AND IRIS_TENSORRT_LIBRARY)
    set(IRIS_HAS_TENSORRT TRUE)
    find_library(IRIS_TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin_10 nvinfer_plugin PATHS "${IRIS_TENSORRT_ROOT}/lib" "${IRIS_TENSORRT_ROOT}/lib/x64" NO_DEFAULT_PATH REQUIRED)
    find_package(Eigen3 CONFIG REQUIRED)
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs imgproc)
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
  src/tools/RigCalibrationTool.cpp
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
  src/stages/output/WebRtcTransport.cpp
  src/stages/output/MjpegResize.cu
  src/stages/output/disk/GpuMp4Writer.cpp src/stages/output/disk/BgrToNv12.cu)
add_library(iris::iris ALIAS iris)
target_compile_features(iris PUBLIC cxx_std_20)
target_compile_definitions(iris PRIVATE IRIS_BUILD_ASSET_DIRECTORY="${IRIS_BUILD_ASSET_DIRECTORY}")
set_target_properties(iris PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(iris PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src "${IRIS_VCPKG_ROOT}/include")
if(IRIS_ENABLE_WEBRTC)
  target_compile_definitions(iris PRIVATE IRIS_HAS_WEBRTC=1 GST_USE_UNSTABLE_API=1)
  target_include_directories(iris PRIVATE "${IRIS_GSTREAMER_INCLUDE_DIR}" "${IRIS_GLIB_INCLUDE_DIR}" "${IRIS_GLIB_CONFIG_INCLUDE_DIR}")
  target_link_libraries(iris PRIVATE ${IRIS_GSTREAMER_LIBRARIES})
endif()
if(IRIS_HAS_TENSORRT)
  target_sources(iris PRIVATE src/tools/da3/src/preprocess.cpp src/tools/da3/src/tensor_rt.cpp)
  target_include_directories(iris PRIVATE "${IRIS_TENSORRT_INCLUDE}")
  target_include_directories(iris PRIVATE src/tools/da3/include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(iris PRIVATE "${IRIS_TENSORRT_LIBRARY}" "${IRIS_TENSORRT_PLUGIN_LIBRARY}" Eigen3::Eigen ${OpenCV_LIBS})
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
    "${CMAKE_CURRENT_SOURCE_DIR}/rtmo_s.engine"
    "$<TARGET_FILE_DIR:iris_app>/assets/rtmo_s.engine"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/da3_base.trt"
    "$<TARGET_FILE_DIR:iris_app>/assets/da3_base.trt"
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
if(WIN32 AND IRIS_ENABLE_WEBRTC)
  set(_iris_gst_runtime "$<TARGET_FILE_DIR:iris_app>/gstreamer-runtime")
  add_custom_command(TARGET iris_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${IRIS_GSTREAMER_RUNTIME_ROOT}/bin"
      "${_iris_gst_runtime}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${IRIS_GSTREAMER_RUNTIME_ROOT}/lib/gstreamer-1.0"
      "${_iris_gst_runtime}/plugins"
    COMMENT "Copying official GStreamer WebRTC runtime bundle")
endif()
if(WIN32)
  file(GLOB _iris_opencv_dlls "${IRIS_VCPKG_ROOT}/bin/opencv_*.dll")
  if(_iris_opencv_dlls)
    add_custom_command(TARGET iris_app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_iris_opencv_dlls} "$<TARGET_FILE_DIR:iris_app>" COMMAND_EXPAND_LISTS COMMENT "Copying OpenCV runtime")
  endif()
endif()
if(IRIS_BUILD_TOOLS)
  include(CTest)
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
  if(IRIS_ENABLE_WEBRTC)
    add_executable(webrtc_runtime_probe src/tools/WebRtcRuntimeProbe.cpp)
    target_compile_definitions(webrtc_runtime_probe PRIVATE GST_USE_UNSTABLE_API=1)
    target_include_directories(webrtc_runtime_probe PRIVATE "${IRIS_GSTREAMER_INCLUDE_DIR}" "${IRIS_GLIB_INCLUDE_DIR}" "${IRIS_GLIB_CONFIG_INCLUDE_DIR}")
    target_link_libraries(webrtc_runtime_probe PRIVATE ${IRIS_GSTREAMER_LIBRARIES})
    add_test(NAME webrtc_runtime_probe COMMAND webrtc_runtime_probe)
    set_tests_properties(webrtc_runtime_probe PROPERTIES ENVIRONMENT
      "GST_PLUGIN_PATH=$<TARGET_FILE_DIR:iris_app>/gstreamer-runtime/plugins;GST_PLUGIN_SYSTEM_PATH=;GST_REGISTRY=$<TARGET_FILE_DIR:iris_app>/gstreamer-runtime/registry.bin"
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:iris_app>/gstreamer-runtime/bin")
  endif()
endif()
if(IRIS_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
