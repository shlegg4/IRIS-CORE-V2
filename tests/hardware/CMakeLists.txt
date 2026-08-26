add_test(NAME camera_enumeration_test COMMAND iris_list_cameras)
set_tests_properties(camera_enumeration_test PROPERTIES
  LABELS "hardware;smoke"
  TIMEOUT 15
  RESOURCE_LOCK camera_bus)

add_test(NAME media_foundation_capture_test
  COMMAND iris_mf_probe --device-index 0 --width 1920 --height 1080 --fps 30 --frames 30)
set_tests_properties(media_foundation_capture_test PROPERTIES
  LABELS "hardware;capture;smoke"
  TIMEOUT 30
  RESOURCE_LOCK camera_0)

add_test(NAME capture_pipeline_validation_test
  COMMAND iris_capture_validation
    --device-index 0 --width 1920 --height 1080 --fps 30 --frames 120
    --max-drop-rate 0.05 --max-p95-latency-ms 75 --max-clock-drift-ppm 250
    --output validation-results/capture-pipeline-validation.json)
set_tests_properties(capture_pipeline_validation_test PROPERTIES
  LABELS "hardware;capture;validation"
  TIMEOUT 60
  RESOURCE_LOCK camera_0
  ATTACHED_FILES_ON_FAIL
    "${CMAKE_CURRENT_BINARY_DIR}/validation-results/capture-pipeline-validation.json")

add_test(NAME capture_pipeline_smoke_test
  COMMAND iris_capture_validation
    --device-index 0 --width 1920 --height 1080 --fps 30 --frames 30
    --max-drop-rate 0.20 --max-decode-failure-rate 0.0
    --max-p95-latency-ms 150 --max-clock-drift-ppm 500
    --min-drift-samples 120
    --output validation-results/capture-pipeline-smoke.json)
set_tests_properties(capture_pipeline_smoke_test PROPERTIES
  LABELS "hardware;capture;smoke"
  TIMEOUT 45
  RESOURCE_LOCK camera_0
  ATTACHED_FILES_ON_FAIL
    "${CMAKE_CURRENT_BINARY_DIR}/validation-results/capture-pipeline-smoke.json")

add_test(NAME capture_backpressure_test
  COMMAND iris_capture_validation
    --device-index 0 --frames 120 --consumer-delay-ms 40
    --max-drop-rate 0.80 --max-p95-latency-ms 200
    --output validation-results/capture-backpressure.json)
set_tests_properties(capture_backpressure_test PROPERTIES
  LABELS "hardware;capture;validation;backpressure"
  TIMEOUT 90
  RESOURCE_LOCK camera_0
  ATTACHED_FILES_ON_FAIL
    "${CMAKE_CURRENT_BINARY_DIR}/validation-results/capture-backpressure.json")

add_test(NAME capture_drift_test
  COMMAND iris_capture_validation
    --device-index 0 --duration 300 --max-drop-rate 0.05
    --max-p95-latency-ms 75 --max-clock-drift-ppm 150
    --output validation-results/capture-drift.json)
set_tests_properties(capture_drift_test PROPERTIES
  LABELS "hardware;capture;validation;drift"
  TIMEOUT 330
  RESOURCE_LOCK camera_0
  ATTACHED_FILES_ON_FAIL
    "${CMAKE_CURRENT_BINARY_DIR}/validation-results/capture-drift.json")

add_test(NAME capture_soak_test
  COMMAND iris_capture_validation
    --device-index 0 --duration 3600 --max-drop-rate 0.05
    --max-p95-latency-ms 75 --max-clock-drift-ppm 150
    --output validation-results/capture-soak.json)
set_tests_properties(capture_soak_test PROPERTIES
  LABELS "hardware;capture;soak"
  TIMEOUT 3700
  RESOURCE_LOCK camera_0
  ATTACHED_FILES_ON_FAIL
    "${CMAKE_CURRENT_BINARY_DIR}/validation-results/capture-soak.json")

add_executable(output_mp4_hardware_test OutputMp4HardwareTest.cpp)
target_link_libraries(output_mp4_hardware_test PRIVATE iris::iris FFmpeg::avformat
  FFmpeg::avcodec FFmpeg::avutil)
add_test(NAME output_mp4_hardware_test COMMAND output_mp4_hardware_test)
set_tests_properties(output_mp4_hardware_test PROPERTIES
  LABELS "hardware;output;smoke"
  TIMEOUT 30
  RESOURCE_LOCK cuda_encoder)
