#include "iris/cli/InteractiveCli.hpp"
#include "iris/runtime/Runtime.hpp"

#include <cassert>
#include <chrono>

int main() {
    std::string error;
    auto status = iris::InteractiveCli::parse("status", error);
    assert(status && std::holds_alternative<iris::GetStatusCommand>(*status));

    auto record =
        iris::InteractiveCli::parse("record start \"recordings/my session.mp4\" 8000000 30", error);
    assert(record && std::holds_alternative<iris::StartRecordingCommand>(*record));
    const auto& recording = std::get<iris::StartRecordingCommand>(*record);
    assert(recording.destination == "recordings/my session.mp4");
    assert(recording.bitrate == 8'000'000);
    assert(recording.frame_rate == 30);

    auto invalid = iris::InteractiveCli::parse("record start out.mp4 nope", error);
    assert(!invalid);
    assert(!error.empty());

    auto capture = iris::InteractiveCli::parse(
        "capture configure --device-index 0 --width 1280 --height 720 --fps 30000/1001 "
        "--format yuy2 --cuda-device 0 --sample-queue 2 --frame-pool 8 "
        "--overflow block --rotation cw90 --allow-fallback true --reconnect false",
        error);
    assert(capture && std::holds_alternative<iris::ConfigureCaptureCommand>(*capture));
    const auto& patch = std::get<iris::ConfigureCaptureCommand>(*capture).patch;
    assert(patch.device_index == 0);
    assert(patch.width == 1280);
    assert(patch.height == 720);
    assert(patch.frame_rate && patch.frame_rate->numerator == 30'000);
    assert(patch.frame_rate && patch.frame_rate->denominator == 1'001);
    assert(patch.format == iris::PixelFormat::Yuy2);
    assert(patch.cuda_device == 0);
    assert(patch.sample_queue_capacity == 2);
    assert(patch.frame_pool_capacity == 8);
    assert(patch.overflow == iris::OverflowPolicy::Block);
    assert(patch.rotation == iris::FrameRotation::Clockwise90);
    assert(patch.allow_format_fallback == true);
    assert(patch.reconnect == false);

    auto add = iris::InteractiveCli::parse("capture add 7 1", error);
    assert(add && std::holds_alternative<iris::AddCameraCommand>(*add));
    assert(std::get<iris::AddCameraCommand>(*add).camera.camera_id == 7);
    assert(std::get<iris::AddCameraCommand>(*add).camera.capture.device_index == 1);

    auto remove = iris::InteractiveCli::parse("capture remove 7", error);
    assert(remove && std::holds_alternative<iris::RemoveCameraCommand>(*remove));
    assert(std::get<iris::RemoveCameraCommand>(*remove).camera_id == 7);

    auto targeted =
        iris::InteractiveCli::parse("capture configure --camera 7 --rotation 180", error);
    assert(targeted && std::holds_alternative<iris::ConfigureCaptureCommand>(*targeted));
    assert(std::get<iris::ConfigureCaptureCommand>(*targeted).camera_id == 7);

    auto sync = iris::InteractiveCli::parse("capture sync 4 6 partial", error);
    assert(sync && std::holds_alternative<iris::ConfigureSynchronizerCommand>(*sync));
    const auto& sync_command = std::get<iris::ConfigureSynchronizerCommand>(*sync);
    assert(sync_command.tolerance == std::chrono::milliseconds(4));
    assert(sync_command.queue_capacity == 6);
    assert(sync_command.incomplete_batch_policy == iris::IncompleteBatchPolicy::EmitPartial);

    iris::Runtime runtime(iris::CaptureConfig{}, 0);
    runtime.start();
    auto response = runtime.execute(iris::GetStatusCommand{});
    assert(response);
    assert(response.snapshot);
    assert(response.snapshot->state == iris::RuntimeState::Stopped);
    assert(response.snapshot->cameras.size() == 1);

    iris::SharedMemoryOutputConfig shared_memory;
    shared_memory.enabled = false;
    response = runtime.execute(iris::ConfigureSharedMemoryCommand{shared_memory});
    assert(response);

    response = runtime.execute(iris::ConfigureCaptureCommand{patch});
    assert(response);

    iris::CameraCaptureConfig second_camera;
    second_camera.camera_id = 7;
    second_camera.capture.device_index = 1;
    response = runtime.execute(iris::AddCameraCommand{second_camera});
    assert(response);
    assert(response.snapshot && response.snapshot->cameras.size() == 2);

    response = runtime.execute(iris::ConfigureCaptureCommand{patch});
    assert(!response);
    assert(response.status == iris::RuntimeCommandStatus::Rejected);

    response = runtime.execute(iris::ConfigureCaptureCommand{patch, 7});
    assert(response);

    response = runtime.execute(iris::RemoveCameraCommand{7});
    assert(response);
    assert(response.snapshot && response.snapshot->cameras.size() == 1);

    response = runtime.execute(iris::RemoveCameraCommand{0});
    assert(!response);

    response = runtime.execute(sync_command);
    assert(response);
    assert(response.snapshot && response.snapshot->sync_tolerance == std::chrono::milliseconds(4));
    assert(response.snapshot && response.snapshot->sync_queue_capacity == 6);

    response = runtime.execute(iris::ConfigureCaptureCommand{});
    assert(!response);
    assert(response.status == iris::RuntimeCommandStatus::Rejected);

    response = runtime.execute(iris::StartRecordingCommand{"test.mp4", {}, {}});
    assert(!response);
    assert(response.status == iris::RuntimeCommandStatus::Rejected);

    response = runtime.execute(iris::ShutdownCommand{});
    assert(response);
    assert(response.snapshot);
    assert(response.snapshot->state == iris::RuntimeState::Shutdown);
    runtime.stop();
}
