#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/OutputStage.hpp"
#include "iris/stages/output/SnapshotBatchSource.hpp"
#include "iris/stages/pose/PoseConfig.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"
#include "iris/stages/capture/SynchronizedVideoConfig.hpp"

#include <memory>
#include <string>

namespace iris {
class RigCalibrationTool;
class Pipeline {
  public:
    Pipeline(CaptureConfig, infrastructure::metrics::MetricRegistry&, PoseConfig = {}, std::shared_ptr<RigCalibrationTool> = {},
             std::shared_ptr<SnapshotBatchSource> = {});
    Pipeline(MultiCameraCaptureConfig, infrastructure::metrics::MetricRegistry&, PoseConfig = {}, std::shared_ptr<RigCalibrationTool> = {},
             std::shared_ptr<SnapshotBatchSource> = {});
    Pipeline(SynchronizedVideoConfig, infrastructure::metrics::MetricRegistry&, PoseConfig = {}, std::shared_ptr<RigCalibrationTool> = {},
             std::shared_ptr<SnapshotBatchSource> = {});
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    void start();
    // Stop capture from producing new frames. Downstream stages are allowed to drain.
    void stop_producing();
    // Stop production, close the pipeline channels, and join all stage workers.
    void wait();
    void stop();
    OutputCommandResult configure_shared_memory(SharedMemoryOutputConfig);
    OutputCommandResult configure_preview(PreviewConfig);
    OutputCommandResult configure_disk(DiskOutputConfig);
    OutputCommandResult start_recording();
    OutputCommandResult stop_recording();
    std::size_t processed_count() const noexcept;
    std::vector<VideoDecodeStatus> video_decode_status() const;
    PreviewTransportHealth preview_health() const;
    bool healthy() const noexcept;
    std::string output_failure() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace iris
