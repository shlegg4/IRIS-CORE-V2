#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/OutputStage.hpp"
#include "iris/stages/pose/PoseConfig.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"

#include <memory>
#include <functional>

namespace iris {
class Pipeline {
  public:
    Pipeline(CaptureConfig, infrastructure::metrics::MetricRegistry&, PoseConfig = {});
    Pipeline(MultiCameraCaptureConfig, infrastructure::metrics::MetricRegistry&, PoseConfig = {});
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
    void set_preview_status_provider(std::function<std::string()>);
    OutputCommandResult configure_disk(DiskOutputConfig);
    OutputCommandResult start_recording();
    OutputCommandResult stop_recording();
    std::size_t processed_count() const noexcept;
    PreviewTransportHealth preview_health() const;
    bool healthy() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace iris
