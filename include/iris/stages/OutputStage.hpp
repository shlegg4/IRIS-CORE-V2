#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/pipeline/Packet.hpp"
#include "iris/stages/output/OutputCommand.hpp"
#include "iris/stages/output/OutputConfig.hpp"
#include "iris/stages/output/PreviewSink.hpp"

#include <cstddef>
#include <memory>

namespace iris {

class OutputStage final {
  public:
    OutputStage(Channel<Packet>& input, infrastructure::metrics::MetricRegistry& metrics,
                OutputConfig config = {});
    ~OutputStage();

    OutputStage(const OutputStage&) = delete;
    OutputStage& operator=(const OutputStage&) = delete;

    void start();
    void stop();

    OutputCommandResult configure_shared_memory(SharedMemoryOutputConfig config);
    OutputCommandResult configure_disk(DiskOutputConfig config);
    OutputCommandResult start_recording();
    OutputCommandResult stop_recording();

    [[nodiscard]] std::size_t processed_count() const noexcept;
    [[nodiscard]] PreviewTransportHealth preview_health() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
