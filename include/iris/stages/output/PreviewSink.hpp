#pragma once

#include "iris/pipeline/Packet.hpp"
#include "iris/stages/output/OutputCommand.hpp"
#include "iris/stages/output/OutputConfig.hpp"

#include <cstddef>
#include <memory>
#include <functional>
#include <string>

namespace iris {

// A shared packet view deliberately retains Frame::buffer::owner until each consumer is done.
using PreviewPacket = std::shared_ptr<const Packet>;

struct PreviewTransportHealth {
    bool enabled{};
    std::size_t connected_clients{};
    std::size_t published_packets{};
    std::size_t dropped_packets{};
    std::string last_error;
    std::string codec;
    int cuda_device{-1};
    std::uint32_t latency_ms{};
};

class PreviewTransport {
  public:
    virtual ~PreviewTransport() = default;
    virtual void start() = 0;
    virtual void publish(PreviewPacket packet) noexcept = 0;
    virtual void stop() noexcept = 0;
    [[nodiscard]] virtual PreviewTransportHealth health() const = 0;
};

class PreviewSink final {
  public:
    explicit PreviewSink(PreviewConfig config = {});
    ~PreviewSink();
    PreviewSink(const PreviewSink&) = delete;
    PreviewSink& operator=(const PreviewSink&) = delete;
    void start();
    void publish(PreviewPacket packet) noexcept;
    void stop() noexcept;
    OutputCommandResult configure(PreviewConfig config);
    void set_status_provider(std::function<std::string()> provider);
    OutputCommandResult configure_shared_memory(SharedMemoryOutputConfig config);
    [[nodiscard]] PreviewTransportHealth shared_memory_health() const;
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
