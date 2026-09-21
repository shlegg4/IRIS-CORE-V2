#pragma once

#include "iris/stages/output/PreviewSink.hpp"
#include "iris/stages/output/H264PreviewProtocol.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace iris::output {

class PreviewHttpServer final {
  public:
    using StatusProvider = std::function<std::string()>;
    PreviewHttpServer(std::string bind_address, std::uint16_t port, StatusProvider status);
    ~PreviewHttpServer();
    PreviewHttpServer(const PreviewHttpServer&) = delete;
    void start();
    void stop() noexcept;
    void set_frame(CameraId camera, std::shared_ptr<const std::vector<std::uint8_t>> jpeg);
    void publish_event(std::string json_envelope);
    void publish_h264(H264PreviewAccessUnit access_unit);
    void set_h264_stream_config(H264PreviewStreamConfig config);
    [[nodiscard]] PreviewTransportHealth health() const;
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

bool is_loopback_address(const std::string& address) noexcept;

} // namespace iris::output
