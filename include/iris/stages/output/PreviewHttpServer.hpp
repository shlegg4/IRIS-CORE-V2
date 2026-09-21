#pragma once

#include "iris/stages/output/PreviewSink.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace iris::output {

class PreviewHttpServer final {
  public:
    using StatusProvider = std::function<std::string()>;
    // A handler is deliberately scoped to one WebSocket.  This keeps WebRTC
    // peer state out of the HTTP server while allowing more than one browser.
    using WebRtcMessageHandler = std::function<std::string(const std::string& message)>;
    using WebRtcHandlerFactory = std::function<WebRtcMessageHandler()>;
    PreviewHttpServer(std::string bind_address, std::uint16_t port, StatusProvider status,
                      WebRtcHandlerFactory webrtc_handler_factory = {});
    ~PreviewHttpServer();
    PreviewHttpServer(const PreviewHttpServer&) = delete;
    void start();
    void stop() noexcept;
    void set_frame(CameraId camera, std::shared_ptr<const std::vector<std::uint8_t>> jpeg);
    void publish_event(std::string json_envelope);
    [[nodiscard]] PreviewTransportHealth health() const;
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

bool is_loopback_address(const std::string& address) noexcept;

} // namespace iris::output
