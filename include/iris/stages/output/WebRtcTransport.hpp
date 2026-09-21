#pragma once

#include "iris/stages/output/PreviewHttpServer.hpp"
#include "iris/stages/output/PreviewSink.hpp"

#include <memory>

namespace iris::output {

// CPU-copy implementation of the preview WebRTC transport.  The public
// handler factory gives each signalling socket its own peer session.
class WebRtcTransport final : public PreviewTransport {
  public:
    explicit WebRtcTransport(WebRtcPreviewConfig config);
    ~WebRtcTransport() override;
    void start() override;
    void publish(PreviewPacket packet) noexcept override;
    void stop() noexcept override;
    [[nodiscard]] PreviewTransportHealth health() const override;
    [[nodiscard]] PreviewHttpServer::WebRtcHandlerFactory signalling_factory();
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris::output
