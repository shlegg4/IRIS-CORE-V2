#pragma once

#include "iris/stages/output/PreviewHttpServer.hpp"
#include "iris/stages/output/PreviewSink.hpp"

#include <memory>

namespace iris::output {
class H264Transport final : public PreviewTransport {
  public:
    H264Transport(H264PreviewConfig config, PreviewHttpServer& server);
    ~H264Transport() override;
    void start() override;
    void publish(PreviewPacket packet) noexcept override;
    void stop() noexcept override;
    [[nodiscard]] PreviewTransportHealth health() const override;
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
