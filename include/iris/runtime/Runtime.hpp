#pragma once

#include "iris/runtime/RuntimeControl.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"

#include <cstdint>
#include <memory>

namespace iris {

class Runtime {
  public:
    explicit Runtime(CaptureConfig = {}, std::uint16_t metrics_port = 9464);
    explicit Runtime(MultiCameraCaptureConfig, std::uint16_t metrics_port = 9464);
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void start();
    RuntimeCommandResponse execute(RuntimeCommand);
    RuntimeSnapshot snapshot() const;
    void stop();
    int run();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
