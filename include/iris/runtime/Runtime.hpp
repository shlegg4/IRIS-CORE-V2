#pragma once

#include "iris/runtime/RuntimeControl.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"
#include "iris/stages/pose/PoseConfig.hpp"
#include "iris/stages/output/SnapshotBatchSource.hpp"

#include <cstdint>
#include <chrono>
#include <vector>
#include <memory>

namespace iris {

class Runtime {
  public:
    explicit Runtime(CaptureConfig = {}, std::uint16_t metrics_port = 9464,
                     PoseConfig = {});
    explicit Runtime(MultiCameraCaptureConfig, std::uint16_t metrics_port = 9464,
                     PoseConfig = {});
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void start();
    RuntimeCommandResponse execute(RuntimeCommand);
    RuntimeSnapshot snapshot() const;
    std::shared_ptr<const Packet> capture_snapshot_batch(const std::vector<CameraId>&, std::chrono::milliseconds) const;
    void stop();
    int run();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
