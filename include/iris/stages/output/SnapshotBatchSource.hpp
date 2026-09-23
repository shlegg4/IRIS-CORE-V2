#pragma once
#include "iris/pipeline/Packet.hpp"
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
namespace iris {
class SnapshotBatchSource {
  public:
    void publish(std::shared_ptr<const Packet>) noexcept;
    std::shared_ptr<const Packet> wait_for_cameras(const std::vector<CameraId>&, std::chrono::milliseconds);
  private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::uint64_t generation_{};
    std::shared_ptr<const Packet> latest_;
};
}
