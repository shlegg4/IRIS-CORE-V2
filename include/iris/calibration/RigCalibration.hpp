#pragma once
#include "iris/pipeline/Frame.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace iris {
struct RigCameraCalibration {
    CameraId camera_id{};
    std::array<float,9> intrinsics{};
    std::array<float,5> distortion{};
    std::array<float,9> R_w2c{};
    std::array<float,3> t_w2c{};
    Extent2D resolution{};
};
struct RigCalibration {
    std::vector<RigCameraCalibration> cameras;
    std::uint64_t revision{};
    std::chrono::system_clock::time_point created_at;
    std::string method{"da3"};
    bool metric_scale{};
};
class CalibrationStore {
  public:
    std::shared_ptr<const RigCalibration> snapshot() const noexcept { return current_.load(); }
    void publish(std::shared_ptr<const RigCalibration> value) noexcept { current_.store(std::move(value)); }
    void clear() noexcept { current_.store(nullptr); }
  private:
    std::atomic<std::shared_ptr<const RigCalibration>> current_;
};
}
