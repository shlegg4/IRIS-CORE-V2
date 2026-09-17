#pragma once
#include "iris/calibration/RigCalibration.hpp"
#include "iris/pipeline/Packet.hpp"
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>

namespace iris {
class RigCalibrationTool {
  public:
    struct Status { std::string state{"idle"}, message; std::uint64_t source_sequence{}; };
    explicit RigCalibrationTool(std::shared_ptr<CalibrationStore>);
    ~RigCalibrationTool();
    void observe(const Packet&);
    bool start(std::filesystem::path engine, std::filesystem::path output);
    void cancel();
    Status status() const;
  private:
    void run(std::stop_token, std::filesystem::path, std::filesystem::path);
    std::shared_ptr<CalibrationStore> store_;
    mutable std::mutex mutex_;
    std::condition_variable_any changed_;
    std::optional<Packet> latest_;
    Status status_;
    std::jthread worker_;
};
}
