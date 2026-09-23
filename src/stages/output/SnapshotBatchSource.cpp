#include "iris/stages/output/SnapshotBatchSource.hpp"
#include <algorithm>
namespace iris {
void SnapshotBatchSource::publish(std::shared_ptr<const Packet> packet) noexcept {
    { std::scoped_lock lock(mutex_); latest_ = std::move(packet); ++generation_; }
    ready_.notify_all();
}
std::shared_ptr<const Packet> SnapshotBatchSource::wait_for_cameras(const std::vector<CameraId>& ids, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    auto observed = generation_;
    while (ready_.wait_until(lock, deadline, [&] { return generation_ > observed; })) {
        observed = generation_;
        if (!latest_ || latest_->frames.size() != ids.size()) continue;
        bool complete = true;
        for (const auto id : ids) if (std::ranges::count(latest_->frames, id, &Frame::camera) != 1) { complete = false; break; }
        if (complete) return latest_;
    }
    return {};
}
}
