#pragma once
#include "iris/infrastructure/metrics/ChannelMetrics.hpp"
#include "iris/pipeline/OverflowPolicy.hpp"
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
namespace iris {
enum class SendResult { Sent, ReplacedOldest, DroppedNewest, Closed };

struct ChannelStats {
    std::size_t depth{}, peak_depth{};
    std::uint64_t sent{}, received{}, dropped{};
    bool closed{};
};
template <typename T> class Channel {
  public:
    explicit Channel(std::size_t capacity = 1, OverflowPolicy policy = OverflowPolicy::Block,
                     std::optional<infrastructure::metrics::ChannelMetrics> metrics = std::nullopt)
        : capacity_(capacity), policy_(policy), metrics_(std::move(metrics)) {
        if (!capacity) {
            throw std::invalid_argument("channel capacity must be greater than zero");
        }
    }
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    SendResult send(T value) {
        std::unique_lock lock(mutex_);
        bool replaced_oldest = false;
        if (policy_ == OverflowPolicy::Block) {
            writable_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
        }
        if (closed_) {
            if (metrics_) {
                metrics_->rejected_closed.increment();
            }
            return SendResult::Closed;
        }
        if (queue_.size() >= capacity_) {
            ++dropped_;
            if (metrics_) {
                metrics_->dropped.increment();
            }
            if (policy_ == OverflowPolicy::DropNewest) {
                if (metrics_) {
                    metrics_->dropped_newest.increment();
                    metrics_->depth.set(static_cast<double>(queue_.size()));
                }
                return SendResult::DroppedNewest;
            }
            queue_.pop();
            enqueue_times_.pop();
            replaced_oldest = true;
            if (metrics_) {
                metrics_->dropped_oldest.increment();
            }
        }
        queue_.push(std::move(value));
        enqueue_times_.push(std::chrono::steady_clock::now());
        ++sent_;
        peak_depth_ = std::max(peak_depth_, queue_.size());
        if (metrics_) {
            metrics_->sent.increment();
            metrics_->depth.set(static_cast<double>(queue_.size()));
            metrics_->peak_depth.set(static_cast<double>(peak_depth_));
        }
        readable_.notify_one();
        return replaced_oldest ? SendResult::ReplacedOldest : SendResult::Sent;
    }
    std::optional<T> receive() {
        std::unique_lock lock(mutex_);
        readable_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        const auto enqueued = enqueue_times_.front();
        queue_.pop();
        enqueue_times_.pop();
        ++received_;
        if (metrics_) {
            metrics_->received.increment();
            metrics_->depth.set(static_cast<double>(queue_.size()));
            const auto residence = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enqueued).count();
            metrics_->residence_ms.observe(residence);
            metrics_->last_residence_ms.set(residence);
        }
        writable_.notify_one();
        return value;
    }
    std::optional<T> try_receive() {
        std::scoped_lock lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        const auto enqueued = enqueue_times_.front();
        queue_.pop();
        enqueue_times_.pop();
        ++received_;
        if (metrics_) {
            metrics_->received.increment();
            metrics_->depth.set(static_cast<double>(queue_.size()));
            const auto residence = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enqueued).count();
            metrics_->residence_ms.observe(residence);
            metrics_->last_residence_ms.set(residence);
        }
        writable_.notify_one();
        return value;
    }
    void close() {
        std::scoped_lock lock(mutex_);
        closed_ = true;
        readable_.notify_all();
        writable_.notify_all();
    }
    ChannelStats stats() const {
        std::scoped_lock lock(mutex_);
        return {queue_.size(), peak_depth_, sent_, received_, dropped_, closed_};
    }

  private:
    const std::size_t capacity_;
    const OverflowPolicy policy_;
    std::optional<infrastructure::metrics::ChannelMetrics> metrics_;
    mutable std::mutex mutex_;
    std::condition_variable readable_, writable_;
    std::queue<T> queue_;
    std::queue<std::chrono::steady_clock::time_point> enqueue_times_;
    std::size_t peak_depth_{};
    std::uint64_t sent_{}, received_{}, dropped_{};
    bool closed_{};
};
} // namespace iris
