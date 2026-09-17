#include "iris/pipeline/Stage.hpp"

#include <utility>

namespace iris {

Stage::Stage(Channel<Packet>& input, Channel<Packet>* output) : input_(input), output_(output) {}

Stage::~Stage() { stop(); }

void Stage::start() {
    if (!running_.exchange(true)) {
        { std::scoped_lock lock(failure_mutex_); failure_ = nullptr; }
        worker_ = std::thread(&Stage::run, this);
    }
}

void Stage::stop() {
    if (running_.exchange(false)) {
        input_.close();
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

void Stage::run() {
    try {
        while (true) {
            auto packet = input_.receive();
            if (!packet) break;
            process(*packet);
            if (output_ && output_->send(std::move(*packet)) == SendResult::Closed) break;
        }
    } catch (...) {
        std::scoped_lock lock(failure_mutex_);
        failure_ = std::current_exception();
    }
    if (output_) {
        output_->close();
    }
}

bool Stage::healthy() const noexcept { std::scoped_lock lock(failure_mutex_); return !failure_; }
std::exception_ptr Stage::failure() const noexcept { std::scoped_lock lock(failure_mutex_); return failure_; }

} // namespace iris
