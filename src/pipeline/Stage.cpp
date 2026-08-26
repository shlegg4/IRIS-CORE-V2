#include "iris/pipeline/Stage.hpp"

#include <utility>

namespace iris {

Stage::Stage(Channel<Packet>& input, Channel<Packet>* output) : input_(input), output_(output) {}

Stage::~Stage() { stop(); }

void Stage::start() {
    if (!running_.exchange(true)) {
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
    while (true) {
        auto packet = input_.receive();
        if (!packet) {
            break;
        }
        process(*packet);
        if (output_ && output_->send(std::move(*packet)) == SendResult::Closed) {
            break;
        }
    }
    if (output_) {
        output_->close();
    }
}

} // namespace iris
