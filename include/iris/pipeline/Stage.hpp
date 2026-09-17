#pragma once

#include "iris/pipeline/Channel.hpp"
#include "iris/pipeline/Packet.hpp"

#include <atomic>
#include <thread>
#include <exception>
#include <mutex>

namespace iris {

class Stage {
  public:
    Stage(Channel<Packet>& input, Channel<Packet>* output = nullptr);
    virtual ~Stage();

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    virtual void start();
    virtual void stop();
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] std::exception_ptr failure() const noexcept;

  protected:
    virtual void process(Packet& packet) = 0;

  private:
    void run();

    Channel<Packet>& input_;
    Channel<Packet>* output_;
    std::atomic_bool running_{false};
    std::thread worker_;
    mutable std::mutex failure_mutex_;
    std::exception_ptr failure_;
};

} // namespace iris
