#pragma once

#include "iris/pipeline/Channel.hpp"
#include "iris/pipeline/Packet.hpp"

#include <atomic>
#include <thread>

namespace iris {

class Stage {
  public:
    Stage(Channel<Packet>& input, Channel<Packet>* output = nullptr);
    virtual ~Stage();

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    virtual void start();
    virtual void stop();

  protected:
    virtual void process(Packet& packet) = 0;

  private:
    void run();

    Channel<Packet>& input_;
    Channel<Packet>* output_;
    std::atomic_bool running_{false};
    std::thread worker_;
};

} // namespace iris
