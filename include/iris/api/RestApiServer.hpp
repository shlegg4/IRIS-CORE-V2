#pragma once

#include "iris/runtime/Runtime.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace iris::api {

class RestApiServer final {
  public:
    explicit RestApiServer(Runtime&, std::string bind_address = "127.0.0.1",
                           std::uint16_t port = 8090);
    ~RestApiServer();
    RestApiServer(const RestApiServer&) = delete;
    RestApiServer& operator=(const RestApiServer&) = delete;

    void start();
    void stop() noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris::api
