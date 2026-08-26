#pragma once

#include "iris/runtime/RuntimeControl.hpp"

#include <iosfwd>
#include <optional>
#include <string>

namespace iris {

class Runtime;

class InteractiveCli {
  public:
    explicit InteractiveCli(Runtime&);
    int run(std::istream& input, std::ostream& output);

    static std::optional<RuntimeCommand> parse(std::string line, std::string& error);
    static std::string help();

  private:
    Runtime& runtime_;
};

} // namespace iris
