#pragma once

#include <string>

namespace iris {

enum class OutputCommandStatus { Applied, Rejected, Failed };

struct OutputCommandResult {
    OutputCommandStatus status{OutputCommandStatus::Applied};
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == OutputCommandStatus::Applied;
    }
};

} // namespace iris
