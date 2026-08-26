#pragma once
#include "iris/pipeline/Frame.hpp"
#include <optional>
#include <vector>
namespace iris {
using FrameBatch = std::vector<Frame>;
struct Pose {
    std::uint64_t source_sequence{};
};
using PoseBatch = std::vector<Pose>;
struct Packet {
    std::uint64_t sequence{};
    FrameBatch frames;
    std::optional<PoseBatch> poses;
};
} // namespace iris
