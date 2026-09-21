#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <string>

namespace iris::output {

inline constexpr std::uint8_t h264_preview_version = 1;
inline constexpr std::uint16_t h264_flag_keyframe = 1U << 0U;
inline constexpr std::uint16_t h264_flag_discontinuity = 1U << 1U;
inline constexpr std::uint16_t h264_flag_config = 1U << 2U;
inline constexpr std::size_t h264_preview_header_size = 32;

struct H264PreviewAccessUnit {
    std::uint16_t flags{};
    std::uint32_t camera{};
    std::uint64_t sequence{};
    std::uint64_t timestamp_us{};
    std::vector<std::uint8_t> payload;
};
struct H264PreviewStreamConfig {
    std::uint32_t camera{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t fps{};
    std::string codec{"avc1.42E01E"};
    std::vector<std::uint8_t> description;
};

std::vector<std::uint8_t> encode_h264_preview_access_unit(const H264PreviewAccessUnit& unit);
bool decode_h264_preview_access_unit(std::span<const std::uint8_t> bytes, H264PreviewAccessUnit& unit);

} // namespace iris::output
