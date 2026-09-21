#include "iris/stages/output/H264PreviewProtocol.hpp"

#include <algorithm>
#include <cstring>

namespace iris::output {
namespace {
template <typename T> void put(std::vector<std::uint8_t>& out, std::size_t offset, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) out[offset + i] = static_cast<std::uint8_t>(value >> (i * 8U));
}
template <typename T> T get(const std::uint8_t* data) {
    T value{};
    for (std::size_t i = 0; i < sizeof(T); ++i) value |= static_cast<T>(data[i]) << (i * 8U);
    return value;
}
}

std::vector<std::uint8_t> encode_h264_preview_access_unit(const H264PreviewAccessUnit& unit) {
    if (unit.payload.size() > UINT32_MAX) return {};
    std::vector<std::uint8_t> out(h264_preview_header_size + unit.payload.size(), 0);
    std::memcpy(out.data(), "IRWS", 4);
    out[4] = h264_preview_version;
    out[5] = static_cast<std::uint8_t>(h264_preview_header_size);
    put<std::uint16_t>(out, 6, unit.flags);
    put<std::uint32_t>(out, 8, unit.camera);
    put<std::uint64_t>(out, 12, unit.sequence);
    put<std::uint64_t>(out, 20, unit.timestamp_us);
    put<std::uint32_t>(out, 28, static_cast<std::uint32_t>(unit.payload.size()));
    std::copy(unit.payload.begin(), unit.payload.end(), out.begin() + h264_preview_header_size);
    return out;
}

bool decode_h264_preview_access_unit(std::span<const std::uint8_t> bytes, H264PreviewAccessUnit& unit) {
    if (bytes.size() < h264_preview_header_size || std::memcmp(bytes.data(), "IRWS", 4) != 0 ||
        bytes[4] != h264_preview_version || bytes[5] != h264_preview_header_size) return false;
    const auto payload_size = get<std::uint32_t>(bytes.data() + 28);
    if (payload_size != bytes.size() - h264_preview_header_size) return false;
    unit.flags = get<std::uint16_t>(bytes.data() + 6);
    unit.camera = get<std::uint32_t>(bytes.data() + 8);
    unit.sequence = get<std::uint64_t>(bytes.data() + 12);
    unit.timestamp_us = get<std::uint64_t>(bytes.data() + 20);
    unit.payload.assign(bytes.begin() + h264_preview_header_size, bytes.end());
    return true;
}
} // namespace iris::output
