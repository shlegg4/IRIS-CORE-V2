#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace iris::output::disk {

void convert_bgr_to_nv12(const std::uint8_t* source, std::size_t source_pitch,
                         std::uint8_t* destination, std::size_t destination_pitch,
                         std::uint32_t width, std::uint32_t height, cudaStream_t stream);

} // namespace iris::output::disk
