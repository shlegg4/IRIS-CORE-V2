#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace iris::capture::video {

void launch_nv12_to_bgr(const std::uint8_t* y_plane, std::size_t y_stride,
                        const std::uint8_t* uv_plane, std::size_t uv_stride,
                        void* bgr, std::size_t bgr_stride, std::uint32_t width,
                        std::uint32_t height, bool full_range, cudaStream_t stream);

} // namespace iris::capture::video
