#include "stages/output/disk/BgrToNv12.hpp"

namespace iris::output::disk {
namespace {

__device__ std::uint8_t clamp_byte(float value) {
    return static_cast<std::uint8_t>(fminf(255.0F, fmaxf(0.0F, value)));
}

__global__ void bgr_to_nv12(const std::uint8_t* source, std::size_t source_pitch,
                            std::uint8_t* y_plane, std::uint8_t* uv_plane,
                            std::size_t destination_pitch, std::uint32_t width,
                            std::uint32_t height) {
    const auto x = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto y = static_cast<std::uint32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    const auto* pixel = source + (y * source_pitch) + (x * 3U);
    const float blue = pixel[0];
    const float green = pixel[1];
    const float red = pixel[2];
    y_plane[(y * destination_pitch) + x] =
        clamp_byte(16.0F + (0.098F * blue) + (0.504F * green) + (0.257F * red));

    if ((x & 1U) == 0 && (y & 1U) == 0) {
        float red_sum = 0.0F;
        float green_sum = 0.0F;
        float blue_sum = 0.0F;
        for (std::uint32_t dy = 0; dy < 2; ++dy) {
            for (std::uint32_t dx = 0; dx < 2; ++dx) {
                const auto* sample = source + ((y + dy) * source_pitch) + ((x + dx) * 3U);
                blue_sum += sample[0];
                green_sum += sample[1];
                red_sum += sample[2];
            }
        }
        const float average_blue = blue_sum * 0.25F;
        const float average_green = green_sum * 0.25F;
        const float average_red = red_sum * 0.25F;
        auto* uv = uv_plane + ((y / 2U) * destination_pitch) + x;
        uv[0] = clamp_byte(128.0F + (0.439F * average_blue) - (0.291F * average_green) -
                           (0.148F * average_red));
        uv[1] = clamp_byte(128.0F - (0.071F * average_blue) - (0.368F * average_green) +
                           (0.439F * average_red));
    }
}

} // namespace

void convert_bgr_to_nv12(const std::uint8_t* source, std::size_t source_pitch,
                         std::uint8_t* destination, std::size_t destination_pitch,
                         std::uint32_t width, std::uint32_t height, cudaStream_t stream) {
    const dim3 threads(16, 16);
    const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    auto* uv_plane = destination + (destination_pitch * height);
    bgr_to_nv12<<<blocks, threads, 0, stream>>>(source, source_pitch, destination, uv_plane,
                                                destination_pitch, width, height);
}

} // namespace iris::output::disk
