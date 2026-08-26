#include "stages/capture/decoding/GpuDecoder.hpp"

#include <cuda_runtime.h>

namespace iris::capture {
namespace {

__global__ void rotate_bgr_kernel(const unsigned char* source, std::size_t source_pitch,
                                  unsigned char* destination, std::size_t destination_pitch,
                                  unsigned source_width, unsigned source_height,
                                  FrameRotation rotation) {
    const unsigned destination_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned destination_y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool axes_swapped =
        rotation == FrameRotation::Clockwise90 || rotation == FrameRotation::CounterClockwise90;
    const unsigned destination_width = axes_swapped ? source_height : source_width;
    const unsigned destination_height = axes_swapped ? source_width : source_height;
    if (destination_x >= destination_width || destination_y >= destination_height) {
        return;
    }

    unsigned source_x{};
    unsigned source_y{};
    switch (rotation) {
    case FrameRotation::Clockwise90:
        source_x = destination_y;
        source_y = source_height - 1 - destination_x;
        break;
    case FrameRotation::Rotate180:
        source_x = source_width - 1 - destination_x;
        source_y = source_height - 1 - destination_y;
        break;
    case FrameRotation::CounterClockwise90:
        source_x = source_width - 1 - destination_y;
        source_y = destination_x;
        break;
    case FrameRotation::None:
        source_x = destination_x;
        source_y = destination_y;
        break;
    }

    const auto* source_pixel = source + source_y * source_pitch + source_x * 3;
    auto* destination_pixel = destination + destination_y * destination_pitch + destination_x * 3;
    destination_pixel[0] = source_pixel[0];
    destination_pixel[1] = source_pixel[1];
    destination_pixel[2] = source_pixel[2];
}

} // namespace

void launch_bgr_rotation(const void* source, std::size_t source_pitch, void* destination,
                         std::size_t destination_pitch, std::uint32_t source_width,
                         std::uint32_t source_height, FrameRotation rotation, cudaStream_t stream) {
    const dim3 block(16, 16);
    const unsigned destination_width = swaps_axes(rotation) ? source_height : source_width;
    const unsigned destination_height = swaps_axes(rotation) ? source_width : source_height;
    const dim3 grid((destination_width + block.x - 1) / block.x,
                    (destination_height + block.y - 1) / block.y);
    rotate_bgr_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(source), source_pitch,
        static_cast<unsigned char*>(destination), destination_pitch, source_width, source_height,
        rotation);
    infrastructure::gpu::check_cuda(cudaGetLastError(), "BGR rotation kernel");
}

} // namespace iris::capture
