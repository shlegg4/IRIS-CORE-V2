#include "stages/capture/video/Nv12ToBgrCuda.hpp"

#include "iris/infrastructure/gpu/CudaResources.hpp"

#include <cuda_runtime.h>

namespace iris::capture::video {
namespace {
__device__ unsigned char clamp_round(float value) {
    const float rounded = value >= 0.0F ? floorf(value + 0.5F) : ceilf(value - 0.5F);
    return static_cast<unsigned char>(rounded < 0.0F ? 0.0F : rounded > 255.0F ? 255.0F : rounded);
}

__global__ void nv12_to_bgr_kernel(const std::uint8_t* y_plane, std::size_t y_stride,
                                   const std::uint8_t* uv_plane, std::size_t uv_stride,
                                   std::uint8_t* bgr, std::size_t bgr_stride,
                                   std::uint32_t width, std::uint32_t height,
                                   bool full_range) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const auto luma = static_cast<float>(y_plane[static_cast<std::size_t>(y) * y_stride + x]);
    const auto* chroma = uv_plane + static_cast<std::size_t>(y / 2) * uv_stride + (x / 2) * 2;
    const float u = static_cast<float>(chroma[0]) - 128.0F;
    const float v = static_cast<float>(chroma[1]) - 128.0F;
    const float yy = full_range ? luma : 1.164383F * (luma - 16.0F);
    auto* pixel = bgr + static_cast<std::size_t>(y) * bgr_stride + static_cast<std::size_t>(x) * 3;
    pixel[0] = clamp_round(yy + 1.772F * u);
    pixel[1] = clamp_round(yy - 0.344136F * u - 0.714136F * v);
    pixel[2] = clamp_round(yy + 1.402F * v);
}
} // namespace

void launch_nv12_to_bgr(const std::uint8_t* y_plane, std::size_t y_stride,
                        const std::uint8_t* uv_plane, std::size_t uv_stride,
                        void* bgr, std::size_t bgr_stride, std::uint32_t width,
                        std::uint32_t height, bool full_range, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    nv12_to_bgr_kernel<<<grid, block, 0, stream>>>(
        y_plane, y_stride, uv_plane, uv_stride, static_cast<std::uint8_t*>(bgr), bgr_stride,
        width, height, full_range);
    infrastructure::gpu::check_cuda(cudaGetLastError(), "NV12 to BGR conversion kernel");
}

} // namespace iris::capture::video
