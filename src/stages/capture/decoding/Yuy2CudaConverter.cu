#include "stages/capture/decoding/GpuDecoder.hpp"
#include <cuda_runtime.h>
namespace iris::capture {
namespace {
__device__ unsigned char clamp(float v) {
    return static_cast<unsigned char>(v < 0 ? 0 : v > 255 ? 255 : v);
}
__global__ void kernel(const unsigned char* src, size_t sp, unsigned char* dst, size_t dp,
                       unsigned w, unsigned h) {
    unsigned x = (blockIdx.x * blockDim.x + threadIdx.x) * 2,
             y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }
    auto p = src + y * sp + x * 2;
    float y0 = p[0], u = p[1] - 128.0f, y1 = p[2], v = p[3] - 128.0f;
    for (unsigned i = 0; i < 2 && x + i < w; ++i) {
        float yy = i ? y1 : y0;
        auto d = dst + y * dp + (x + i) * 3;
        d[0] = clamp(yy + 1.772f * u);
        d[1] = clamp(yy - 0.344136f * u - 0.714136f * v);
        d[2] = clamp(yy + 1.402f * v);
    }
}
} // namespace
void launch_yuy2_to_bgr(const std::uint8_t* s, std::size_t sp, void* d, std::size_t dp,
                        std::uint32_t w, std::uint32_t h, cudaStream_t stream) {
    dim3 block(16, 16), grid((w / 2 + 15) / 16, (h + 15) / 16);
    kernel<<<grid, block, 0, stream>>>(s, sp, static_cast<unsigned char*>(d), dp, w, h);
    infrastructure::gpu::check_cuda(cudaGetLastError(), "YUY2 conversion kernel");
}
} // namespace iris::capture
