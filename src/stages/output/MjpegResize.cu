#include <cuda_runtime_api.h>
#include <cstddef>

namespace iris::output {
namespace {
__global__ void resize_bgr_kernel(const unsigned char* source, std::size_t source_pitch,
                                  unsigned char* destination, std::size_t destination_pitch,
                                  unsigned source_width, unsigned source_height,
                                  unsigned destination_width, unsigned destination_height) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) return;
    const auto sx = x * source_width / destination_width;
    const auto sy = y * source_height / destination_height;
    const auto* input = source + sy * source_pitch + sx * 3;
    auto* output = destination + y * destination_pitch + x * 3;
    output[0] = input[0]; output[1] = input[1]; output[2] = input[2];
}
}
void resize_bgr(const unsigned char* source, std::size_t source_pitch, unsigned char* destination,
                std::size_t destination_pitch, unsigned source_width, unsigned source_height,
                unsigned destination_width, unsigned destination_height, cudaStream_t stream) {
    const dim3 block(16,16), grid((destination_width+15)/16,(destination_height+15)/16);
    resize_bgr_kernel<<<grid,block,0,stream>>>(source,source_pitch,destination,destination_pitch,
                                                source_width,source_height,destination_width,destination_height);
}
} // namespace iris::output
