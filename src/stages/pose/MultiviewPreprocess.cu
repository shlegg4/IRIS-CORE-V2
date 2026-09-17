#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

__global__ void preprocess_kernel(const std::uint8_t* const* src, const std::size_t* pitch,
                                  const std::uint32_t* w, const std::uint32_t* h,
                                  const float* source_k, const float* target_k, const float* distortion,
                                  float* dst) {
        const int view = blockIdx.z;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= 640 || y >= 640) return;
        const float scale = fminf(640.0f / static_cast<float>(w[view]), 640.0f / static_cast<float>(h[view]));
        const int rw = max(1, static_cast<int>(w[view] * scale + 0.5f));
        const int rh = max(1, static_cast<int>(h[view] * scale + 0.5f));
        const int px = x - (640 - rw) / 2, py = y - (640 - rh) / 2;
        const std::size_t plane = 640ull * 640ull;
        if (px < 0 || py < 0 || px >= rw || py >= rh) {
            for (int c = 0; c < 3; ++c) dst[(view * 3 + c) * plane + y * 640 + x] = 0.0f;
            return;
        }
        const float fx = target_k[view * 9], fy = target_k[view * 9 + 4];
        const float cx = target_k[view * 9 + 2], cy = target_k[view * 9 + 5];
        const float xu = (static_cast<float>(x) - cx) / fx;
        const float yu = (static_cast<float>(y) - cy) / fy;
        const float r2 = xu * xu + yu * yu;
        const float k1 = distortion[view * 5], k2 = distortion[view * 5 + 1];
        const float p1 = distortion[view * 5 + 2], p2 = distortion[view * 5 + 3], k3 = distortion[view * 5 + 4];
        const float radial = 1.0f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        const float xd = xu * radial + 2.0f * p1 * xu * yu + p2 * (r2 + 2.0f * xu * xu);
        const float yd = yu * radial + p1 * (r2 + 2.0f * yu * yu) + 2.0f * p2 * xu * yu;
        const int sx = min(static_cast<int>(w[view]) - 1, max(0, static_cast<int>(source_k[view * 9] * xd + source_k[view * 9 + 2])));
        const int sy = min(static_cast<int>(h[view]) - 1, max(0, static_cast<int>(source_k[view * 9 + 4] * yd + source_k[view * 9 + 5])));
        const auto* row = src[view] + sy * pitch[view];
        for (int c = 0; c < 3; ++c) dst[(view * 3 + c) * plane + y * 640 + x] = static_cast<float>(row[sx * 3 + c]);
}

extern "C" void iris_multiview_preprocess(const void* const* sources, const std::size_t* strides,
                                            const std::uint32_t* widths, const std::uint32_t* heights,
                                            const float* source_k, const float* target_k, const float* distortion,
                                            float* destination, cudaStream_t stream) {
    const std::uint8_t* device_sources[3] = {static_cast<const std::uint8_t*>(sources[0]), static_cast<const std::uint8_t*>(sources[1]), static_cast<const std::uint8_t*>(sources[2])};
    const std::size_t device_strides[3] = {strides[0], strides[1], strides[2]};
    const std::uint8_t** source_device = nullptr; std::size_t* stride_device = nullptr; std::uint32_t* width_device = nullptr; std::uint32_t* height_device = nullptr; float *source_k_device = nullptr, *target_k_device = nullptr, *distortion_device = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&source_device), sizeof(device_sources), stream); cudaMallocAsync(reinterpret_cast<void**>(&stride_device), sizeof(device_strides), stream); cudaMallocAsync(reinterpret_cast<void**>(&width_device), 3 * sizeof(std::uint32_t), stream); cudaMallocAsync(reinterpret_cast<void**>(&height_device), 3 * sizeof(std::uint32_t), stream); cudaMallocAsync(reinterpret_cast<void**>(&source_k_device), 27 * sizeof(float), stream); cudaMallocAsync(reinterpret_cast<void**>(&target_k_device), 27 * sizeof(float), stream); cudaMallocAsync(reinterpret_cast<void**>(&distortion_device), 15 * sizeof(float), stream);
    cudaMemcpyAsync(source_device, device_sources, sizeof(device_sources), cudaMemcpyHostToDevice, stream); cudaMemcpyAsync(stride_device, device_strides, sizeof(device_strides), cudaMemcpyHostToDevice, stream); cudaMemcpyAsync(width_device, widths, 3 * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream); cudaMemcpyAsync(height_device, heights, 3 * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream); cudaMemcpyAsync(source_k_device, source_k, 27 * sizeof(float), cudaMemcpyHostToDevice, stream); cudaMemcpyAsync(target_k_device, target_k, 27 * sizeof(float), cudaMemcpyHostToDevice, stream); cudaMemcpyAsync(distortion_device, distortion, 15 * sizeof(float), cudaMemcpyHostToDevice, stream);
    preprocess_kernel<<<dim3(40, 40, 3), dim3(16, 16), 0, stream>>>(source_device, stride_device, width_device, height_device, source_k_device, target_k_device, distortion_device, destination);
    cudaFreeAsync(source_device, stream); cudaFreeAsync(stride_device, stream); cudaFreeAsync(width_device, stream); cudaFreeAsync(height_device, stream); cudaFreeAsync(source_k_device, stream); cudaFreeAsync(target_k_device, stream); cudaFreeAsync(distortion_device, stream);
}
