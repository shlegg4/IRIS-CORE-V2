#pragma once
#include <cuda_runtime_api.h>
namespace iris::infrastructure::gpu {
void check_cuda(cudaError_t result, const char* operation);
class CudaStream {
  public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_{};
};
class CudaEvent {
  public:
    CudaEvent();
    ~CudaEvent();
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    void record(cudaStream_t);
    void wait(cudaStream_t) const;
    void synchronize() const;
    bool complete() const;
    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_{};
};
} // namespace iris::infrastructure::gpu
