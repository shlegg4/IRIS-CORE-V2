#include "iris/infrastructure/gpu/CudaResources.hpp"
#include <stdexcept>
#include <string>
namespace iris::infrastructure::gpu {
void check_cuda(cudaError_t r, const char* op) {
    if (r != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(r));
    }
}
CudaStream::CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
}
CudaStream::~CudaStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}
CudaEvent::CudaEvent() {
    check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
               "cudaEventCreateWithFlags");
}
CudaEvent::~CudaEvent() {
    if (event_) {
        cudaEventDestroy(event_);
    }
}
void CudaEvent::record(cudaStream_t s) {
    check_cuda(cudaEventRecord(event_, s), "cudaEventRecord");
}
void CudaEvent::wait(cudaStream_t s) const {
    check_cuda(cudaStreamWaitEvent(s, event_), "cudaStreamWaitEvent");
}
void CudaEvent::synchronize() const {
    check_cuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
}
bool CudaEvent::complete() const {
    auto r = cudaEventQuery(event_);
    if (r == cudaSuccess) {
        return true;
    }
    if (r == cudaErrorNotReady) {
        return false;
    }
    check_cuda(r, "cudaEventQuery");
    return false;
}
} // namespace iris::infrastructure::gpu
