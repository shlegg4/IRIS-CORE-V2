extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
}
#include <cuda_runtime_api.h>
#include <iostream>

namespace {
std::string error_string(int result) {
    char buffer[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(result, buffer, sizeof(buffer));
    return buffer;
}
}

int main() {
    const auto* encoder = avcodec_find_encoder_by_name("h264_nvenc");
    if (!encoder) { std::cerr << "h264_nvenc encoder unavailable\n"; return 2; }
    std::cout << "encoder=h264_nvenc\ncodec_id=" << encoder->id << "\n";

    const auto* decoder = avcodec_find_decoder(AV_CODEC_ID_H264);
    bool cuda_decoder_available = false;
    if (decoder) {
        for (int index = 0;; ++index) {
            const auto* config = avcodec_get_hw_config(decoder, index);
            if (!config) break;
            const auto* type = av_hwdevice_get_type_name(config->device_type);
            std::cout << "decoder_config=" << (type ? type : "unknown")
                      << " methods=" << config->methods
                      << " pixel_format=" << av_get_pix_fmt_name(config->pix_fmt) << "\n";
            cuda_decoder_available |= config->device_type == AV_HWDEVICE_TYPE_CUDA;
        }
    }
    std::cout << "h264_cuda_decoder_config=" << (cuda_decoder_available ? "yes" : "no") << "\n";

    const auto runtime_result = cudaSetDevice(0);
    std::cout << "cuda_runtime_set_device=" << cudaGetErrorString(runtime_result) << "\n";
    if (runtime_result != cudaSuccess) return 4;

    struct DeviceOption { const char* label; const char* device; int flags; };
    const DeviceOption options[] = {
        {"default-device/default-context", nullptr, 0},
        {"explicit-device/default-context", "0", 0},
        {"default-device/primary-context", nullptr, AV_CUDA_USE_PRIMARY_CONTEXT},
        {"explicit-device/primary-context", "0", AV_CUDA_USE_PRIMARY_CONTEXT},
        {"default-device/current-context", nullptr, AV_CUDA_USE_CURRENT_CONTEXT},
        {"explicit-device/current-context", "0", AV_CUDA_USE_CURRENT_CONTEXT},
    };
    bool any_cuda_device = false;
    for (const auto& option : options) {
        AVBufferRef* device{};
        const auto result = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA,
                                                   option.device, nullptr, option.flags);
        std::cout << "cuda_device_test=" << option.label << " result=" << result;
        if (result < 0) std::cout << " error=" << error_string(result);
        else any_cuda_device = true;
        std::cout << "\n";
        av_buffer_unref(&device);
    }
    return any_cuda_device ? 0 : 3;
}
