extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}
#include <iostream>

int main() {
    const auto* encoder = avcodec_find_encoder_by_name("h264_nvenc");
    if (!encoder) { std::cerr << "h264_nvenc encoder unavailable\n"; return 2; }
    AVBufferRef* device{};
    const auto result = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (result < 0) { std::cerr << "CUDA FFmpeg device unavailable: " << result << "\n"; return 3; }
    std::cout << "encoder=h264_nvenc\ncodec_id=" << encoder->id << "\n";
    av_buffer_unref(&device);
    return 0;
}
