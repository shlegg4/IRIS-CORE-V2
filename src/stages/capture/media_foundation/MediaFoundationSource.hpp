#pragma once
#include "iris/stages/capture/CaptureConfig.hpp"
#include "stages/capture/CaptureSample.hpp"
#include <condition_variable>
#include <deque>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mutex>
#include <string>
#include <vector>
#include <wrl/client.h>
namespace iris::capture {
class SourceReaderCallback;
struct CameraInfo {
    std::wstring name, symbolic_link;
};
struct NegotiatedFormat {
    Extent2D extent{};
    FrameRate frame_rate{};
    PixelFormat format{PixelFormat::Unknown};
};
class MediaFoundationSource {
  public:
    MediaFoundationSource();
    ~MediaFoundationSource();
    MediaFoundationSource(const MediaFoundationSource&) = delete;
    static std::vector<CameraInfo> enumerate();
    NegotiatedFormat open(const CaptureConfig&);
    std::optional<CaptureSample> read();
    void close();

  private:
    friend class SourceReaderCallback;
    void on_sample(HRESULT status, DWORD flags, LONGLONG timestamp, IMFSample* sample);
    bool com_initialised_{};
    bool mf_initialised_{};
    CameraId camera_{};
    std::uint64_t sequence_{};
    NegotiatedFormat format_{};
    Microsoft::WRL::ComPtr<IMFMediaSource> source_;
    Microsoft::WRL::ComPtr<IMFSourceReader> reader_;
    Microsoft::WRL::ComPtr<IMFSourceReaderCallback> callback_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    // Media Foundation may invoke the async callback while CaptureStage is
    // stopping and replacing a camera. Keep the owner and COM interfaces
    // alive until every callback has left the object.
    std::condition_variable callback_cv_;
    std::size_t callbacks_inflight_{};
    std::deque<CaptureSample> queue_;
    HRESULT async_error_{S_OK};
    bool closing_{};
};
} // namespace iris::capture
