#include "stages/capture/media_foundation/MediaFoundationSource.hpp"
#include <mfapi.h>
#include <mferror.h>
#include <stdexcept>
namespace iris::capture {
namespace {
constexpr DWORD kVideoStream = static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
void hrcheck(HRESULT hr, const char* op) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(op) + " failed: 0x" +
                                 std::to_string(static_cast<unsigned long>(hr)));
    }
}
GUID subtype(PixelFormat f) {
    return f == PixelFormat::Mjpeg  ? MFVideoFormat_MJPG
           : f == PixelFormat::Yuy2 ? MFVideoFormat_YUY2
                                    : MFVideoFormat_ARGB32;
}
PixelFormat pixel(GUID g) {
    if (g == MFVideoFormat_MJPG) {
        return PixelFormat::Mjpeg;
    }
    if (g == MFVideoFormat_YUY2) {
        return PixelFormat::Yuy2;
    }
    if (g == MFVideoFormat_ARGB32 || g == MFVideoFormat_RGB32) {
        return PixelFormat::Bgra8;
    }
    return PixelFormat::Unknown;
}
struct ThreadCom {
    bool owned{};
    ThreadCom() {
        auto hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        owned = SUCCEEDED(hr);
    }
    ~ThreadCom() {
        if (owned) {
            CoUninitialize();
        }
    }
};
} // namespace

class SourceReaderCallback final : public IMFSourceReaderCallback {
  public:
    explicit SourceReaderCallback(MediaFoundationSource* owner) : owner_(owner) {}
    STDMETHODIMP QueryInterface(REFIID iid, void** object) override {
        if (!object) {
            return E_POINTER;
        }
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IMFSourceReaderCallback)) {
            *object = static_cast<IMFSourceReaderCallback*>(this);
            AddRef();
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++references_; }
    STDMETHODIMP_(ULONG) Release() override {
        auto remaining = --references_;
        if (!remaining) {
            delete this;
        }
        return remaining;
    }
    STDMETHODIMP OnReadSample(HRESULT status, DWORD, DWORD flags, LONGLONG timestamp,
                              IMFSample* sample) override {
        owner_->on_sample(status, flags, timestamp, sample);
        std::scoped_lock lock(owner_->queue_mutex_);
        if (!owner_->closing_ && owner_->reader_) {
            owner_->reader_->ReadSample(kVideoStream, 0, nullptr, nullptr, nullptr, nullptr);
        }
        return S_OK;
    }
    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent*) override { return S_OK; }
    STDMETHODIMP OnFlush(DWORD) override { return S_OK; }

  private:
    std::atomic_ulong references_{1};
    MediaFoundationSource* owner_;
};

MediaFoundationSource::MediaFoundationSource() {
    auto hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    com_initialised_ = SUCCEEDED(hr);
    hrcheck(MFStartup(MF_VERSION, MFSTARTUP_LITE), "MFStartup");
    mf_initialised_ = true;
}
MediaFoundationSource::~MediaFoundationSource() {
    close();
    if (mf_initialised_) {
        MFShutdown();
    }
    if (com_initialised_) {
        CoUninitialize();
    }
}
std::vector<CameraInfo> MediaFoundationSource::enumerate() {
    Microsoft::WRL::ComPtr<IMFAttributes> a;
    hrcheck(MFCreateAttributes(&a, 1), "MFCreateAttributes");
    hrcheck(a->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                       MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),
            "SetGUID");
    IMFActivate** devices{};
    UINT32 count{};
    hrcheck(MFEnumDeviceSources(a.Get(), &devices, &count), "MFEnumDeviceSources");
    std::vector<CameraInfo> out;
    for (UINT32 i = 0; i < count; ++i) {
        WCHAR *n{}, *s{};
        UINT32 nl{}, sl{};
        devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &n, &nl);
        devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &s,
                                       &sl);
        out.push_back({n ? n : L"", s ? s : L""});
        CoTaskMemFree(n);
        CoTaskMemFree(s);
        devices[i]->Release();
    }
    CoTaskMemFree(devices);
    return out;
}
NegotiatedFormat MediaFoundationSource::open(const CaptureConfig& c) {
    close();
    auto cameras = enumerate();
    std::size_t index = c.device_index.value_or(0);
    if (!c.device_symbolic_link.empty()) {
        auto target = std::wstring(c.device_symbolic_link.begin(), c.device_symbolic_link.end());
        auto it = std::find_if(cameras.begin(), cameras.end(),
                               [&](auto& x) { return x.symbolic_link == target; });
        if (it == cameras.end()) {
            throw std::runtime_error("capture device symbolic link not found");
        }
        index = static_cast<std::size_t>(it - cameras.begin());
    }
    if (index >= cameras.size()) {
        throw std::runtime_error("capture device index not found");
    }
    Microsoft::WRL::ComPtr<IMFAttributes> a;
    hrcheck(MFCreateAttributes(&a, 2), "MFCreateAttributes");
    hrcheck(a->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                       MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID),
            "source type");
    hrcheck(a->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                         cameras[index].symbolic_link.c_str()),
            "symbolic link");
    hrcheck(MFCreateDeviceSource(a.Get(), &source_), "MFCreateDeviceSource");
    Microsoft::WRL::ComPtr<IMFAttributes> ra;
    hrcheck(MFCreateAttributes(&ra, 3), "reader attributes");
    callback_.Attach(new SourceReaderCallback(this));
    hrcheck(ra->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, callback_.Get()),
            "Set source reader callback");
    ra->SetUINT32(MF_LOW_LATENCY, TRUE);
    hrcheck(MFCreateSourceReaderFromMediaSource(source_.Get(), ra.Get(), &reader_),
            "MFCreateSourceReader");
    Microsoft::WRL::ComPtr<IMFMediaType> selected;
    Microsoft::WRL::ComPtr<IMFMediaType> fallback;
    for (DWORD type_index = 0;; ++type_index) {
        Microsoft::WRL::ComPtr<IMFMediaType> candidate;
        auto result = reader_->GetNativeMediaType(kVideoStream, type_index, &candidate);
        if (result == MF_E_NO_MORE_TYPES) {
            break;
        }
        hrcheck(result, "GetNativeMediaType");
        GUID candidate_subtype{};
        UINT32 candidate_width{}, candidate_height{}, candidate_num{}, candidate_den{};
        if (FAILED(candidate->GetGUID(MF_MT_SUBTYPE, &candidate_subtype)) ||
            FAILED(MFGetAttributeSize(candidate.Get(), MF_MT_FRAME_SIZE, &candidate_width,
                                      &candidate_height)) ||
            FAILED(MFGetAttributeRatio(candidate.Get(), MF_MT_FRAME_RATE, &candidate_num,
                                       &candidate_den))) {
            continue;
        }
        if (candidate_subtype != subtype(c.format)) {
            continue;
        }
        if (!fallback) {
            fallback = candidate;
        }
        if (candidate_width == c.extent.width && candidate_height == c.extent.height &&
            candidate_num * c.frame_rate.denominator == c.frame_rate.numerator * candidate_den) {
            selected = candidate;
            break;
        }
    }
    if (!selected && c.allow_format_fallback) {
        selected = fallback;
    }
    if (!selected) {
        throw std::runtime_error("requested camera mode is not advertised natively");
    }
    hrcheck(reader_->SetCurrentMediaType(kVideoStream, nullptr, selected.Get()),
            "native media type negotiation");
    Microsoft::WRL::ComPtr<IMFMediaType> actual;
    hrcheck(reader_->GetCurrentMediaType(kVideoStream, &actual), "GetCurrentMediaType");
    UINT32 w{}, h{}, n{}, d{};
    GUID st{};
    MFGetAttributeSize(actual.Get(), MF_MT_FRAME_SIZE, &w, &h);
    MFGetAttributeRatio(actual.Get(), MF_MT_FRAME_RATE, &n, &d);
    actual->GetGUID(MF_MT_SUBTYPE, &st);
    format_ = {{w, h}, {n, d}, pixel(st)};
    if (!c.allow_format_fallback &&
        (format_.extent.width != c.extent.width || format_.extent.height != c.extent.height ||
         format_.format != c.format ||
         n * c.frame_rate.denominator != c.frame_rate.numerator * d)) {
        throw std::runtime_error(
            "camera negotiated a format different from the requested exact mode");
    }
    camera_ = static_cast<CameraId>(index);
    sequence_ = 0;
    {
        std::scoped_lock lock(queue_mutex_);
        queue_.clear();
        async_error_ = S_OK;
        closing_ = false;
    }
    hrcheck(reader_->ReadSample(kVideoStream, 0, nullptr, nullptr, nullptr, nullptr),
            "Begin asynchronous capture");
    return format_;
}
std::optional<CaptureSample> MediaFoundationSource::read() {
    std::unique_lock lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return closing_ || FAILED(async_error_) || !queue_.empty(); });
    if (FAILED(async_error_)) {
        hrcheck(async_error_, "asynchronous camera read");
    }
    if (queue_.empty()) {
        return std::nullopt;
    }
    auto result = std::move(queue_.front());
    queue_.pop_front();
    return result;
}
void MediaFoundationSource::on_sample(HRESULT status, DWORD flags, LONGLONG ts, IMFSample* s) {
    if (FAILED(status) || (flags & MF_SOURCE_READERF_ERROR)) {
        std::scoped_lock lock(queue_mutex_);
        async_error_ = FAILED(status) ? status : E_FAIL;
        queue_cv_.notify_all();
        return;
    }
    if (!s || (flags & MF_SOURCE_READERF_STREAMTICK)) {
        return;
    }
    Microsoft::WRL::ComPtr<IMFMediaBuffer> b;
    if (FAILED(s->ConvertToContiguousBuffer(&b))) {
        return;
    }
    BYTE* p{};
    DWORD len{};
    if (FAILED(b->Lock(&p, nullptr, &len))) {
        return;
    }
    const auto copy_begin = std::chrono::steady_clock::now();
    CaptureSample out;
    out.camera = camera_;
    out.sequence = sequence_++;
    out.bytes.assign(p, p + len);
    out.format = format_.format;
    out.extent = format_.extent;
    out.source_timestamp = std::chrono::nanoseconds(ts * 100);
    out.host_arrival = std::chrono::steady_clock::now();
    out.host_copy_ms = std::chrono::duration<double, std::milli>(out.host_arrival - copy_begin).count();
    out.discontinuity = (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) != 0;
    b->Unlock();
    {
        std::scoped_lock lock(queue_mutex_);
        if (!closing_) {
            if (queue_.size() >= 2) {
                queue_.pop_front();
            }
            queue_.push_back(std::move(out));
        }
    }
    queue_cv_.notify_one();
}
void MediaFoundationSource::close() {
    {
        std::scoped_lock lock(queue_mutex_);
        closing_ = true;
    }
    queue_cv_.notify_all();
    if (reader_) {
        reader_->Flush(kVideoStream);
    }
    reader_.Reset();
    callback_.Reset();
    if (source_) {
        source_->Shutdown();
    }
    source_.Reset();
    sequence_ = 0;
}
} // namespace iris::capture
