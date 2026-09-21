#include "iris/stages/output/WebRtcTransport.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <iostream>

#include <cuda_runtime_api.h>
#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef IRIS_HAS_WEBRTC
#include <gst/app/gstappsrc.h>
#include <gst/sdp/gstsdpmessage.h>
#include <gst/webrtc/webrtc.h>
#include <gst/gst.h>
#endif

namespace iris::output {
using json = nlohmann::json;
static const char* env_or_empty(const char* name) { const auto* value = std::getenv(name); return value ? value : ""; }

class WebRtcTransport::Impl {
  public:
    explicit Impl(WebRtcPreviewConfig value) : config(std::move(value)) {
        if (!config.queue_capacity || !config.max_fps || !config.max_width || !config.bitrate)
            throw std::invalid_argument("invalid WebRTC preview configuration");
    }
    ~Impl() { stop(); }

    void start() {
#ifdef IRIS_HAS_WEBRTC
        std::scoped_lock guard(lock);
        if (running.exchange(true)) return;
        int argc = 0; char** argv = nullptr; gst_init(&argc, &argv);
        std::cerr << "GStreamer compile/runtime version=" << GST_VERSION_MAJOR << "." << GST_VERSION_MINOR << "." << GST_VERSION_MICRO
                  << "/" << gst_version_string() << "\n"
                  << "GST_PLUGIN_PATH=" << env_or_empty("GST_PLUGIN_PATH") << "\n"
                  << "GST_PLUGIN_SYSTEM_PATH=" << env_or_empty("GST_PLUGIN_SYSTEM_PATH") << "\n"
                  << "GST_REGISTRY=" << env_or_empty("GST_REGISTRY") << "\n";
        for (const char* element : {"appsrc", "videoconvert", "videoscale", "vp8enc", "rtpvp8pay", "webrtcbin", "nicesrc", "nicesink", "dtlssrtpenc", "dtlssrtpdec", "srtpenc", "srtpdec", "rtpbin"}) {
            GstElement* probe = gst_element_factory_make(element, nullptr);
            if (!probe) {
                running = false;
                error = std::string("GStreamer element unavailable: ") + element + " GST_PLUGIN_PATH=" + env_or_empty("GST_PLUGIN_PATH") + " GStreamer version=" + gst_version_string();
                std::cerr << error << "\n";
                return;
            }
            if (const auto* factory = gst_element_get_factory(probe)) {
                if (!std::strcmp(element, "webrtcbin") || !std::strcmp(element, "nicesrc") || !std::strcmp(element, "vp8enc"))
                    std::cerr << "GStreamer factory " << element << " plugin=" << gst_plugin_get_filename(gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(factory))) << "\n";
            }
            gst_object_unref(probe);
        }
#else
        throw std::runtime_error("IRIS was built without GStreamer WebRTC support");
#endif
    }
    void stop() noexcept {
        std::scoped_lock guard(lock);
        running = false;
#ifdef IRIS_HAS_WEBRTC
        sessions.clear();
#endif
    }
    void publish(PreviewPacket packet) noexcept {
        if (!running || !packet) return;
        std::scoped_lock guard(lock);
        for (const auto& frame : packet->frames) {
            if (frame.format != PixelFormat::Bgr8 || !frame.buffer.data || !frame.extent.width || !frame.extent.height) continue;
            if (std::find(cameras.begin(), cameras.end(), frame.camera) == cameras.end()) {
                cameras.push_back(frame.camera); std::sort(cameras.begin(), cameras.end());
            }
            dimensions[frame.camera] = {frame.extent.width, frame.extent.height};
            ready.notify_all();
#ifdef IRIS_HAS_WEBRTC
            for (auto& session : sessions) {
                try {
                    const auto source = session->sources.find(frame.camera);
                    if (source != session->sources.end()) push_frame(source->second, frame);
                } catch (const std::exception& exception) {
                    ++dropped;
                    error = exception.what();
                } catch (...) {
                    ++dropped;
                    error = "unknown WebRTC publish exception";
                }
            }
#endif
            ++published;
        }
    }
    PreviewTransportHealth health() const {
        std::scoped_lock guard(lock);
        return {config.enabled, sessions.size(), published, dropped, error};
    }
    PreviewHttpServer::WebRtcMessageHandler make_handler() {
        return [this, session = std::shared_ptr<Session>{}](const std::string& text) mutable -> std::string {
            std::unique_lock guard(lock);
            try {
                if (!running) return failure(error.empty() ? "WebRTC runtime is unavailable" : error);
                const auto message = json::parse(text);
                const auto type = message.value("type", "");
                if (type == "hello") {
                    if (message.value("version", 0) != 1) return failure("unsupported WebRTC signalling version");
                    if (message.contains("cameras") && message["cameras"].is_array()) {
                        for (const auto& camera : message["cameras"]) {
                            const auto id = camera.is_object() ? camera.value("camera_id", 0U) : camera.get<CameraId>();
                            if (std::find(cameras.begin(), cameras.end(), id) == cameras.end()) cameras.push_back(id);
                            if (camera.is_object()) {
                                dimensions[id] = {camera.value("width", config.max_width), camera.value("height", 720U)};
                            } else if (!dimensions.contains(id)) {
                                dimensions[id] = {config.max_width, 720U};
                            }
                        }
                        std::sort(cameras.begin(), cameras.end());
                    }
                    if (!session) {
#ifdef IRIS_HAS_WEBRTC
                        session = create_session();
                        for (const auto camera : cameras) {
                            const auto size = dimensions.find(camera);
                            if (size != dimensions.end()) add_source(*session, camera, size->second.first, size->second.second);
                        }
                        sessions.push_back(session);
#else
                        return failure("GStreamer WebRTC support is not compiled in");
#endif
                    }
                    return json{{"type", "tracks"}, {"cameras", cameras}}.dump();
                }
                if (!session) return failure("send hello before signalling");
                if (type == "offer") {
#ifdef IRIS_HAS_WEBRTC
                    if (cameras.empty()) return failure("No camera tracks are available");
                    return answer(*session, message.value("sdp", ""));
#else
                    return failure("GStreamer WebRTC support is not compiled in");
#endif
                }
                if (type == "ice") {
#ifdef IRIS_HAS_WEBRTC
                    const auto candidate = message.value("candidate", "");
                    if (candidate.empty()) return failure("ICE candidate is empty");
                    g_signal_emit_by_name(session->webrtc, "add-ice-candidate", message.value("sdpMLineIndex", 0), candidate.c_str());
#endif
                    return {};
                }
                if (type == "close") { close_session(session); session.reset(); return {}; }
                return failure("unknown signalling message type");
            } catch (const std::exception& exception) { error = exception.what(); return failure(exception.what()); }
            catch (...) { error = "unknown WebRTC signalling exception"; return failure(error); }
        };
    }
  private:
    static std::string failure(const std::string& message) { return json{{"type", "error"}, {"message", message}}.dump(); }
#ifdef IRIS_HAS_WEBRTC
    struct Source { GstElement* appsrc{}; GstElement* convert{}; GstElement* scale{}; GstElement* encoder{}; GstElement* pay{}; GstPad* sink_pad{}; std::uint64_t frame_index{}; };
    struct Session {
        GstElement* pipeline{}; GstElement* webrtc{};
        std::unordered_map<CameraId, Source> sources;
        ~Session() {
            if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
            for (auto& [_, source] : sources) if (source.sink_pad && webrtc) gst_element_release_request_pad(webrtc, source.sink_pad);
            if (pipeline) gst_object_unref(pipeline);
        }
    };
    std::shared_ptr<Session> create_session() {
        auto result = std::make_shared<Session>();
        result->pipeline = gst_pipeline_new(nullptr); result->webrtc = gst_element_factory_make("webrtcbin", "peer");
        if (!result->pipeline || !result->webrtc) throw std::runtime_error("could not create webrtcbin pipeline");
        g_object_set(result->webrtc, "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, nullptr);
        gst_bin_add(GST_BIN(result->pipeline), result->webrtc);
        return result;
    }
    void add_source(Session& session, CameraId camera, std::uint32_t frame_width, std::uint32_t frame_height) {
        auto* source = gst_element_factory_make("appsrc", nullptr); auto* convert = gst_element_factory_make("videoconvert", nullptr);
        auto* scale = gst_element_factory_make("videoscale", nullptr); auto* encoder = gst_element_factory_make("vp8enc", nullptr); auto* pay = gst_element_factory_make("rtpvp8pay", nullptr);
        if (!source || !convert || !scale || !encoder || !pay) throw std::runtime_error("could not create VP8 media elements");
        // Keep the initial CPU-copy path lossless. videoscale stays in the
        // graph for a later negotiated/max-width caps filter.
        const auto width = frame_width;
        const auto height = frame_height;
        auto* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", "width", G_TYPE_INT, static_cast<int>(width), "height", G_TYPE_INT, static_cast<int>(height), "framerate", GST_TYPE_FRACTION, static_cast<int>(config.max_fps), 1, nullptr);
        g_object_set(source, "caps", caps, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", FALSE, nullptr); gst_caps_unref(caps);
        g_object_set(encoder, "target-bitrate", static_cast<int>(config.bitrate), "deadline", 1, nullptr);
        gst_bin_add_many(GST_BIN(session.pipeline), source, convert, scale, encoder, pay, nullptr);
        if (!gst_element_link_many(source, convert, scale, encoder, pay, nullptr)) throw std::runtime_error("could not link VP8 media pipeline");
        // Requesting sink_%u is the operation that creates and associates the
        // send transceiver. Do it while the pipeline is still in NULL state;
        // adding a transceiver first attempts to allocate the stream twice.
        auto* sink_pad = gst_element_request_pad_simple(session.webrtc, "sink_%u");
        if (!sink_pad) throw std::runtime_error("could not request webrtcbin sink_%u pad");
        auto* src_pad = gst_element_get_static_pad(pay, "src");
        const auto linked = src_pad && gst_pad_link(src_pad, sink_pad) == GST_PAD_LINK_OK;
        if (src_pad) gst_object_unref(src_pad);
        if (!linked)
            throw std::runtime_error("could not link VP8 stream to webrtcbin");
        gst_element_sync_state_with_parent(source); gst_element_sync_state_with_parent(convert); gst_element_sync_state_with_parent(scale); gst_element_sync_state_with_parent(encoder); gst_element_sync_state_with_parent(pay);
        session.sources.emplace(camera, Source{source, convert, scale, encoder, pay, sink_pad});
    }
    std::string answer(Session& session, const std::string& offer) {
        if (offer.empty()) return failure("offer SDP is empty");
        // Build all advertised camera branches before handing the offer to
        // webrtcbin; accepting an offer on an uninitialized bin leaves it in
        // the CLOSED state on recent GStreamer releases.
        for (const auto camera : cameras) {
            const auto size = dimensions.find(camera);
            if (size != dimensions.end() && !session.sources.contains(camera))
                add_source(session, camera, size->second.first, size->second.second);
        }
        gst_element_set_state(session.pipeline, GST_STATE_PLAYING);
        GstSDPMessage* parsed{}; if (gst_sdp_message_new_from_text(offer.c_str(), &parsed) != GST_SDP_OK) return failure("invalid offer SDP");
        auto* remote = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, parsed);
        auto* remote_promise = gst_promise_new();
        g_signal_emit_by_name(session.webrtc, "set-remote-description", remote, remote_promise);
        gst_promise_wait(remote_promise);
        const auto* remote_reply = gst_promise_get_reply(remote_promise);
        if (remote_reply && gst_structure_has_field(remote_reply, "error")) {
            const GValue* error_value = gst_structure_get_value(remote_reply, "error");
            const char* error_text = "invalid remote SDP";
            if (error_value && G_VALUE_HOLDS_STRING(error_value)) error_text = g_value_get_string(error_value);
            else if (error_value && G_VALUE_HOLDS_BOXED(error_value)) {
                const auto* detail = static_cast<const GError*>(g_value_get_boxed(error_value));
                if (detail && detail->message) error_text = detail->message;
            }
            gst_promise_unref(remote_promise); gst_webrtc_session_description_free(remote);
            return failure(error_text ? error_text : "invalid remote SDP");
        }
        gst_promise_unref(remote_promise); gst_webrtc_session_description_free(remote);
        GstPromise* promise = gst_promise_new(); g_signal_emit_by_name(session.webrtc, "create-answer", nullptr, promise); gst_promise_wait(promise);
         const GstStructure* reply = gst_promise_get_reply(promise); GstWebRTCSessionDescription* local{};
         if (!reply) { gst_promise_unref(promise); return failure("GStreamer returned no SDP answer"); }
         gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &local, nullptr); gst_promise_unref(promise);
        if (!local) return failure("GStreamer did not produce an SDP answer");
        g_signal_emit_by_name(session.webrtc, "set-local-description", local, nullptr);
        // Let webrtcbin finish host-candidate gathering before returning SDP.
        // This keeps the initial implementation usable without a second
        // server-to-browser signalling channel for trickle ICE.
        const auto gather_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < gather_deadline) {
            while (g_main_context_iteration(nullptr, FALSE)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        GstWebRTCSessionDescription* gathered = nullptr;
        g_object_get(session.webrtc, "local-description", &gathered, nullptr);
        const auto* description = gathered ? gathered->sdp : local->sdp;
        char* text = gst_sdp_message_as_text(description); std::string result = json{{"type", "answer"}, {"sdp", text ? text : ""}}.dump(); g_free(text);
        if (gathered) gst_webrtc_session_description_free(gathered); gst_webrtc_session_description_free(local); return result;
    }
    void push_frame(Source& source, const Frame& frame) noexcept {
        try {
            if (frame.ready) frame.ready->synchronize();
            const auto width = frame.extent.width;
            const auto height = frame.extent.height;
            GstBuffer* buffer = gst_buffer_new_allocate(nullptr, static_cast<gsize>(width) * height * 3, nullptr); GstMapInfo mapped{};
            if (!buffer || !gst_buffer_map(buffer, &mapped, GST_MAP_WRITE)) throw std::runtime_error("could not allocate WebRTC frame buffer");
            if (cudaMemcpy2D(mapped.data, width * 3, frame.buffer.data, frame.buffer.stride_bytes, width * 3, height, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("WebRTC CUDA frame copy failed");
            gst_buffer_unmap(buffer, &mapped); GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(source.frame_index++, GST_SECOND, config.max_fps); GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, config.max_fps);
            if (gst_app_src_push_buffer(GST_APP_SRC(source.appsrc), buffer) != GST_FLOW_OK) ++dropped;
        } catch (const std::exception& exception) { ++dropped; error = exception.what(); }
    }
    void close_session(const std::shared_ptr<Session>& session) { sessions.erase(std::remove(sessions.begin(), sessions.end(), session), sessions.end()); }
#else
    struct Session {};
    void close_session(const std::shared_ptr<Session>&) {}
#endif
    WebRtcPreviewConfig config; mutable std::mutex lock; std::condition_variable ready; std::atomic_bool running{false}; std::vector<CameraId> cameras; std::unordered_map<CameraId, std::pair<std::uint32_t, std::uint32_t>> dimensions; std::size_t published{}, dropped{}; std::string error;
    std::vector<std::shared_ptr<Session>> sessions;
};

WebRtcTransport::WebRtcTransport(WebRtcPreviewConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
WebRtcTransport::~WebRtcTransport() = default;
void WebRtcTransport::start() { impl_->start(); }
void WebRtcTransport::publish(PreviewPacket packet) noexcept { impl_->publish(std::move(packet)); }
void WebRtcTransport::stop() noexcept { impl_->stop(); }
PreviewTransportHealth WebRtcTransport::health() const { return impl_->health(); }
PreviewHttpServer::WebRtcHandlerFactory WebRtcTransport::signalling_factory() { return [this] { return impl_->make_handler(); }; }

} // namespace iris::output
