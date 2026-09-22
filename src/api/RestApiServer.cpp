#include "iris/api/RestApiServer.hpp"

#include "iris/pipeline/Frame.hpp"
#include "iris/pipeline/OverflowPolicy.hpp"
#include "iris/stages/output/PreviewHttpServer.hpp"

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <thread>

namespace iris::api {
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

namespace {
const char* command_status(RuntimeCommandStatus s) {
    switch (s) { case RuntimeCommandStatus::Applied: return "applied"; case RuntimeCommandStatus::Rejected: return "rejected"; case RuntimeCommandStatus::Failed: return "failed"; }
    return "failed";
}
const char* pixel_format(PixelFormat v) { switch (v) { case PixelFormat::Mjpeg: return "mjpeg"; case PixelFormat::Yuy2: return "yuy2"; case PixelFormat::Bgra8: return "bgra8"; case PixelFormat::Bgr8: return "bgr8"; default: return "unknown"; } }
const char* overflow(OverflowPolicy v) { switch (v) { case OverflowPolicy::Block: return "block"; case OverflowPolicy::DropOldest: return "drop-oldest"; case OverflowPolicy::DropNewest: return "drop-newest"; } return "unknown"; }
const char* rotation(FrameRotation v) { switch (v) { case FrameRotation::None: return "none"; case FrameRotation::Clockwise90: return "cw90"; case FrameRotation::Rotate180: return "180"; case FrameRotation::CounterClockwise90: return "ccw90"; } return "none"; }
std::optional<std::uint32_t> number(const json& body, const char* key) { if (!body.contains(key) || !body[key].is_number_unsigned()) return std::nullopt; return body[key].get<std::uint32_t>(); }
std::optional<FrameRate> frame_rate(const json& b) { if (!b.contains("frame_rate")) return std::nullopt; if (b["frame_rate"].is_number_unsigned()) return FrameRate{b["frame_rate"].get<std::uint32_t>(), 1}; if (!b["frame_rate"].is_object()) return std::nullopt; return FrameRate{b["frame_rate"].value("numerator", 0U), b["frame_rate"].value("denominator", 0U)}; }
std::optional<PixelFormat> parse_format(const json& b) { if (!b.contains("format") || !b["format"].is_string()) return std::nullopt; const auto v=b["format"].get<std::string>(); if(v=="mjpeg")return PixelFormat::Mjpeg; if(v=="yuy2")return PixelFormat::Yuy2; if(v=="bgra8")return PixelFormat::Bgra8; return std::nullopt; }
std::optional<OverflowPolicy> parse_overflow(const json& b) { if (!b.contains("overflow") || !b["overflow"].is_string()) return std::nullopt; const auto v=b["overflow"].get<std::string>(); if(v=="block")return OverflowPolicy::Block; if(v=="drop-oldest")return OverflowPolicy::DropOldest; if(v=="drop-newest")return OverflowPolicy::DropNewest; return std::nullopt; }
std::optional<FrameRotation> parse_rotation(const json& b) { if (!b.contains("rotation") || !b["rotation"].is_string()) return std::nullopt; const auto v=b["rotation"].get<std::string>(); if(v=="none")return FrameRotation::None; if(v=="cw90")return FrameRotation::Clockwise90; if(v=="180")return FrameRotation::Rotate180; if(v=="ccw90")return FrameRotation::CounterClockwise90; return std::nullopt; }
std::optional<CaptureConfigPatch> capture_patch(const json& b) { CaptureConfigPatch p; if(!b.is_object())return std::nullopt; if(b.contains("device_symbolic_link"))p.device_symbolic_link=b.value("device_symbolic_link",""); if(b.contains("device_index"))p.device_index=number(b,"device_index"); if(b.contains("width"))p.width=number(b,"width"); if(b.contains("height"))p.height=number(b,"height"); if(b.contains("frame_rate"))p.frame_rate=frame_rate(b); if(b.contains("format"))p.format=parse_format(b); if(b.contains("cuda_device"))p.cuda_device=b.value("cuda_device",0); if(b.contains("sample_queue_capacity"))p.sample_queue_capacity=number(b,"sample_queue_capacity"); if(b.contains("frame_pool_capacity"))p.frame_pool_capacity=number(b,"frame_pool_capacity"); if(b.contains("overflow"))p.overflow=parse_overflow(b); if(b.contains("rotation"))p.rotation=parse_rotation(b); if(b.contains("allow_format_fallback"))p.allow_format_fallback=b.value("allow_format_fallback",false); if(b.contains("reconnect"))p.reconnect=b.value("reconnect",true); return p; }
std::optional<CameraId> path_camera(const std::string& path) { const std::string prefix="/api/v1/cameras/"; if(!path.starts_with(prefix))return std::nullopt; try { std::size_t used{}; auto v=std::stoul(path.substr(prefix.size()),&used); if(used!=path.size()-prefix.size()||v>UINT32_MAX)return std::nullopt; return static_cast<CameraId>(v); } catch(...) { return std::nullopt; } }

json metrics(const infrastructure::metrics::MetricsSnapshot& m) {
    json result{{"counters", m.counters}, {"gauges", m.gauges}, {"histograms", json::object()}};
    for (const auto& [name, h] : m.histograms) result["histograms"][name] = { {"bounds", h.bounds}, {"counts", h.counts}, {"count", h.count}, {"sum", h.sum} };
    return result;
}
json camera(const CameraCaptureConfig& c) {
    const auto& x = c.capture;
    return {{"camera_id", c.camera_id}, {"device_symbolic_link", x.device_symbolic_link}, {"device_index", x.device_index ? json(*x.device_index) : json(nullptr)}, {"width", x.extent.width}, {"height", x.extent.height}, {"frame_rate", { {"numerator", x.frame_rate.numerator}, {"denominator", x.frame_rate.denominator}, {"value", x.frame_rate.value()} }}, {"format", pixel_format(x.format)}, {"cuda_device", x.cuda_device}, {"sample_queue_capacity", x.sample_queue_capacity}, {"frame_pool_capacity", x.frame_pool_capacity}, {"overflow", overflow(x.overflow)}, {"rotation", rotation(x.rotation)}, {"allow_format_fallback", x.allow_format_fallback}, {"reconnect", x.reconnect}};
}
json snapshot(const RuntimeSnapshot& s) {
    json cameras = json::array(); for (const auto& c : s.cameras) cameras.push_back(camera(c));
    return {{"state", to_string(s.state)}, {"recording", s.recording}, {"recording_path", s.recording_path.string()}, {"shared_memory_destination", s.shared_memory_destination}, {"shared_memory_enabled", s.shared_memory_enabled}, {"processed_packets", s.processed_packets}, {"cameras", cameras}, {"sync_tolerance_ms", s.sync_tolerance.count()}, {"sync_queue_capacity", s.sync_queue_capacity}, {"incomplete_batch_policy", s.incomplete_batch_policy == IncompleteBatchPolicy::DropBatch ? "drop" : "partial"}, {"pose_backend", s.pose_backend}, {"pose_model_path", s.pose_model_path.string()}, {"pose_engine_path", s.pose_engine_path.string()}, {"preview", {{"enabled", s.preview.enabled}, {"bind_address", s.preview.bind_address}, {"port", s.preview.port}, {"published_packets", s.preview.published_packets}, {"dropped_packets", s.preview.dropped_packets}, {"connected_clients", s.preview.connected_clients}, {"event_clients", s.preview.event_clients}, {"h264_clients", s.preview.h264_clients}, {"mjpeg_clients", s.preview.mjpeg_clients}, {"last_error", s.preview.last_error}}}, {"last_error", s.last_error}, {"metrics", metrics(s.metrics)}};
}
json response(const RuntimeCommandResponse& r) { json out{{"status", command_status(r.status)}, {"message", r.message}}; if (r.snapshot) out["snapshot"] = snapshot(*r.snapshot); return out; }
}

class RestApiServer::Impl {
  public:
    Impl(Runtime& runtime, std::string bind, std::uint16_t port) : runtime_(runtime), bind_(std::move(bind)), port_(port), acceptor_(context_) {}
    ~Impl() { stop(); }
    void start() {
        if (running_.exchange(true)) return;
        if (!output::is_loopback_address(bind_)) { running_ = false; throw std::invalid_argument("API server must bind to a loopback address"); }
        boost::system::error_code ec; auto address = asio::ip::make_address(bind_ == "localhost" ? "127.0.0.1" : bind_, ec); if (ec) throw std::runtime_error(ec.message());
        acceptor_.open(tcp::v4(), ec); if (!ec) acceptor_.set_option(asio::socket_base::reuse_address(true), ec); if (!ec) acceptor_.bind({address, port_}, ec); if (!ec) acceptor_.listen(asio::socket_base::max_listen_connections, ec); if (ec) { running_=false; throw std::runtime_error("API listen: "+ec.message()); }
        thread_ = std::thread([this] { while (running_) { tcp::socket socket(context_); boost::system::error_code e; acceptor_.accept(socket, e); if (!e) std::thread(&Impl::session, this, std::move(socket)).detach(); } });
    }
    void stop() noexcept { if (!running_.exchange(false)) return; boost::system::error_code ec; acceptor_.close(ec); if (thread_.joinable()) thread_.join(); }
  private:
    void session(tcp::socket socket) noexcept {
        try { beast::flat_buffer buffer; http::request<http::string_body> req; http::read(socket, buffer, req); auto [code, body] = handle(req); http::response<http::string_body> res{code, req.version()}; res.set(http::field::content_type, "application/json"); res.set(http::field::access_control_allow_origin, "*"); res.body() = body.dump(); res.prepare_payload(); res.keep_alive(false); http::write(socket, res); } catch (...) {}
    }
    std::pair<http::status, json> handle(const http::request<http::string_body>& req) {
        const std::string target(req.target()); const auto query_start = target.find('?'); const std::string path = target.substr(0, query_start);
        if (path == "/api/v1/status" && req.method() == http::verb::get) return {http::status::ok, snapshot(runtime_.snapshot())};
        if (path == "/api/v1/metrics" && req.method() == http::verb::get) { auto result=metrics(runtime_.snapshot().metrics); if(query_start!=std::string::npos) { const auto query=target.substr(query_start+1); const std::string key="prefix="; if(query.starts_with(key)) { const auto prefix=query.substr(key.size()); for(auto group: {"counters","gauges","histograms"}) for(auto it=result[group].begin();it!=result[group].end();) { if(!it.key().starts_with(prefix)) it=result[group].erase(it); else ++it; } } } return {http::status::ok, result}; }
        if (path == "/api/v1/cameras" && req.method() == http::verb::get) { json out=json::array(); for (const auto& c:runtime_.snapshot().cameras) out.push_back(camera(c)); return {http::status::ok, out}; }
        if (path == "/api/v1/recording" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"recording",s.recording},{"path",s.recording_path.string()}}}; }
        if (path == "/api/v1/synchronizer" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"tolerance_ms",s.sync_tolerance.count()},{"queue_capacity",s.sync_queue_capacity},{"incomplete_batch_policy",s.incomplete_batch_policy==IncompleteBatchPolicy::EmitPartial?"partial":"drop"}}}; }
        if (path == "/api/v1/outputs/preview" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"enabled",s.preview.enabled},{"bind_address",s.preview.bind_address},{"port",s.preview.port},{"published_packets",s.preview.published_packets},{"dropped_packets",s.preview.dropped_packets},{"connected_clients",s.preview.connected_clients},{"last_error",s.preview.last_error}}}; }
        RuntimeCommand command;
        if (path == "/api/v1/pipeline/start" && req.method() == http::verb::post) command=StartPipelineCommand{};
        else if (path == "/api/v1/pipeline/stop" && req.method() == http::verb::post) command=StopPipelineCommand{};
        else if (path == "/api/v1/recording/stop" && req.method() == http::verb::post) command=StopRecordingCommand{};
        else if (path == "/api/v1/shutdown" && req.method() == http::verb::post) command=ShutdownCommand{};
        else if (path == "/api/v1/recording/start" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); if (b.is_discarded() || !b.contains("destination") || !b["destination"].is_string()) return error(http::status::bad_request,"destination is required"); command=StartRecordingCommand{b["destination"].get<std::string>(),number(b,"bitrate"),number(b,"frame_rate")}; }
        else if (path == "/api/v1/cameras" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded()||!b.contains("camera_id"))return error(http::status::bad_request,"camera_id is required"); CameraCaptureConfig c; c.camera_id=b.value("camera_id",0U); auto p=capture_patch(b); if(!p)return error(http::status::bad_request,"invalid camera body"); c.capture.device_symbolic_link=p->device_symbolic_link.value_or(""); c.capture.device_index=p->device_index.value_or(0); c.capture.extent.width=p->width.value_or(c.capture.extent.width); c.capture.extent.height=p->height.value_or(c.capture.extent.height); if(p->frame_rate)c.capture.frame_rate=*p->frame_rate; if(p->format)c.capture.format=*p->format; if(p->cuda_device)c.capture.cuda_device=*p->cuda_device; if(p->sample_queue_capacity)c.capture.sample_queue_capacity=*p->sample_queue_capacity; if(p->frame_pool_capacity)c.capture.frame_pool_capacity=*p->frame_pool_capacity; if(p->overflow)c.capture.overflow=*p->overflow; if(p->rotation)c.capture.rotation=*p->rotation; if(p->allow_format_fallback)c.capture.allow_format_fallback=*p->allow_format_fallback; if(p->reconnect)c.capture.reconnect=*p->reconnect; command=AddCameraCommand{c}; }
        else if (auto id=path_camera(path); id && req.method()==http::verb::delete_) command=RemoveCameraCommand{*id};
        else if (auto id=path_camera(path); id && req.method()==http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); auto p=capture_patch(b); if(b.is_discarded()||!p||p->empty())return error(http::status::bad_request,"camera patch is empty or invalid"); command=ConfigureCaptureCommand{*p,*id}; }
        else if (path == "/api/v1/pose" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded()||!b.contains("backend"))return error(http::status::bad_request,"backend is required"); ConfigurePoseCommand p; const auto v=b.value("backend",""); if(v=="off")p.backend=ConfigurePoseCommand::Backend::Off; else if(v=="monocular")p.backend=ConfigurePoseCommand::Backend::Monocular; else if(v=="2d")p.backend=ConfigurePoseCommand::Backend::TwoDimensional; else if(v=="multiview")p.backend=ConfigurePoseCommand::Backend::Multiview; else return error(http::status::bad_request,"invalid pose backend"); p.model_path=b.value("model_path",""); p.engine_path=b.value("engine_path",""); p.calibration_path=b.value("calibration_path",""); command=std::move(p); }
        else if (path == "/api/v1/calibration/start" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); command=StartRigCalibrationCommand{b.is_object()?b.value("output_path", "rig-calibration.json"):"rig-calibration.json"}; }
        else if (path == "/api/v1/calibration/cancel" && req.method() == http::verb::post) command=CancelRigCalibrationCommand{};
        else if (path == "/api/v1/calibration/clear" && req.method() == http::verb::post) command=ClearRigCalibrationCommand{};
        else if (path == "/api/v1/calibration" && req.method() == http::verb::get) command=GetRigCalibrationStatusCommand{};
        else if (path == "/api/v1/outputs/shared-memory" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded())return error(http::status::bad_request,"invalid shared-memory body"); SharedMemoryOutputConfig c; c.enabled=b.value("enabled",false); c.destination=b.value("destination",c.destination); c.capacity_bytes=b.value("capacity_bytes",c.capacity_bytes); c.legacy_v1=b.value("legacy_v1",true); command=ConfigureSharedMemoryCommand{c}; }
        else if (path == "/api/v1/synchronizer" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded())return error(http::status::bad_request,"invalid synchronizer body"); ConfigureSynchronizerCommand c; c.tolerance=std::chrono::milliseconds(b.value("tolerance_ms",3)); c.queue_capacity=b.value("queue_capacity",4U); c.incomplete_batch_policy=b.value("incomplete_batch_policy", "drop")=="partial" ? IncompleteBatchPolicy::EmitPartial : IncompleteBatchPolicy::DropBatch; command= c; }
        else if (path == "/api/v1/outputs/preview" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded())return error(http::status::bad_request,"invalid preview body"); PreviewConfig c; c.http.enabled=b.value("http_enabled",true); c.mjpeg.enabled=b.value("mjpeg_enabled",true); c.h264.enabled=b.value("h264_enabled",true); c.http.bind_address=b.value("bind_address",c.http.bind_address); c.http.port=b.value("port",c.http.port); c.h264.max_fps=b.value("max_fps",c.h264.max_fps); c.h264.max_width=b.value("max_width",c.h264.max_width); c.mjpeg.jpeg_quality=b.value("jpeg_quality",c.mjpeg.jpeg_quality); c.h264.bitrate=b.value("bitrate",c.h264.bitrate); c.http.queue_capacity=b.value("queue_capacity",c.http.queue_capacity); command=ConfigurePreviewCommand{c}; }
        else return error(http::status::not_found,"endpoint not found");
        auto r=runtime_.execute(std::move(command)); auto code=r.status==RuntimeCommandStatus::Applied?http::status::ok:(r.status==RuntimeCommandStatus::Rejected?http::status::unprocessable_entity:http::status::internal_server_error); return {code,response(r)};
    }
    static std::pair<http::status,json> error(http::status s, std::string message) { return {s, {{"status","rejected"},{"message",std::move(message)}}}; }
    Runtime& runtime_; std::string bind_; std::uint16_t port_; asio::io_context context_; tcp::acceptor acceptor_; std::atomic_bool running_{false}; std::thread thread_;
};

RestApiServer::RestApiServer(Runtime& r, std::string bind, std::uint16_t port) : impl_(std::make_unique<Impl>(r,std::move(bind),port)) {}
RestApiServer::~RestApiServer() = default;
void RestApiServer::start() { impl_->start(); }
void RestApiServer::stop() noexcept { impl_->stop(); }
} // namespace iris::api
