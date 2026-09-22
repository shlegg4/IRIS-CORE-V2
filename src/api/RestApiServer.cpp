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
std::optional<std::uint32_t> number(const json& body, const char* key) { if (!body.contains(key) || !body[key].is_number_unsigned()) return std::nullopt; return body[key].get<std::uint32_t>(); }
std::optional<FrameRate> frame_rate(const json& b) { if (!b.contains("frame_rate")) return std::nullopt; if (b["frame_rate"].is_number_unsigned()) return FrameRate{b["frame_rate"].get<std::uint32_t>(), 1}; if (!b["frame_rate"].is_object()) return std::nullopt; return FrameRate{b["frame_rate"].value("numerator", 0U), b["frame_rate"].value("denominator", 0U)}; }
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
        const std::string path(req.target()); if (path == "/api/v1/status" && req.method() == http::verb::get) return {http::status::ok, snapshot(runtime_.snapshot())};
        if (path == "/api/v1/metrics" && req.method() == http::verb::get) return {http::status::ok, metrics(runtime_.snapshot().metrics)};
        if (path == "/api/v1/cameras" && req.method() == http::verb::get) { json out=json::array(); for (const auto& c:runtime_.snapshot().cameras) out.push_back(camera(c)); return {http::status::ok, out}; }
        if (path == "/api/v1/recording" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"recording",s.recording},{"path",s.recording_path.string()}}}; }
        RuntimeCommand command;
        if (path == "/api/v1/pipeline/start" && req.method() == http::verb::post) command=StartPipelineCommand{};
        else if (path == "/api/v1/pipeline/stop" && req.method() == http::verb::post) command=StopPipelineCommand{};
        else if (path == "/api/v1/recording/stop" && req.method() == http::verb::post) command=StopRecordingCommand{};
        else if (path == "/api/v1/shutdown" && req.method() == http::verb::post) command=ShutdownCommand{};
        else if (path == "/api/v1/recording/start" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); if (b.is_discarded() || !b.contains("destination") || !b["destination"].is_string()) return error(http::status::bad_request,"destination is required"); command=StartRecordingCommand{b["destination"].get<std::string>(),number(b,"bitrate"),number(b,"frame_rate")}; }
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
