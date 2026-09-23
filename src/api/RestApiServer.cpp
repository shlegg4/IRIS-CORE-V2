#include "iris/api/RestApiServer.hpp"

#include "iris/pipeline/Frame.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/pipeline/OverflowPolicy.hpp"
#include "iris/stages/output/PreviewHttpServer.hpp"
#ifdef _WIN32
#include "stages/capture/media_foundation/MediaFoundationSource.hpp"
#include <windows.h>
#endif

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cuda_runtime_api.h>

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
#ifdef _WIN32
std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const auto length = static_cast<int>(value.size());
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, value.data(), length, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) return {};
    std::string result(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), length, result.data(), bytes, nullptr, nullptr);
    return result;
}
#endif

json metrics(const infrastructure::metrics::MetricsSnapshot& m) {
    json result{{"counters", m.counters}, {"gauges", m.gauges}, {"histograms", json::object()}};
    for (const auto& [name, h] : m.histograms) result["histograms"][name] = { {"bounds", h.bounds}, {"counts", h.counts}, {"count", h.count}, {"sum", h.sum} };
    return result;
}
json camera(const CameraCaptureConfig& c, const infrastructure::metrics::MetricsSnapshot& metrics_snapshot) {
    const auto& x = c.capture;
    const auto prefix = "iris_capture_camera_" + std::to_string(c.camera_id);
    const auto counter = [&metrics_snapshot, &prefix](const std::string& suffix) { const auto it = metrics_snapshot.counters.find(prefix + suffix); return it == metrics_snapshot.counters.end() ? std::uint64_t{} : it->second; };
    const auto gauge = [&metrics_snapshot, &prefix](const std::string& suffix) { const auto it = metrics_snapshot.gauges.find(prefix + suffix); return it == metrics_snapshot.gauges.end() ? 0.0 : it->second; };
    return {{"camera_id", c.camera_id}, {"device_symbolic_link", x.device_symbolic_link}, {"device_index", x.device_index ? json(*x.device_index) : json(nullptr)}, {"width", x.extent.width}, {"height", x.extent.height}, {"frame_rate", { {"numerator", x.frame_rate.numerator}, {"denominator", x.frame_rate.denominator}, {"value", x.frame_rate.value()} }}, {"format", pixel_format(x.format)}, {"cuda_device", x.cuda_device}, {"sample_queue_capacity", x.sample_queue_capacity}, {"frame_pool_capacity", x.frame_pool_capacity}, {"overflow", overflow(x.overflow)}, {"rotation", rotation(x.rotation)}, {"allow_format_fallback", x.allow_format_fallback}, {"reconnect", x.reconnect}, {"connected", gauge("_up") > 0.0}, {"frames_received", counter("_samples_received_total")}, {"frames_dropped", counter("_pool_exhaustions_total")}, {"source_errors", counter("_source_errors_total")}, {"last_frame_age_ms", gauge("_last_frame_age_ms")}, {"reconnect_count", counter("_reconnects_total")}, {"last_error", nullptr}};
}
json snapshot(const RuntimeSnapshot& s) {
    json cameras = json::array(); for (const auto& c : s.cameras) cameras.push_back(camera(c, s.metrics));
    json video_inputs = json::array(); for (const auto& v : s.video_inputs) video_inputs.push_back({{"camera_id", v.camera_id}, {"path", v.path.string()}, {"rotation", rotation(v.rotation)}});
    json decode_status = json::array(); for (const auto& status : s.video_decode_status) decode_status.push_back({{"camera_id", status.camera_id}, {"codec", status.codec}, {"backend", status.backend}, {"detail", status.detail}});
    return {{"state", to_string(s.state)}, {"input_mode", s.input_mode}, {"video_inputs", video_inputs}, {"video_decode_status", decode_status}, {"video_cuda_device", s.video_cuda_device}, {"video_frame_pool_capacity", s.video_frame_pool_capacity}, {"video_realtime", s.video_realtime}, {"video_loop", s.video_loop}, {"recording", s.recording}, {"recording_path", s.recording_path.string()}, {"shared_memory_destination", s.shared_memory_destination}, {"shared_memory_enabled", s.shared_memory_enabled}, {"processed_packets", s.processed_packets}, {"cameras", cameras}, {"sync_tolerance_ms", s.sync_tolerance.count()}, {"sync_queue_capacity", s.sync_queue_capacity}, {"incomplete_batch_policy", s.incomplete_batch_policy == IncompleteBatchPolicy::DropBatch ? "drop" : "partial"}, {"pose_backend", s.pose_backend}, {"pose_model_path", s.pose_model_path.string()}, {"pose_engine_path", s.pose_engine_path.string()}, {"preview", {{"enabled", s.preview.enabled}, {"bind_address", s.preview.bind_address}, {"port", s.preview.port}, {"published_packets", s.preview.published_packets}, {"dropped_packets", s.preview.dropped_packets}, {"connected_clients", s.preview.connected_clients}, {"event_clients", s.preview.event_clients}, {"h264_clients", s.preview.h264_clients}, {"mjpeg_clients", s.preview.mjpeg_clients}, {"last_error", s.preview.last_error}}}, {"last_error", s.last_error}, {"metrics", metrics(s.metrics)}};
}
json response(const RuntimeCommandResponse& r) { json out{{"status", command_status(r.status)}, {"message", r.message}}; if (r.snapshot) out["snapshot"] = snapshot(*r.snapshot); return out; }
void write_bmp(const std::filesystem::path& path, const Frame& frame) {
    if (frame.format != PixelFormat::Bgr8 || !frame.buffer.data || !frame.extent.width || !frame.extent.height || frame.buffer.stride_bytes < static_cast<std::size_t>(frame.extent.width) * 3)
        throw std::runtime_error("snapshot requires a valid full-resolution BGR frame for camera " + std::to_string(frame.camera));
    if (frame.ready) frame.ready->synchronize();
    if (cudaSetDevice(frame.buffer.device_id) != cudaSuccess) throw std::runtime_error("could not select camera CUDA device");
    const auto width = frame.extent.width, height = frame.extent.height;
    const std::size_t source_stride = static_cast<std::size_t>(width) * 3;
    std::vector<std::uint8_t> pixels(source_stride * height);
    if (cudaMemcpy2D(pixels.data(), source_stride, frame.buffer.data, frame.buffer.stride_bytes, source_stride, height, cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("could not copy full-resolution snapshot from GPU for camera " + std::to_string(frame.camera));
    const std::uint32_t row_size = (width * 3U + 3U) & ~3U;
    const std::uint32_t image_size = row_size * height;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not create snapshot image: " + path.string());
    auto u16 = [&out](std::uint16_t v) { const char b[2]{static_cast<char>(v), static_cast<char>(v >> 8)}; out.write(b, 2); };
    auto u32 = [&out](std::uint32_t v) { const char b[4]{static_cast<char>(v), static_cast<char>(v >> 8), static_cast<char>(v >> 16), static_cast<char>(v >> 24)}; out.write(b, 4); };
    out.put('B'); out.put('M'); u32(54U + image_size); u16(0); u16(0); u32(54); u32(40); u32(width); u32(height); u16(1); u16(24); u32(0); u32(image_size); u32(2835); u32(2835); u32(0); u32(0);
    std::vector<char> row(row_size, 0);
    for (std::uint32_t y = height; y-- > 0;) { std::memcpy(row.data(), pixels.data() + static_cast<std::size_t>(y) * source_stride, source_stride); out.write(row.data(), row.size()); }
    if (!out) throw std::runtime_error("failed while writing snapshot image: " + path.string());
}
json snapshot_pose(const Packet& packet, const std::string& backend) {
    json pose{{"schema_version", 1}, {"packet_sequence", packet.sequence}, {"backend", backend}, {"coordinate_frame", "IRIS rig calibration world"}, {"panoptic_units", "mm"}, {"multiview_units", "calibration units"}, {"panoptic", json::array()}, {"multiview", json::array()}, {"views_2d", json::array()}};
    if (packet.poses) for (const auto& person : *packet.poses) { json joints=json::array(), confidence=json::array(); for (const auto& joint : person.joints_3d_mm) joints.push_back({joint[0],joint[1],joint[2]}); for (const auto score : person.joint_confidence) confidence.push_back(score); pose["panoptic"].push_back({{"source_sequence",person.source_sequence},{"score",person.score},{"joints3d",joints},{"confidence",confidence}}); }
    if (packet.multiview_poses) for (const auto& person : *packet.multiview_poses) { json joints=json::array(), valid=json::array(); for (std::size_t i=0;i<person.joints_3d.size();++i) { const auto& joint=person.joints_3d[i]; joints.push_back({joint[0],joint[1],joint[2]}); valid.push_back(person.joint_valid[i]); } pose["multiview"].push_back({{"active",person.active},{"joints3d",joints},{"valid",valid}}); }
    if (packet.view_poses_2d) for (const auto& view : *packet.view_poses_2d) { json points=json::array(), scores=json::array(), valid=json::array(); for (std::size_t i=0;i<view.points_px.size();++i) { points.push_back({view.points_px[i][0],view.points_px[i][1]}); scores.push_back(view.scores[i]); valid.push_back(view.valid[i]); } pose["views_2d"].push_back({{"camera_id",view.camera_id},{"person_id",view.person_id},{"points_px",points},{"scores",scores},{"valid",valid}}); }
    return pose;
}
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
        if (path == "/api/v1/cameras/discover" && req.method() == http::verb::get) {
#ifdef _WIN32
            try {
                iris::capture::MediaFoundationSource source;
                const auto devices = source.enumerate();
                json out = json::array();
                for (std::size_t index = 0; index < devices.size(); ++index)
                    out.push_back({{"name", utf8(devices[index].name)}, {"device_index", index},
                                   {"device_symbolic_link", utf8(devices[index].symbolic_link)}});
                return {http::status::ok, out};
            } catch (const std::exception& cause) {
                return error(http::status::internal_server_error, cause.what());
            }
#else
            return error(http::status::not_implemented, "Camera discovery is available on Windows only");
#endif
        }
        if (path == "/api/v1/cameras" && req.method() == http::verb::get) { const auto s=runtime_.snapshot(); json out=json::array(); for (const auto& c:s.cameras) out.push_back(camera(c, s.metrics)); return {http::status::ok, out}; }
        if (path == "/api/v1/video-source" && req.method() == http::verb::get) { const auto s=runtime_.snapshot(); json videos=json::array(); for(const auto& v:s.video_inputs) videos.push_back({{"camera_id",v.camera_id},{"path",v.path.string()},{"rotation",rotation(v.rotation)}}); json decoders=json::array(); for(const auto& d:s.video_decode_status) decoders.push_back({{"camera_id",d.camera_id},{"codec",d.codec},{"backend",d.backend},{"detail",d.detail}}); return {http::status::ok, {{"input_mode",s.input_mode},{"cuda_device",s.video_cuda_device},{"frame_pool_capacity",s.video_frame_pool_capacity},{"realtime",s.video_realtime},{"loop",s.video_loop},{"cameras",videos},{"decode_status",decoders}}}; }
        if (path == "/api/v1/recording" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"recording",s.recording},{"path",s.recording_path.string()}}}; }
        if (path == "/api/v1/synchronizer" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"tolerance_ms",s.sync_tolerance.count()},{"queue_capacity",s.sync_queue_capacity},{"incomplete_batch_policy",s.incomplete_batch_policy==IncompleteBatchPolicy::EmitPartial?"partial":"drop"}}}; }
        if (path == "/api/v1/outputs/preview" && req.method() == http::verb::get) { auto s=runtime_.snapshot(); return {http::status::ok, {{"enabled",s.preview.enabled},{"bind_address",s.preview.bind_address},{"port",s.preview.port},{"published_packets",s.preview.published_packets},{"dropped_packets",s.preview.dropped_packets},{"connected_clients",s.preview.connected_clients},{"last_error",s.preview.last_error}}}; }
        if (path == "/api/v1/capture/snapshot" && req.method() == http::verb::post) {
            auto b=json::parse(req.body(),nullptr,false);
            if(b.is_discarded()||!b.is_object()||!b.contains("session_id")||!b["session_id"].is_string()||!b.contains("batch_id")||!b["batch_id"].is_string()||!b.contains("orientation_degrees")||!b["orientation_degrees"].is_number()||!b.contains("camera_ids")||!b["camera_ids"].is_array()||!b.contains("output_dir")||!b["output_dir"].is_string()) return error(http::status::bad_request,"session_id, batch_id, orientation_degrees, camera_ids, and output_dir are required");
            const auto session_id=b["session_id"].get<std::string>(), batch_id=b["batch_id"].get<std::string>();
            if(session_id.empty()||session_id.size()>80||batch_id.empty()||batch_id.size()>32||!std::all_of(session_id.begin(),session_id.end(),[](unsigned char c){return std::isalnum(c)||c=='-'||c=='_';})||!std::all_of(batch_id.begin(),batch_id.end(),[](unsigned char c){return std::isalnum(c)||c=='-'||c=='_';})) return error(http::status::bad_request,"invalid session or batch ID");
            const auto angle=b["orientation_degrees"].get<int>(); if(angle!=0&&angle!=45&&angle!=90&&angle!=135) return error(http::status::bad_request,"unsupported orientation");
            std::vector<CameraId> ids;std::set<CameraId> unique;for(const auto& value:b["camera_ids"]){if(!value.is_number_unsigned())return error(http::status::bad_request,"camera_ids must contain unsigned IDs");const auto id=value.get<CameraId>();if(!unique.insert(id).second)return error(http::status::bad_request,"camera IDs must be unique");ids.push_back(id);}if(ids.size()!=4)return error(http::status::bad_request,"snapshot batches require exactly four cameras");
            const auto status=runtime_.snapshot();if(status.state!=RuntimeState::Running)return error(http::status::service_unavailable,"IRIS capture pipeline is not running");
            const auto packet=runtime_.capture_snapshot_batch(ids,std::chrono::milliseconds(5000));if(!packet)return error(http::status::service_unavailable,"timed out waiting for the next synchronized four-camera frame batch");
            std::vector<const Frame*> selected;selected.reserve(ids.size());std::vector<std::int64_t> timestamps;for(const auto id:ids){const auto frame=std::find_if(packet->frames.begin(),packet->frames.end(),[id](const Frame& f){return f.camera==id;});if(frame==packet->frames.end())return error(http::status::conflict,"synchronized frame batch is missing a selected camera");selected.push_back(&*frame);const auto capture=frame->timing.estimated_capture_time.time_since_epoch().count()?frame->timing.estimated_capture_time:frame->timing.host_arrival_time;timestamps.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(capture.time_since_epoch()).count());}
            const auto [min_time,max_time]=std::minmax_element(timestamps.begin(),timestamps.end());const double skew_ms=static_cast<double>(*max_time-*min_time)/1.0e6;if(skew_ms>static_cast<double>(status.sync_tolerance.count()))return error(http::status::conflict,"synchronized frame batch exceeded the configured timestamp tolerance");
            std::filesystem::path destination=b["output_dir"].get<std::string>();if(!destination.is_absolute())return error(http::status::bad_request,"output_dir must be an absolute path");destination=std::filesystem::absolute(destination).lexically_normal();std::error_code ec;std::filesystem::create_directories(destination.parent_path(),ec);if(ec)return error(http::status::internal_server_error,"could not create snapshot parent directory: "+ec.message());if(std::filesystem::exists(destination,ec)){if(ec||!std::filesystem::is_directory(destination)||!std::filesystem::is_empty(destination))return error(http::status::conflict,"snapshot output directory already exists and is not empty");std::filesystem::remove(destination,ec);if(ec)return error(http::status::conflict,"could not reserve snapshot output directory: "+ec.message());}
            const auto pending=destination.parent_path()/(destination.filename().string()+".pending-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));if(!std::filesystem::create_directory(pending,ec)||ec)return error(http::status::internal_server_error,"could not create atomic snapshot staging directory");
            try {
                json frames=json::array();
                for(std::size_t i=0;i<selected.size();++i){const auto& frame=*selected[i];const std::string filename="camera_"+std::to_string(frame.camera)+".bmp";write_bmp(pending/filename,frame);const auto image=(destination/filename).string();frames.push_back({{"camera_id",frame.camera},{"image_path",image},{"frame_sequence",frame.sequence},{"timestamp_ns",timestamps[i]},{"timestamp_seconds",static_cast<double>(timestamps[i])/1.0e9},{"width",frame.extent.width},{"height",frame.extent.height}});}
                const auto pose=snapshot_pose(*packet,status.pose_backend);{std::ofstream pose_file(pending/"pose.json",std::ios::binary|std::ios::trunc);if(!pose_file)throw std::runtime_error("could not create pose metadata");pose_file<<pose.dump(2);if(!pose_file)throw std::runtime_error("failed writing pose metadata");}
                std::filesystem::rename(pending,destination,ec);if(ec)throw std::runtime_error("could not atomically commit snapshot batch: "+ec.message());
                return {http::status::ok,{{"schema_version",1},{"session_id",session_id},{"batch_id",batch_id},{"orientation_degrees",angle},{"packet_sequence",packet->sequence},{"captured_at",std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()},{"synchronization_skew_ms",skew_ms},{"pose_path",(destination/"pose.json").string()},{"pose",pose},{"frames",frames}}};
            } catch(const std::exception& cause){std::filesystem::remove_all(pending,ec);return error(http::status::internal_server_error,cause.what());}
        }
        RuntimeCommand command;
        if (path == "/api/v1/pipeline/start" && req.method() == http::verb::post) command=StartPipelineCommand{};
        else if (path == "/api/v1/pipeline/stop" && req.method() == http::verb::post) command=StopPipelineCommand{};
        else if (path == "/api/v1/video-source" && req.method() == http::verb::delete_) command=UseLiveCaptureCommand{};
        else if (path == "/api/v1/video-source" && req.method() == http::verb::post) {
            auto b=json::parse(req.body(),nullptr,false);
            if(b.is_discarded() || !b.contains("cameras") || !b["cameras"].is_array()) return error(http::status::bad_request,"cameras array is required");
            if((b.contains("cuda_device") && !b["cuda_device"].is_number_integer()) ||
               (b.contains("frame_pool_capacity") && !b["frame_pool_capacity"].is_number_unsigned()) ||
               (b.contains("realtime") && !b["realtime"].is_boolean()) ||
               (b.contains("loop") && !b["loop"].is_boolean())) return error(http::status::bad_request,"video source options have invalid types");
            SynchronizedVideoConfig c;
            c.cuda_device=b.value("cuda_device",0);
            c.frame_pool_capacity=b.value("frame_pool_capacity",8U);
            c.realtime=b.value("realtime",false);
            c.loop=b.value("loop",false);
            for(const auto& item:b["cameras"]) {
                if(!item.is_object() || !item.contains("camera_id") || !item["camera_id"].is_number_unsigned() || !item.contains("path") || !item["path"].is_string()) return error(http::status::bad_request,"each video camera requires camera_id and path");
                const auto video_rotation = parse_rotation(item);
                if (item.contains("rotation") && !video_rotation) return error(http::status::bad_request,"video rotation must be none, cw90, 180, or ccw90");
                c.cameras.push_back({item["camera_id"].get<CameraId>(),item["path"].get<std::string>(),video_rotation.value_or(FrameRotation::None)});
            }
            command=ConfigureVideoIngestionCommand{std::move(c)};
        }
        else if (path == "/api/v1/recording/stop" && req.method() == http::verb::post) command=StopRecordingCommand{};
        else if (path == "/api/v1/shutdown" && req.method() == http::verb::post) command=ShutdownCommand{};
        else if (path == "/api/v1/recording/start" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); if (b.is_discarded() || !b.contains("destination") || !b["destination"].is_string()) return error(http::status::bad_request,"destination is required"); command=StartRecordingCommand{b["destination"].get<std::string>(),number(b,"bitrate"),number(b,"frame_rate")}; }
        else if (path == "/api/v1/cameras" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded()||!b.contains("camera_id"))return error(http::status::bad_request,"camera_id is required"); CameraCaptureConfig c; c.camera_id=b.value("camera_id",0U); auto p=capture_patch(b); if(!p)return error(http::status::bad_request,"invalid camera body"); c.capture.device_symbolic_link=p->device_symbolic_link.value_or(""); c.capture.device_index=p->device_index.value_or(0); c.capture.extent.width=p->width.value_or(c.capture.extent.width); c.capture.extent.height=p->height.value_or(c.capture.extent.height); if(p->frame_rate)c.capture.frame_rate=*p->frame_rate; if(p->format)c.capture.format=*p->format; if(p->cuda_device)c.capture.cuda_device=*p->cuda_device; if(p->sample_queue_capacity)c.capture.sample_queue_capacity=*p->sample_queue_capacity; if(p->frame_pool_capacity)c.capture.frame_pool_capacity=*p->frame_pool_capacity; if(p->overflow)c.capture.overflow=*p->overflow; if(p->rotation)c.capture.rotation=*p->rotation; if(p->allow_format_fallback.has_value())c.capture.allow_format_fallback=*p->allow_format_fallback; if(p->reconnect.has_value())c.capture.reconnect=*p->reconnect; command=AddCameraCommand{c}; }
        else if (auto id=path_camera(path); id && req.method()==http::verb::delete_) command=RemoveCameraCommand{*id};
        else if (auto id=path_camera(path); id && req.method()==http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); auto p=capture_patch(b); if(b.is_discarded()||!p||p->empty())return error(http::status::bad_request,"camera patch is empty or invalid"); command=ConfigureCaptureCommand{*p,*id}; }
        else if (path == "/api/v1/pose" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded()||!b.contains("backend"))return error(http::status::bad_request,"backend is required"); ConfigurePoseCommand p; const auto v=b.value("backend",""); if(v=="off")p.backend=ConfigurePoseCommand::Backend::Off; else if(v=="monocular")p.backend=ConfigurePoseCommand::Backend::Monocular; else if(v=="2d")p.backend=ConfigurePoseCommand::Backend::TwoDimensional; else if(v=="multiview")p.backend=ConfigurePoseCommand::Backend::Multiview; else return error(http::status::bad_request,"invalid pose backend"); p.model_path=b.value("model_path",""); p.engine_path=b.value("engine_path",""); p.calibration_path=b.value("calibration_path",""); command=std::move(p); }
        else if (path == "/api/v1/calibration/start" && req.method() == http::verb::post) { auto b=json::parse(req.body(),nullptr,false); command=StartRigCalibrationCommand{b.is_object()?b.value("output_path", "rig-calibration.json"):"rig-calibration.json"}; }
        else if (path == "/api/v1/calibration/cancel" && req.method() == http::verb::post) command=CancelRigCalibrationCommand{};
        else if (path == "/api/v1/calibration/clear" && req.method() == http::verb::post) command=ClearRigCalibrationCommand{};
        else if (path == "/api/v1/calibration" && req.method() == http::verb::get) command=GetRigCalibrationStatusCommand{};
        else if (path == "/api/v1/outputs/shared-memory" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded())return error(http::status::bad_request,"invalid shared-memory body"); SharedMemoryOutputConfig c; c.enabled=b.value("enabled",false); c.destination=b.value("destination",c.destination); c.capacity_bytes=b.value("capacity_bytes",c.capacity_bytes); c.legacy_v1=b.value("legacy_v1",true); command=ConfigureSharedMemoryCommand{c}; }
        else if (path == "/api/v1/synchronizer" && req.method() == http::verb::patch) { auto b=json::parse(req.body(),nullptr,false); if(b.is_discarded())return error(http::status::bad_request,"invalid synchronizer body"); ConfigureSynchronizerCommand c; c.tolerance=std::chrono::milliseconds(b.value("tolerance_ms",20)); c.queue_capacity=b.value("queue_capacity",4U); c.incomplete_batch_policy=b.value("incomplete_batch_policy", "drop")=="partial" ? IncompleteBatchPolicy::EmitPartial : IncompleteBatchPolicy::DropBatch; command= c; }
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
