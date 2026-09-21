#include "iris/stages/output/PreviewHttpServer.hpp"

#include <boost/asio/ip/address.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>

namespace iris::output {
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;
std::string base64_encode(const std::vector<std::uint8_t>& input) { static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out; for (std::size_t i = 0; i < input.size(); i += 3) { const auto a = input[i]; const auto b = i + 1 < input.size() ? input[i + 1] : 0; const auto c = i + 2 < input.size() ? input[i + 2] : 0; out += table[a >> 2]; out += table[((a & 3) << 4) | (b >> 4)]; out += i + 1 < input.size() ? table[((b & 15) << 2) | (c >> 6)] : '='; out += i + 2 < input.size() ? table[c & 63] : '='; } return out; }

bool is_loopback_address(const std::string& value) noexcept {
    if (value == "localhost") return true;
    boost::system::error_code error;
    const auto address = asio::ip::make_address(value, error);
    return !error && address.is_loopback();
}

class PreviewHttpServer::Impl {
  public:
    Impl(std::string bind, std::uint16_t port, StatusProvider provider)
        : bind_(std::move(bind)), port_(port), status_(std::move(provider)), acceptor_(context_) {}
    ~Impl() { stop(); }
    void start() {
        if (running_.exchange(true)) return;
        if (!is_loopback_address(bind_)) { running_ = false; throw std::invalid_argument("preview server must bind to a loopback address"); }
        boost::system::error_code error;
        auto address = bind_ == "localhost" ? asio::ip::address_v4::loopback() : asio::ip::make_address(bind_, error);
        if (error) { running_ = false; throw std::runtime_error(error.message()); }
        acceptor_.open(address.is_v6() ? tcp::v6() : tcp::v4(), error);
        if (!error) acceptor_.set_option(asio::socket_base::reuse_address(true), error);
        if (!error) acceptor_.bind({address, port_}, error);
        if (!error) acceptor_.listen(asio::socket_base::max_listen_connections, error);
        if (error) { running_ = false; throw std::runtime_error("preview server listen: " + error.message()); }
        std::cerr << "IRIS preview listening on " << bind_ << ':' << port_ << "\n";
        accept_thread_ = std::thread([this] { accept_loop(); });
    }
    void stop() noexcept {
        if (!running_.exchange(false)) return;
        boost::system::error_code ignored; acceptor_.close(ignored); changed_.notify_all();
        if (accept_thread_.joinable()) accept_thread_.join();
        std::vector<std::thread> sessions;
        { std::scoped_lock lock(session_mutex_); sessions.swap(session_threads_); }
        for (auto& session : sessions) if (session.joinable()) session.join();
    }
    void set_frame(CameraId camera, std::shared_ptr<const std::vector<std::uint8_t>> jpeg) {
        { std::scoped_lock lock(mutex_); frames_[camera] = std::move(jpeg); ++generation_; }
        changed_.notify_all();
    }
    void event(std::string message) { { std::scoped_lock lock(mutex_); events_.push_back(std::move(message)); if (events_.size() > 64) events_.pop_front(); ++generation_; } changed_.notify_all(); }
    void h264(H264PreviewAccessUnit unit) { { std::scoped_lock lock(mutex_); h264_packets_.push_back(std::move(unit)); if (h264_packets_.size() > 128) h264_packets_.pop_front(); ++generation_; } changed_.notify_all(); }
    void h264_config(H264PreviewStreamConfig config) { { std::scoped_lock lock(mutex_); h264_configs_[config.camera] = std::move(config); ++generation_; } changed_.notify_all(); }
    PreviewTransportHealth health() const { std::scoped_lock lock(mutex_); return {running_, clients_, published_, 0, error_}; }
  private:
    void accept_loop() noexcept {
        while (running_) { tcp::socket socket(context_); boost::system::error_code error; acceptor_.accept(socket, error); if (error) { if (running_) { set_error(error.message()); std::cerr << "IRIS preview accept error: " << error.message() << "\n"; } continue; } std::cerr << "IRIS preview accepted connection\n"; std::scoped_lock lock(session_mutex_); session_threads_.emplace_back(&Impl::session, this, std::move(socket)); }
    }
    static std::optional<CameraId> camera_from_target(beast::string_view target) {
        const std::string prefix = "/api/preview/", suffix = ".mjpeg"; std::string path(target);
        if (!path.starts_with(prefix) || !path.ends_with(suffix)) return std::nullopt;
        try { std::size_t used{}; auto value=std::stoul(path.substr(prefix.size(), path.size()-prefix.size()-suffix.size()), &used); if (used != path.size()-prefix.size()-suffix.size() || value > UINT32_MAX) return std::nullopt; return static_cast<CameraId>(value); } catch (...) { return std::nullopt; }
    }
    void session(tcp::socket socket) noexcept {
        try { beast::flat_buffer buffer; http::request<http::string_body> request; http::read(socket, buffer, request);
            if (websocket::is_upgrade(request) && request.target() == "/api/events") { websocket_session(std::move(socket), std::move(request)); return; }
            if (websocket::is_upgrade(request) && request.target() == "/api/preview/stream") { h264_session(std::move(socket), std::move(request)); return; }
            const auto camera=camera_from_target(request.target()); if (request.method()!=http::verb::get || !camera) { http::response<http::string_body> r{http::status::not_found,request.version()}; r.set(http::field::content_type,"text/plain"); r.body()="not found"; r.prepare_payload(); http::write(socket,r); return; }
            mjpeg_session(std::move(socket), request.version(), *camera);
        } catch (const std::exception& exception) { std::cerr << "IRIS preview session error: " << exception.what() << "\n"; }
        catch (...) { std::cerr << "IRIS preview session error: unknown exception\n"; }
    }
    void mjpeg_session(tcp::socket socket, unsigned version, CameraId camera) {
        { std::scoped_lock lock(mutex_); ++clients_; }
        struct Guard { Impl* self; ~Guard(){std::scoped_lock lock(self->mutex_);--self->clients_;} } guard{this};
        const std::string boundary="iris-preview"; http::response<http::empty_body> header{http::status::ok,version}; header.set(http::field::content_type,"multipart/x-mixed-replace; boundary="+boundary); header.keep_alive(true); http::serializer<false,http::empty_body> serializer{header}; http::write_header(socket,serializer);
        std::size_t seen{}; while(running_) { std::shared_ptr<const std::vector<std::uint8_t>> jpeg; { std::unique_lock lock(mutex_); changed_.wait_for(lock,std::chrono::seconds(1),[&]{return !running_ || generation_ != seen;}); seen=generation_; if(auto it=frames_.find(camera);it!=frames_.end()) jpeg=it->second; } if(!jpeg) continue; std::string part="--"+boundary+"\r\nContent-Type: image/jpeg\r\nContent-Length: "+std::to_string(jpeg->size())+"\r\n\r\n"; asio::write(socket,asio::buffer(part)); asio::write(socket,asio::buffer(*jpeg)); asio::write(socket,asio::buffer(std::string("\r\n"))); ++published_; }
    }
    void websocket_session(tcp::socket socket, http::request<http::string_body> request) {
        { std::scoped_lock lock(mutex_); ++clients_; }
        struct Guard { Impl* self; ~Guard(){std::scoped_lock lock(self->mutex_);--self->clients_;} } guard{this}; websocket::stream<tcp::socket> ws(std::move(socket)); ws.accept(request); ws.text(true); std::size_t sent{};
        while(running_) { std::vector<std::string> messages; { std::unique_lock lock(mutex_); changed_.wait_for(lock,std::chrono::seconds(1),[&]{return !running_ || generation_ != sent;}); sent=generation_; messages.assign(events_.begin(),events_.end()); } ws.write(asio::buffer(std::string("{\"version\":1,\"type\":\"status\",\"data\":" + status_() + "}"))); for(const auto& message:messages) ws.write(asio::buffer(message)); }
        std::cerr << "IRIS preview events session ended\n";
    }
    void h264_session(tcp::socket socket, http::request<http::string_body> request) {
        websocket::stream<tcp::socket> ws(std::move(socket));
        ws.accept(request);
        ws.text(true);
        beast::flat_buffer hello_buffer;
        boost::system::error_code error;
        ws.read(hello_buffer, error);
        if (error) return;
        std::vector<CameraId> cameras;
        try {
            const auto hello = nlohmann::json::parse(beast::buffers_to_string(hello_buffer.data()));
            std::cerr << "IRIS preview H.264 hello: " << hello.dump() << "\n";
            if (hello.value("type", "") != "hello" || hello.value("version", 0) != 1) {
                ws.write(asio::buffer(std::string(R"({"version":1,"type":"error","message":"invalid preview hello"})")), error);
                return;
            }
            if (hello.contains("cameras") && hello["cameras"].is_array()) {
                for (const auto& value : hello["cameras"]) if (value.is_number_unsigned()) cameras.push_back(value.get<CameraId>());
            }
        } catch (...) {
            ws.write(asio::buffer(std::string(R"({"version":1,"type":"error","message":"malformed preview hello"})")), error);
            return;
        }
        {
            std::unique_lock lock(mutex_);
            changed_.wait_for(lock, std::chrono::seconds(2), [&] { if (!running_) return true; if (cameras.empty()) return !h264_configs_.empty(); return std::all_of(cameras.begin(), cameras.end(), [&](const auto camera) { return h264_configs_.contains(camera); }); });
        }
        nlohmann::json streams = nlohmann::json::array();
        { std::scoped_lock lock(mutex_); for (const auto camera : cameras) { auto it = h264_configs_.find(camera); if (it == h264_configs_.end()) { streams.push_back({{"camera_id", camera}, {"codec", "avc1.42E01E"}, {"format", "avc"}}); continue; } const auto& config = it->second; streams.push_back({{"camera_id", camera}, {"width", config.width}, {"height", config.height}, {"fps", config.fps}, {"codec", config.codec}, {"format", "avc"}, {"description", base64_encode(config.description)}}); } }
        ws.write(asio::buffer(nlohmann::json{{"version", 1}, {"type", "config"}, {"streams", streams}}.dump()), error);
        std::cerr << "IRIS preview H.264 config streams=" << streams.size() << "\n";
        if (error) return;
        { std::scoped_lock lock(mutex_); ++clients_; }
        struct Guard { Impl* self; ~Guard(){ std::scoped_lock lock(self->mutex_); --self->clients_; } } guard{this};
        std::uint64_t seen_generation = 0;
        std::unordered_map<CameraId, std::uint64_t> last_sequence;
        std::unordered_map<CameraId, bool> awaiting_keyframe;
        while (running_) {
            std::vector<std::vector<std::uint8_t>> outgoing;
            std::unordered_map<CameraId, std::vector<H264PreviewAccessUnit>> pending;
            {
                std::unique_lock lock(mutex_);
                changed_.wait_for(lock, std::chrono::seconds(1), [&] { return !running_ || generation_ != seen_generation; });
                seen_generation = generation_;
                for (const auto& packet : h264_packets_) {
                    if (!cameras.empty() && std::find(cameras.begin(), cameras.end(), packet.camera) == cameras.end()) continue;
                    const auto previous = last_sequence.find(packet.camera);
                    if (previous != last_sequence.end() && packet.sequence <= previous->second) continue;
                    if (awaiting_keyframe[packet.camera] && !(packet.flags & h264_flag_keyframe)) continue;
                    auto next = packet;
                    if (previous == last_sequence.end() && !(packet.flags & h264_flag_keyframe)) { awaiting_keyframe[packet.camera] = true; continue; }
                    if (previous != last_sequence.end() && packet.sequence > previous->second + 1) { awaiting_keyframe[packet.camera] = true; if (!(packet.flags & h264_flag_keyframe)) continue; }
                    if (packet.flags & h264_flag_discontinuity) { awaiting_keyframe[packet.camera] = true; if (!(packet.flags & h264_flag_keyframe)) continue; }
                    awaiting_keyframe[packet.camera] = false;
                    last_sequence[packet.camera] = packet.sequence;
                    auto& client_queue = pending[packet.camera];
                    if (client_queue.size() >= 2) {
                        client_queue.erase(client_queue.begin());
                        awaiting_keyframe[packet.camera] = true;
                    }
                    if (awaiting_keyframe[packet.camera] && !(next.flags & h264_flag_keyframe)) continue;
                    if (awaiting_keyframe[packet.camera]) { next.flags |= h264_flag_discontinuity; awaiting_keyframe[packet.camera] = false; }
                    client_queue.push_back(std::move(next));
                }
            }
            for (auto& [camera, packets] : pending) for (auto& packet : packets) outgoing.push_back(encode_h264_preview_access_unit(packet));
            for (const auto& packet : outgoing) {
                ws.binary(true);
                ws.write(asio::buffer(packet), error);
                if (error) return;
            }
        }
    }
    void set_error(std::string error) { std::scoped_lock lock(mutex_); error_=std::move(error); }
    std::string bind_; std::uint16_t port_; StatusProvider status_; asio::io_context context_; tcp::acceptor acceptor_; std::atomic_bool running_{false}; std::thread accept_thread_; std::mutex session_mutex_; std::vector<std::thread> session_threads_; mutable std::mutex mutex_; std::condition_variable changed_; std::unordered_map<CameraId,std::shared_ptr<const std::vector<std::uint8_t>>> frames_; std::deque<std::string> events_; std::deque<H264PreviewAccessUnit> h264_packets_; std::unordered_map<CameraId,H264PreviewStreamConfig> h264_configs_; std::size_t generation_{},clients_{},published_{}; std::string error_;
};
PreviewHttpServer::PreviewHttpServer(std::string bind, std::uint16_t port, StatusProvider status):impl_(std::make_unique<Impl>(std::move(bind),port,std::move(status))){} PreviewHttpServer::~PreviewHttpServer()=default; void PreviewHttpServer::start(){impl_->start();} void PreviewHttpServer::stop() noexcept{impl_->stop();} void PreviewHttpServer::set_frame(CameraId c,std::shared_ptr<const std::vector<std::uint8_t>> j){impl_->set_frame(c,std::move(j));} void PreviewHttpServer::publish_event(std::string e){impl_->event(std::move(e));} void PreviewHttpServer::publish_h264(H264PreviewAccessUnit u){impl_->h264(std::move(u));} void PreviewHttpServer::set_h264_stream_config(H264PreviewStreamConfig c){impl_->h264_config(std::move(c));} PreviewTransportHealth PreviewHttpServer::health()const{return impl_->health();}
} // namespace iris::output
