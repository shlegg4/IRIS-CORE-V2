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

namespace iris::output {
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

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
    PreviewTransportHealth health() const { std::scoped_lock lock(mutex_); return {running_, clients_, published_, 0, error_}; }
  private:
    void accept_loop() noexcept {
        while (running_) { tcp::socket socket(context_); boost::system::error_code error; acceptor_.accept(socket, error); if (error) { if (running_) set_error(error.message()); continue; } std::scoped_lock lock(session_mutex_); session_threads_.emplace_back(&Impl::session, this, std::move(socket)); }
    }
    static std::optional<CameraId> camera_from_target(beast::string_view target) {
        const std::string prefix = "/api/preview/", suffix = ".mjpeg"; std::string path(target);
        if (!path.starts_with(prefix) || !path.ends_with(suffix)) return std::nullopt;
        try { std::size_t used{}; auto value=std::stoul(path.substr(prefix.size(), path.size()-prefix.size()-suffix.size()), &used); if (used != path.size()-prefix.size()-suffix.size() || value > UINT32_MAX) return std::nullopt; return static_cast<CameraId>(value); } catch (...) { return std::nullopt; }
    }
    void session(tcp::socket socket) noexcept {
        try { beast::flat_buffer buffer; http::request<http::string_body> request; http::read(socket, buffer, request);
            if (websocket::is_upgrade(request) && request.target() == "/api/events") { websocket_session(std::move(socket), std::move(request)); return; }
            const auto camera=camera_from_target(request.target()); if (request.method()!=http::verb::get || !camera) { http::response<http::string_body> r{http::status::not_found,request.version()}; r.set(http::field::content_type,"text/plain"); r.body()="not found"; r.prepare_payload(); http::write(socket,r); return; }
            mjpeg_session(std::move(socket), request.version(), *camera);
        } catch (const std::exception&) {}
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
    }
    void set_error(std::string error) { std::scoped_lock lock(mutex_); error_=std::move(error); }
    std::string bind_; std::uint16_t port_; StatusProvider status_; asio::io_context context_; tcp::acceptor acceptor_; std::atomic_bool running_{false}; std::thread accept_thread_; std::mutex session_mutex_; std::vector<std::thread> session_threads_; mutable std::mutex mutex_; std::condition_variable changed_; std::unordered_map<CameraId,std::shared_ptr<const std::vector<std::uint8_t>>> frames_; std::deque<std::string> events_; std::size_t generation_{},clients_{},published_{}; std::string error_;
};
PreviewHttpServer::PreviewHttpServer(std::string bind, std::uint16_t port, StatusProvider status):impl_(std::make_unique<Impl>(std::move(bind),port,std::move(status))){} PreviewHttpServer::~PreviewHttpServer()=default; void PreviewHttpServer::start(){impl_->start();} void PreviewHttpServer::stop() noexcept{impl_->stop();} void PreviewHttpServer::set_frame(CameraId c,std::shared_ptr<const std::vector<std::uint8_t>> j){impl_->set_frame(c,std::move(j));} void PreviewHttpServer::publish_event(std::string e){impl_->event(std::move(e));} PreviewTransportHealth PreviewHttpServer::health()const{return impl_->health();}
} // namespace iris::output
