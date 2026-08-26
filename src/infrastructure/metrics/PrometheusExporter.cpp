#include "iris/infrastructure/metrics/PrometheusExporter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <WS2tcpip.h>
#include <WinSock2.h>
#endif

namespace iris::infrastructure::metrics {
namespace {

template <typename Map> std::vector<std::string> sorted_names(const Map& values) {
    std::vector<std::string> names;
    names.reserve(values.size());
    for (const auto& [name, unused] : values) {
        static_cast<void>(unused);
        names.push_back(name);
    }
    std::ranges::sort(names);
    return names;
}

void append_response(std::uintptr_t socket_handle, std::string_view status,
                     std::string_view content_type, std::string_view body) {
#ifdef _WIN32
    std::ostringstream headers;
    headers << "HTTP/1.1 " << status << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n\r\n";
    const auto header_text = headers.str();
    const auto socket = static_cast<SOCKET>(socket_handle);
    send(socket, header_text.data(), static_cast<int>(header_text.size()), 0);
    send(socket, body.data(), static_cast<int>(body.size()), 0);
#else
    static_cast<void>(socket_handle);
    static_cast<void>(status);
    static_cast<void>(content_type);
    static_cast<void>(body);
#endif
}

} // namespace

std::string format_prometheus(const MetricsSnapshot& snapshot) {
    std::ostringstream output;
    output << std::setprecision(17);
    for (const auto& name : sorted_names(snapshot.counters)) {
        output << "# TYPE " << name << " counter\n";
        output << name << ' ' << snapshot.counters.at(name) << '\n';
    }
    for (const auto& name : sorted_names(snapshot.gauges)) {
        output << "# TYPE " << name << " gauge\n";
        output << name << ' ' << snapshot.gauges.at(name) << '\n';
    }
    for (const auto& name : sorted_names(snapshot.histograms)) {
        const auto& histogram = snapshot.histograms.at(name);
        output << "# TYPE " << name << " histogram\n";
        std::uint64_t cumulative = 0;
        for (std::size_t index = 0; index < histogram.bounds.size(); ++index) {
            cumulative += histogram.counts.at(index);
            output << name << "_bucket{le=\"" << histogram.bounds[index] << "\"} " << cumulative
                   << '\n';
        }
        output << name << "_bucket{le=\"+Inf\"} " << histogram.count << '\n';
        output << name << "_sum " << histogram.sum << '\n';
        output << name << "_count " << histogram.count << '\n';
    }
    return output.str();
}

class PrometheusExporter::Impl {
  public:
    Impl(const MetricRegistry& registry, std::uint16_t port) : registry_(registry), port_(port) {}

    ~Impl() { stop(); }

    void start() {
        if (worker_.joinable()) {
            return;
        }
#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed for Prometheus exporter");
        }
        winsock_started_ = true;
        listen_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_socket_ == INVALID_SOCKET) {
            stop();
            throw std::runtime_error("could not create Prometheus listen socket");
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port_);
        if (bind(listen_socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
                SOCKET_ERROR ||
            listen(listen_socket_, 8) == SOCKET_ERROR) {
            const auto error = WSAGetLastError();
            stop();
            throw std::runtime_error(
                "could not bind Prometheus exporter to 127.0.0.1:" + std::to_string(port_) +
                " (Winsock " + std::to_string(error) + ")");
        }
        u_long non_blocking = 1;
        ioctlsocket(listen_socket_, FIONBIO, &non_blocking);
        worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
#else
        throw std::runtime_error("Prometheus exporter is currently implemented for Windows");
#endif
    }

    void stop() {
        if (worker_.joinable()) {
            worker_.request_stop();
            worker_.join();
        }
#ifdef _WIN32
        if (listen_socket_ != INVALID_SOCKET) {
            closesocket(listen_socket_);
            listen_socket_ = INVALID_SOCKET;
        }
        if (winsock_started_) {
            WSACleanup();
            winsock_started_ = false;
        }
#endif
    }

  private:
    void run(std::stop_token stop) {
#ifdef _WIN32
        while (!stop.stop_requested()) {
            const SOCKET client = accept(listen_socket_, nullptr, nullptr);
            if (client == INVALID_SOCKET) {
                const auto error = WSAGetLastError();
                if (error != WSAEWOULDBLOCK) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            const DWORD timeout_ms = 1000;
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                       sizeof(timeout_ms));
            std::array<char, 4096> request{};
            const int received = recv(client, request.data(), static_cast<int>(request.size()), 0);
            const std::string_view text(request.data(), received > 0 ? received : 0);
            if (text.starts_with("GET /metrics ")) {
                const auto body = format_prometheus(registry_.snapshot());
                append_response(static_cast<std::uintptr_t>(client), "200 OK",
                                "text/plain; version=0.0.4; charset=utf-8", body);
            } else if (text.starts_with("GET /healthz ")) {
                append_response(static_cast<std::uintptr_t>(client), "200 OK",
                                "text/plain; charset=utf-8", "ok\n");
            } else {
                append_response(static_cast<std::uintptr_t>(client), "404 Not Found",
                                "text/plain; charset=utf-8", "not found\n");
            }
            closesocket(client);
        }
#else
        static_cast<void>(stop);
#endif
    }

    const MetricRegistry& registry_;
    std::uint16_t port_;
    std::jthread worker_;
#ifdef _WIN32
    SOCKET listen_socket_{INVALID_SOCKET};
    bool winsock_started_{};
#endif
};

PrometheusExporter::PrometheusExporter(const MetricRegistry& registry, std::uint16_t port)
    : impl_(std::make_unique<Impl>(registry, port)) {}

PrometheusExporter::~PrometheusExporter() = default;

void PrometheusExporter::start() { impl_->start(); }

void PrometheusExporter::stop() { impl_->stop(); }

} // namespace iris::infrastructure::metrics
