#include "iris/stages/output/PreviewSink.hpp"

#include "iris/pipeline/Channel.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace iris {
namespace {
template <typename T> void append(std::vector<std::byte>& out, const T& value) {
    const auto* p = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), p, p + sizeof(value));
}

// Explicit field encoding: this is a wire format, never a C++ struct dump.
std::vector<std::byte> v1(const Packet& p) {
    std::vector<std::byte> out; const std::uint32_t version = 1; append(out, version);
    append(out, p.sequence); const auto nf = static_cast<std::uint32_t>(p.frames.size());
    const auto np = static_cast<std::uint32_t>(p.poses ? p.poses->size() : 0); append(out,nf); append(out,np);
    for (const auto& f : p.frames) { append(out,f.camera); append(out,f.sequence); append(out,f.extent.width); append(out,f.extent.height); append(out,f.format); append(out,f.buffer.stride_bytes); append(out,f.buffer.size_bytes); cudaIpcMemHandle_t h{}; if (!f.buffer.data || cudaIpcGetMemHandle(&h,f.buffer.data)!=cudaSuccess) throw std::runtime_error("could not export output frame through CUDA IPC"); const auto* b=reinterpret_cast<const std::byte*>(&h); out.insert(out.end(),b,b+sizeof h); }
    if (p.poses) for (const auto& pose:*p.poses) append(out,pose.source_sequence); return out;
}
std::vector<std::byte> v2(const Packet& p) {
    std::vector<std::byte> out; const std::uint32_t magic=0x32535249U, version=2; append(out,magic); append(out,version);
    const std::uint64_t payload_size_offset = 0; append(out,payload_size_offset); append(out,p.sequence);
    const auto frame_count=static_cast<std::uint32_t>(p.frames.size()), pose_count=static_cast<std::uint32_t>(p.poses?p.poses->size():0); append(out,frame_count); append(out,pose_count);
    for(const auto& f:p.frames) { append(out,f.camera); append(out,f.sequence); append(out,f.extent.width); append(out,f.extent.height); append(out,f.format); append(out,f.buffer.stride_bytes); append(out,f.buffer.size_bytes); append(out,f.buffer.device_id); const auto source_ns=f.timing.source_time.count(); append(out,source_ns); cudaIpcMemHandle_t h{}; if(!f.buffer.data || cudaIpcGetMemHandle(&h,f.buffer.data)!=cudaSuccess) throw std::runtime_error("could not export output frame through CUDA IPC"); const auto* b=reinterpret_cast<const std::byte*>(&h); out.insert(out.end(),b,b+sizeof h); }
    if(p.poses) for(const auto& pose:*p.poses) append(out,pose.source_sequence); const auto n=static_cast<std::uint64_t>(out.size()); std::memcpy(out.data()+sizeof(magic)+sizeof(version),&n,sizeof n); return out;
}
#ifdef _WIN32
std::wstring widen(const std::string& value) { const auto n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),nullptr,0); if(n<=0) throw std::runtime_error("shared-memory destination is not valid UTF-8"); std::wstring out(n,L'\0'); MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),out.data(),n); return out; }
struct Header { alignas(8) volatile LONG64 sequence_lock{}; std::uint64_t packet_sequence{}; std::uint64_t payload_size{}; };
class Mapping { public: Mapping(const std::string& name,std::size_t cap):cap_(cap) { if(cap<=sizeof(Header)) throw std::invalid_argument("shared-memory capacity is too small"); auto n=widen(name); h_=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,static_cast<DWORD>(cap>>32U),static_cast<DWORD>(cap),n.c_str()); if(!h_) throw std::runtime_error("CreateFileMappingW failed"); v_=MapViewOfFile(h_,FILE_MAP_ALL_ACCESS,0,0,cap); if(!v_) {CloseHandle(h_); throw std::runtime_error("MapViewOfFile failed");} } ~Mapping(){if(v_)UnmapViewOfFile(v_);if(h_)CloseHandle(h_);} void write(std::uint64_t sequence,const std::vector<std::byte>& bytes){if(bytes.size()>cap_-sizeof(Header)) throw std::runtime_error("packet exceeds configured shared-memory capacity");auto* hd=static_cast<Header*>(v_);InterlockedIncrement64(&hd->sequence_lock);hd->packet_sequence=sequence;hd->payload_size=bytes.size();std::memcpy(hd+1,bytes.data(),bytes.size());MemoryBarrier();InterlockedIncrement64(&hd->sequence_lock);} private: HANDLE h_{};void* v_{};std::size_t cap_{};};
#endif
} // namespace

class PreviewSink::Impl {
 public:
  explicit Impl(PreviewConfig c): config(std::move(c)), queue(config.shared_memory_queue_capacity,OverflowPolicy::DropOldest) { if (config.shared_memory_queue_capacity == 0) throw std::invalid_argument("preview queue capacity must be greater than zero"); }
  ~Impl(){stop();}
  void start(){if(running.exchange(true))return; {std::scoped_lock l(lock); configure_locked(config.shared_memory);} worker=std::thread([this]{while(auto p=queue.receive()){try{std::scoped_lock l(lock); if(!config.shared_memory.enabled)continue; auto b=v2(**p);
#ifdef _WIN32
if(v2map)v2map->write((*p)->sequence,b); if(v1map)v1map->write((*p)->sequence,v1(**p));
#endif
++published;}catch(const std::exception& e){std::scoped_lock l(lock); error=e.what();}}});}
  void stop() noexcept {if(running.exchange(false)){queue.close();} if(worker.joinable())worker.join();}
  void publish(PreviewPacket p) noexcept {if(!running || !config.shared_memory.enabled)return; try{const auto result=queue.send(std::move(p)); if(result==SendResult::ReplacedOldest || result==SendResult::DroppedNewest) ++dropped; if(result!=SendResult::Closed) ++submitted;}catch(...){++dropped;}}
  OutputCommandResult configure(SharedMemoryOutputConfig c){std::scoped_lock l(lock);try{configure_locked(c);config.shared_memory=std::move(c);return {OutputCommandStatus::Applied,"shared-memory configuration applied"};}catch(const std::exception&e){error=e.what();return {OutputCommandStatus::Failed,error};}}
  PreviewTransportHealth health()const{std::scoped_lock l(lock);return {config.shared_memory.enabled,0,published.load(),dropped.load(),error};}
 private:
  void configure_locked(const SharedMemoryOutputConfig& c){
#ifdef _WIN32
std::unique_ptr<Mapping> legacy,versioned; if(c.enabled){versioned=std::make_unique<Mapping>(c.destination+"_v2",c.capacity_bytes);if(c.legacy_v1)legacy=std::make_unique<Mapping>(c.destination,c.capacity_bytes);} v1map=std::move(legacy);v2map=std::move(versioned);
#else
if(c.enabled) throw std::runtime_error("shared-memory output is currently implemented for Windows only");
#endif
}
  PreviewConfig config; Channel<PreviewPacket> queue; std::atomic_bool running{false}; std::thread worker; mutable std::mutex lock; std::atomic_size_t submitted{0},published{0},dropped{0};std::string error;
#ifdef _WIN32
std::unique_ptr<Mapping> v1map,v2map;
#endif
 };
PreviewSink::PreviewSink(PreviewConfig c):impl_(std::make_unique<Impl>(std::move(c))){} PreviewSink::~PreviewSink()=default; void PreviewSink::start(){impl_->start();} void PreviewSink::publish(PreviewPacket p) noexcept{impl_->publish(std::move(p));} void PreviewSink::stop() noexcept{impl_->stop();} OutputCommandResult PreviewSink::configure_shared_memory(SharedMemoryOutputConfig c){return impl_->configure(std::move(c));} PreviewTransportHealth PreviewSink::shared_memory_health()const{return impl_->health();}
} // namespace iris
