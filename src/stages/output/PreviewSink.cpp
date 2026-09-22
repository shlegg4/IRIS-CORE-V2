#include "iris/stages/output/PreviewSink.hpp"
#include "iris/stages/output/PreviewHttpServer.hpp"
#include "iris/stages/output/H264Transport.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"

#include "iris/pipeline/Channel.hpp"

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <mutex>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <chrono>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace iris {
namespace output { void resize_bgr(const unsigned char*, std::size_t, unsigned char*, std::size_t, unsigned, unsigned, unsigned, unsigned, cudaStream_t); }
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
    if(p.poses) for(const auto& pose:*p.poses) append(out,pose.source_sequence);
     const auto nm=static_cast<std::uint32_t>(p.multiview_poses?p.multiview_poses->size():0);append(out,nm);
     if(p.multiview_poses)for(const auto& pose:*p.multiview_poses){const std::uint8_t active=pose.active?1:0;append(out,active);for(const auto& joint:pose.joints_3d)for(float value:joint)append(out,value);for(bool valid:pose.joint_valid){const std::uint8_t value=valid?1:0;append(out,value);}for(const auto& view:pose.joint_scores)for(float score:view)append(out,score);}
    const auto n=static_cast<std::uint64_t>(out.size()); std::memcpy(out.data()+sizeof(magic)+sizeof(version),&n,sizeof n); return out;
}
template <std::size_t N>
void append_pose_json(std::ostringstream& out, const std::array<std::array<float, 3>, N>& joints, const std::array<bool, N>& valid, std::size_t id) {
    out << "{\"id\":" << id << ",\"joints3d\":[";
    std::array<bool, N> json_valid = valid;
    for(std::size_t i=0;i<N;++i){
        if(i)out<<',';
        const auto finite=std::isfinite(joints[i][0])&&std::isfinite(joints[i][1])&&std::isfinite(joints[i][2]);
        json_valid[i]=json_valid[i]&&finite;
        if(finite)out<<'['<<joints[i][0]<<','<<joints[i][1]<<','<<joints[i][2]<<']';
        else out<<"[0,0,0]";
    }
    out << "],\"valid\":["; for(std::size_t i=0;i<N;++i){if(i)out<<',';out<<(json_valid[i]?"true":"false");} out << "]}";
}
template <std::size_t N>
std::string pose_event(std::uint64_t sequence, const std::array<std::array<float, 3>, N>& joints, const std::array<bool, N>& valid) {
    std::ostringstream out; out << "{\"version\":1,\"type\":\"pose\",\"data\":{\"sequence\":" << sequence << ',';
    std::ostringstream person; append_pose_json(person,joints,valid,0);
    const auto json=person.str(); out << json.substr(1,json.size()-2) << "}}"; return out.str();
}
std::string pose_event(const Packet&, const Pose& pose) { std::array<bool,panoptic_joint_count> valid{}; for(std::size_t i=0;i<valid.size();++i)valid[i]=pose.joint_confidence[i]>0.0F; return pose_event(pose.source_sequence,pose.joints_3d_mm,valid); }
std::string multiview_pose_event(const Packet& packet) {
    const auto* pose3d = packet.multiview_poses && !packet.multiview_poses->empty() ? &(*packet.multiview_poses)[0] : nullptr;
    std::ostringstream out; out << "{\"version\":1,\"type\":\"pose\",\"data\":{\"sequence\":" << packet.sequence << ",\"joints3d\":[";
    for (std::size_t joint = 0; joint < coco_joint_count; ++joint) {
        if (joint) out << ',';
        const auto& value = pose3d ? pose3d->joints_3d[joint] : std::array<float, 3>{};
        out << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
    }
    out << "],\"valid\":[";
    for (std::size_t joint = 0; joint < coco_joint_count; ++joint) {
        if (joint) out << ',';
        out << (pose3d && pose3d->joint_valid[joint] ? "true" : "false");
    }
    out << "],\"people\":[";
    if (packet.multiview_poses) {
        bool first_person = true;
        for (std::size_t person_id = 0; person_id < packet.multiview_poses->size(); ++person_id) {
            const auto& pose = (*packet.multiview_poses)[person_id];
            if (!pose.active) continue;
            if (!first_person) out << ',';
            first_person = false;
            out << "{\"id\":" << person_id << ",\"joints3d\":[";
            for (std::size_t joint = 0; joint < coco_joint_count; ++joint) {
                if (joint) out << ',';
                const auto& value = pose.joints_3d[joint];
                out << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
            }
            out << "],\"valid\":[";
            for (std::size_t joint = 0; joint < coco_joint_count; ++joint) {
                if (joint) out << ',';
                out << (pose.joint_valid[joint] ? "true" : "false");
            }
            out << "]}";
        }
    }
    out << "],\"views\":[";
    bool first=true;
    if (packet.view_poses_2d) for (const auto& pose : *packet.view_poses_2d) {
        const auto frame = std::find_if(packet.frames.begin(), packet.frames.end(), [&pose](const Frame& value) { return value.camera == pose.camera_id; });
        if (frame == packet.frames.end()) continue;
        if (!first) out << ','; first = false;
        out << "{\"camera_id\":" << pose.camera_id << ",\"person_id\":" << pose.person_id
            << ",\"frame_sequence\":" << frame->sequence
            << ",\"width\":" << frame->extent.width << ",\"height\":" << frame->extent.height << ",\"points\":[";
        for(std::size_t joint=0;joint<pose.points_px.size();++joint){if(joint)out<<','; const auto& p=pose.points_px[joint]; out<<'['<<p[0]<<','<<p[1]<<']';}
        out << "],\"scores\":["; for(std::size_t joint=0;joint<pose.scores.size();++joint){if(joint)out<<',';out<<pose.scores[joint];}
        out << "],\"valid\":["; for(std::size_t joint=0;joint<pose.valid.size();++joint){if(joint)out<<',';out<<(pose.valid[joint]?"true":"false");} out << "]}";
    }
    out << "]}}"; return out.str();
}
#ifdef _WIN32
std::wstring widen(const std::string& value) { const auto n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),nullptr,0); if(n<=0) throw std::runtime_error("shared-memory destination is not valid UTF-8"); std::wstring out(n,L'\0'); MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),out.data(),n); return out; }
struct Header { alignas(8) volatile LONG64 sequence_lock{}; std::uint64_t packet_sequence{}; std::uint64_t payload_size{}; };
class Mapping { public: Mapping(const std::string& name,std::size_t cap):cap_(cap) { if(cap<=sizeof(Header)) throw std::invalid_argument("shared-memory capacity is too small"); auto n=widen(name); h_=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,static_cast<DWORD>(cap>>32U),static_cast<DWORD>(cap),n.c_str()); if(!h_) throw std::runtime_error("CreateFileMappingW failed"); v_=MapViewOfFile(h_,FILE_MAP_ALL_ACCESS,0,0,cap); if(!v_) {CloseHandle(h_); throw std::runtime_error("MapViewOfFile failed");} } ~Mapping(){if(v_)UnmapViewOfFile(v_);if(h_)CloseHandle(h_);} void write(std::uint64_t sequence,const std::vector<std::byte>& bytes){if(bytes.size()>cap_-sizeof(Header)) throw std::runtime_error("packet exceeds configured shared-memory capacity");auto* hd=static_cast<Header*>(v_);InterlockedIncrement64(&hd->sequence_lock);hd->packet_sequence=sequence;hd->payload_size=bytes.size();std::memcpy(hd+1,bytes.data(),bytes.size());MemoryBarrier();InterlockedIncrement64(&hd->sequence_lock);} private: HANDLE h_{};void* v_{};std::size_t cap_{};};
#endif
} // namespace

class MjpegTransport final : public PreviewTransport {
  public:
    MjpegTransport(MjpegPreviewConfig config, output::PreviewHttpServer& server)
        : config_(std::move(config)), server_(server), queue_(config_.queue_capacity, OverflowPolicy::DropOldest) {
        if (!config_.queue_capacity || !config_.max_fps || !config_.max_width || config_.jpeg_quality > 100) throw std::invalid_argument("invalid MJPEG preview configuration");
    }
    ~MjpegTransport() override { stop(); }
    void start() override { if (running_.exchange(true)) return; worker_=std::thread([this]{run();}); }
    void stop() noexcept override { if(running_.exchange(false)) queue_.close(); if(worker_.joinable())worker_.join(); }
    void publish(PreviewPacket packet) noexcept override { if(!running_)return; const auto r=queue_.send(std::move(packet)); if(r==SendResult::ReplacedOldest||r==SendResult::DroppedNewest)++dropped_; }
    PreviewTransportHealth health() const override { std::scoped_lock lock(lock_); return {config_.enabled,0,0,0,0,published_,dropped_,error_}; }
  private:
    void run() noexcept { while(auto packet=queue_.receive()) { for(const auto& frame:(*packet)->frames) { try { if(frame.format != PixelFormat::Bgr8 || !frame.buffer.data) continue; if(cudaSetDevice(frame.buffer.device_id) != cudaSuccess) throw std::runtime_error("preview cudaSetDevice failed"); const auto now=std::chrono::steady_clock::now(); const auto interval=std::chrono::milliseconds(1000/config_.max_fps); if(auto it=last_.find(frame.camera);it!=last_.end() && now-it->second<interval) continue; if(frame.ready) frame.ready->synchronize(); auto jpeg=encode(frame); server_.set_frame(frame.camera,std::move(jpeg)); last_[frame.camera]=now; ++published_; } catch(const std::exception& e) { std::scoped_lock lock(lock_); error_=e.what(); } } } }
    std::shared_ptr<const std::vector<std::uint8_t>> encode(const Frame& frame) {
        // nvJPEG consumes the captured GPU buffer directly after its readiness event completes.
        if (!frame.extent.width || !frame.extent.height || frame.buffer.stride_bytes < static_cast<std::size_t>(frame.extent.width) * 3) {
            throw std::runtime_error("invalid preview frame layout camera=" + std::to_string(frame.camera) +
                                     " extent=" + std::to_string(frame.extent.width) + "x" +
                                     std::to_string(frame.extent.height) + " stride=" +
                                     std::to_string(frame.buffer.stride_bytes));
        }
        cudaPointerAttributes attributes{};
        const auto pointer_status = cudaPointerGetAttributes(&attributes, frame.buffer.data);
        if (pointer_status != cudaSuccess || attributes.type != cudaMemoryTypeDevice ||
            attributes.device != frame.buffer.device_id) {
            throw std::runtime_error("invalid preview GPU buffer camera=" + std::to_string(frame.camera) +
                                     " device=" + std::to_string(frame.buffer.device_id));
        }
        nvjpegHandle_t handle{}; nvjpegEncoderState_t state{}; nvjpegEncoderParams_t params{}; cudaStream_t stream{};
        auto cleanup=[&]{if(stream)cudaStreamDestroy(stream);if(params)nvjpegEncoderParamsDestroy(params);if(state)nvjpegEncoderStateDestroy(state);if(handle)nvjpegDestroy(handle);};
        if(nvjpegCreateSimple(&handle)!=NVJPEG_STATUS_SUCCESS) throw std::runtime_error("nvjpegCreateSimple failed");
        if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess){cleanup();throw std::runtime_error("cudaStreamCreateWithFlags failed");}
        if(nvjpegEncoderStateCreate(handle,&state,stream)!=NVJPEG_STATUS_SUCCESS ||
           nvjpegEncoderParamsCreate(handle,&params,stream)!=NVJPEG_STATUS_SUCCESS ||
           nvjpegEncoderParamsSetQuality(params,static_cast<int>(config_.jpeg_quality),stream)!=NVJPEG_STATUS_SUCCESS ||
           nvjpegEncoderParamsSetSamplingFactors(params,NVJPEG_CSS_420,stream)!=NVJPEG_STATUS_SUCCESS){cleanup();throw std::runtime_error("nvjpeg encoder setup failed");}
        const auto width=frame.extent.width > config_.max_width ? config_.max_width : frame.extent.width;
        const auto height=static_cast<std::uint32_t>((static_cast<std::uint64_t>(frame.extent.height)*width)/frame.extent.width);
        unsigned char* resized{}; std::size_t resized_pitch{};
        if(width != frame.extent.width && cudaMallocPitch(reinterpret_cast<void**>(&resized),&resized_pitch,width*3,height)!=cudaSuccess){cleanup();throw std::runtime_error("preview resize allocation failed");}
        if(resized) output::resize_bgr(static_cast<unsigned char*>(frame.buffer.data),frame.buffer.stride_bytes,resized,resized_pitch,frame.extent.width,frame.extent.height,width,height,stream);
        nvjpegImage_t image{}; image.channel[0]=resized ? resized : static_cast<unsigned char*>(frame.buffer.data); image.pitch[0]=static_cast<unsigned int>(resized ? resized_pitch : frame.buffer.stride_bytes);
        const auto result=nvjpegEncodeImage(handle,state,params,&image,NVJPEG_INPUT_BGRI,static_cast<int>(width),static_cast<int>(height),stream);
        if(result!=NVJPEG_STATUS_SUCCESS){if(resized)cudaFree(resized);cleanup();throw std::runtime_error("nvjpegEncodeImage failed (status " + std::to_string(static_cast<int>(result)) + ", camera=" + std::to_string(frame.camera) + ", extent=" + std::to_string(width) + "x" + std::to_string(height) + ", pitch=" + std::to_string(image.pitch[0]) + ")");}
        if(cudaStreamSynchronize(stream)!=cudaSuccess){if(resized)cudaFree(resized);cleanup();throw std::runtime_error("CUDA JPEG encoding failed");}
        std::size_t bytes{}; if(nvjpegEncodeRetrieveBitstream(handle,state,nullptr,&bytes,stream)!=NVJPEG_STATUS_SUCCESS){if(resized)cudaFree(resized);cleanup();throw std::runtime_error("nvjpeg bitstream size failed");}
        auto output=std::make_shared<std::vector<std::uint8_t>>(bytes); if(nvjpegEncodeRetrieveBitstream(handle,state,output->data(),&bytes,stream)!=NVJPEG_STATUS_SUCCESS){if(resized)cudaFree(resized);cleanup();throw std::runtime_error("nvjpeg bitstream retrieval failed");} output->resize(bytes); if(resized)cudaFree(resized); cleanup(); return output;
    }
    MjpegPreviewConfig config_; output::PreviewHttpServer& server_; Channel<PreviewPacket> queue_; std::atomic_bool running_{false}; std::thread worker_; mutable std::mutex lock_; std::unordered_map<CameraId,std::chrono::steady_clock::time_point> last_; std::size_t published_{},dropped_{}; std::string error_;
};

class PreviewSink::Impl {
 public:
  explicit Impl(PreviewConfig c, infrastructure::metrics::MetricRegistry* metrics): config(std::move(c)), queue(config.shared_memory_queue_capacity,OverflowPolicy::DropOldest), event_queue(config.http.queue_capacity,OverflowPolicy::DropOldest), metrics_enabled(metrics != nullptr),
      pose_video_skew_ms(metrics ? metrics->histogram("iris_preview_pose_video_publish_abs_skew_ms", {1,2,5,10,20,33,50,75,100,150,250,500,1000,2000}) : infrastructure::metrics::Histogram{}),
      last_pose_video_skew_ms(metrics ? metrics->gauge("iris_preview_last_pose_video_publish_skew_ms") : infrastructure::metrics::Gauge{}) { if (config.shared_memory_queue_capacity == 0 || config.http.queue_capacity == 0 || !config.h264.queue_capacity) throw std::invalid_argument("preview queue capacity must be greater than zero"); }
  ~Impl(){stop();}
  void start(){if(running.exchange(true))return; {std::scoped_lock l(lock); configure_locked(config.shared_memory);} start_network(); worker=std::thread([this]{while(auto p=queue.receive()){try{std::scoped_lock l(lock); if(!config.shared_memory.enabled)continue; auto b=v2(**p);
#ifdef _WIN32
if(v2map)v2map->write((*p)->sequence,b); if(v1map)v1map->write((*p)->sequence,v1(**p));
#endif
++published;}catch(const std::exception& e){std::scoped_lock l(lock); error=e.what();}}}); event_worker=std::thread([this]{std::unordered_map<CameraId,std::uint64_t> last_frame_sequence; while(auto packet=event_queue.receive()){{std::scoped_lock l(lock); if(!server)continue; bool in_order=true; for(const auto& frame:(*packet)->frames){const auto previous=last_frame_sequence.find(frame.camera); if(previous!=last_frame_sequence.end()&&frame.sequence<=previous->second){in_order=false;break;}} if(!in_order)continue; for(const auto& frame:(*packet)->frames)last_frame_sequence[frame.camera]=frame.sequence; if((*packet)->poses)for(const auto& pose:*(*packet)->poses)server->publish_event(pose_event(**packet,pose)); if((*packet)->multiview_poses)server->publish_event(multiview_pose_event(**packet)); } if((*packet)->poses||(*packet)->multiview_poses){const auto published_at=std::chrono::steady_clock::now();for(const auto& frame:(*packet)->frames)record_pose_publish(frame.camera,frame.sequence,published_at);}}});}
  void stop() noexcept {if(running.exchange(false)){queue.close();event_queue.close();} if(worker.joinable())worker.join();if(event_worker.joinable())event_worker.join();stop_network();}
  void publish(PreviewPacket p) noexcept {if(!running)return; try{std::scoped_lock l(lock);if(config.shared_memory.enabled){const auto result=queue.send(p);if(result==SendResult::ReplacedOldest||result==SendResult::DroppedNewest)++dropped;if(result!=SendResult::Closed)++submitted;}if(mjpeg)mjpeg->publish(p);if(h264)h264->publish(p);if(server){const auto result=event_queue.send(std::move(p));if(result==SendResult::ReplacedOldest||result==SendResult::DroppedNewest)++event_dropped;}}catch(...){++dropped;}}
  OutputCommandResult configure(SharedMemoryOutputConfig c){std::scoped_lock l(lock);try{configure_locked(c);config.shared_memory=std::move(c);return {OutputCommandStatus::Applied,"shared-memory configuration applied"};}catch(const std::exception&e){error=e.what();return {OutputCommandStatus::Failed,error};}}
  OutputCommandResult configure_preview(PreviewConfig c){try {if(!output::is_loopback_address(c.http.bind_address))throw std::invalid_argument("preview server must bind to a loopback address");if (c.h264.enabled && (!c.h264.bitrate || !c.h264.max_fps || !c.h264.max_width || !c.h264.queue_capacity)) throw std::invalid_argument("invalid H.264 preview configuration");if(running)stop_network();{std::scoped_lock l(lock);config=std::move(c);}if(running)start_network();return {OutputCommandStatus::Applied,"preview configuration applied"};}catch(const std::exception&e){std::scoped_lock l(lock);error=e.what();return {OutputCommandStatus::Failed,error};}}
  void set_status(std::function<std::string()> provider){std::scoped_lock l(lock);status_provider=std::move(provider);}
  PreviewTransportHealth health()const{std::scoped_lock l(lock);auto result=PreviewTransportHealth{config.shared_memory.enabled || config.mjpeg.enabled || config.http.enabled || config.h264.enabled,0,0,0,0,published.load(),dropped.load(),error};if(mjpeg){const auto video=mjpeg->health();result.published_packets+=video.published_packets;result.dropped_packets+=video.dropped_packets;if(!video.last_error.empty())result.last_error=video.last_error;}if(h264){const auto video=h264->health();result.published_packets+=video.published_packets;result.dropped_packets+=video.dropped_packets;if(!video.last_error.empty())result.last_error=video.last_error;result.codec=video.codec;}if(server){auto web=server->health();result.connected_clients=web.connected_clients;result.event_clients=web.event_clients;result.h264_clients=web.h264_clients;result.mjpeg_clients=web.mjpeg_clients;if(!web.last_error.empty())result.last_error=web.last_error;}return result;}
 private:
  void start_network(){std::scoped_lock l(lock);if(!config.http.enabled)return;server=std::make_unique<output::PreviewHttpServer>(config.http.bind_address,config.http.port,[this]{return status_provider?status_provider():std::string("{}");});server->start();if(config.h264.enabled){h264=std::make_unique<output::H264Transport>(config.h264,*server,[this](CameraId camera,std::uint64_t sequence,std::chrono::steady_clock::time_point at){record_video_publish(camera,sequence,at);});h264->start();}if(config.mjpeg.enabled){mjpeg=std::make_unique<MjpegTransport>(config.mjpeg,*server);mjpeg->start();}}
  void stop_network() noexcept {std::unique_ptr<MjpegTransport> old_mjpeg;std::unique_ptr<output::H264Transport> old_h264;std::unique_ptr<output::PreviewHttpServer> old_server;{std::scoped_lock l(lock);old_mjpeg=std::move(mjpeg);old_h264=std::move(h264);old_server=std::move(server);}if(old_mjpeg)old_mjpeg->stop();if(old_h264)old_h264->stop();if(old_server)old_server->stop();}
  void configure_locked(const SharedMemoryOutputConfig& c){
#ifdef _WIN32
std::unique_ptr<Mapping> legacy,versioned; if(c.enabled){versioned=std::make_unique<Mapping>(c.destination+"_v2",c.capacity_bytes);if(c.legacy_v1)legacy=std::make_unique<Mapping>(c.destination,c.capacity_bytes);} v1map=std::move(legacy);v2map=std::move(versioned);
#else
if(c.enabled) throw std::runtime_error("shared-memory output is currently implemented for Windows only");
#endif
}
  struct PublishPair { std::optional<std::chrono::steady_clock::time_point> pose, video; };
  void record_pose_publish(CameraId camera, std::uint64_t sequence, std::chrono::steady_clock::time_point at) { record_publish(camera,sequence,at,true); }
  void record_video_publish(CameraId camera, std::uint64_t sequence, std::chrono::steady_clock::time_point at) { record_publish(camera,sequence,at,false); }
  void record_publish(CameraId camera, std::uint64_t sequence, std::chrono::steady_clock::time_point at, bool is_pose) {
      std::scoped_lock l(timing_lock); auto& pair=publish_pairs[{camera,sequence}]; if(is_pose)pair.pose=at;else pair.video=at;
      if(pair.pose&&pair.video){const auto signed_ms=std::chrono::duration<double,std::milli>(*pair.pose-*pair.video).count();if(metrics_enabled){pose_video_skew_ms.observe(std::abs(signed_ms));last_pose_video_skew_ms.set(signed_ms);}publish_pairs.erase({camera,sequence});}
      while(publish_pairs.size()>512)publish_pairs.erase(publish_pairs.begin());
  }
  PreviewConfig config; Channel<PreviewPacket> queue,event_queue; std::atomic_bool running{false}; std::thread worker,event_worker; mutable std::mutex lock; std::atomic_size_t submitted{0},published{0},dropped{0},event_dropped{0};std::string error;std::function<std::string()> status_provider;std::unique_ptr<output::PreviewHttpServer> server;std::unique_ptr<MjpegTransport> mjpeg;std::unique_ptr<output::H264Transport> h264;
  bool metrics_enabled{};std::mutex timing_lock;std::map<std::pair<CameraId,std::uint64_t>,PublishPair> publish_pairs;infrastructure::metrics::Histogram pose_video_skew_ms;infrastructure::metrics::Gauge last_pose_video_skew_ms;
#ifdef _WIN32
std::unique_ptr<Mapping> v1map,v2map;
#endif
 };
PreviewSink::PreviewSink(PreviewConfig c,infrastructure::metrics::MetricRegistry* metrics):impl_(std::make_unique<Impl>(std::move(c),metrics)){} PreviewSink::~PreviewSink()=default; void PreviewSink::start(){impl_->start();} void PreviewSink::publish(PreviewPacket p) noexcept{impl_->publish(std::move(p));} void PreviewSink::stop() noexcept{impl_->stop();} OutputCommandResult PreviewSink::configure(PreviewConfig c){return impl_->configure_preview(std::move(c));} void PreviewSink::set_status_provider(std::function<std::string()> p){impl_->set_status(std::move(p));} OutputCommandResult PreviewSink::configure_shared_memory(SharedMemoryOutputConfig c){return impl_->configure(std::move(c));} PreviewTransportHealth PreviewSink::shared_memory_health()const{return impl_->health();}
} // namespace iris
