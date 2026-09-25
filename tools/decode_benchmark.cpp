#include "stages/capture/media_foundation/MediaFoundationSource.hpp"
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <wincodec.h>
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
using Bytes = std::vector<unsigned char>;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
const char* nvjpeg_status_name(nvjpegStatus_t status) {
    switch (status) {
    case NVJPEG_STATUS_SUCCESS: return "SUCCESS";
    case NVJPEG_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case NVJPEG_STATUS_INVALID_PARAMETER: return "INVALID_PARAMETER";
    case NVJPEG_STATUS_BAD_JPEG: return "BAD_JPEG";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "JPEG_NOT_SUPPORTED";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "ALLOCATOR_FAILURE";
    case NVJPEG_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case NVJPEG_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case NVJPEG_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "IMPLEMENTATION_NOT_SUPPORTED";
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM: return "INCOMPLETE_BITSTREAM";
    default: return "UNKNOWN";
    }
}
void nj(nvjpegStatus_t e, const char* operation = nullptr) {
    if (e != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation ? operation : "nvJPEG") + ": " +
                                 nvjpeg_status_name(e) + " (" + std::to_string(e) + ")");
    }
}
double ms(Clock::time_point begin) { return std::chrono::duration<double, std::milli>(Clock::now()-begin).count(); }

void hr(HRESULT e) { if(FAILED(e)) throw std::runtime_error("WIC HRESULT " + std::to_string(e)); }
struct Com {
    Com() { hr(CoInitializeEx(nullptr,COINIT_MULTITHREADED)); }
    ~Com() { CoUninitialize(); }
};
struct CpuJpeg {
    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    CpuJpeg() { hr(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory))); }
    Bytes decode(const Bytes& bytes, int& width, int& height, unsigned char* destination=nullptr) {
        Microsoft::WRL::ComPtr<IWICStream> stream;
        hr(factory->CreateStream(&stream));
        hr(stream->InitializeFromMemory(const_cast<BYTE*>(bytes.data()),static_cast<DWORD>(bytes.size())));
        Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
        hr(factory->CreateDecoderFromStream(stream.Get(),nullptr,WICDecodeMetadataCacheOnLoad,&decoder));
        Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
        hr(decoder->GetFrame(0,&frame));
        UINT w{},h{}; hr(frame->GetSize(&w,&h));
        if(!w || !h || w>16384 || h>16384) throw std::runtime_error("unsupported JPEG dimensions");
        if(destination && (width!=static_cast<int>(w) || height!=static_cast<int>(h))) throw std::runtime_error("JPEG dimensions changed");
        width=w; height=h;
        Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
        hr(factory->CreateFormatConverter(&converter));
        hr(converter->Initialize(frame.Get(),GUID_WICPixelFormat24bppBGR,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom));
        Bytes result; if(!destination) { result.resize(size_t(w)*h*3); destination=result.data(); }
        hr(converter->CopyPixels(nullptr,w*3,w*h*3,destination));
        return result;
    }
    Bytes synthetic(int width,int height,int n) {
        Bytes pixels(size_t(width)*height*3);
        for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
            auto i=(size_t(y)*width+x)*3;
            pixels[i]=(x+n*7)%256; pixels[i+1]=(y+n*11)%256; pixels[i+2]=(x+y+n*13)%256;
        }
        Microsoft::WRL::ComPtr<IStream> stream;
        hr(CreateStreamOnHGlobal(nullptr,TRUE,&stream));
        Microsoft::WRL::ComPtr<IWICBitmapEncoder> encoder;
        hr(factory->CreateEncoder(GUID_ContainerFormatJpeg,nullptr,&encoder));
        hr(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache));
        Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> frame;
        hr(encoder->CreateNewFrame(&frame,nullptr)); hr(frame->Initialize(nullptr));
        hr(frame->SetSize(width,height));
        auto format=GUID_WICPixelFormat24bppBGR; hr(frame->SetPixelFormat(&format));
        if(format!=GUID_WICPixelFormat24bppBGR) throw std::runtime_error("unexpected encoder pixel format");
        hr(frame->WritePixels(height,width*3,static_cast<UINT>(pixels.size()),pixels.data()));
        hr(frame->Commit()); hr(encoder->Commit());
        STATSTG stat{}; hr(stream->Stat(&stat,STATFLAG_NONAME));
        Bytes result(static_cast<size_t>(stat.cbSize.QuadPart));
        LARGE_INTEGER zero{}; hr(stream->Seek(zero,STREAM_SEEK_SET,nullptr));
        ULONG read{}; hr(stream->Read(result.data(),static_cast<ULONG>(result.size()),&read));
        if(read!=result.size()) throw std::runtime_error("short encoded JPEG read");
        return result;
    }
};

// Construct first, initialize second: partial initialization also gets RAII cleanup.
struct Decoder {
    cudaStream_t stream{};
    nvjpegHandle_t handle{};
    nvjpegJpegState_t state{};
    nvjpegJpegDecoder_t decoder{};
    nvjpegJpegStream_t jpeg{};
    nvjpegDecodeParams_t params{};
    nvjpegBufferPinned_t pinned{};
    nvjpegBufferDevice_t scratch{};
    unsigned char *output{}, *input{}, *cpu{};
    size_t pitch{};
    int width{}, height{};
    std::string mode;
    std::unique_ptr<CpuJpeg> cpu_decoder;
    ~Decoder() {
        if (stream) cudaStreamSynchronize(stream);
        if (state) nvjpegJpegStateDestroy(state);
        if (pinned) nvjpegBufferPinnedDestroy(pinned);
        if (scratch) nvjpegBufferDeviceDestroy(scratch);
        if (params) nvjpegDecodeParamsDestroy(params);
        if (jpeg) nvjpegJpegStreamDestroy(jpeg);
        if (decoder) nvjpegDecoderDestroy(decoder);
        if (handle) nvjpegDestroy(handle);
        if (output) cudaFree(output);
        if (input) cudaFreeHost(input);
        if (cpu) cudaFreeHost(cpu);
        if (stream) cudaStreamDestroy(stream);
    }
    void init(std::string name, int w, int h, size_t max_bytes) {
        mode=name; width=w; height=h;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        ck(cudaMallocPitch(reinterpret_cast<void**>(&output),&pitch,w*3,h));
        ck(cudaMallocHost(reinterpret_cast<void**>(&input),max_bytes));
        ck(cudaMallocHost(reinterpret_cast<void**>(&cpu),size_t(w)*h*3));
        if (mode=="cpu-upload") return;
        if (mode=="hardware") {
            nj(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE,nullptr,nullptr,0,&handle),
               "nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE)");
            unsigned engines{},cores{};
            nj(nvjpegGetHardwareDecoderInfo(handle,&engines,&cores),
               "nvjpegGetHardwareDecoderInfo");
            if (!engines || !cores) throw std::runtime_error("no hardware JPEG engines");
            nj(nvjpegJpegStateCreate(handle,&state));
            nj(nvjpegDecodeBatchedInitialize(handle,state,1,1,NVJPEG_OUTPUT_BGRI));
        } else {
            nj(nvjpegCreateSimple(&handle));
            if (mode=="gpu-hybrid") {
                nj(nvjpegDecoderCreate(handle,NVJPEG_BACKEND_GPU_HYBRID,&decoder));
                nj(nvjpegDecoderStateCreate(handle,decoder,&state));
                nj(nvjpegJpegStreamCreate(handle,&jpeg));
                nj(nvjpegDecodeParamsCreate(handle,&params));
                nj(nvjpegDecodeParamsSetOutputFormat(params,NVJPEG_OUTPUT_BGRI));
                nj(nvjpegBufferPinnedCreate(handle,nullptr,&pinned));
                nj(nvjpegBufferDeviceCreate(handle,nullptr,&scratch));
                nj(nvjpegStateAttachPinnedBuffer(state,pinned));
                nj(nvjpegStateAttachDeviceBuffer(state,scratch));
            } else nj(nvjpegJpegStateCreate(handle,&state));
        }
    }
    void run(const Bytes& bytes) {
        nvjpegImage_t image{}; image.channel[0]=output; image.pitch[0]=static_cast<unsigned>(pitch);
        const unsigned char* data=bytes.data();
        if(mode=="simple-pinned") { std::memcpy(input,data,bytes.size()); data=input; }
        if(mode=="cpu-upload") {
            if(!cpu_decoder) cpu_decoder=std::make_unique<CpuJpeg>();
            cpu_decoder->decode(bytes,width,height,cpu);
            ck(cudaMemcpy2DAsync(output,pitch,cpu,width*3,width*3,height,cudaMemcpyHostToDevice,stream));
            ck(cudaStreamSynchronize(stream)); // decoded remains alive until upload completes
            return;
        }
        if(mode=="gpu-hybrid") {
            nj(nvjpegJpegStreamParse(handle,data,bytes.size(),0,0,jpeg));
            int unsupported{}; nj(nvjpegDecoderJpegSupported(decoder,jpeg,params,&unsupported));
            if(unsupported) throw std::runtime_error("GPU hybrid does not support this JPEG");
            nj(nvjpegDecodeJpegHost(handle,decoder,state,params,jpeg));
            nj(nvjpegDecodeJpegTransferToDevice(handle,decoder,state,jpeg,stream));
            nj(nvjpegDecodeJpegDevice(handle,decoder,state,&image,stream));
        } else if(mode=="hardware") {
            size_t length=bytes.size();
            nj(nvjpegDecodeBatched(handle,state,&data,&length,&image,stream));
        } else {
            int components{}, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
            nvjpegChromaSubsampling_t sub;
            nj(nvjpegGetImageInfo(handle,data,bytes.size(),&components,&sub,widths,heights));
            nj(nvjpegDecode(handle,state,data,bytes.size(),NVJPEG_OUTPUT_BGRI,&image,stream));
        }
        ck(cudaStreamSynchronize(stream));
    }
};

void report_hardware_decoder_info() {
    nvjpegHandle_t handle{};
    const auto create_status = nvjpegCreateSimple(&handle);
    if (create_status != NVJPEG_STATUS_SUCCESS) {
        std::cout << "nvJPEG hardware decoder query: could not create query handle: "
                  << nvjpeg_status_name(create_status) << " (" << create_status << ")\n";
        return;
    }

    unsigned engines{}, cores_per_engine{};
    const auto query_status = nvjpegGetHardwareDecoderInfo(handle, &engines, &cores_per_engine);
    if (query_status == NVJPEG_STATUS_SUCCESS) {
        std::cout << "nvJPEG hardware decoder query: engines=" << engines
                  << ", cores_per_engine=" << cores_per_engine << '\n';
    } else {
        std::cout << "nvJPEG hardware decoder query: " << nvjpeg_status_name(query_status)
                  << " (" << query_status << ")\n";
    }
    nvjpegDestroy(handle);
}

int main(int argc,char** argv) try {
    Com com;
    CpuJpeg reference_decoder;
    std::filesystem::path directory;
    iris::CaptureConfig config;
    int frames=32, iterations=300, warmup=30, lanes=4, device=0;
    bool capture=false, self_test=false;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--help") {
            std::cout<<"--input DIR [--capture --device-index N --width W --height H --fps N --frames 32]\n"
                       "[--iterations 300 --warmup 30 --lanes 4 --cuda-device 0] [--self-test]\n"
                       "Capture writes a new directory of camera JPEGs; omit --capture to replay. Stop other camera apps first.\n";
            return 0;
        }
        if(a=="--capture") { capture=true; continue; }
        if(a=="--self-test") { self_test=true; continue; }
        if(++i>=argc) throw std::runtime_error("missing argument value");
        if(a=="--input") { directory=argv[i]; continue; }
        int v=std::stoi(argv[i]);
        if(v<0) throw std::runtime_error("negative argument");
        if(a=="--device-index") config.device_index=v;
        else if(a=="--width") config.extent.width=v;
        else if(a=="--height") config.extent.height=v;
        else if(a=="--fps") config.frame_rate.numerator=v;
        else if(a=="--frames") frames=v;
        else if(a=="--iterations") iterations=v;
        else if(a=="--warmup") warmup=v;
        else if(a=="--lanes") lanes=v;
        else if(a=="--cuda-device") device=v;
        else throw std::runtime_error("unknown option "+a);
    }
    if((directory.empty() && !self_test) || (self_test && capture) || frames<1 || iterations<1 || warmup<1 || lanes<1 || lanes>32 || !config.extent.width || !config.extent.height || !config.frame_rate.numerator)
        throw std::runtime_error("provide --input DIR and positive dimensions/counts (lanes <=32)");
    if(capture) {
        if(std::filesystem::exists(directory)) throw std::runtime_error("capture directory must not already exist");
        iris::capture::MediaFoundationSource source;
        source.open(config);
        std::filesystem::create_directories(directory);
        for(int i=0;i<frames;) {
            auto sample=source.read(); if(!sample) continue;
            auto path=directory/(std::to_string(i++)+".jpg");
            std::ofstream out(path,std::ios::binary);
            out.write(reinterpret_cast<const char*>(sample->bytes.data()),sample->bytes.size());
            if(!out) throw std::runtime_error("cannot save camera JPEG");
        }
        source.close();
    }
    std::vector<std::filesystem::path> paths;
    if(!self_test) for(auto& e:std::filesystem::directory_iterator(directory)) {
        auto ext=e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
        if(e.is_regular_file() && (ext==".jpg" || ext==".jpeg")) paths.push_back(e.path());
    }
    std::sort(paths.begin(),paths.end());
    std::vector<Bytes> corpus;
    int width=0,height=0; size_t maximum=0;
    if(self_test) {
        width=640; height=360;
        for(int n=0;n<8;++n) {
            Bytes b=reference_decoder.synthetic(width,height,n);
            maximum=std::max(maximum,b.size()); corpus.push_back(std::move(b));
        }
        std::cout<<"SYNTHETIC SMOKE TEST: not camera performance evidence\n";
    }
    for(auto& path:paths) {
        std::ifstream in(path,std::ios::binary);
        Bytes b((std::istreambuf_iterator<char>(in)),{});
        int w{},h{};
        try {
            reference_decoder.decode(b,w,h);
        } catch (const std::exception& e) {
            std::cerr << "Skipping " << path.filename().string()
                      << ": WIC could not decode this JPEG: " << e.what() << '\n';
            continue;
        }
        if(width && (width!=w || height!=h)) throw std::runtime_error("use a corpus with uniform dimensions");
        width=w; height=h; maximum=std::max(maximum,b.size()); corpus.push_back(std::move(b));
    }
    if(corpus.empty()) throw std::runtime_error("no JPEGs in input directory");
    ck(cudaSetDevice(device)); cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,device));
    std::cout<<"GPU: "<<prop.name<<"; corpus="<<corpus.size()<<"; "<<width<<"x"<<height
             <<"; warmup="<<warmup<<"; iterations per lane="<<iterations<<"\n";
    report_hardware_decoder_info();
    std::cout<<"mode,lanes,samples,median_ms,p95_ms,images_per_second\n";
    for(auto mode:{"simple","simple-pinned","gpu-hybrid","hardware","cpu-upload"}) {
        for(int count:std::vector<int>{1,lanes}) {
            std::vector<std::unique_ptr<Decoder>> decoders;
            try {
                for(int l=0;l<count;++l) {
                    auto d=std::make_unique<Decoder>(); d->init(mode,width,height,maximum);
                    // Compare every corpus image to a CPU reference outside the timed region.
                    for(const auto& bytes:corpus) {
                        d->run(bytes);
                        Bytes actual(size_t(width)*height*3);
                        ck(cudaMemcpy2D(actual.data(),width*3,d->output,d->pitch,width*3,height,cudaMemcpyDeviceToHost));
                        auto reference=reference_decoder.decode(bytes,width,height);
                        double error=0;
                        for(size_t i=0;i<actual.size();++i) error+=std::abs(int(actual[i])-int(reference[i]));
                        const double mae=error/actual.size();
                        if(mae>8) throw std::runtime_error("output validation failed: mean absolute byte error="+std::to_string(mae));
                    }
                    for(int n=0;n<warmup;++n) d->run(corpus[n%corpus.size()]);
                    decoders.push_back(std::move(d));
                }
            } catch(const std::exception& e) { std::cerr<<mode<<" lanes="<<count<<" UNAVAILABLE: "<<e.what()<<'\n'; break; }
            for(auto& d:decoders) d->cpu_decoder.reset();
            std::barrier gate(count+1);
            Clock::time_point measured_begin;
            std::vector<std::future<std::vector<double>>> jobs;
            for(int l=0;l<count;++l) jobs.push_back(std::async(std::launch::async,[&,l] {
                std::unique_ptr<Com> worker_com;
                std::exception_ptr setup_error;
                try {
                    ck(cudaSetDevice(device)); worker_com=std::make_unique<Com>();
                    for(int n=0;n<warmup;++n) decoders[l]->run(corpus[(n+l)%corpus.size()]);
                } catch(...) { setup_error=std::current_exception(); }
                gate.arrive_and_wait();
                gate.arrive_and_wait();
                if(setup_error) std::rethrow_exception(setup_error);
                std::vector<double> times; times.reserve(iterations);
                for(int n=0;n<iterations;++n) {
                    auto begin=Clock::now(); decoders[l]->run(corpus[(n+l)%corpus.size()]); times.push_back(ms(begin));
                }
                decoders[l]->cpu_decoder.reset();
                return times;
            }));
            gate.arrive_and_wait();
            measured_begin=Clock::now(); gate.arrive_and_wait();
            std::vector<double> times;
            for(auto& job:jobs) { auto t=job.get(); times.insert(times.end(),t.begin(),t.end()); }
            auto elapsed=ms(measured_begin); std::sort(times.begin(),times.end());
            auto percentile=[&](double q){return times[static_cast<size_t>(std::ceil(q*times.size()))-1];};
            std::cout<<mode<<','<<count<<','<<times.size()<<','<<percentile(.5)<<','<<percentile(.95)<<','<<times.size()*1000/elapsed<<std::endl;
            if(lanes==1) break;
        }
    }
    return 0;
} catch(const std::exception& e) { std::cerr<<"Benchmark failed: "<<e.what()<<'\n'; return 1; }
