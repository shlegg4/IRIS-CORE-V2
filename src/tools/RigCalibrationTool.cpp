#include "iris/tools/RigCalibrationTool.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "da3/preprocess.hpp"
#include "da3/ply_writer.hpp"
#include "da3/reconstruct.hpp"
#include "da3/tensor_rt.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace iris {
namespace {
Eigen::Matrix4f load_w2c(const da3::TensorOutput& extrinsics, std::size_t view) {
    Eigen::Matrix4f matrix;
    const std::size_t base = view * 16;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            matrix(row, column) = extrinsics.values[base + static_cast<std::size_t>(row * 4 + column)];
    return matrix;
}

nlohmann::json matrix_to_json(const Eigen::Matrix4f& matrix) {
    nlohmann::json rows = nlohmann::json::array();
    for (int row = 0; row < 4; ++row) {
        nlohmann::json values = nlohmann::json::array();
        for (int column = 0; column < 4; ++column) values.push_back(matrix(row, column));
        rows.push_back(std::move(values));
    }
    return rows;
}

nlohmann::json vector_to_json(const Eigen::Vector3f& vector) {
    return nlohmann::json::array({vector.x(), vector.y(), vector.z()});
}

void validate_camera(const RigCameraCalibration& camera) {
    for (const auto value : camera.intrinsics)
        if (!std::isfinite(value)) throw std::runtime_error("DA3 returned non-finite intrinsics");
    for (const auto value : camera.R_w2c)
        if (!std::isfinite(value)) throw std::runtime_error("DA3 returned a non-finite rotation");
    for (const auto value : camera.t_w2c)
        if (!std::isfinite(value)) throw std::runtime_error("DA3 returned a non-finite translation");
    if (camera.intrinsics[0] <= 0.0F || camera.intrinsics[4] <= 0.0F)
        throw std::runtime_error("DA3 returned invalid focal lengths");

    constexpr float tolerance = 0.05F;
    for (int row = 0; row < 3; ++row) {
        for (int other = 0; other < 3; ++other) {
            float dot = 0.0F;
            for (int column = 0; column < 3; ++column)
                dot += camera.R_w2c[row * 3 + column] * camera.R_w2c[other * 3 + column];
            const float expected = row == other ? 1.0F : 0.0F;
            if (std::abs(dot - expected) > tolerance)
                throw std::runtime_error("DA3 returned a non-orthonormal rotation");
        }
    }
    const auto& r = camera.R_w2c;
    const float determinant = r[0] * (r[4] * r[8] - r[5] * r[7]) -
                              r[1] * (r[3] * r[8] - r[5] * r[6]) +
                              r[2] * (r[3] * r[7] - r[4] * r[6]);
    if (std::abs(determinant - 1.0F) > tolerance)
        throw std::runtime_error("DA3 returned an improper rotation");
}
} // namespace

RigCalibrationTool::RigCalibrationTool(std::shared_ptr<CalibrationStore> store) : store_(std::move(store)) {}
RigCalibrationTool::~RigCalibrationTool() { cancel(); }
void RigCalibrationTool::observe(const Packet& packet) {
    std::scoped_lock lock(mutex_);
    latest_ = packet;
    changed_.notify_all();
}
bool RigCalibrationTool::start(std::filesystem::path engine, std::filesystem::path output) {
    if (worker_.joinable()) {
        { std::scoped_lock lock(mutex_); if (status_.state=="waiting-for-frames" || status_.state=="preparing" || status_.state=="inference") return false; }
        worker_.join();
    }
    std::scoped_lock lock(mutex_);
    status_ = {"waiting-for-frames", {}, 0};
    worker_ = std::jthread([this, engine=std::move(engine), output=std::move(output)](std::stop_token stop) mutable { run(stop, std::move(engine), std::move(output)); });
    return true;
}
void RigCalibrationTool::cancel() { if (worker_.joinable()) { worker_.request_stop(); changed_.notify_all(); worker_.join(); } }
RigCalibrationTool::Status RigCalibrationTool::status() const { std::scoped_lock lock(mutex_); return status_; }

void RigCalibrationTool::run(std::stop_token stop, std::filesystem::path engine, std::filesystem::path output) {
    try {
        Packet packet;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, stop, [this] { return latest_.has_value(); });
            if (stop.stop_requested()) { status_.state="cancelled"; return; }
            packet = std::move(*latest_);
            latest_.reset();
            status_ = {"preparing", {}, packet.sequence};
        }
        if (packet.frames.size() < 2 || packet.frames.size() > 10) throw std::runtime_error("DA3 base calibration requires 2-10 synchronized cameras");
        std::set<CameraId> ids;
        std::vector<cv::Mat> images; std::vector<std::filesystem::path> labels;
        for (const auto& frame : packet.frames) {
            if (!ids.insert(frame.camera).second || frame.format != PixelFormat::Bgr8 || !frame.buffer.data) throw std::runtime_error("DA3 calibration requires unique valid BGR frames");
            if (frame.ready) frame.ready->synchronize();
            cv::Mat image(static_cast<int>(frame.extent.height), static_cast<int>(frame.extent.width), CV_8UC3);
            infrastructure::gpu::check_cuda(cudaMemcpy2D(image.data, image.step, frame.buffer.data, frame.buffer.stride_bytes, frame.extent.width*3, frame.extent.height, cudaMemcpyDeviceToHost), "copy DA3 calibration frame");
            images.push_back(std::move(image)); labels.emplace_back("camera-"+std::to_string(frame.camera));
        }
        { std::scoped_lock lock(mutex_); status_.state="inference"; }
        da3::PreprocessConfig preprocess; preprocess.expected_height=da3::kBaseInputHeight; preprocess.expected_width=da3::kBaseInputWidth;
        auto batch = da3::PrepareInputBatch(images, labels, preprocess);
        da3::TensorRtEngine trt(engine, true);
        const auto outputs = trt.InferBase(static_cast<int>(images.size()), batch.images);
        if (outputs.extrinsics.values.size() != images.size()*16 || outputs.intrinsics.values.size() != images.size()*9) {
            const auto format_dims = [](const std::vector<std::int64_t>& dims) {
                std::ostringstream text;
                text << '[';
                for (std::size_t i = 0; i < dims.size(); ++i) {
                    if (i != 0) text << ',';
                    text << dims[i];
                }
                text << ']';
                return text.str();
            };
            std::ostringstream message;
            message << "DA3 camera tensor shape mismatch for " << images.size() << " views: expected extrinsics [1," << images.size()
                    << ",4,4] (" << images.size() * 16 << " values) and intrinsics [1," << images.size() << ",3,3] ("
                    << images.size() * 9 << " values), got extrinsics " << format_dims(outputs.extrinsics.dims) << " ("
                    << outputs.extrinsics.values.size() << " values) and intrinsics " << format_dims(outputs.intrinsics.dims)
                    << " (" << outputs.intrinsics.values.size() << " values). Check that the DA3 engine was built from the current dynamic-view ONNX export.";
            throw std::runtime_error(message.str());
        }

        da3::ReconstructionConfig reconstruction_config;
        reconstruction_config.viewer_align = true;
        const da3::ReconstructionResult reconstruction =
            da3::ReconstructPointCloudBase(outputs, batch.views, reconstruction_config);

        auto calibration = std::make_shared<RigCalibration>();
        calibration->revision = store_->snapshot() ? store_->snapshot()->revision + 1 : 1;
        calibration->created_at = std::chrono::system_clock::now(); calibration->method="da3"; calibration->metric_scale=false;
        std::size_t view=0;
        for (const auto& frame : packet.frames) {
            RigCameraCalibration camera; camera.camera_id=frame.camera; camera.resolution=frame.extent;
            const auto& pv=batch.views[view]; const std::size_t kb=view*9;
            camera.intrinsics={outputs.intrinsics.values[kb]/pv.scale_x,outputs.intrinsics.values[kb+1]/pv.scale_x,(outputs.intrinsics.values[kb+2]-pv.content_x)/pv.scale_x,outputs.intrinsics.values[kb+3]/pv.scale_y,outputs.intrinsics.values[kb+4]/pv.scale_y,(outputs.intrinsics.values[kb+5]-pv.content_y)/pv.scale_y,outputs.intrinsics.values[kb+6],outputs.intrinsics.values[kb+7],outputs.intrinsics.values[kb+8]};

            // Reconstruction's alignment maps DA3's source world into the ground-aligned
            // world. Convert w2c so it still maps points in the newly published world.
            const Eigen::Matrix4f aligned_w2c =
                load_w2c(outputs.extrinsics, view) * reconstruction.alignment.inverse();
            for(int r=0;r<3;++r){for(int c=0;c<3;++c)camera.R_w2c[r*3+c]=aligned_w2c(r,c);camera.t_w2c[r]=aligned_w2c(r,3);}
            validate_camera(camera);
            calibration->cameras.push_back(camera); ++view;
        }
        if(!output.parent_path().empty())std::filesystem::create_directories(output.parent_path()); const auto temporary=output.string()+".tmp";
        const auto ply_path = output.parent_path() / (output.stem().string() + ".ply");
        da3::WriteBinaryPly(ply_path, reconstruction.vertices);

        nlohmann::json json={{"version",1},{"revision",calibration->revision},{"method","da3"},{"scale","relative"},{"cameras",nlohmann::json::array()}};
        for(const auto& c:calibration->cameras)json["cameras"].push_back({{"camera_id",c.camera_id},{"image_width",c.resolution.width},{"image_height",c.resolution.height},{"intrinsics",c.intrinsics},{"distortion",c.distortion},{"R_w2c",c.R_w2c},{"t_w2c",c.t_w2c}});
        const auto& ground = reconstruction.ground_alignment;
        json["reconstruction"] = {
            {"viewer_align", reconstruction_config.viewer_align},
            {"alignment", matrix_to_json(reconstruction.alignment)},
            {"point_count", reconstruction.vertices.size()},
            {"confidence_threshold", reconstruction.applied_conf_threshold},
            {"ply", ply_path.string()},
            {"ground_alignment", {
                {"plane_found", ground.plane_found},
                {"sampled_point_count", ground.sampled_point_count},
                {"inlier_count", ground.inlier_count},
                {"inlier_ratio", ground.inlier_ratio},
                {"plane_normal_world", vector_to_json(ground.plane_normal)},
                {"plane_point_world", vector_to_json(ground.plane_point)},
                {"camera_centroid_world", vector_to_json(ground.camera_centroid)},
                {"origin_world", vector_to_json(ground.origin)}
            }}
        };
        { std::ofstream file(temporary); file<<json.dump(2); if(!file)throw std::runtime_error("could not write rig calibration"); }
#ifdef _WIN32
        if (!MoveFileExW(std::filesystem::path(temporary).c_str(), output.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) throw std::runtime_error("could not atomically replace rig calibration");
#else
        std::filesystem::rename(temporary,output);
#endif
        store_->publish(calibration);
        { std::scoped_lock lock(mutex_); status_={"complete",output.string(),packet.sequence}; }
    } catch(const std::exception& error) { std::scoped_lock lock(mutex_); status_={"failed",error.what(),0}; }
}
}
