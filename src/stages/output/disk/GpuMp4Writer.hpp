#pragma once

#include "iris/pipeline/Frame.hpp"
#include "iris/stages/output/OutputConfig.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace iris::output::disk {

class GpuMp4Writer {
  public:
    GpuMp4Writer(const DiskOutputConfig&, const Frame& first_frame);
    ~GpuMp4Writer();

    GpuMp4Writer(const GpuMp4Writer&) = delete;
    GpuMp4Writer& operator=(const GpuMp4Writer&) = delete;

    std::uint64_t write(const Frame&);
    void finalize();
    [[nodiscard]] std::uint64_t encoded_frames() const noexcept;
    [[nodiscard]] std::uint64_t bytes_written() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris::output::disk
