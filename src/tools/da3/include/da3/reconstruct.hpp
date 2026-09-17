#pragma once

#include "da3/types.hpp"

namespace da3 {

ReconstructionResult ReconstructPointCloud(
    const Mv4Outputs& outputs,
    const std::vector<ProcessedView>& processed_views,
    const ReconstructionConfig& config = {}
);

ReconstructionResult ReconstructPointCloudBase(
    const MvBaseOutputs& outputs,
    const std::vector<ProcessedView>& processed_views,
    const ReconstructionConfig& config = {}
);

}  // namespace da3
