#pragma once

#include "da3/types.hpp"

namespace da3 {

void WriteBinaryPly(const fs::path& output_path, const std::vector<PointVertex>& vertices);

}  // namespace da3
