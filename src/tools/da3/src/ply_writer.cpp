#include "da3/ply_writer.hpp"

#include <fstream>
#include <stdexcept>

namespace da3 {

void WriteBinaryPly(const fs::path& output_path, const std::vector<PointVertex>& vertices) {
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream stream(output_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open PLY output for writing: " + output_path.string());
    }

    stream << "ply\n";
    stream << "format binary_little_endian 1.0\n";
    stream << "element vertex " << vertices.size() << "\n";
    stream << "property float x\n";
    stream << "property float y\n";
    stream << "property float z\n";
    stream << "property uchar red\n";
    stream << "property uchar green\n";
    stream << "property uchar blue\n";
    stream << "end_header\n";

    for (const PointVertex& vertex : vertices) {
        const float x = vertex.position.x();
        const float y = vertex.position.y();
        const float z = vertex.position.z();
        stream.write(reinterpret_cast<const char*>(&x), sizeof(float));
        stream.write(reinterpret_cast<const char*>(&y), sizeof(float));
        stream.write(reinterpret_cast<const char*>(&z), sizeof(float));
        stream.write(
            reinterpret_cast<const char*>(vertex.color.data()),
            static_cast<std::streamsize>(vertex.color.size())
        );
    }

    if (!stream) {
        throw std::runtime_error("Failed while writing PLY output: " + output_path.string());
    }
}

}  // namespace da3
