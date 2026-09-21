#include "iris/stages/output/H264PreviewProtocol.hpp"

#include <cassert>
#include <cstdint>

int main() {
    iris::output::H264PreviewAccessUnit input;
    input.flags = iris::output::h264_flag_keyframe;
    input.camera = 7;
    input.sequence = 42;
    input.timestamp_us = 123456;
    input.payload = {0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1f};
    const auto encoded = iris::output::encode_h264_preview_access_unit(input);
    iris::output::H264PreviewAccessUnit output;
    assert(iris::output::decode_h264_preview_access_unit(encoded, output));
    assert(output.flags == input.flags && output.camera == input.camera && output.sequence == input.sequence);
    assert(output.timestamp_us == input.timestamp_us && output.payload == input.payload);
    auto invalid = encoded;
    invalid[0] = 'X';
    assert(!iris::output::decode_h264_preview_access_unit(invalid, output));
    invalid = encoded;
    invalid.pop_back();
    assert(!iris::output::decode_h264_preview_access_unit(invalid, output));
}
