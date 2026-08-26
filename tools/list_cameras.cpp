#include "stages/capture/media_foundation/MediaFoundationSource.hpp"
#include <iostream>
int main() {
    try {
        iris::capture::MediaFoundationSource session;
        auto cameras = session.enumerate();
        for (std::size_t i = 0; i < cameras.size(); ++i) {
            std::wcout << i << L": " << cameras[i].name << L"\n   " << cameras[i].symbolic_link
                       << L'\n';
        }
        return cameras.empty() ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
