#include "face_service.hpp"

#include <csignal>
#include <iostream>

namespace {

void signal_handler(int) {
    lxfu::FaceService::instance().stop();
}

}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        return lxfu::FaceService::instance().run();
    } catch (const std::exception& ex) {
        std::cerr << "FaceService fatal error: " << ex.what() << std::endl;
        return 1;
    }
}

