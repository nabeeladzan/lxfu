#include "face_engine.hpp"
#include "face_detector.hpp"
#include "config.hpp"
#include "lmdb_store.hpp"

#include <security/pam_appl.h>
#include <security/pam_ext.h>
#include <security/pam_modules.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <syslog.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ModuleOptions {
    std::optional<std::string> source_path;
    std::optional<std::string> device_path;
    double threshold = 0.90;
    bool debug = false;
};

ModuleOptions parse_options(pam_handle_t* pamh, int argc, const char** argv) {
    ModuleOptions opts;
    for (int i = 0; i < argc; ++i) {
        std::string arg{argv[i] ? argv[i] : ""};
        if (arg == "debug") {
            opts.debug = true;
            continue;
        }
        auto pos = arg.find('=');
        if (pos == std::string::npos || pos == arg.size() - 1) {
            pam_syslog(pamh, LOG_WARNING, "pam_lxfu: ignoring malformed option '%s'", arg.c_str());
            continue;
        }
        std::string key = arg.substr(0, pos);
        std::string value = arg.substr(pos + 1);

        if (key == "source") {
            opts.source_path = value;
        } else if (key == "device") {
            opts.device_path = value;
        } else if (key == "threshold") {
            try {
                opts.threshold = std::stod(value);
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid threshold '%s'", value.c_str());
            }
        } else {
            pam_syslog(pamh, LOG_WARNING, "pam_lxfu: unknown option '%s'", key.c_str());
        }
    }

    if (opts.threshold <= 0.0 || opts.threshold > 1.0) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: threshold %.3f out of range, resetting to 0.90", opts.threshold);
        opts.threshold = 0.90;
    }

    return opts;
}

cv::Mat capture_frame(const std::string& device, pam_handle_t* pamh, bool debug) {
    cv::VideoCapture cap;
    int device_id = -1;
    if (device.rfind("/dev/video", 0) == 0) {
        try {
            device_id = std::stoi(device.substr(10));
        } catch (const std::exception&) {
            device_id = -1;
        }
    }

    if (device_id >= 0) {
        cap.open(device_id);
    } else {
        cap.open(device);
    }

    if (!cap.isOpened()) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: failed to open capture device '%s'", device.c_str());
        throw std::runtime_error("capture device open failure");
    }

    cv::Mat frame;
    cap >> frame;
    cap.release();

    if (frame.empty()) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: empty frame captured from '%s'", device.c_str());
        throw std::runtime_error("empty frame");
    }

    if (debug) {
        pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: captured frame %dx%d", frame.cols, frame.rows);
    }

    return frame.clone();
}

cv::Mat load_source(const ModuleOptions& opts, pam_handle_t* pamh, const Config& config) {
    if (opts.source_path) {
        cv::Mat image = cv::imread(*opts.source_path);
        if (image.empty()) {
            pam_syslog(pamh, LOG_ERR, "pam_lxfu: failed to load image '%s'", opts.source_path->c_str());
            throw std::runtime_error("image load failure");
        }
        return image;
    }

    std::string device = opts.device_path.value_or(config.get("default_device", "/dev/video0"));
    return capture_frame(device, pamh, opts.debug);
}

FaceDetector& shared_face_detector() {
    static FaceDetector detector(/*verbose=*/false);
    return detector;
}

FaceEngine& shared_face_engine(const std::string& model_path) {
    static std::unique_ptr<FaceEngine> engine;
    static std::string cached_model_path;

    if (!engine || cached_model_path != model_path) {
        engine = std::make_unique<FaceEngine>(model_path, /*verbose=*/false);
        cached_model_path = model_path;
    }
    return *engine;
}

int match_user_with_face(pam_handle_t* pamh, const std::string& username, const ModuleOptions& opts) {
    Config config = load_config(false);
    cv::Mat frame = load_source(opts, pamh, config);

    FaceDetector& detector = shared_face_detector();
    if (!detector.is_initialized()) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: face detector not available; using full frame");
    }

    cv::Mat face_image = detector.crop_to_face(frame);

    FaceEngine& engine = shared_face_engine(config.get("model_path"));

    std::vector<float> embedding = engine.extract_embedding(face_image);
    if (embedding.empty()) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: embedding extraction failed");
        return PAM_AUTHINFO_UNAVAIL;
    }

    std::string faiss_path = config.get_faiss_index_path();
    engine.init_index(faiss_path, static_cast<int>(embedding.size()));

    if (engine.get_index_size() == 0) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: FAISS index is empty; no faces enrolled");
        return PAM_AUTHINFO_UNAVAIL;
    }

    auto result = engine.search(embedding);
    if (result.id < 0) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: no similar face found");
        return PAM_AUTH_ERR;
    }

    if (result.similarity < static_cast<float>(opts.threshold)) {
        pam_syslog(pamh, LOG_INFO, "pam_lxfu: similarity %.2f below threshold %.2f", result.similarity, opts.threshold);
        return PAM_AUTH_ERR;
    }

    LMDBStore store(config.get_lmdb_path(), LMDBStore::Mode::ReadOnly);
    std::string stored_name = store.get_name(result.id);

    if (stored_name.empty()) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: face id %ld has no associated username", static_cast<long>(result.id));
        return PAM_AUTH_ERR;
    }

    if (stored_name != username) {
        pam_syslog(pamh, LOG_INFO, "pam_lxfu: best match '%s' does not match user '%s'", stored_name.c_str(), username.c_str());
        return PAM_AUTH_ERR;
    }

    if (opts.debug) {
        pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: user '%s' matched with similarity %.2f (id %ld)",
                   username.c_str(), result.similarity, static_cast<long>(result.id));
    }

    return PAM_SUCCESS;
}

} // namespace

extern "C" {

PAM_EXTERN int pam_sm_authenticate(pam_handle_t* pamh, int, int argc, const char** argv) {
    ModuleOptions opts = parse_options(pamh, argc, argv);

    const char* user = nullptr;
    int pam_rc = pam_get_user(pamh, &user, nullptr);
    if (pam_rc != PAM_SUCCESS || !user || *user == '\0') {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: unable to determine username");
        return PAM_USER_UNKNOWN;
    }

    try {
        return match_user_with_face(pamh, user, opts);
    } catch (const std::exception& ex) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: exception: %s", ex.what());
        return PAM_AUTHINFO_UNAVAIL;
    } catch (...) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: unknown exception during authentication");
        return PAM_AUTHINFO_UNAVAIL;
    }
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t*, int, int, const char**) {
    return PAM_SUCCESS;
}

} // extern "C"

