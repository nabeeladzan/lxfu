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
#include <ctime>

namespace {

struct ModuleOptions {
    std::optional<std::string> source_path;
    std::optional<std::string> device_path;
    std::optional<std::string> target_name;
    double threshold = 0.90;
    bool debug = false;
    bool allow_all = false;
    int retries = 1;
    double interval_seconds = 0.0;
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
        } else if (key == "name") {
            opts.target_name = value;
        } else if (key == "allow_all" || key == "all") {
            std::string lowered = value;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            opts.allow_all = (lowered == "1" || lowered == "true" || lowered == "yes");
        } else if (key == "threshold") {
            try {
                opts.threshold = std::stod(value);
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid threshold '%s'", value.c_str());
            }
        } else if (key == "retries") {
            try {
                int r = std::stoi(value);
                if (r < 1) {
                    pam_syslog(pamh, LOG_WARNING, "pam_lxfu: retries must be >=1, received %d", r);
                } else {
                    opts.retries = r;
                }
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid retries '%s'", value.c_str());
            }
        } else if (key == "interval") {
            try {
                double seconds = std::stod(value);
                if (seconds < 0.0) {
                    pam_syslog(pamh, LOG_WARNING, "pam_lxfu: interval must be >=0, received %f", seconds);
                } else {
                    opts.interval_seconds = seconds;
                }
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid interval '%s'", value.c_str());
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

    int64_t index_size = static_cast<int64_t>(engine.get_index_size());
    int k = static_cast<int>(std::min<int64_t>(index_size, 10));
    auto candidates = engine.search_many(embedding, k);
    if (candidates.empty()) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: no similar face found");
        return PAM_AUTH_ERR;
    }

    std::string desired_name = opts.allow_all ? std::string() : opts.target_name.value_or(username);

    LMDBStore store(config.get_lmdb_path(), LMDBStore::Mode::ReadOnly);

    std::optional<FaceEngine::SearchResult> chosen;
    std::string matched_name;

    for (const auto& candidate : candidates) {
        std::string stored_name = store.get_name(candidate.id);
        if (stored_name.empty()) {
            continue;
        }
        if (!opts.allow_all && stored_name != desired_name) {
            continue;
        }
        chosen = candidate;
        matched_name = stored_name;
        break;
    }

    if (!chosen) {
        if (opts.allow_all) {
            pam_syslog(pamh, LOG_INFO, "pam_lxfu: best matches do not satisfy allow_all criteria");
        } else {
            pam_syslog(pamh, LOG_INFO, "pam_lxfu: no match for requested name '%s'", desired_name.c_str());
        }
        return PAM_AUTH_ERR;
    }

    if (chosen->similarity < static_cast<float>(opts.threshold)) {
        pam_syslog(pamh, LOG_INFO, "pam_lxfu: similarity %.2f below threshold %.2f", chosen->similarity, opts.threshold);
        return PAM_AUTH_ERR;
    }

    if (!opts.allow_all && matched_name != desired_name) {
        pam_syslog(pamh, LOG_INFO, "pam_lxfu: matched name '%s' does not match requested '%s'", matched_name.c_str(), desired_name.c_str());
        return PAM_AUTH_ERR;
    }

    if (opts.debug) {
        pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: matched name '%s' with similarity %.2f (id %ld)",
                   matched_name.c_str(), chosen->similarity, static_cast<long>(chosen->id));
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

    int attempts = std::max(1, opts.retries);
    for (int attempt = 1; attempt <= attempts; ++attempt) {
        try {
            int result = match_user_with_face(pamh, user, opts);
            if (result == PAM_SUCCESS) {
                return PAM_SUCCESS;
            }

            if (result == PAM_AUTHINFO_UNAVAIL) {
                return result;
            }

            if (attempt < attempts) {
                pam_syslog(pamh, LOG_INFO, "pam_lxfu: attempt %d/%d failed for user '%s'", attempt, attempts, user);
                pam_prompt(pamh, PAM_TEXT_INFO, nullptr, "Face not recognized, please try again.");
                if (opts.interval_seconds > 0.0) {
                    const int usec = static_cast<int>(opts.interval_seconds * 1'000'000.0);
                    struct timespec ts {
                        usec / 1'000'000,
                        static_cast<long>((usec % 1'000'000) * 1000)
                    };
                    nanosleep(&ts, nullptr);
                }
            }

        } catch (const std::exception& ex) {
            pam_syslog(pamh, LOG_ERR, "pam_lxfu: exception: %s", ex.what());
            return PAM_AUTHINFO_UNAVAIL;
        } catch (...) {
            pam_syslog(pamh, LOG_ERR, "pam_lxfu: unknown exception during authentication");
            return PAM_AUTHINFO_UNAVAIL;
        }
    }
    return PAM_AUTH_ERR;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t*, int, int, const char**) {
    return PAM_SUCCESS;
}

} // extern "C"
