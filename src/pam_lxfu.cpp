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
#include <numeric>
#include <tuple>
#include <chrono>
#include <thread>

namespace {

struct ModuleOptions {
    std::optional<std::string> source_path;
    std::optional<std::string> device_path;
    std::optional<std::string> target_name;
    double threshold = 0.75;
    bool debug = false;
    bool allow_all = false;
    int retries = 1;
    double interval_seconds = 0.0;
    double warmup_delay_seconds = 0.0;
    double capture_duration_seconds = 2.0;
    double frame_interval_seconds = 0.1;
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
        } else if (key == "warmup_delay") {
            try {
                double seconds = std::stod(value);
                if (seconds < 0.0) {
                    pam_syslog(pamh, LOG_WARNING, "pam_lxfu: warmup_delay must be >=0, received %f", seconds);
                } else {
                    opts.warmup_delay_seconds = seconds;
                }
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid warmup_delay '%s'", value.c_str());
            }
        } else if (key == "capture_duration") {
            try {
                double seconds = std::stod(value);
                if (seconds < 0.0) {
                    pam_syslog(pamh, LOG_WARNING, "pam_lxfu: capture_duration must be >=0, received %f", seconds);
                } else {
                    opts.capture_duration_seconds = seconds;
                }
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid capture_duration '%s'", value.c_str());
            }
        } else if (key == "frame_interval") {
            try {
                double seconds = std::stod(value);
                if (seconds < 0.0) {
                    pam_syslog(pamh, LOG_WARNING, "pam_lxfu: frame_interval must be >=0, received %f", seconds);
                } else {
                    opts.frame_interval_seconds = seconds;
                }
            } catch (const std::exception&) {
                pam_syslog(pamh, LOG_WARNING, "pam_lxfu: invalid frame_interval '%s'", value.c_str());
            }
        } else {
            pam_syslog(pamh, LOG_WARNING, "pam_lxfu: unknown option '%s'", key.c_str());
        }
    }

    if (opts.threshold <= 0.0 || opts.threshold > 1.0) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: threshold %.3f out of range, resetting to 0.75", opts.threshold);
        opts.threshold = 0.75;
    }

    return opts;
}

bool parse_device_index(const std::string& path, int& index) {
    if (path.rfind("/dev/video", 0) != 0) {
        return false;
    }
    try {
        index = std::stoi(path.substr(10));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool open_video_capture(cv::VideoCapture& cap, const std::string& source, pam_handle_t* pamh, bool debug) {
    cap.release();

    if (cap.open(source, cv::CAP_V4L2)) {
        if (debug) {
            pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: opened device '%s' via CAP_V4L2", source.c_str());
        }
        return true;
    }
    if (cap.open(source)) {
        if (debug) {
            pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: opened device '%s' via default backend", source.c_str());
        }
        return true;
    }

    int index = -1;
    if (parse_device_index(source, index) && cap.open(index)) {
        if (debug) {
            pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: opened device '%s' via numeric index %d", source.c_str(), index);
        }
        return true;
    }

    pam_syslog(pamh, LOG_ERR, "pam_lxfu: failed to open capture device '%s'", source.c_str());
    return false;
}

void apply_camera_defaults(cv::VideoCapture& cap) {
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
}

void warm_up_camera(cv::VideoCapture& cap, double warmup_delay_seconds, pam_handle_t* pamh, bool debug) {
    cv::Mat dummy;
    const int default_warmup_frames = 12;

    if (warmup_delay_seconds <= 0.0) {
        for (int i = 0; i < default_warmup_frames; ++i) {
            if (!cap.read(dummy) || dummy.empty()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        return;
    }

    auto start = std::chrono::steady_clock::now();
    const auto warmup_duration = std::chrono::duration<double>(warmup_delay_seconds);
    int frames_read = 0;
    while (std::chrono::steady_clock::now() - start < warmup_duration) {
        if (!cap.read(dummy) || dummy.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }
        ++frames_read;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    if (debug) {
        pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: warmup captured %d frames over %.2fs", frames_read, warmup_delay_seconds);
    }
}

std::vector<cv::Mat> capture_faces_from_device(const std::string& device,
                                               const ModuleOptions& opts,
                                               pam_handle_t* pamh,
                                               FaceDetector& detector) {
    cv::VideoCapture cap;
    if (!open_video_capture(cap, device, pamh, opts.debug)) {
        throw std::runtime_error("capture device open failure");
    }

    apply_camera_defaults(cap);
    warm_up_camera(cap, opts.warmup_delay_seconds, pamh, opts.debug);

    const double capture_duration = std::max(0.0, opts.capture_duration_seconds);
    const double frame_interval = std::max(0.0, opts.frame_interval_seconds);
    const auto start_time = std::chrono::steady_clock::now();
    const int max_faces = 60;

    std::vector<cv::Mat> face_images;
    std::vector<cv::Mat> fallback_frames;

    int total_frames = 0;
    int frames_with_faces = 0;
    int consecutive_failures = 0;
    const int max_consecutive_failures = 20;

    while (true) {
        if (capture_duration > 0.0) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= capture_duration) {
                break;
            }
        } else if (total_frames > 0) {
            break;
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            ++consecutive_failures;
            if (opts.debug && (consecutive_failures == 1 || consecutive_failures % 5 == 0)) {
                pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: failed to capture frame (%d)", consecutive_failures);
            }
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            if (frame_interval > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::duration<double>(frame_interval)));
            }
            continue;
        }

        consecutive_failures = 0;
        ++total_frames;

        auto face = detector.crop_to_face(frame);
        if (face) {
            face_images.push_back(*face);
            ++frames_with_faces;
        } else if (face_images.empty()) {
            fallback_frames.push_back(frame);
        }

        if (face_images.size() >= static_cast<std::size_t>(max_faces)) {
            break;
        }

        if (frame_interval > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::duration<double>(frame_interval)));
        }
    }

    cap.release();

    if (face_images.empty() && !fallback_frames.empty()) {
        if (auto face = detector.crop_to_face(fallback_frames.back())) {
            face_images.push_back(*face);
        }
    }

    if (opts.debug) {
        pam_syslog(pamh, LOG_DEBUG,
                   "pam_lxfu: captured %d frames, %d with detected faces",
                   total_frames, frames_with_faces);
    }

    return face_images;
}

std::vector<cv::Mat> load_faces(const ModuleOptions& opts,
                                pam_handle_t* pamh,
                                const Config& config,
                                FaceDetector& detector) {
    if (opts.source_path) {
        cv::Mat image = cv::imread(*opts.source_path);
        if (image.empty()) {
            pam_syslog(pamh, LOG_ERR, "pam_lxfu: failed to load image '%s'", opts.source_path->c_str());
            throw std::runtime_error("image load failure");
        }
        auto face = detector.crop_to_face(image);
        if (!face) {
            pam_syslog(pamh, LOG_WARNING, "pam_lxfu: no face detected in source image '%s'", opts.source_path->c_str());
            return {};
        }
        return { *face };
    }

    std::string device = opts.device_path.value_or(config.get("default_device", "/dev/video0"));
    if (opts.debug) {
        pam_syslog(pamh, LOG_DEBUG,
                   "pam_lxfu: capturing from device '%s' (duration %.2fs, frame_interval %.2fs, warmup %.2fs)",
                   device.c_str(),
                   std::max(0.0, opts.capture_duration_seconds),
                   std::max(0.0, opts.frame_interval_seconds),
                   std::max(0.0, opts.warmup_delay_seconds));
    }
    return capture_faces_from_device(device, opts, pamh, detector);
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

    FaceDetector& detector = shared_face_detector();
    if (!detector.is_initialized()) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: face detector not available; using full frame");
    }

    std::vector<cv::Mat> face_images;
    try {
        face_images = load_faces(opts, pamh, config, detector);
    } catch (const std::exception& ex) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: capture error: %s", ex.what());
        return PAM_AUTHINFO_UNAVAIL;
    }

    if (face_images.empty()) {
        pam_syslog(pamh, LOG_INFO, "pam_lxfu: no valid face frames captured");
        return PAM_AUTH_ERR;
    }

    FaceEngine& engine = shared_face_engine(config.get("model_path"));

    std::vector<std::vector<float>> query_embeddings;
    query_embeddings.reserve(face_images.size());
    for (const auto& face : face_images) {
        std::vector<float> embedding = engine.extract_embedding(face);
        if (!embedding.empty()) {
            query_embeddings.push_back(std::move(embedding));
        }
    }

    if (query_embeddings.empty()) {
        pam_syslog(pamh, LOG_ERR, "pam_lxfu: embedding extraction failed for captured frames");
        return PAM_AUTHINFO_UNAVAIL;
    }

    LMDBStore store(config.get_embeddings_path(), LMDBStore::Mode::ReadOnly);
    auto entries = store.get_all_embeddings();
    if (entries.empty()) {
        pam_syslog(pamh, LOG_WARNING, "pam_lxfu: no enrolled profiles available");
        return PAM_AUTHINFO_UNAVAIL;
    }

    auto compute_best = [&](const std::vector<std::pair<std::string, LMDBStore::EmbeddingList>>& profiles,
                            const std::optional<std::string>& target)
        -> std::tuple<std::string, float, float> {
        float best_avg = -1.0f;
        float best_max = -1.0f;
        std::string best_name;

        for (const auto& [name, embeddings] : profiles) {
            if (target && name != *target) {
                continue;
            }
            if (embeddings.empty()) {
                continue;
            }

            bool dimension_mismatch = std::any_of(
                embeddings.begin(), embeddings.end(),
                [&](const auto& emb) {
                    return emb.size() != query_embeddings.front().size();
                });
            if (dimension_mismatch) {
                continue;
            }

            double sum_similarity = 0.0;
            float max_similarity = -1.0f;
            std::size_t count = 0;

            for (const auto& stored : embeddings) {
                for (const auto& query_emb : query_embeddings) {
                    float sim = std::inner_product(query_emb.begin(), query_emb.end(),
                                                   stored.begin(), 0.0f);
                    sim = (sim + 1.0f) * 0.5f;
                    sum_similarity += static_cast<double>(sim);
                    max_similarity = std::max(max_similarity, sim);
                    ++count;
                }
            }

            if (count == 0) {
                continue;
            }

            float avg_similarity = static_cast<float>(sum_similarity / static_cast<double>(count));
            if (avg_similarity > best_avg) {
                best_avg = avg_similarity;
                best_max = max_similarity;
                best_name = name;
            }
        }

        return {best_name, best_avg, best_max};
    };

    if (!opts.allow_all) {
        std::string desired = opts.target_name.value_or(username);
        auto [matched_name, avg_similarity, max_similarity] = compute_best(entries, desired);
        if (avg_similarity < 0.0f) {
            pam_syslog(pamh, LOG_INFO, "pam_lxfu: no match for requested name '%s'", desired.c_str());
            return PAM_AUTH_ERR;
        }
        if (avg_similarity < static_cast<float>(opts.threshold)) {
            pam_syslog(pamh, LOG_INFO, "pam_lxfu: similarity %.2f below threshold %.2f for '%s'",
                       avg_similarity, opts.threshold, desired.c_str());
            return PAM_AUTH_ERR;
        }
        if (opts.debug) {
            pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: user '%s' matched avg %.2f (max %.2f) using %zu frame(s)",
                       desired.c_str(), avg_similarity, max_similarity, query_embeddings.size());
        }
        return PAM_SUCCESS;
    }

    auto [best_name, best_avg, best_max] = compute_best(entries, std::nullopt);
    if (best_name.empty() || best_avg < static_cast<float>(opts.threshold)) {
        pam_syslog(pamh, LOG_INFO, "pam_lxfu: no profile exceeded threshold %.2f", opts.threshold);
        return PAM_AUTH_ERR;
    }

    if (opts.debug) {
        pam_syslog(pamh, LOG_DEBUG, "pam_lxfu: matched profile '%s' avg %.2f (max %.2f) using %zu frame(s)",
                   best_name.c_str(), best_avg, best_max, query_embeddings.size());
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
