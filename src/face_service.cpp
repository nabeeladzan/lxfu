#include "face_service.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <systemd/sd-bus.h>

#include <chrono>
#include <algorithm>
#include <csignal>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <numeric>

namespace lxfu {

namespace {

constexpr const char* kServiceName = "dev.nabeeladzan.lxfu";
constexpr const char* kManagerPath = "/dev/nabeeladzan/lxfu";
constexpr const char* kManagerInterface = "dev.nabeeladzan.lxfu.Manager";
constexpr const char* kDevicePath = "/dev/nabeeladzan/lxfu/Device0";
constexpr const char* kDeviceInterface = "dev.nabeeladzan.lxfu.Device";

bool parse_bool(const std::string& value, bool fallback) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return fallback;
}

double parse_double(const std::string& value, double fallback) {
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        return fallback;
    }
}

float parse_float(const std::string& value, float fallback) {
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        return fallback;
    }
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

bool open_video_capture(cv::VideoCapture& cap, const std::string& source) {
    cap.release();

    if (cap.open(source, cv::CAP_V4L2)) {
        return true;
    }
    if (cap.open(source)) {
        return true;
    }

    int index = -1;
    if (parse_device_index(source, index) && cap.open(index)) {
        return true;
    }

    return false;
}

void apply_camera_defaults(cv::VideoCapture& cap) {
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
}

void idle_sleep(double seconds) {
    if (seconds <= 0.0) {
        return;
    }
    auto duration = std::chrono::duration<double>(seconds);
    std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(duration));
}

} // namespace

static const sd_bus_vtable manager_vtable[] = {
    SD_BUS_VTABLE_START(0),
    SD_BUS_METHOD("GetDefaultDevice", "", "o", &FaceService::on_get_default_device, SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_SIGNAL("DeviceListChanged", "ao", 0),
    SD_BUS_VTABLE_END
};

static const sd_bus_vtable device_vtable[] = {
    SD_BUS_VTABLE_START(0),
    SD_BUS_METHOD("Claim", "", "", &FaceService::on_claim, SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("Release", "", "", &FaceService::on_release, SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("VerifyStart", "s", "", &FaceService::on_verify_start, SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("VerifyStop", "", "", &FaceService::on_verify_stop, SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_SIGNAL("VerificationStatus", "ss", 0),
    SD_BUS_VTABLE_END
};

FaceService::FaceService()
    : bus_(nullptr),
      manager_slot_(nullptr),
      device_slot_(nullptr),
      service_name_(kServiceName),
      manager_path_(kManagerPath),
      manager_interface_(kManagerInterface),
      device_path_(kDevicePath),
      device_interface_(kDeviceInterface),
      running_(false),
      claimed_(false),
      verifying_(false),
      stop_requested_(false),
      config_(load_config(false)),
      detector_(false),
      default_warmup_(1.0),
      default_capture_(2.0),
      default_interval_(0.1),
      default_threshold_(config_.get_threshold()) {

    model_path_ = config_.get("model_path");
    db_path_ = config_.get_embeddings_path();

    default_warmup_ = parse_double(config_.get("service_warmup_delay", "1.0"), 1.0);
    default_capture_ = parse_double(config_.get("service_capture_duration", "2.0"), 2.0);
    default_interval_ = parse_double(config_.get("service_frame_interval", "0.1"), 0.1);
    default_threshold_ = parse_float(config_.get("service_threshold"), default_threshold_);
}

FaceService::~FaceService() {
    stop();
}

FaceService& FaceService::instance() {
    static FaceService service;
    return service;
}

bool FaceService::ensure_resources_ready() {
    if (!detector_.is_initialized()) {
        detector_ = FaceDetector(false);
    }

    if (!detector_.is_initialized()) {
        std::cerr << "FaceService: face detector unavailable" << std::endl;
        return false;
    }
    return true;
}

void FaceService::start_bus() {
    int r = sd_bus_open_system(&bus_);
    if (r < 0) {
        throw std::runtime_error("Failed to connect to system bus: " + std::string(strerror(-r)));
    }

    r = sd_bus_request_name(bus_, service_name_.c_str(), 0);
    if (r < 0) {
        throw std::runtime_error("Failed to request bus name: " + std::string(strerror(-r)));
    }
}

void FaceService::register_objects() {
    int r = sd_bus_add_object_vtable(bus_, &manager_slot_, manager_path_.c_str(), manager_interface_.c_str(), manager_vtable, nullptr);
    if (r < 0) {
        throw std::runtime_error("Failed to register manager object: " + std::string(strerror(-r)));
    }

    r = sd_bus_add_object_vtable(bus_, &device_slot_, device_path_.c_str(), device_interface_.c_str(), device_vtable, nullptr);
    if (r < 0) {
        throw std::runtime_error("Failed to register device object: " + std::string(strerror(-r)));
    }
}

void FaceService::unregister_objects() {
    if (manager_slot_) {
        sd_bus_slot_unref(manager_slot_);
        manager_slot_ = nullptr;
    }
    if (device_slot_) {
        sd_bus_slot_unref(device_slot_);
        device_slot_ = nullptr;
    }
}

int FaceService::run() {
    start_bus();
    register_objects();

    running_.store(true);

    while (running_.load()) {
        int r = sd_bus_process(bus_, nullptr);
        if (r < 0) {
            std::cerr << "FaceService: sd_bus_process error: " << strerror(-r) << std::endl;
            break;
        }

        if (r == 0) {
            r = sd_bus_wait(bus_, 1'000'000); // 1s
            if (r < 0) {
                std::cerr << "FaceService: sd_bus_wait error: " << strerror(-r) << std::endl;
                break;
            }
        }
    }

    request_stop_verification();
    unregister_objects();

    if (bus_) {
        sd_bus_close(bus_);
        sd_bus_unref(bus_);
        bus_ = nullptr;
    }

    running_.store(false);
    return 0;
}

void FaceService::stop() {
    running_.store(false);
    request_stop_verification();
    if (bus_) {
        sd_bus_close(bus_);
    }
}

int FaceService::handle_get_default_device(sd_bus_message* m, sd_bus_error*) {
    return sd_bus_reply_method_return(m, "o", device_path_.c_str());
}

int FaceService::handle_claim(sd_bus_message* m, sd_bus_error* error) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (claimed_.load()) {
        return sd_bus_error_set_errno(error, EBUSY);
    }
    claimed_.store(true);
    return sd_bus_reply_method_return(m, "");
}

int FaceService::handle_release(sd_bus_message* m, sd_bus_error*) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!claimed_.load()) {
            return sd_bus_reply_method_return(m, "");
        }
        request_stop_verification();
        claimed_.store(false);
    }
    return sd_bus_reply_method_return(m, "");
}

int FaceService::handle_verify_start(sd_bus_message* m, sd_bus_error* error) {
    const char* mode = "any";
    int r = sd_bus_message_read(m, "s", &mode);
    if (r < 0) {
        return r;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!claimed_.load()) {
            return sd_bus_error_set_errno(error, EPERM);
        }
        if (verifying_.load()) {
            return sd_bus_error_set_errno(error, EALREADY);
        }
        verifying_.store(true);
        stop_requested_.store(false);
    }

    start_verification(mode ? mode : "any");
    return sd_bus_reply_method_return(m, "");
}

int FaceService::handle_verify_stop(sd_bus_message* m, sd_bus_error*) {
    request_stop_verification();
    return sd_bus_reply_method_return(m, "");
}

std::vector<cv::Mat> FaceService::capture_faces(const std::string& device_path,
                                                double warmup_delay,
                                                double capture_duration,
                                                double frame_interval,
                                                std::atomic_bool& stop_flag,
                                                int& total_frames,
                                                int& frames_with_faces) {
    std::vector<cv::Mat> face_images;
    std::vector<cv::Mat> fallback_frames;

    cv::VideoCapture cap;
    if (!open_video_capture(cap, device_path)) {
        throw std::runtime_error("Failed to open device: " + device_path);
    }

    apply_camera_defaults(cap);

    cv::Mat dummy;
    auto warmup_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(warmup_delay);
    while (std::chrono::steady_clock::now() < warmup_end && !stop_flag.load()) {
        if (!cap.read(dummy) || dummy.empty()) {
            idle_sleep(0.03);
            continue;
        }
        idle_sleep(0.03);
    }

    const auto start_time = std::chrono::steady_clock::now();
    total_frames = 0;
    frames_with_faces = 0;

    while (!stop_flag.load()) {
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
            idle_sleep(frame_interval > 0.0 ? frame_interval : 0.05);
            continue;
        }

        ++total_frames;

        auto face = detector_.crop_to_face(frame);
        if (face) {
            face_images.push_back(*face);
            ++frames_with_faces;
        } else if (face_images.empty()) {
            fallback_frames.push_back(frame);
        }

        if (frame_interval > 0.0) {
            idle_sleep(frame_interval);
        }
    }

    cap.release();

    if (face_images.empty() && !fallback_frames.empty()) {
        if (auto face = detector_.crop_to_face(fallback_frames.back())) {
            face_images.push_back(*face);
        }
    }

    return face_images;
}

std::optional<std::tuple<std::string, float, float>>
FaceService::compute_best_match(const std::vector<std::vector<float>>& embeddings,
                                LMDBStore& store,
                                const std::optional<std::string>& required_name,
                                bool allow_all) {
    if (embeddings.empty()) {
        return std::nullopt;
    }

    auto entries = store.get_all_embeddings();
    if (entries.empty()) {
        return std::nullopt;
    }

    const std::size_t dim = embeddings.front().size();
    float best_avg = -1.0f;
    float best_max = -1.0f;
    std::string best_name;

    for (const auto& [name, stored_list] : entries) {
        if (!allow_all && required_name && name != *required_name) {
            continue;
        }
        if (stored_list.empty()) {
            continue;
        }

        bool dimension_mismatch = std::any_of(
            stored_list.begin(), stored_list.end(),
            [&](const auto& emb) { return emb.size() != dim; });
        if (dimension_mismatch) {
            continue;
        }

        double sum_similarity = 0.0;
        float max_similarity = -1.0f;
        std::size_t count = 0;

        for (const auto& stored : stored_list) {
            for (const auto& query : embeddings) {
                float sim = std::inner_product(query.begin(), query.end(), stored.begin(), 0.0f);
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

    if (best_avg < 0.0f) {
        return std::nullopt;
    }

    return std::make_optional(std::make_tuple(best_name, best_avg, best_max));
}

void FaceService::emit_status_signal(const std::string& status, const std::string& message) {
    if (!bus_) {
        return;
    }
    int r = sd_bus_emit_signal(bus_, device_path_.c_str(), device_interface_.c_str(), "VerificationStatus", "ss",
                               status.c_str(), message.c_str());
    if (r < 0) {
        std::cerr << "FaceService: failed to emit status signal: " << strerror(-r) << std::endl;
    }
}

void FaceService::verification_worker(double warmup_delay,
                                      double capture_duration,
                                      double frame_interval,
                                      float threshold,
                                      std::string device_path,
                                      std::string target_name,
                                      bool allow_all) {
    try {
        if (!ensure_resources_ready()) {
            emit_status_signal("verify-error", "resources-unavailable");
            verifying_.store(false);
            return;
        }

        std::unique_ptr<FaceEngine> engine;
        try {
            engine = std::make_unique<FaceEngine>(model_path_, /*verbose=*/false);
        } catch (const std::exception& ex) {
            emit_status_signal("verify-error", ex.what());
            verifying_.store(false);
            return;
        }

        emit_status_signal("verify-started", "");

        int total_frames = 0;
        int frames_with_faces = 0;
        auto faces = capture_faces(device_path, warmup_delay, capture_duration, frame_interval, stop_requested_, total_frames, frames_with_faces);

        if (stop_requested_.load()) {
            emit_status_signal("verify-cancelled", "");
            verifying_.store(false);
            stop_requested_.store(false);
            return;
        }

        if (faces.empty()) {
            emit_status_signal("verify-no-face", "no-valid-frames");
            verifying_.store(false);
            stop_requested_.store(false);
            return;
        }

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(faces.size());
        for (const auto& face : faces) {
            std::vector<float> embedding = engine->extract_embedding(face);
            if (!embedding.empty()) {
                embeddings.push_back(std::move(embedding));
            }
        }

        if (embeddings.empty()) {
            emit_status_signal("verify-error", "embedding-failed");
            verifying_.store(false);
            stop_requested_.store(false);
            return;
        }

        LMDBStore store(db_path_, LMDBStore::Mode::ReadOnly);
        auto result = compute_best_match(embeddings, store, allow_all ? std::optional<std::string>{} : std::optional<std::string>{target_name}, allow_all);
        if (!result) {
            emit_status_signal("verify-no-match", "no-enrollment");
            verifying_.store(false);
            stop_requested_.store(false);
            return;
        }

        const auto& [matched_name, avg_similarity, max_similarity] = *result;
        if (avg_similarity >= threshold) {
            emit_status_signal("verify-match", matched_name + ":" + std::to_string(avg_similarity));
        } else {
            emit_status_signal("verify-no-match", matched_name + ":" + std::to_string(avg_similarity));
        }

    } catch (const std::exception& ex) {
        emit_status_signal("verify-error", ex.what());
    }

    verifying_.store(false);
    stop_requested_.store(false);
}

void FaceService::start_verification(const std::string& mode) {
    (void)mode;

    std::string device_path = config_.get("service_device", config_.get("default_device", "/dev/video0"));
    std::string target_name = config_.get("service_profile", config_.get("default_profile", "default"));
    bool allow_all = parse_bool(config_.get("service_allow_all", "false"), false);
    float threshold = default_threshold_;

    try {
        threshold = parse_float(config_.get("service_threshold"), default_threshold_);
    } catch (const std::exception&) {
        threshold = default_threshold_;
    }

    double warmup_delay = parse_double(config_.get("service_warmup_delay"), default_warmup_);
    double capture_duration = parse_double(config_.get("service_capture_duration"), default_capture_);
    double frame_interval = parse_double(config_.get("service_frame_interval"), default_interval_);

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    worker_thread_ = std::thread([this, warmup_delay, capture_duration, frame_interval, threshold, device_path, target_name, allow_all]() {
        verification_worker(warmup_delay, capture_duration, frame_interval, threshold, device_path, target_name, allow_all);
    });
}

void FaceService::request_stop_verification() {
    stop_requested_.store(true);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    verifying_.store(false);
    stop_requested_.store(false);
}

int FaceService::on_get_default_device(sd_bus_message* m, void* userdata, sd_bus_error* error) {
    (void)userdata;
    return FaceService::instance().handle_get_default_device(m, error);
}

int FaceService::on_claim(sd_bus_message* m, void* userdata, sd_bus_error* error) {
    (void)userdata;
    return FaceService::instance().handle_claim(m, error);
}

int FaceService::on_release(sd_bus_message* m, void* userdata, sd_bus_error* error) {
    (void)userdata;
    return FaceService::instance().handle_release(m, error);
}

int FaceService::on_verify_start(sd_bus_message* m, void* userdata, sd_bus_error* error) {
    (void)userdata;
    return FaceService::instance().handle_verify_start(m, error);
}

int FaceService::on_verify_stop(sd_bus_message* m, void* userdata, sd_bus_error* error) {
    (void)userdata;
    return FaceService::instance().handle_verify_stop(m, error);
}

} // namespace lxfu
