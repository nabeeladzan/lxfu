#pragma once

#include "config.hpp"
#include "face_detector.hpp"
#include "face_engine.hpp"
#include "lmdb_store.hpp"

#include <systemd/sd-bus.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// Forward declaration
struct sd_bus;

namespace lxfu {

class FaceService {
public:
    FaceService();
    ~FaceService();

    FaceService(const FaceService&) = delete;
    FaceService& operator=(const FaceService&) = delete;

    int run();

    static FaceService& instance();

    // DBus handlers
    int handle_get_default_device(sd_bus_message* m, sd_bus_error* error);
    int handle_claim(sd_bus_message* m, sd_bus_error* error);
    int handle_release(sd_bus_message* m, sd_bus_error* error);
    int handle_verify_start(sd_bus_message* m, sd_bus_error* error);
    int handle_verify_stop(sd_bus_message* m, sd_bus_error* error);

    void stop();

    static int on_get_default_device(sd_bus_message* m, void* userdata, sd_bus_error* error);
    static int on_claim(sd_bus_message* m, void* userdata, sd_bus_error* error);
    static int on_release(sd_bus_message* m, void* userdata, sd_bus_error* error);
    static int on_verify_start(sd_bus_message* m, void* userdata, sd_bus_error* error);
    static int on_verify_stop(sd_bus_message* m, void* userdata, sd_bus_error* error);

private:
    void start_bus();
    void register_objects();
    void unregister_objects();

    void start_verification(const std::string& mode);
    void request_stop_verification();
    void verification_worker(double warmup_delay,
                             double capture_duration,
                             double frame_interval,
                             float threshold,
                             std::string device_path,
                             std::string target_name,
                             bool allow_all);

    void emit_status_signal(const std::string& status,
                            const std::string& message = {});

    bool ensure_resources_ready();

    std::vector<cv::Mat> capture_faces(const std::string& device_path,
                                       double warmup_delay,
                                       double capture_duration,
                                       double frame_interval,
                                       std::atomic_bool& stop_flag,
                                       int& total_frames,
                                       int& frames_with_faces);

    std::optional<std::tuple<std::string, float, float>>
    compute_best_match(const std::vector<std::vector<float>>& embeddings,
                       LMDBStore& store,
                       const std::optional<std::string>& required_name,
                       bool allow_all);

private:
    sd_bus* bus_;
    sd_bus_slot* manager_slot_;
    sd_bus_slot* device_slot_;

    std::string service_name_;
    std::string manager_path_;
    std::string manager_interface_;
    std::string device_path_;
    std::string device_interface_;

    std::atomic_bool running_;
    std::atomic_bool claimed_;
    std::atomic_bool verifying_;
    std::atomic_bool stop_requested_;

    std::thread worker_thread_;
    std::mutex state_mutex_;

    Config config_;
    std::string model_path_;
    std::string db_path_;
    FaceDetector detector_;

    double default_warmup_;
    double default_capture_;
    double default_interval_;
    float default_threshold_;
};

} // namespace lxfu
