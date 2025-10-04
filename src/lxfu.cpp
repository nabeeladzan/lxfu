#include "face_engine.hpp"
#include "lmdb_store.hpp"
#include "config.hpp"
#include "face_detector.hpp"

#include <iostream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

Config g_config;
FaceDetector g_face_detector;

void print_usage(const char* program_name) {
    std::cout << "LXFU - Linux Face Utility\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " [--preview] enroll [--device PATH|--file PATH] [--name NAME]\n";
    std::cout << "  " << program_name << " [--preview] query [--device PATH|--file PATH] [--name NAME|--all]\n";
    std::cout << "  " << program_name << " list\n";
    std::cout << "  " << program_name << " delete --name NAME [--confirm]\n";
    std::cout << "  " << program_name << " clear [--confirm]\n";
    std::cout << "  " << program_name << " config\n\n";
    std::cout << "Legacy positional fallback:\n";
    std::cout << "  " << program_name << " enroll <device|image_path> <name>\n";
    std::cout << "  " << program_name << " query <device|image_path> [name]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --preview       Show camera preview window (press SPACE to capture, ESC to cancel)\n";
    std::cout << "  --device PATH   Capture from camera device (defaults to config setting)\n";
    std::cout << "  --file PATH     Load from image file instead of a device\n";
    std::cout << "  --name NAME     Specify profile name (defaults to 'default')\n";
    std::cout << "  --all           Query mode: allow matches for any enrolled name\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " enroll --device /dev/video0 --name alice\n";
    std::cout << "  " << program_name << " query --device /dev/video0 --name alice\n";
    std::cout << "  " << program_name << " query --device /dev/video0 --all\n";
    std::cout << "  " << program_name << " list\n";
    std::cout << "  " << program_name << " delete --name alice --confirm\n";
    std::cout << "  " << program_name << " clear --confirm\n";
    std::cout << "  " << program_name << " config\n";
}

struct EnrollOptions {
    std::string source;
    std::string name{ "default" };
    bool show_preview{false};
};

struct QueryOptions {
    std::string source;
    std::optional<std::string> target_name;
    bool match_all{false};
    bool show_preview{false};
};

namespace {

bool is_flag(const std::string& arg, std::initializer_list<const char*> names) {
    return std::any_of(names.begin(), names.end(), [&](const char* n) { return arg == n; });
}

std::string require_value(const std::vector<std::string>& args, size_t& i, const char* flag) {
    if (i + 1 >= args.size()) {
        throw std::runtime_error(std::string("Missing value for ") + flag);
    }
    return args[++i];
}

bool confirm_action(bool auto_confirm, const std::string& prompt) {
    if (auto_confirm) {
        return true;
    }
    std::cout << prompt << " Type 'yes' to continue: ";
    std::cout.flush();
    std::string response;
    if (!std::getline(std::cin, response)) {
        return false;
    }
    return response == "yes";
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

    // Try explicit path first so that non-sequential devices (e.g. IR cameras) work.
    if (cap.open(source, cv::CAP_V4L2)) {
        return true;
    }
    if (cap.open(source)) {
        return true;
    }

    int index = 0;
    if (parse_device_index(source, index) && cap.open(index)) {
        return true;
    }

    return false;
}

void apply_camera_defaults(cv::VideoCapture& cap) {
    struct CameraProp { int property; double value; };
    const CameraProp desired_props[] = {
        {cv::CAP_PROP_FRAME_WIDTH, 640},
        {cv::CAP_PROP_FRAME_HEIGHT, 480},
        {cv::CAP_PROP_FPS, 30},
    };

    for (const auto& prop : desired_props) {
        cap.set(prop.property, prop.value);
    }
}

void warm_up_camera(cv::VideoCapture& cap, int frames_to_discard = 10) {
    cv::Mat dummy;
    for (int i = 0; i < frames_to_discard; ++i) {
        if (!cap.read(dummy) || dummy.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}

}

cv::Mat capture_from_device(const std::string& device_path, bool show_preview = false) {
    cv::VideoCapture cap;
    if (!open_video_capture(cap, device_path)) {
        throw std::runtime_error("Failed to open device: " + device_path);
    }

    apply_camera_defaults(cap);

    cv::Mat frame;

    if (show_preview) {
        const char* display = std::getenv("DISPLAY");
        const char* wayland = std::getenv("WAYLAND_DISPLAY");
        if (!display && !wayland) {
            std::cout << "⚠ Warning: --preview requested but no display detected (headless system)" << std::endl;
            std::cout << "⚠ Falling back to instant capture mode..." << std::endl;
            show_preview = false;
        }
    }

    if (show_preview) {
        const std::string window_name = "LXFU Preview - Press SPACE to capture, ESC to cancel";
        try {
            cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        } catch (const cv::Exception&) {
            std::cout << "⚠ Warning: Could not create preview window (headless system?)" << std::endl;
            std::cout << "⚠ Falling back to instant capture mode..." << std::endl;
            show_preview = false;
        }
    }

    if (show_preview) {
        std::cout << "Preview mode: Press SPACE to capture, ESC to cancel..." << std::endl;
        bool captured = false;
        while (!captured) {
            cv::Mat current_frame;
            if (!cap.read(current_frame) || current_frame.empty()) {
                cap.release();
                cv::destroyAllWindows();
                throw std::runtime_error("Failed to capture frame from device");
            }

            cv::putText(current_frame, "Press SPACE to capture, ESC to cancel",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 0), 2);

            cv::Mat preview_frame = current_frame.clone();
            g_face_detector.draw_faces(preview_frame);

            try {
                cv::imshow("LXFU Preview - Press SPACE to capture, ESC to cancel", preview_frame);
            } catch (const cv::Exception&) {
                std::cout << "⚠ Warning: Preview display failed, switching to instant capture" << std::endl;
                frame = current_frame.clone();
                captured = true;
                break;
            }

            int key = cv::waitKey(30);
            if (key == 32) { // SPACE
                frame = current_frame.clone();
                captured = true;
                std::cout << "✓ Frame captured!" << std::endl;
            } else if (key == 27) { // ESC
                cap.release();
                cv::destroyAllWindows();
                throw std::runtime_error("Capture cancelled by user");
            }
        }

        cv::destroyAllWindows();
        cv::waitKey(1);
    } else {
        if (!cap.read(frame) || frame.empty()) {
            throw std::runtime_error("Failed to capture frame");
        }
        frame = frame.clone();
        std::cout << "✓ Frame captured (instant mode)" << std::endl;
    }

    cap.release();
    return frame;
}

cv::Mat load_image_or_capture(const std::string& source, bool show_preview) {
    if (source.rfind("/dev/video", 0) == 0) {
        return capture_from_device(source, show_preview);
    }

    if (!fs::exists(source)) {
        throw std::runtime_error("File not found: " + source);
    }

    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + source);
    }

    if (show_preview) {
        const char* display = std::getenv("DISPLAY");
        const char* wayland = std::getenv("WAYLAND_DISPLAY");
        if (!display && !wayland) {
            std::cout << "⚠ Warning: --preview requested but no display detected (headless system)" << std::endl;
            std::cout << "⚠ Skipping image preview..." << std::endl;
        } else {
            try {
                const std::string window_name = "LXFU Image Preview - Press any key to continue";
                cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
                cv::imshow(window_name, image);
                std::cout << "Loaded image. Press any key to continue..." << std::endl;
                cv::waitKey(0);
                cv::destroyAllWindows();
                cv::waitKey(1);
            } catch (const cv::Exception&) {
                std::cout << "⚠ Warning: Could not display preview (headless system?)" << std::endl;
                std::cout << "⚠ Continuing without preview..." << std::endl;
            }
        }
    }

    return image;
}

void enroll(const EnrollOptions& opts) {
    try {
        std::string model_path = g_config.get("model_path");
        FaceEngine engine(model_path);

        // Check if source is a device (camera) or file
        bool is_device = (opts.source.rfind("/dev/video", 0) == 0);
        
        std::vector<cv::Mat> face_images;

        if (is_device) {
            // Multi-frame capture mode for camera
            std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
            std::cout << "║  ENROLLMENT - Multi-Frame Capture Mode            ║" << std::endl;
            std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
            std::cout << "\nInstructions:" << std::endl;
            std::cout << "  • Look at the camera and stay centered" << std::endl;
            std::cout << "  • VERY SLIGHTLY move and adjust your head" << std::endl;
            std::cout << "  • Try small turns left/right and slight up/down" << std::endl;
            std::cout << "  • Keep your face visible at all times" << std::endl;
            std::cout << "\nCapturing frames for 10 seconds..." << std::endl;

            cv::VideoCapture cap;
            if (!open_video_capture(cap, opts.source)) {
                throw std::runtime_error("Failed to open device: " + opts.source);
            }

            // Set camera properties for better quality
            apply_camera_defaults(cap);

            // Warm up camera
            std::cout << "\nWarming up camera..." << std::endl;
            warm_up_camera(cap);

            bool show_preview = opts.show_preview;
            if (show_preview) {
                const char* display = std::getenv("DISPLAY");
                const char* wayland = std::getenv("WAYLAND_DISPLAY");
                if (!display && !wayland) {
                    std::cout << "⚠ Warning: No display detected, disabling preview" << std::endl;
                    show_preview = false;
                }
            }

            std::string window_name;
            if (show_preview) {
                window_name = "LXFU Enrollment - Keep face visible";
                try {
                    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
                } catch (const cv::Exception&) {
                    std::cout << "⚠ Warning: Could not create preview window" << std::endl;
                    show_preview = false;
                }
            }

            auto start_time = std::chrono::steady_clock::now();
            const int capture_duration_sec = 10;
            int frames_captured = 0;
            int frames_with_faces = 0;
            int last_second_shown = -1;
            int consecutive_failures = 0;
            int reopen_attempts = 0;
            const int max_reopen_attempts = 2;
            const int max_consecutive_failures = 45; // ~4.5s with 100ms sleep
            const int failure_reopen_threshold = 15;

            std::cout << "\nStarting capture..." << std::endl;
            g_face_detector = FaceDetector(false); // Disable verbose for frame-by-frame

            while (true) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                
                if (elapsed >= capture_duration_sec) {
                    break;
                }

                // Show countdown every second
                int current_second = static_cast<int>(elapsed);
                if (current_second != last_second_shown) {
                    int remaining = capture_duration_sec - current_second;
                    std::cout << "⏱  " << remaining << " seconds remaining... "
                              << "(captured " << frames_with_faces << " valid frames)" << std::endl;
                    last_second_shown = current_second;
                }

                cv::Mat frame;
                if (!cap.read(frame) || frame.empty()) {
                    consecutive_failures++;

                    if (consecutive_failures == 1 || consecutive_failures % 5 == 0) {
                        std::cout << "⚠ Warning: Failed to capture frame, retrying..." << std::endl;
                    }

                    if (consecutive_failures == failure_reopen_threshold && reopen_attempts < max_reopen_attempts) {
                        std::cout << "⚠ Attempting to reinitialize device..." << std::endl;
                        ++reopen_attempts;
                        if (!open_video_capture(cap, opts.source)) {
                            throw std::runtime_error("Failed to reinitialize device: " + opts.source);
                        }
                        apply_camera_defaults(cap);
                        warm_up_camera(cap, 5);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }

                    if (consecutive_failures >= max_consecutive_failures) {
                        throw std::runtime_error("Camera did not produce frames. Check cable and pixel format settings for " + opts.source);
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(60));
                    continue;
                }

                consecutive_failures = 0;

                frames_captured++;

                // Try to detect and crop face
                auto face_image = g_face_detector.crop_to_face(frame);
                
                if (show_preview) {
                    cv::Mat preview_frame = frame.clone();
                    g_face_detector.draw_faces(preview_frame);
                    
                    // Draw countdown on preview
                    int remaining = capture_duration_sec - current_second;
                    std::string countdown_text = std::to_string(remaining) + "s";
                    cv::putText(preview_frame, countdown_text,
                                cv::Point(preview_frame.cols - 100, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 2.0,
                                cv::Scalar(0, 255, 255), 3);
                    
                    // Draw frame counter
                    std::string counter_text = "Valid: " + std::to_string(frames_with_faces);
                    cv::putText(preview_frame, counter_text,
                                cv::Point(10, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                cv::Scalar(0, 255, 0), 2);
                    
                    try {
                        cv::imshow(window_name, preview_frame);
                        cv::waitKey(1);
                    } catch (const cv::Exception&) {
                        show_preview = false;
                    }
                }

                if (face_image) {
                    face_images.push_back(*face_image);
                    frames_with_faces++;
                }

                // Small delay to control frame rate (~10 FPS for processing)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            cap.release();
            if (show_preview) {
                cv::destroyAllWindows();
                cv::waitKey(1);
            }

            std::cout << "\n✓ Capture complete!" << std::endl;
            std::cout << "  Total frames processed: " << frames_captured << std::endl;
            std::cout << "  Frames with detected faces: " << frames_with_faces << std::endl;
            std::cout << "  Detection rate: " << std::fixed << std::setprecision(1)
                      << (100.0 * frames_with_faces / std::max(1, frames_captured)) << "%" << std::endl;

            g_face_detector = FaceDetector(true); // Re-enable verbose

            if (face_images.empty()) {
                std::cout << "\n✗ Enrollment failed: No valid faces detected during capture" << std::endl;
                std::cout << "  Please ensure:" << std::endl;
                std::cout << "  • Your face is clearly visible and well-lit" << std::endl;
                std::cout << "  • You're facing the camera" << std::endl;
                std::cout << "  • The camera is working properly" << std::endl;
                return;
            }

        } else {
            // Single image mode for file input
            std::cout << "Loading image from file..." << std::endl;
            cv::Mat image = load_image_or_capture(opts.source, opts.show_preview);
            std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;

            auto face_image = g_face_detector.crop_to_face(image);
            if (!face_image) {
                std::cout << "✗ Enrollment aborted: no face detected in image" << std::endl;
                return;
            }

            face_images.push_back(*face_image);
        }

        // Extract embeddings for all captured faces
        std::cout << "\nExtracting embeddings from " << face_images.size() << " frame(s)..." << std::endl;
        
        std::string lmdb_path = g_config.get_embeddings_path();
        LMDBStore store(lmdb_path);
        
        int embeddings_stored = 0;
        for (size_t i = 0; i < face_images.size(); ++i) {
            if ((i + 1) % 10 == 0 || i == 0 || i == face_images.size() - 1) {
                std::cout << "  Processing frame " << (i + 1) << "/" << face_images.size() << "..." << std::endl;
            }
            
            std::vector<float> embedding = engine.extract_embedding(face_images[i]);
            store.store_embedding(opts.name, embedding);
            embeddings_stored++;
        }

        std::size_t total_samples = store.get_embeddings(opts.name).size();

        std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  ✓ ENROLLMENT SUCCESSFUL!                          ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "\n  Profile: " << opts.name << std::endl;
        std::cout << "  Embedding dimensions: " << (face_images.empty() ? 0 : 384) << std::endl;
        std::cout << "  New samples added: " << embeddings_stored << std::endl;
        std::cout << "  Total samples for profile: " << total_samples << std::endl;
        std::cout << "  Total profiles in database: " << store.size() << std::endl;
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during enrollment: " << e.what() << std::endl;
        throw;
    }
}

void query(const QueryOptions& opts) {
    try {
        std::string model_path = g_config.get("model_path");
        FaceEngine engine(model_path);

        std::cout << "Loading/capturing face..." << std::endl;
        cv::Mat image = load_image_or_capture(opts.source, opts.show_preview);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;

        auto face_image = g_face_detector.crop_to_face(image);
        if (!face_image) {
            std::cout << "✗ Query aborted: no face detected" << std::endl;
            return;
        }

        std::cout << "Extracting face embedding..." << std::endl;
        std::vector<float> embedding = engine.extract_embedding(*face_image);

        std::string lmdb_path = g_config.get_embeddings_path();
        LMDBStore store(lmdb_path, LMDBStore::Mode::ReadOnly);
        auto entries = store.get_all_embeddings();

        if (entries.empty()) {
            std::cout << "\n⚠ No profiles enrolled yet." << std::endl;
            return;
        }

        bool require_specific = !opts.match_all;
        std::string desired = opts.target_name.value_or("default");

        std::cout << "\nComparing against enrolled profiles:" << std::endl;

        float best_avg_similarity = -1.0f;
        float best_max_similarity = -1.0f;
        std::string best_name;
        bool considered_any = false;
        bool matched_name_present = false;

        for (const auto& [name, stored_embedding] : entries) {
            if (require_specific && name != desired) {
                continue;
            }
            if (stored_embedding.empty()) {
                continue;
            }

            bool dimension_mismatch = std::any_of(
                stored_embedding.begin(), stored_embedding.end(),
                [&](const auto& emb) { return emb.size() != embedding.size(); });
            if (dimension_mismatch) {
                continue;
            }

            considered_any = true;
            if (require_specific) {
                matched_name_present = true;
            }

            std::vector<float> similarities;
            similarities.reserve(stored_embedding.size());
            for (const auto& emb : stored_embedding) {
                float sim = std::inner_product(embedding.begin(), embedding.end(),
                                               emb.begin(), 0.0f);
                sim = (sim + 1.0f) * 0.5f;
                similarities.push_back(sim);
            }

            if (similarities.empty()) {
                continue;
            }

            float avg_similarity = std::accumulate(similarities.begin(), similarities.end(), 0.0f) /
                                   static_cast<float>(similarities.size());
            float max_similarity = *std::max_element(similarities.begin(), similarities.end());

            std::cout << "  " << name << ": avg "
                      << std::fixed << std::setprecision(2) << (avg_similarity * 100.0f)
                      << "% (samples: " << similarities.size()
                      << ", max: " << (max_similarity * 100.0f) << "%)" << std::endl;

            if (avg_similarity > best_avg_similarity) {
                best_avg_similarity = avg_similarity;
                best_max_similarity = max_similarity;
                best_name = name;
            }
        }

        if (!considered_any || best_avg_similarity < 0.0f) {
            if (require_specific && !matched_name_present) {
                std::cout << "\n⚠ No enrolled samples for name '" << desired << "'" << std::endl;
            } else {
                std::cout << "\n⚠ No match found" << std::endl;
            }
            return;
        }

        const float threshold = g_config.get_threshold();

        std::cout << "\nBest match: " << best_name << std::endl;
        if (require_specific) {
            std::cout << "  Requested name: " << desired << std::endl;
        }
        std::cout << "  Average similarity: " << std::fixed << std::setprecision(2)
                  << (best_avg_similarity * 100.0f) << "%" << std::endl;
        std::cout << "  Max similarity: " << std::fixed << std::setprecision(2)
                  << (best_max_similarity * 100.0f) << "%" << std::endl;
        std::cout << "  Threshold: " << std::fixed << std::setprecision(2)
                  << (threshold * 100.0f) << "%" << std::endl;

        if (best_avg_similarity >= threshold) {
            std::cout << "\n✓ Authentication successful" << std::endl;
        } else {
            std::cout << "\n✗ Authentication failed: best match below threshold" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error during query: " << e.what() << std::endl;
        throw;
    }
}

void list_profiles() {
    try {
        std::string lmdb_path = g_config.get_embeddings_path();
        if (!fs::exists(lmdb_path)) {
            std::cout << "No profiles enrolled." << std::endl;
            return;
        }
        LMDBStore store(lmdb_path, LMDBStore::Mode::ReadOnly);
        auto entries = store.get_all_embeddings();
        if (entries.empty()) {
            std::cout << "No profiles enrolled." << std::endl;
            return;
        }

        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        std::cout << std::left << std::setw(24) << "Name"
                  << std::setw(12) << "Samples"
                  << "Dim" << std::endl;
        std::cout << std::string(48, '-') << std::endl;
        for (const auto& [name, embedding_list] : entries) {
            std::size_t samples = embedding_list.size();
            std::size_t dimension = samples > 0 ? embedding_list.front().size() : 0;
            std::cout << std::left << std::setw(24) << (name.empty() ? "<unnamed>" : name)
                      << std::setw(12) << samples
                      << dimension << std::endl;
        }
        std::cout << "\nTotal profiles: " << entries.size() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error while listing profiles: " << ex.what() << std::endl;
    }
}

void delete_profile(const std::vector<std::string>& args) {
    try {
        bool confirm_flag = false;
        std::optional<std::string> target_name;

        for (size_t i = 0; i < args.size(); ++i) {
            const std::string& arg = args[i];
            if (arg == "--confirm") {
                confirm_flag = true;
            } else if (is_flag(arg, {"--name"})) {
                target_name = require_value(args, i, "--name");
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("Unknown delete option: " + arg);
            } else if (!target_name) {
                target_name = arg;
            } else {
                throw std::runtime_error("Unexpected argument: " + arg);
            }
        }

        if (!target_name) {
            throw std::runtime_error("delete requires --name NAME");
        }

        std::string lmdb_path = g_config.get_embeddings_path();
        if (!fs::exists(lmdb_path)) {
            std::cout << "No profiles enrolled." << std::endl;
            return;
        }

        LMDBStore store(lmdb_path);

        if (!confirm_action(confirm_flag, "This will delete profile '" + *target_name + "'.")) {
            std::cout << "Deletion cancelled." << std::endl;
            return;
        }

        bool removed = store.delete_embedding(*target_name);
        if (removed) {
            std::cout << "Profile '" << *target_name << "' removed." << std::endl;
        } else {
            std::cout << "No profile named '" << *target_name << "' found." << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error while deleting profile: " << ex.what() << std::endl;
    }
}

void clear_profiles(const std::vector<std::string>& args) {
    try {
        bool confirm_flag = false;
        for (const auto& arg : args) {
            if (arg == "--confirm") {
                confirm_flag = true;
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("Unknown clear option: " + arg);
            }
        }

        std::string lmdb_path = g_config.get_embeddings_path();
        if (!fs::exists(lmdb_path)) {
            std::cout << "Nothing to clear." << std::endl;
            return;
        }

        LMDBStore store(lmdb_path);
        std::size_t profile_count = store.size();
        if (profile_count == 0) {
            std::cout << "Nothing to clear." << std::endl;
            return;
        }

        if (!confirm_action(confirm_flag, "This will remove all profiles (" + std::to_string(profile_count) + ").")) {
            std::cout << "Clear cancelled." << std::endl;
            return;
        }

        store.clear();
        std::cout << "All profiles cleared." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error while clearing profiles: " << ex.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        g_config = load_config();

        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        bool show_preview = false;
        int arg_offset = 1;
        if (std::string(argv[1]) == "--preview") {
            show_preview = true;
            arg_offset = 2;
            if (argc < 3) {
                print_usage(argv[0]);
                return 1;
            }
        }

        if (arg_offset >= argc) {
            print_usage(argv[0]);
            return 1;
        }

        std::string command = argv[arg_offset];

        std::vector<std::string> args;
        for (int i = arg_offset + 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        if (show_preview && command != "enroll" && command != "query") {
            std::cout << "⚠ '--preview' flag is ignored for command '" << command << "'." << std::endl;
        }

        if (command == "enroll") {
            try {
                EnrollOptions options;
                options.show_preview = show_preview;
                options.source = g_config.get("default_device", "/dev/video0");

                bool source_from_flag = false;
                bool name_from_flag = false;
                for (size_t i = 0; i < args.size(); ++i) {
                    const std::string& arg = args[i];
                    if (is_flag(arg, {"--name"})) {
                        options.name = require_value(args, i, "--name");
                        name_from_flag = true;
                    } else if (is_flag(arg, {"--device"})) {
                        options.source = require_value(args, i, "--device");
                        source_from_flag = true;
                    } else if (is_flag(arg, {"--file", "--source"})) {
                        options.source = require_value(args, i, "--file");
                        source_from_flag = true;
                    } else if (!arg.empty() && arg[0] == '-') {
                        throw std::runtime_error("Unknown enroll option: " + arg);
                    } else if (!source_from_flag) {
                        options.source = arg;
                        source_from_flag = true;
                    } else if (!name_from_flag) {
                        options.name = arg;
                        name_from_flag = true;
                    } else {
                        throw std::runtime_error("Unexpected argument: " + arg);
                    }
                }

                if (options.source.empty()) {
                    throw std::runtime_error("No capture source provided");
                }

                enroll(options);
            } catch (const std::exception& ex) {
                std::cerr << "Error: " << ex.what() << "\n";
                print_usage(argv[0]);
                return 1;
            }

        } else if (command == "query") {
            try {
                QueryOptions options;
                options.show_preview = show_preview;
                options.source = g_config.get("default_device", "/dev/video0");

                bool source_from_flag = false;
                bool name_from_flag = false;
                for (size_t i = 0; i < args.size(); ++i) {
                    const std::string& arg = args[i];
                    if (is_flag(arg, {"--name"})) {
                        options.target_name = require_value(args, i, "--name");
                        name_from_flag = true;
                    } else if (is_flag(arg, {"--device"})) {
                        options.source = require_value(args, i, "--device");
                        source_from_flag = true;
                    } else if (is_flag(arg, {"--file", "--source"})) {
                        options.source = require_value(args, i, "--file");
                        source_from_flag = true;
                    } else if (arg == "--all") {
                        options.match_all = true;
                        options.target_name.reset();
                    } else if (!arg.empty() && arg[0] == '-') {
                        throw std::runtime_error("Unknown query option: " + arg);
                    } else if (!source_from_flag) {
                        options.source = arg;
                        source_from_flag = true;
                    } else if (!name_from_flag) {
                        options.target_name = arg;
                        name_from_flag = true;
                    } else {
                        throw std::runtime_error("Unexpected argument: " + arg);
                    }
                }

                if (!options.match_all) {
                    options.target_name = options.target_name.value_or("default");
                }

                if (options.source.empty()) {
                    throw std::runtime_error("No capture source provided");
                }

                query(options);
            } catch (const std::exception& ex) {
                std::cerr << "Error: " << ex.what() << "\n";
                print_usage(argv[0]);
                return 1;
            }

        } else if (command == "list") {
            if (!args.empty()) {
                std::cerr << "Error: 'list' does not accept additional arguments" << std::endl;
                return 1;
            }
            list_profiles();

        } else if (command == "config") {
            if (!args.empty()) {
                std::cerr << "Error: 'config' does not accept additional arguments" << std::endl;
                return 1;
            }
            g_config.print_config();

        } else if (command == "delete") {
            delete_profile(args);

        } else if (command == "clear") {
            clear_profiles(args);

        } else {
            std::cerr << "Error: Unknown command '" << command << "'\n";
            print_usage(argv[0]);
            return 1;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
