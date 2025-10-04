#include "face_engine.hpp"
#include "lmdb_store.hpp"
#include "config.hpp"
#include "face_detector.hpp"

#include <iostream>
#include <string>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <initializer_list>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

// Global config and face detector
Config g_config;
FaceDetector g_face_detector;

void print_usage(const char* program_name) {
    std::cout << "LXFU - Linux Face Utility\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " [--preview] enroll [--device PATH|--file PATH] [--name NAME]\n";
    std::cout << "  " << program_name << " [--preview] query [--device PATH|--file PATH] [--name NAME|--all]\n";
    std::cout << "  " << program_name << " list\n";
    std::cout << "  " << program_name << " delete (--name NAME | --id ID) [--confirm]\n";
    std::cout << "  " << program_name << " clear [--confirm]\n\n";
    std::cout << "Positional fallback (legacy):\n";
    std::cout << "  " << program_name << " [--preview] enroll <device|image_path> <name>\n";
    std::cout << "  " << program_name << " [--preview] query <device|image_path> [name]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --preview       Show camera preview window (press SPACE to capture, ESC to cancel)\n";
    std::cout << "  --device PATH   Capture from camera device (defaults to config setting)\n";
    std::cout << "  --file PATH     Load from image file instead of a device\n";
    std::cout << "  --name NAME     Specify profile name (defaults to 'default')\n";
    std::cout << "  --all           Query mode: allow matches for any enrolled name\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " enroll --device /dev/video0 --name nabeel\n";
    std::cout << "  " << program_name << " enroll face.jpg nabeel\n";
    std::cout << "  " << program_name << " query --device /dev/video0 --name nabeel\n";
    std::cout << "  " << program_name << " query --device /dev/video0 --all\n";
    std::cout << "  " << program_name << " list\n";
    std::cout << "  " << program_name << " delete --name nabeel --confirm\n";
    std::cout << "  " << program_name << " clear --confirm\n";
}

struct EnrollOptions {
    std::string source;
    std::string name;
    bool show_preview{false};
};

struct QueryOptions {
    std::string source;
    std::optional<std::string> target_name;
    bool match_all{false};
    bool show_preview{false};
};

namespace {

bool is_flag(const std::string& arg, const std::initializer_list<const char*>& names) {
    for (const char* n : names) {
        if (arg == n) {
            return true;
        }
    }
    return false;
}

std::string require_value(const std::vector<std::string>& args, size_t& i, const char* flag) {
    if (i + 1 >= args.size()) {
        throw std::runtime_error(std::string("Missing value for ") + flag);
    }
    return args[++i];
}

} // namespace

bool is_camera_source(const std::string& source) {
    return source.rfind("/dev/video", 0) == 0;
}

cv::Mat load_image_or_capture(const std::string& source, bool show_preview = false);

bool face_detected(const cv::Mat& original, const cv::Mat& cropped) {
    return cropped.cols != original.cols || cropped.rows != original.rows;
}

cv::Mat capture_with_retry(const std::string& source, bool show_preview, cv::Mat& raw_image) {
    bool camera = is_camera_source(source);
    bool detector_available = g_face_detector.is_initialized();

    int attempt = 0;
    while (true) {
        ++attempt;
        raw_image = load_image_or_capture(source, show_preview);

        if (!camera || !detector_available) {
            return g_face_detector.crop_to_face(raw_image);
        }

        cv::Mat face_image = g_face_detector.crop_to_face(raw_image);
        if (face_detected(raw_image, face_image)) {
            if (attempt > 1) {
                std::cout << "✓ Face detected after " << attempt << " attempts" << std::endl;
            }
            return face_image;
        }

        std::cout << "⚠ No face detected; retrying capture (Ctrl+C to abort)..." << std::endl;
    }
}

std::string join_ids(const std::vector<int64_t>& ids) {
    if (ids.empty()) {
        return "-";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << ids[i];
    }
    return oss.str();
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

cv::Mat capture_from_device(const std::string& device_path, bool show_preview = false) {
    // Extract device number from path like /dev/video0
    int device_id = 0;
    if (device_path.find("/dev/video") == 0) {
        device_id = std::stoi(device_path.substr(10));
    }
    
    cv::VideoCapture cap(device_id);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open device: " + device_path);
    }
    
    cv::Mat frame;
    
    if (show_preview) {
        // Check if display is available (X11/Wayland)
        const char* display = std::getenv("DISPLAY");
        const char* wayland = std::getenv("WAYLAND_DISPLAY");
        
        if (!display && !wayland) {
            // Headless system detected - fall back to instant capture
            std::cout << "⚠ Warning: --preview requested but no display detected (headless system)" << std::endl;
            std::cout << "⚠ Falling back to instant capture mode..." << std::endl;
            show_preview = false;
        }
    }
    
    if (show_preview) {
        // Preview mode: show window and wait for user confirmation
        const std::string window_name = "LXFU Preview - Press SPACE to capture, ESC to cancel";
        
        try {
            cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        } catch (const cv::Exception& e) {
            // Window creation failed (likely headless)
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
            cap >> current_frame;
            
            if (current_frame.empty()) {
                cap.release();
                cv::destroyAllWindows();
                throw std::runtime_error("Failed to capture frame from device");
            }
            
            // Draw instruction text on frame
            cv::putText(current_frame, "Press SPACE to capture, ESC to cancel", 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                       0.7, cv::Scalar(0, 255, 0), 2);
            
            // Draw face detection boxes if available
            cv::Mat preview_frame = current_frame.clone();
            g_face_detector.draw_faces(preview_frame);
            
            try {
                cv::imshow("LXFU Preview - Press SPACE to capture, ESC to cancel", preview_frame);
            } catch (const cv::Exception& e) {
                // Display failed mid-operation
                std::cout << "⚠ Warning: Preview display failed, switching to instant capture" << std::endl;
                frame = current_frame.clone();
                captured = true;
                break;
            }
            
            int key = cv::waitKey(30);
            if (key == 32) { // SPACE key
                frame = current_frame.clone();
                captured = true;
                std::cout << "✓ Frame captured!" << std::endl;
            } else if (key == 27) { // ESC key
                cap.release();
                cv::destroyAllWindows();
                throw std::runtime_error("Capture cancelled by user");
            }
        }
        
        // Clean up properly to prevent segfault
        cv::destroyAllWindows();
        cv::waitKey(1); // Process window events
    } else {
        // Headless mode: capture immediately
        cap >> frame;
        if (frame.empty()) {
            throw std::runtime_error("Failed to capture frame");
        }
        frame = frame.clone();
        std::cout << "✓ Frame captured (instant mode)" << std::endl;
    }
    
    // Always release the capture device
    cap.release();
    
    return frame;
}

cv::Mat load_image_or_capture(const std::string& source, bool show_preview) {
    // Check if it's a device path
    if (source.find("/dev/video") == 0) {
        return capture_from_device(source, show_preview);
    }
    
    // Try to load as image file
    if (!fs::exists(source)) {
        throw std::runtime_error("File not found: " + source);
    }
    
    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + source);
    }
    
    // If preview requested for image file, show it
    if (show_preview) {
        // Check if display is available
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
                cv::waitKey(1); // Process window events
            } catch (const cv::Exception& e) {
                std::cout << "⚠ Warning: Could not display preview (headless system?)" << std::endl;
                std::cout << "⚠ Continuing without preview..." << std::endl;
            }
        }
    }
    
    return image;
}

void enroll(const EnrollOptions& opts) {
    try {
        // Initialize face engine
        std::string model_path = g_config.get("model_path");
        FaceEngine engine(model_path);
        
        // Load or capture image (retrying for camera sources until a face is detected)
        std::cout << "Loading/capturing face..." << std::endl;
        cv::Mat image;
        cv::Mat face_image = capture_with_retry(opts.source, opts.show_preview, image);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        // Extract embedding
        std::cout << "Extracting face embedding..." << std::endl;
        std::vector<float> embedding = engine.extract_embedding(face_image);
        std::cout << "Embedding extracted: " << embedding.size() << " dimensions" << std::endl;
        
        // Initialize FAISS index
        std::string faiss_path = g_config.get_faiss_index_path();
        engine.init_index(faiss_path, embedding.size());
        
        // Add to FAISS
        int64_t face_id = engine.add_embedding(embedding);
        std::cout << "Face added to index with ID: " << face_id << std::endl;
        
        // Store name in LMDB
        std::string lmdb_path = g_config.get_lmdb_path();
        LMDBStore store(lmdb_path);
        store.store_name(face_id, opts.name);
        std::cout << "Name '" << opts.name << "' associated with face ID " << face_id << std::endl;
        
        // Save index
        engine.save_index(faiss_path);
        
        std::cout << "\n✓ Enrollment successful!" << std::endl;
        std::cout << "  Name: " << opts.name << std::endl;
        std::cout << "  Face ID: " << face_id << std::endl;
        std::cout << "  Total faces in database: " << engine.get_index_size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during enrollment: " << e.what() << std::endl;
        throw;
    }
}

void query(const QueryOptions& opts) {
    try {
        // Initialize face engine
        std::string model_path = g_config.get("model_path");
        FaceEngine engine(model_path);
        
        // Load or capture image (retrying for cameras until a face is detected)
        std::cout << "Loading/capturing face..." << std::endl;
        cv::Mat image;
        cv::Mat face_image = capture_with_retry(opts.source, opts.show_preview, image);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        // Extract embedding
        std::cout << "Extracting face embedding..." << std::endl;
        std::vector<float> embedding = engine.extract_embedding(face_image);
        
        // Initialize FAISS index
        std::string faiss_path = g_config.get_faiss_index_path();
        engine.init_index(faiss_path, embedding.size());
        
        if (engine.get_index_size() == 0) {
            std::cout << "\n⚠ No faces enrolled yet. Use 'enroll' command first." << std::endl;
            return;
        }
        
        // Search for similar face
        std::cout << "Searching for similar faces..." << std::endl;
        int64_t total = static_cast<int64_t>(engine.get_index_size());
        int k = static_cast<int>(std::min<int64_t>(total, 10));
        auto candidates = engine.search_many(embedding, k);

        if (candidates.empty()) {
            std::cout << "\n⚠ No match found" << std::endl;
            return;
        }

        std::string lmdb_path = g_config.get_lmdb_path();
        LMDBStore store(lmdb_path, LMDBStore::Mode::ReadOnly);

        std::unordered_map<std::string, std::vector<FaceEngine::SearchResult>> grouped;
        for (const auto& candidate : candidates) {
            std::string stored = store.get_name(candidate.id);
            if (!stored.empty()) {
                grouped[stored].push_back(candidate);
            }
        }

        if (grouped.empty()) {
            std::cout << "\n⚠ No match found" << std::endl;
            return;
        }

        auto compute_average = [](const std::vector<FaceEngine::SearchResult>& results) {
            if (results.empty()) {
                return 0.0f;
            }
            float sum = 0.0f;
            for (const auto& r : results) {
                sum += r.similarity;
            }
            return sum / results.size();
        };

        if (!opts.match_all) {
            std::string desired_name = opts.target_name.value();
            auto it = grouped.find(desired_name);
            if (it == grouped.end()) {
                std::cout << "\n⚠ No match found for name '" << desired_name << "'" << std::endl;
                return;
            }
            float average = compute_average(it->second);
            float percentage = average * 100.0f;
            std::cout << "\n✓ Face recognized!" << std::endl;
            std::cout << "  Name: " << desired_name << " (requested)" << std::endl;
            std::cout << "  Average similarity: " << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
            std::cout << "  Samples considered: " << it->second.size() << std::endl;
            return;
        }

        std::string best_name;
        float best_average = -2.0f;
        size_t best_count = 0;
        for (const auto& [name, results] : grouped) {
            float average = compute_average(results);
            if (average > best_average) {
                best_average = average;
                best_name = name;
                best_count = results.size();
            }
        }

        if (best_name.empty()) {
            std::cout << "\n⚠ No match found" << std::endl;
            return;
        }

        std::cout << "\n✓ Face recognized!" << std::endl;
        std::cout << "  Name: " << best_name << std::endl;
        std::cout << "  Average similarity: " << std::fixed << std::setprecision(2) << (best_average * 100.0f) << "%" << std::endl;
        std::cout << "  Samples considered: " << best_count << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during query: " << e.what() << std::endl;
        throw;
    }
}

bool list_profiles() {
    try {
        std::string lmdb_path = g_config.get_lmdb_path();
        if (!std::filesystem::exists(lmdb_path)) {
            std::cout << "No faces enrolled." << std::endl;
            return true;
        }

        LMDBStore store(lmdb_path, LMDBStore::Mode::ReadOnly);
        auto entries = store.get_all_entries();

        if (entries.empty()) {
            std::cout << "No faces enrolled." << std::endl;
            return true;
        }

        std::unordered_map<std::string, std::vector<int64_t>> grouped;
        for (const auto& entry : entries) {
            grouped[entry.second].push_back(entry.first);
        }

        std::cout << std::left << std::setw(24) << "Name" << std::setw(8) << "Count" << "IDs" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::vector<std::pair<std::string, std::vector<int64_t>>> ordered(grouped.begin(), grouped.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        for (auto& group : ordered) {
            const std::string& name = group.first.empty() ? std::string("<unnamed>") : group.first;
            const auto& ids = group.second;
            std::cout << std::left << std::setw(24) << name
                      << std::setw(8) << ids.size()
                      << join_ids(ids) << std::endl;
        }

        std::cout << "\nTotal faces: " << entries.size() << std::endl;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "Error while listing profiles: " << ex.what() << std::endl;
        return false;
    }
    return false;
}

bool delete_profiles(const std::vector<std::string>& args) {
    try {
        bool confirm_flag = false;
        std::optional<std::string> target_name;
        std::optional<int64_t> target_id;
        std::vector<std::string> positional;

        for (size_t i = 0; i < args.size(); ++i) {
            const std::string& arg = args[i];
            if (arg == "--confirm") {
                confirm_flag = true;
            } else if (is_flag(arg, {"--name"})) {
                target_name = require_value(args, i, "--name");
            } else if (is_flag(arg, {"--id"})) {
                std::string value = require_value(args, i, "--id");
                try {
                    target_id = std::stoll(value);
                } catch (const std::exception&) {
                    throw std::runtime_error("Invalid value for --id: " + value);
                }
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("Unknown delete option: " + arg);
            } else {
                positional.push_back(arg);
            }
        }

        if (!positional.empty()) {
            if (!target_name && !target_id) {
                // Treat first positional argument as name for convenience
                target_name = positional[0];
            } else {
                throw std::runtime_error("Unexpected positional argument: " + positional[0]);
            }
        }

        if ((target_name.has_value() && target_id.has_value()) || (!target_name && !target_id)) {
            throw std::runtime_error("Specify exactly one of --name or --id for delete");
        }

        std::string lmdb_path = g_config.get_lmdb_path();
        if (!std::filesystem::exists(lmdb_path)) {
            std::cout << "No faces enrolled." << std::endl;
            return true;
        }

        LMDBStore store(lmdb_path, LMDBStore::Mode::ReadWrite);
        auto entries = store.get_all_entries();

        if (entries.empty()) {
            std::cout << "No faces enrolled." << std::endl;
            return true;
        }

        std::unordered_set<int64_t> remove_ids;

        if (target_id) {
            auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
                return entry.first == target_id.value();
            });
            if (it == entries.end()) {
                std::cout << "No entry found with ID " << target_id.value() << std::endl;
                return true;
            }
            remove_ids.insert(target_id.value());
        } else if (target_name) {
            for (const auto& entry : entries) {
                if (entry.second == target_name.value()) {
                    remove_ids.insert(entry.first);
                }
            }
            if (remove_ids.empty()) {
                std::cout << "No entries found for name '" << target_name.value() << "'" << std::endl;
                return true;
            }
        }

        size_t remove_count = remove_ids.size();
        if (remove_count == 0) {
            std::cout << "Nothing to delete." << std::endl;
            return true;
        }

        std::string prompt;
        if (target_id) {
            prompt = "This will delete face ID " + std::to_string(target_id.value()) + ".";
        } else {
            prompt = "This will delete " + std::to_string(remove_count) + " face(s) for name '" + target_name.value() + "'.";
        }

        if (!confirm_action(confirm_flag, prompt)) {
            std::cout << "Deletion cancelled." << std::endl;
            return true;
        }

        std::vector<std::pair<int64_t, std::string>> keep_entries;
        keep_entries.reserve(entries.size() - remove_count);
        for (const auto& entry : entries) {
            if (remove_ids.count(entry.first) == 0) {
                keep_entries.push_back(entry);
            }
        }

        std::string faiss_path = g_config.get_faiss_index_path();

        if (keep_entries.empty()) {
            store.clear();
            if (std::filesystem::exists(faiss_path)) {
                std::filesystem::remove(faiss_path);
            }
            std::cout << "All faces removed. Database is now empty." << std::endl;
            return true;
        }

        if (!std::filesystem::exists(faiss_path)) {
            throw std::runtime_error("FAISS index not found; cannot rebuild after deletion");
        }

        std::unique_ptr<faiss::Index> raw_index(faiss::read_index(faiss_path.c_str()));
        auto* flat = dynamic_cast<faiss::IndexFlatIP*>(raw_index.get());
        if (!flat) {
            throw std::runtime_error("Expected IndexFlatIP for FAISS index");
        }

        int64_t dim = flat->d;
        std::vector<float> buffer(dim);
        std::vector<float> embeddings;
        embeddings.reserve(static_cast<size_t>(keep_entries.size()) * dim);

        for (const auto& entry : keep_entries) {
            flat->reconstruct(entry.first, buffer.data());
            embeddings.insert(embeddings.end(), buffer.begin(), buffer.end());
        }

        faiss::IndexFlatIP rebuilt(dim);
        if (!embeddings.empty()) {
            rebuilt.add(keep_entries.size(), embeddings.data());
        }
        faiss::write_index(&rebuilt, faiss_path.c_str());

        store.clear();
        int64_t new_id = 0;
        for (const auto& entry : keep_entries) {
            store.store_name(new_id++, entry.second);
        }

        std::cout << "Removed " << remove_count << " face(s). Remaining: " << keep_entries.size() << std::endl;
        std::cout << "IDs have been reindexed sequentially; run 'lxfu list' to inspect." << std::endl;
        return true;

    } catch (const std::exception& ex) {
        std::cerr << "Error while deleting faces: " << ex.what() << std::endl;
        return false;
    }
    return false;
}

bool clear_profiles(const std::vector<std::string>& args) {
    try {
        bool confirm_flag = false;
        for (const auto& arg : args) {
            if (arg == "--confirm") {
                confirm_flag = true;
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("Unknown clear option: " + arg);
            }
        }

        std::string lmdb_path = g_config.get_lmdb_path();
        std::string faiss_path = g_config.get_faiss_index_path();

        bool has_index = std::filesystem::exists(faiss_path);
        size_t entry_count = 0;
        bool has_lmdb = std::filesystem::exists(lmdb_path);

        if (has_lmdb) {
            try {
                LMDBStore store(lmdb_path, LMDBStore::Mode::ReadOnly);
                entry_count = store.size();
            } catch (...) {
                entry_count = 0;
            }
        }

        if (!has_index && entry_count == 0) {
            std::cout << "Nothing to clear." << std::endl;
            return true;
        }

        std::string prompt = "This will remove all enrolled faces";
        if (entry_count > 0) {
            prompt += " (" + std::to_string(entry_count) + " entries)";
        }
        prompt += ".";

        if (!confirm_action(confirm_flag, prompt)) {
            std::cout << "Clear cancelled." << std::endl;
            return true;
        }

        if (has_lmdb) {
            LMDBStore store(lmdb_path, LMDBStore::Mode::ReadWrite);
            store.clear();
        }

        if (has_index) {
            std::filesystem::remove(faiss_path);
        }

        std::cout << "All facial data cleared." << std::endl;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "Error while clearing database: " << ex.what() << std::endl;
        return false;
    }
    return false;
}

int main(int argc, char* argv[]) {
    try {
        // Load configuration
        g_config = load_config();
        
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }
        
        // Check for --preview flag
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
                options.name = "default";

                bool source_from_flag = false;
                bool name_from_flag = false;
                std::vector<std::string> positional;

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
                    } else {
                        positional.push_back(arg);
                    }
                }

                if (!positional.empty()) {
                    if (!source_from_flag) {
                        options.source = positional[0];
                        source_from_flag = true;
                    }
                    if (positional.size() > 1 && !name_from_flag) {
                        options.name = positional[1];
                        name_from_flag = true;
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
                options.match_all = false;

                bool source_from_flag = false;
                bool name_from_flag = false;
                std::vector<std::string> positional;

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
                    } else {
                        positional.push_back(arg);
                    }
                }

                if (!positional.empty()) {
                    if (!source_from_flag) {
                        options.source = positional[0];
                        source_from_flag = true;
                    }
                    if (positional.size() > 1 && !name_from_flag && !options.match_all) {
                        options.target_name = positional[1];
                        name_from_flag = true;
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

        } else {
            if (command == "list") {
                if (!args.empty()) {
                    std::cerr << "Error: 'list' does not accept additional arguments" << std::endl;
                    return 1;
                }
                if (!list_profiles()) {
                    return 1;
                }
            } else if (command == "delete") {
                if (!delete_profiles(args)) {
                    return 1;
                }
            } else if (command == "clear") {
                if (!clear_profiles(args)) {
                    return 1;
                }
            } else {
                std::cerr << "Error: Unknown command '" << command << "'\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
