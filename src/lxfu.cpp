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

namespace fs = std::filesystem;

// Global config and face detector
Config g_config;
FaceDetector g_face_detector;

void print_usage(const char* program_name) {
    std::cout << "LXFU - Linux Face Utility\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " [--preview] enroll [--device PATH|--file PATH] [--name NAME]\n";
    std::cout << "  " << program_name << " [--preview] query [--device PATH|--file PATH] [--name NAME|--all]\n\n";
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

cv::Mat load_image_or_capture(const std::string& source, bool show_preview = false) {
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
        
        // Load or capture image
        std::cout << "Loading/capturing face..." << std::endl;
        cv::Mat image = load_image_or_capture(opts.source, opts.show_preview);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        // Detect and crop to face
        cv::Mat face_image = g_face_detector.crop_to_face(image);
        
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
        
        // Load or capture image
        std::cout << "Loading/capturing face..." << std::endl;
        cv::Mat image = load_image_or_capture(opts.source, opts.show_preview);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        // Detect and crop to face
        cv::Mat face_image = g_face_detector.crop_to_face(image);
        
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

        bool require_specific = !opts.match_all;
        std::string desired_name = opts.target_name.value_or("default");

        std::optional<FaceEngine::SearchResult> chosen;
        std::string matched_name;

        for (const auto& candidate : candidates) {
            std::string stored = store.get_name(candidate.id);
            if (stored.empty()) {
                continue;
            }
            if (require_specific && stored != desired_name) {
                continue;
            }
            chosen = candidate;
            matched_name = stored;
            break;
        }

        if (!chosen) {
            if (require_specific) {
                std::cout << "\n⚠ No match found for name '" << desired_name << "'" << std::endl;
            } else {
                std::cout << "\n⚠ No match found" << std::endl;
            }
            return;
        }

        float similarity_percentage = chosen->similarity * 100.0f;

        std::cout << "\n✓ Face recognized!" << std::endl;
        if (opts.match_all) {
            std::cout << "  Name: " << matched_name << std::endl;
        } else {
            std::cout << "  Name: " << matched_name << " (requested)" << std::endl;
        }
        std::cout << "  Similarity: " << std::fixed << std::setprecision(2)
                  << similarity_percentage << "%" << std::endl;
        std::cout << "  Face ID: " << chosen->id << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during query: " << e.what() << std::endl;
        throw;
    }
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
