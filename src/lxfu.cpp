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
    std::cout << "  " << program_name << " clear [--confirm]\n\n";
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

}

cv::Mat capture_from_device(const std::string& device_path, bool show_preview = false) {
    int device_id = 0;
    if (device_path.rfind("/dev/video", 0) == 0) {
        device_id = std::stoi(device_path.substr(10));
    }

    cv::VideoCapture cap(device_id);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open device: " + device_path);
    }

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
            cap >> current_frame;
            if (current_frame.empty()) {
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
        cap >> frame;
        if (frame.empty()) {
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

        std::cout << "Loading/capturing face..." << std::endl;
        cv::Mat image = load_image_or_capture(opts.source, opts.show_preview);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;

        cv::Mat face_image = g_face_detector.crop_to_face(image);

        std::cout << "Extracting face embedding..." << std::endl;
        std::vector<float> embedding = engine.extract_embedding(face_image);
        std::cout << "Embedding extracted: " << embedding.size() << " dimensions" << std::endl;

        std::string lmdb_path = g_config.get_embeddings_path();
        LMDBStore store(lmdb_path);
        store.store_embedding(opts.name, embedding);

        std::cout << "\n✓ Enrollment successful!" << std::endl;
        std::cout << "  Name: " << opts.name << std::endl;
        std::cout << "  Embedding dimensions: " << embedding.size() << std::endl;
        std::cout << "  Total profiles: " << store.size() << std::endl;

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

        cv::Mat face_image = g_face_detector.crop_to_face(image);

        std::cout << "Extracting face embedding..." << std::endl;
        std::vector<float> embedding = engine.extract_embedding(face_image);

        std::string lmdb_path = g_config.get_embeddings_path();
        LMDBStore store(lmdb_path, LMDBStore::Mode::ReadOnly);
        auto entries = store.get_all_embeddings();

        if (entries.empty()) {
            std::cout << "\n⚠ No profiles enrolled yet." << std::endl;
            return;
        }

        bool require_specific = !opts.match_all;
        std::string desired = opts.target_name.value_or("default");

        float best_similarity = -1.0f;
        std::string best_name;

        for (const auto& [name, stored_embedding] : entries) {
            if (require_specific && name != desired) {
                continue;
            }
            if (stored_embedding.size() != embedding.size()) {
                continue;
            }
            float similarity = std::inner_product(embedding.begin(), embedding.end(),
                                                  stored_embedding.begin(), 0.0f);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_name = name;
            }
        }

        if (best_similarity < 0.0f) {
            if (require_specific) {
                std::cout << "\n⚠ No match found for name '" << desired << "'" << std::endl;
            } else {
                std::cout << "\n⚠ No match found" << std::endl;
            }
            return;
        }

        std::cout << "\n✓ Face recognized!" << std::endl;
        if (require_specific) {
            std::cout << "  Name: " << best_name << " (requested)" << std::endl;
        } else {
            std::cout << "  Name: " << best_name << std::endl;
        }
        std::cout << "  Similarity: " << std::fixed << std::setprecision(2)
                  << (best_similarity * 100.0f) << "%" << std::endl;

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

        std::cout << std::left << std::setw(24) << "Name" << "Dimensions" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        for (const auto& [name, embedding] : entries) {
            std::cout << std::left << std::setw(24) << (name.empty() ? "<unnamed>" : name)
                      << embedding.size() << std::endl;
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
