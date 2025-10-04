#include "face_engine.hpp"

#include <iostream>
#include <filesystem>

int main(int argc, const char* argv[]) {
    try {
        const std::string model_path = (argc > 1) ? argv[1] : "dino.pt";
        const std::string image_path = (argc > 2) ? argv[2] : "";

        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found at: " << model_path << '\n';
            std::cerr << "Usage: " << argv[0] << " [model.pt] [image.jpg]" << '\n';
            return 1;
        }

        FaceEngine engine(model_path);

        cv::Mat image;
        if (!image_path.empty()) {
            image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_path << '\n';
                return 1;
            }
            std::cout << "Loaded image: " << image.cols << "x" << image.rows << '\n';
        } else {
            std::cout << "No image provided; using a random tensor." << std::endl;
            image = cv::Mat(224, 224, CV_8UC3);
            cv::randu(image, 0, 255);
        }

        auto embedding = engine.extract_embedding(image);
        std::cout << "Embedding dimension: " << embedding.size() << '\n';
        std::cout << "First 5 values:";
        for (size_t i = 0; i < std::min<std::size_t>(5, embedding.size()); ++i) {
            std::cout << ' ' << embedding[i];
        }
        std::cout << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}

