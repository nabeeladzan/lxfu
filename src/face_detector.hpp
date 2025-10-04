#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <filesystem>

class FaceDetector {
private:
    cv::CascadeClassifier face_cascade_;
    bool initialized_;
    bool verbose_;
    
    // Try multiple possible locations for Haar cascade
    std::string find_cascade_file() {
        std::vector<std::string> possible_paths = {
            // System-wide installations
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            // Flatpak
            "/app/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            // Homebrew (macOS)
            "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            // Local development
            "./haarcascade_frontalface_default.xml",
            "../haarcascade_frontalface_default.xml",
        };
        
        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                return path;
            }
        }
        
        return "";
    }
    
public:
    explicit FaceDetector(bool verbose = true) : initialized_(false), verbose_(verbose) {
        std::string cascade_path = find_cascade_file();

        if (cascade_path.empty()) {
            if (verbose_) {
                std::cerr << "⚠ Warning: Haar cascade file not found. Face detection disabled." << std::endl;
                std::cerr << "⚠ Install opencv-data package or download haarcascade_frontalface_default.xml" << std::endl;
            }
            return;
        }

        if (!face_cascade_.load(cascade_path)) {
            if (verbose_) {
                std::cerr << "⚠ Warning: Could not load Haar cascade from: " << cascade_path << std::endl;
                std::cerr << "⚠ Face detection disabled." << std::endl;
            }
            return;
        }

        initialized_ = true;
        if (verbose_) {
            std::cout << "✓ Face detector initialized using: " << cascade_path << std::endl;
        }
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    // Detect faces and return the largest one (assumed to be the primary face)
    cv::Rect detect_largest_face(const cv::Mat& image) {
        if (!initialized_) {
            // Return full image rect if detector not available
            return cv::Rect(0, 0, image.cols, image.rows);
        }
        
        // Convert to grayscale for detection
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        // Equalize histogram for better detection
        cv::equalizeHist(gray, gray);
        
        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(
            gray,
            faces,
            1.1,    // scaleFactor
            3,      // minNeighbors
            0,      // flags
            cv::Size(30, 30)  // minSize
        );
        
        if (faces.empty()) {
            if (verbose_) {
                std::cout << "⚠ No face detected, using full image" << std::endl;
            }
            return cv::Rect(0, 0, image.cols, image.rows);
        }
        
        // Find largest face
        cv::Rect largest_face = faces[0];
        for (const auto& face : faces) {
            if (face.area() > largest_face.area()) {
                largest_face = face;
            }
        }
        
        if (verbose_) {
            std::cout << "✓ Face detected at (" << largest_face.x << ", " << largest_face.y 
                      << ") size " << largest_face.width << "x" << largest_face.height << std::endl;
        }

        if (verbose_ && faces.size() > 1) {
            std::cout << "  Note: " << faces.size() << " faces detected, using largest" << std::endl;
        }
        
        return largest_face;
    }
    
    // Crop image to face region with optional padding
    cv::Mat crop_to_face(const cv::Mat& image, float padding = 0.2f) {
        cv::Rect face_rect = detect_largest_face(image);
        
        // If no face detected or detector unavailable, return original
        if (face_rect.width == image.cols && face_rect.height == image.rows) {
            return image.clone();
        }
        
        // Add padding around face
        int pad_x = static_cast<int>(face_rect.width * padding);
        int pad_y = static_cast<int>(face_rect.height * padding);
        
        face_rect.x = std::max(0, face_rect.x - pad_x);
        face_rect.y = std::max(0, face_rect.y - pad_y);
        face_rect.width = std::min(image.cols - face_rect.x, face_rect.width + 2 * pad_x);
        face_rect.height = std::min(image.rows - face_rect.y, face_rect.height + 2 * pad_y);
        
        // Crop to face region
        cv::Mat cropped = image(face_rect).clone();
        
        if (verbose_) {
            std::cout << "✓ Cropped to face region: " << cropped.cols << "x" << cropped.rows 
                      << " (from " << image.cols << "x" << image.rows << ")" << std::endl;
        }
        
        return cropped;
    }
    
    // Draw detected faces on image (for preview mode)
    void draw_faces(cv::Mat& image) {
        if (!initialized_) {
            return;
        }
        
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        cv::equalizeHist(gray, gray);
        
        std::vector<cv::Rect> faces;
        face_cascade_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        
        // Draw rectangles around faces
        for (const auto& face : faces) {
            cv::rectangle(image, face, cv::Scalar(0, 255, 0), 2);
            
            // Draw padding preview
            float padding = 0.2f;
            int pad_x = static_cast<int>(face.width * padding);
            int pad_y = static_cast<int>(face.height * padding);
            
            cv::Rect padded_face(
                std::max(0, face.x - pad_x),
                std::max(0, face.y - pad_y),
                std::min(image.cols - face.x + pad_x, face.width + 2 * pad_x),
                std::min(image.rows - face.y + pad_y, face.height + 2 * pad_y)
            );
            
            cv::rectangle(image, padded_face, cv::Scalar(255, 255, 0), 1, cv::LINE_DASHED);
        }
        
        // Draw face count
        if (!faces.empty()) {
            std::string text = "Faces: " + std::to_string(faces.size());
            cv::putText(image, text, cv::Point(10, image.rows - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
    }
};
