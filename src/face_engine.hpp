#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>
#include <cmath>
#include <algorithm>

class FaceEngine {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    int feature_dim_;
    bool verbose_;
    
    // Helper to preprocess image
    torch::Tensor preprocess_image(const cv::Mat& image) {
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        const int target_size = 224;
        const float crop_pct = 0.875f;
        const int resize_size = static_cast<int>(std::round(target_size / crop_pct));

        int resize_width = 0;
        int resize_height = 0;
        if (rgb_image.cols >= rgb_image.rows) {
            resize_height = resize_size;
            resize_width = static_cast<int>(std::round(
                resize_size * static_cast<float>(rgb_image.cols) / static_cast<float>(rgb_image.rows)));
        } else {
            resize_width = resize_size;
            resize_height = static_cast<int>(std::round(
                resize_size * static_cast<float>(rgb_image.rows) / static_cast<float>(rgb_image.cols)));
        }

        resize_width = std::max(resize_width, target_size);
        resize_height = std::max(resize_height, target_size);

        cv::Mat resized;
        cv::resize(rgb_image, resized, cv::Size(resize_width, resize_height), 0.0, 0.0, cv::INTER_CUBIC);

        const int raw_x0 = (resized.cols - target_size) / 2;
        const int raw_y0 = (resized.rows - target_size) / 2;
        const int max_x = std::max(0, resized.cols - target_size);
        const int max_y = std::max(0, resized.rows - target_size);
        const int x0 = std::clamp(raw_x0, 0, max_x);
        const int y0 = std::clamp(raw_y0, 0, max_y);
        const int crop_width = std::min(target_size, resized.cols);
        const int crop_height = std::min(target_size, resized.rows);
        cv::Rect crop{x0, y0, crop_width, crop_height};
        cv::Mat cropped = resized(crop).clone();
        if (cropped.cols != target_size || cropped.rows != target_size) {
            cv::resize(cropped, cropped, cv::Size(target_size, target_size), 0.0, 0.0, cv::INTER_CUBIC);
        }

        // Convert to float and normalize to [0, 1]
        cv::Mat float_image;
        cropped.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
        
        // ImageNet normalization
        const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        const std::vector<float> std = {0.229f, 0.224f, 0.225f};
        
        std::vector<cv::Mat> channels(3);
        cv::split(float_image, channels);
        
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean[i]) / std[i];
        }
        
        cv::merge(channels, float_image);
        
        // Convert to tensor: HWC -> CHW
        torch::Tensor tensor = torch::from_blob(
            float_image.data,
            {float_image.rows, float_image.cols, 3},
            torch::kFloat32
        ).clone();
        
        tensor = tensor.permute({2, 0, 1}).unsqueeze(0);
        return tensor.to(device_);
    }
    
public:
    FaceEngine(const std::string& model_path, bool verbose = true) 
        : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          feature_dim_(0),
          verbose_(verbose) {

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file not found: " + model_path);
        }

        if (verbose_) {
            std::cout << "Loading DINOv2 model on " 
                      << (torch::cuda::is_available() ? "CUDA" : "CPU") << "..." << std::endl;
        }
        
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
    }
    
    // Extract embedding from image
    std::vector<float> extract_embedding(const cv::Mat& image) {
        torch::Tensor input = preprocess_image(image);
        
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(std::move(input));
        
        torch::Tensor output = model_.forward(inputs).toTensor();
        output = output.cpu();
        
        // Flatten if needed
        if (output.dim() > 2) {
            output = output.flatten(1);
        }
        
        // L2 normalize for cosine similarity
        output = torch::nn::functional::normalize(output, 
            torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        
        // Convert to vector
        float* data = output.data_ptr<float>();
        int64_t size = output.size(1);
        feature_dim_ = static_cast<int>(size);

        return std::vector<float>(data, data + size);
    }

    int embedding_dim() const { return feature_dim_; }
};
