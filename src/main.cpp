#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// Helper function to convert OpenCV Mat to PyTorch Tensor
torch::Tensor mat_to_tensor(const cv::Mat& image, const torch::Device& device) {
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // Resize to 224x224 (DINOv3 input size)
    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(224, 224));
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    
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
    
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0); // CHW and add batch dimension
    return tensor.to(device);
}

int main(int argc, const char* argv[]) {
    try {
        const std::string default_model{"dino.pt"};
        const std::string model_path = (argc > 1) ? argv[1] : default_model;
        const std::string image_path = (argc > 2) ? argv[2] : "";

        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found at: " << model_path << '\n';
            std::cerr << "Usage: " << argv[0] << " [model.pt] [image.jpg]" << '\n';
            return 1;
        }

        const bool cuda_available = torch::cuda::is_available();
        const torch::Device device = cuda_available ? torch::kCUDA : torch::kCPU;
        std::cout << "Loading model on " << (cuda_available ? "CUDA" : "CPU") << "..." << '\n';

        torch::jit::script::Module module = torch::jit::load(model_path, device);
        module.eval();

        torch::Tensor input;
        
        if (!image_path.empty() && std::filesystem::exists(image_path)) {
            std::cout << "Loading image: " << image_path << '\n';
            cv::Mat image = cv::imread(image_path);
            
            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_path << '\n';
                return 1;
            }
            
            std::cout << "Image size: " << image.cols << "x" << image.rows << '\n';
            input = mat_to_tensor(image, device);
        } else {
            std::cout << "No image provided. Using random tensor..." << '\n';
            input = torch::rand({1, 3, 224, 224}, torch::TensorOptions{}.device(device).dtype(torch::kFloat32));
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(std::move(input));

        std::cout << "Running inference..." << '\n';
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << "Output tensor shape: " << output.sizes() << '\n';
        
        // Convert PyTorch tensor to FAISS-compatible format
        output = output.cpu(); // Ensure it's on CPU
        
        // Flatten if needed and get dimensions
        if (output.dim() > 2) {
            output = output.flatten(1); // Keep batch dim, flatten rest
        }
        
        const int64_t batch_size = output.size(0);
        const int64_t feature_dim = output.size(1);
        
        std::cout << "Feature dimension: " << feature_dim << '\n';
        
        // Normalize embeddings for cosine similarity (L2 normalize)
        // Cosine similarity is better for image embeddings
        output = torch::nn::functional::normalize(output, 
            torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        
        std::cout << "Embeddings L2-normalized for cosine similarity" << '\n';
        
        // Create a FAISS index with Inner Product (equivalent to cosine similarity on normalized vectors)
        // IndexFlatIP = Inner Product index (cosine similarity when vectors are normalized)
        std::unique_ptr<faiss::IndexFlatIP> index(new faiss::IndexFlatIP(feature_dim));
        
        std::cout << "FAISS index created (Inner Product / Cosine Similarity)" << '\n';
        std::cout << "Is trained: " << index->is_trained << '\n';
        
        // Get raw pointer to tensor data
        float* embedding_data = output.data_ptr<float>();
        
        // Add the embedding to the FAISS index
        index->add(batch_size, embedding_data);
        
        std::cout << "Added " << index->ntotal << " vector(s) to FAISS index" << '\n';
        
        // Example: Search for the k nearest neighbors (self-search as demo)
        const int k = 1; // Number of nearest neighbors
        std::vector<faiss::idx_t> labels(k * batch_size);
        std::vector<float> similarities(k * batch_size);
        
        index->search(batch_size, embedding_data, k, similarities.data(), labels.data());
        
        std::cout << "\nNearest neighbor search results (cosine similarity):" << '\n';
        for (int i = 0; i < batch_size; i++) {
            std::cout << "Query " << i << " nearest neighbor: ID=" << labels[i] 
                      << ", similarity=" << similarities[i] << '\n';
        }
        
        // Save the index to disk in current working directory
        const std::string index_path = "dinov3_features.index";
        std::filesystem::path abs_index_path = std::filesystem::absolute(index_path);
        
        faiss::write_index(index.get(), index_path.c_str());
        std::cout << "\nFAISS index saved to: " << abs_index_path << '\n';
        std::cout << "Index contains " << index->ntotal << " vectors of dimension " << feature_dim << '\n';

        return 0;
    } catch (const c10::Error& err) {
        std::cerr << "PyTorch error: " << err.what() << '\n';
        return 1;
    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << '\n';
        return 1;
    }
}
