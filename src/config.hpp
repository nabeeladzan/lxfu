#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>
#include <iostream>
#include <cstdlib>

class Config {
private:
    std::map<std::string, std::string> values_;
    std::string config_source_;
    
    std::string expand_path(const std::string& path) const {
        if (path.empty()) return path;
        
        // Expand ~ to home directory
        if (path[0] == '~') {
            const char* home = std::getenv("HOME");
            if (home) {
                return std::string(home) + path.substr(1);
            }
        }
        return path;
    }
    
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }
    
public:
    Config() : config_source_("built-in defaults") {
        // Set defaults
        values_["model_path"] = "/usr/share/lxfu/dino.pt";
        values_["db_path"] = "~/.lxfu";
        values_["default_device"] = "/dev/video0";
        values_["threshold"] = "0.75";
    }
    
    bool load_from_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            // Parse key=value
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                values_[key] = value;
            }
        }
        
        return true;
    }
    
    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return expand_path(it->second);
        }
        return default_value;
    }
    
    void set(const std::string& key, const std::string& value) {
        values_[key] = value;
    }

    void ensure_db_directory() {
        std::string db_path = get("db_path");
        std::filesystem::create_directories(db_path);
    }

    std::string get_embeddings_path() const {
        return expand_path(get("db_path")) + "/embeddings";
    }

    float get_threshold(float default_value = 0.75f) const {
        try {
            return std::stof(get("threshold", std::to_string(default_value)));
        } catch (const std::exception&) {
            return default_value;
        }
    }

    std::string get_config_source() const {
        return config_source_;
    }

    void set_config_source(const std::string& source) {
        config_source_ = source;
    }

    const std::map<std::string, std::string>& get_all_values() const {
        return values_;
    }

    void print_config() const {
        std::cout << "Configuration source: " << config_source_ << std::endl;
        std::cout << "\nCurrent settings:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        // Print in a specific order for better readability
        std::vector<std::string> ordered_keys = {
            "model_path", "db_path", "default_device", "threshold"
        };
        
        for (const auto& key : ordered_keys) {
            auto it = values_.find(key);
            if (it != values_.end()) {
                std::string expanded = expand_path(it->second);
                if (expanded != it->second) {
                    std::cout << "  " << key << " = " << it->second 
                              << " (" << expanded << ")" << std::endl;
                } else {
                    std::cout << "  " << key << " = " << it->second << std::endl;
                }
            }
        }
        
        // Print any additional keys not in the ordered list
        for (const auto& [key, value] : values_) {
            if (std::find(ordered_keys.begin(), ordered_keys.end(), key) == ordered_keys.end()) {
                std::string expanded = expand_path(value);
                if (expanded != value) {
                    std::cout << "  " << key << " = " << value 
                              << " (" << expanded << ")" << std::endl;
                } else {
                    std::cout << "  " << key << " = " << value << std::endl;
                }
            }
        }
        
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "\nEmbeddings path: " << get_embeddings_path() << std::endl;
    }
};

// Load configuration from standard locations
inline Config load_config(bool verbose = true) {
    Config config;

    // Try to load from system config (standard location)
    if (config.load_from_file("/etc/lxfu/lxfu.conf")) {
        config.set_config_source("/etc/lxfu/lxfu.conf");
        if (verbose) {
            std::cout << "Loaded config from /etc/lxfu/lxfu.conf" << std::endl;
        }
    }
    // Try local config (development)
    else if (config.load_from_file("lxfu.conf")) {
        config.set_config_source("./lxfu.conf");
        if (verbose) {
            std::cout << "Loaded config from ./lxfu.conf" << std::endl;
        }
    }

    // Ensure database directory exists
    config.ensure_db_directory();

    return config;
}
