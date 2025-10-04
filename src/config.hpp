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
    Config() {
        // Set defaults
        values_["model_path"] = "/usr/share/lxfu/dino.pt";
        values_["db_path"] = "~/.lxfu";
        values_["default_device"] = "/dev/video0";
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
    
    std::string get_faiss_index_path() const {
        return expand_path(get("db_path")) + "/lxfu_faces.index";
    }
    
    std::string get_lmdb_path() const {
        return expand_path(get("db_path")) + "/lxfu_metadata.db";
    }
};

// Load configuration from standard locations
inline Config load_config(bool verbose = true) {
    Config config;

    // Try to load from system config
    if (config.load_from_file("/etc/lxfu/lxfu.conf")) {
        if (verbose) {
            std::cout << "Loaded config from /etc/lxfu/lxfu.conf" << std::endl;
        }
    }
    // Try user config (overrides system config)
    else if (config.load_from_file(std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.config/lxfu/lxfu.conf")) {
        if (verbose) {
            std::cout << "Loaded config from ~/.config/lxfu/lxfu.conf" << std::endl;
        }
    }
    // Try local config (development)
    else if (config.load_from_file("lxfu.conf")) {
        if (verbose) {
            std::cout << "Loaded config from ./lxfu.conf" << std::endl;
        }
    }

    // Ensure database directory exists
    config.ensure_db_directory();

    return config;
}
