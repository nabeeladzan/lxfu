#pragma once

#include <lmdb.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <filesystem>

class LMDBStore {
public:
    enum class Mode { ReadWrite, ReadOnly };

private:
    MDB_env* env_;
    MDB_dbi dbi_;
    std::string db_path_;
    Mode mode_;

public:
    LMDBStore(const std::string& db_path, Mode mode = Mode::ReadWrite)
        : env_(nullptr), dbi_(0), db_path_(db_path), mode_(mode) {
        if (mode_ == Mode::ReadWrite) {
            std::filesystem::create_directories(db_path);
        } else if (!std::filesystem::exists(db_path)) {
            throw std::runtime_error("LMDB directory not found: " + db_path);
        }

        int rc = mdb_env_create(&env_);
        if (rc != 0) {
            throw std::runtime_error("Failed to create LMDB environment: " + std::string(mdb_strerror(rc)));
        }

        // Set map size (10GB should be enough for face embeddings)
        mdb_env_set_mapsize(env_, 10UL * 1024UL * 1024UL * 1024UL);

        unsigned int flags = (mode_ == Mode::ReadOnly) ? MDB_RDONLY : 0;
        rc = mdb_env_open(env_, db_path.c_str(), flags, 0664);
        if (rc != 0) {
            throw std::runtime_error("Failed to open LMDB environment: " + std::string(mdb_strerror(rc)));
        }

        MDB_txn* txn;
        rc = mdb_txn_begin(env_, nullptr, (mode_ == Mode::ReadOnly) ? MDB_RDONLY : 0, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }

        unsigned int db_flags = (mode_ == Mode::ReadWrite) ? MDB_CREATE : 0;
        rc = mdb_dbi_open(txn, nullptr, db_flags, &dbi_);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to open database: " + std::string(mdb_strerror(rc)));
        }

        if (mode_ == Mode::ReadOnly) {
            mdb_txn_abort(txn);
        } else {
            mdb_txn_commit(txn);
        }
    }
    
    ~LMDBStore() {
        if (env_) {
            mdb_dbi_close(env_, dbi_);
            mdb_env_close(env_);
        }
    }
    
    // Store a name with an ID
    void store_name(int64_t id, const std::string& name) {
        if (mode_ == Mode::ReadOnly) {
            throw std::runtime_error("LMDBStore opened in read-only mode cannot store entries");
        }
        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }
        
        MDB_val key, val;
        key.mv_size = sizeof(id);
        key.mv_data = &id;
        val.mv_size = name.size();
        val.mv_data = (void*)name.c_str();
        
        rc = mdb_put(txn, dbi_, &key, &val, 0);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to store name: " + std::string(mdb_strerror(rc)));
        }
        
        mdb_txn_commit(txn);
    }
    
    // Retrieve a name by ID
    std::string get_name(int64_t id) {
        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }
        
        MDB_val key, val;
        key.mv_size = sizeof(id);
        key.mv_data = &id;
        
        rc = mdb_get(txn, dbi_, &key, &val);
        if (rc == MDB_NOTFOUND) {
            mdb_txn_abort(txn);
            return "";
        }
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to retrieve name: " + std::string(mdb_strerror(rc)));
        }
        
        std::string name((char*)val.mv_data, val.mv_size);
        mdb_txn_abort(txn);
        return name;
    }
    
    // Get total number of entries
    size_t size() {
        MDB_txn* txn;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) {
            return 0;
        }
        
        MDB_stat stat;
        rc = mdb_stat(txn, dbi_, &stat);
        mdb_txn_abort(txn);
        
        if (rc != 0) {
            return 0;
        }
        
        return stat.ms_entries;
    }
};
