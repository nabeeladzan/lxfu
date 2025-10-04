#pragma once

#include <lmdb.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <filesystem>
#include <utility>
#include <cstdint>

class LMDBStore {
public:
    enum class Mode { ReadWrite, ReadOnly };

private:
    MDB_env* env_;
    MDB_dbi dbi_;
    std::string db_path_;
    Mode mode_;

    static std::vector<std::uint8_t> serialize_embedding(const std::vector<float>& embedding) {
        const std::int32_t dim = static_cast<std::int32_t>(embedding.size());
        std::vector<std::uint8_t> buffer(sizeof(dim) + embedding.size() * sizeof(float));
        std::memcpy(buffer.data(), &dim, sizeof(dim));
        std::memcpy(buffer.data() + sizeof(dim), embedding.data(), embedding.size() * sizeof(float));
        return buffer;
    }

    static std::vector<float> deserialize_embedding(const MDB_val& value) {
        if (value.mv_size < sizeof(std::int32_t)) {
            throw std::runtime_error("LMDB value too small to contain embedding");
        }
        std::int32_t dim = 0;
        std::memcpy(&dim, value.mv_data, sizeof(dim));
        if (dim <= 0) {
            throw std::runtime_error("Invalid embedding dimension stored in LMDB");
        }
        const std::size_t expected = sizeof(dim) + static_cast<std::size_t>(dim) * sizeof(float);
        if (value.mv_size != expected) {
            throw std::runtime_error("LMDB embedding payload size mismatch");
        }
        std::vector<float> embedding(static_cast<std::size_t>(dim));
        std::memcpy(embedding.data(), static_cast<std::uint8_t*>(value.mv_data) + sizeof(dim), embedding.size() * sizeof(float));
        return embedding;
    }

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

        mdb_env_set_mapsize(env_, 1ULL * 1024 * 1024 * 1024); // 1GB map size

        unsigned int flags = (mode_ == Mode::ReadOnly) ? MDB_RDONLY : 0;
        rc = mdb_env_open(env_, db_path.c_str(), flags, 0664);
        if (rc != 0) {
            throw std::runtime_error("Failed to open LMDB environment: " + std::string(mdb_strerror(rc)));
        }

        MDB_txn* txn = nullptr;
        rc = mdb_txn_begin(env_, nullptr, (mode_ == Mode::ReadOnly) ? MDB_RDONLY : 0, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }

        unsigned int db_flags = (mode_ == Mode::ReadWrite) ? MDB_CREATE : 0;
        rc = mdb_dbi_open(txn, nullptr, db_flags, &dbi_);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to open LMDB database: " + std::string(mdb_strerror(rc)));
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

    void store_embedding(const std::string& name, const std::vector<float>& embedding) {
        if (mode_ == Mode::ReadOnly) {
            throw std::runtime_error("Attempted to write to LMDB opened read-only");
        }
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }

        MDB_val key;
        key.mv_size = name.size();
        key.mv_data = const_cast<char*>(name.data());

        auto buffer = serialize_embedding(embedding);
        MDB_val val;
        val.mv_size = buffer.size();
        val.mv_data = buffer.data();

        rc = mdb_put(txn, dbi_, &key, &val, 0);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to store embedding: " + std::string(mdb_strerror(rc)));
        }

        mdb_txn_commit(txn);
    }

    std::vector<std::pair<std::string, std::vector<float>>> get_all_embeddings() const {
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin read transaction: " + std::string(mdb_strerror(rc)));
        }

        MDB_cursor* cursor = nullptr;
        rc = mdb_cursor_open(txn, dbi_, &cursor);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to open LMDB cursor: " + std::string(mdb_strerror(rc)));
        }

        std::vector<std::pair<std::string, std::vector<float>>> entries;
        MDB_val key, val;
        rc = mdb_cursor_get(cursor, &key, &val, MDB_FIRST);
        while (rc == 0) {
            std::string name(static_cast<char*>(key.mv_data), key.mv_size);
            entries.emplace_back(name, deserialize_embedding(val));
            rc = mdb_cursor_get(cursor, &key, &val, MDB_NEXT);
        }

        if (rc != MDB_NOTFOUND) {
            mdb_cursor_close(cursor);
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed while iterating LMDB: " + std::string(mdb_strerror(rc)));
        }

        mdb_cursor_close(cursor);
        mdb_txn_abort(txn);
        return entries;
    }

    bool delete_embedding(const std::string& name) {
        if (mode_ == Mode::ReadOnly) {
            throw std::runtime_error("Attempted to delete from LMDB opened read-only");
        }
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }

        MDB_val key;
        key.mv_size = name.size();
        key.mv_data = const_cast<char*>(name.data());

        rc = mdb_del(txn, dbi_, &key, nullptr);
        if (rc == MDB_NOTFOUND) {
            mdb_txn_abort(txn);
            return false;
        }
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to delete embedding: " + std::string(mdb_strerror(rc)));
        }

        mdb_txn_commit(txn);
        return true;
    }

    void clear() {
        if (mode_ == Mode::ReadOnly) {
            throw std::runtime_error("Attempted to clear LMDB opened read-only");
        }
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin transaction: " + std::string(mdb_strerror(rc)));
        }

        rc = mdb_drop(txn, dbi_, 0);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to drop LMDB database: " + std::string(mdb_strerror(rc)));
        }

        mdb_txn_commit(txn);
    }

    std::size_t size() const {
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) {
            return 0;
        }
        MDB_stat stat{};
        rc = mdb_stat(txn, dbi_, &stat);
        mdb_txn_abort(txn);
        if (rc != 0) {
            return 0;
        }
        return static_cast<std::size_t>(stat.ms_entries);
    }
};

