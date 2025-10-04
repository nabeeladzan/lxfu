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
    using Embedding = std::vector<float>;
    using EmbeddingList = std::vector<Embedding>;

private:
    MDB_env* env_;
    MDB_dbi dbi_;
    std::string db_path_;
    Mode mode_;

    static std::vector<std::uint8_t> serialize_embeddings(const EmbeddingList& embeddings) {
        const std::int32_t count = static_cast<std::int32_t>(embeddings.size());
        const std::int32_t dim = count > 0 ? static_cast<std::int32_t>(embeddings.front().size()) : 0;

        std::size_t payload = sizeof(count) + sizeof(dim);
        payload += static_cast<std::size_t>(count) * static_cast<std::size_t>(dim) * sizeof(float);

        std::vector<std::uint8_t> buffer(payload);
        std::size_t offset = 0;

        std::memcpy(buffer.data() + offset, &count, sizeof(count));
        offset += sizeof(count);
        std::memcpy(buffer.data() + offset, &dim, sizeof(dim));
        offset += sizeof(dim);

        for (const auto& embedding : embeddings) {
            if (static_cast<std::int32_t>(embedding.size()) != dim) {
                throw std::runtime_error("Inconsistent embedding dimension for serialization");
            }
            std::memcpy(buffer.data() + offset, embedding.data(), embedding.size() * sizeof(float));
            offset += embedding.size() * sizeof(float);
        }

        return buffer;
    }

    static EmbeddingList deserialize_embeddings(const MDB_val& value) {
        if (value.mv_size < sizeof(std::int32_t)) {
            throw std::runtime_error("LMDB value too small to contain embedding metadata");
        }

        const std::uint8_t* data = static_cast<const std::uint8_t*>(value.mv_data);
        std::int32_t first = 0;
        std::memcpy(&first, data, sizeof(first));

        if (value.mv_size >= 2 * sizeof(std::int32_t)) {
            std::int32_t second = 0;
            std::memcpy(&second, data + sizeof(std::int32_t), sizeof(second));
            if (first > 0 && second > 0) {
                std::size_t expected = sizeof(std::int32_t) * 2 +
                    static_cast<std::size_t>(first) * static_cast<std::size_t>(second) * sizeof(float);
                if (expected == value.mv_size) {
                    EmbeddingList embeddings(static_cast<std::size_t>(first), Embedding(static_cast<std::size_t>(second)));
                    const float* payload = reinterpret_cast<const float*>(data + sizeof(std::int32_t) * 2);
                    for (std::int32_t i = 0; i < first; ++i) {
                        std::copy(payload + i * second, payload + (i + 1) * second, embeddings[static_cast<std::size_t>(i)].begin());
                    }
                    return embeddings;
                }
            }
        }

        // Legacy format (single embedding)
        std::int32_t dim = first;
        if (dim <= 0) {
            throw std::runtime_error("Invalid embedding dimension stored in legacy LMDB entry");
        }
        std::size_t expected_old = sizeof(std::int32_t) + static_cast<std::size_t>(dim) * sizeof(float);
        if (expected_old != value.mv_size) {
            throw std::runtime_error("LMDB embedding payload size mismatch");
        }
        EmbeddingList embeddings;
        embeddings.emplace_back(static_cast<std::size_t>(dim));
        const float* payload = reinterpret_cast<const float*>(data + sizeof(std::int32_t));
        std::copy(payload, payload + dim, embeddings.front().begin());
        return embeddings;
    }

    EmbeddingList fetch_embeddings_for_key(MDB_txn* txn, MDB_val& key) const {
        EmbeddingList embeddings;
        MDB_val existing_value;
        int rc = mdb_get(txn, dbi_, &key, &existing_value);
        if (rc == MDB_NOTFOUND) {
            return embeddings;
        }
        if (rc != 0) {
            throw std::runtime_error("Failed to read existing embedding: " + std::string(mdb_strerror(rc)));
        }
        return deserialize_embeddings(existing_value);
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

    std::size_t store_embedding(const std::string& name, const Embedding& embedding) {
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

        EmbeddingList embeddings = fetch_embeddings_for_key(txn, key);
        if (!embeddings.empty() && embeddings.front().size() != embedding.size()) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Embedding dimension mismatch while appending to existing profile");
        }
        embeddings.push_back(embedding);

        auto buffer = serialize_embeddings(embeddings);
        MDB_val val;
        val.mv_size = buffer.size();
        val.mv_data = buffer.data();

        rc = mdb_put(txn, dbi_, &key, &val, 0);
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to store embedding: " + std::string(mdb_strerror(rc)));
        }

        mdb_txn_commit(txn);
        return embeddings.size();
    }

    std::vector<std::pair<std::string, EmbeddingList>> get_all_embeddings() const {
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

        std::vector<std::pair<std::string, EmbeddingList>> entries;
        MDB_val key, val;
        rc = mdb_cursor_get(cursor, &key, &val, MDB_FIRST);
        while (rc == 0) {
            std::string name(static_cast<char*>(key.mv_data), key.mv_size);
            entries.emplace_back(name, deserialize_embeddings(val));
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

    EmbeddingList get_embeddings(const std::string& name) const {
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc != 0) {
            throw std::runtime_error("Failed to begin read transaction: " + std::string(mdb_strerror(rc)));
        }

        MDB_val key;
        key.mv_size = name.size();
        key.mv_data = const_cast<char*>(name.data());

        MDB_val value;
        rc = mdb_get(txn, dbi_, &key, &value);
        if (rc == MDB_NOTFOUND) {
            mdb_txn_abort(txn);
            return {};
        }
        if (rc != 0) {
            mdb_txn_abort(txn);
            throw std::runtime_error("Failed to read embeddings: " + std::string(mdb_strerror(rc)));
        }

        EmbeddingList result = deserialize_embeddings(value);
        mdb_txn_abort(txn);
        return result;
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
