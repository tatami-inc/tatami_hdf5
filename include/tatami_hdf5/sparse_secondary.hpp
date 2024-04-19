#ifndef TATAMI_HDF5_SPARSE_SECONDARY_HPP
#define TATAMI_HDF5_SPARSE_SECONDARY_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include "sparse_primary.hpp"
#include "serialize.hpp"
#include "utils.hpp"

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

// We don't use CachedIndex_, as this may be too small to store the other
// dimension; use Index_ instead as this is guaranteed.
template<bool oracle_, typename Index_, typename CachedValue_>
class SecondaryBase {
public:
    SecondaryBase(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        Index_ extract_length,
        size_t cache_size, 
        bool require_minimum_cache,
        bool needs_cached_value) :
        pointers(ptrs),
        dim_stats(std::move(cache_dim_stats)),
        needs_cached_value(needs_cached_value),
        cache_workspace(
            dim_stats.chunk_length, 
            extract_length, 
            cache_size /  size_of_element<Index_, CachedValue_>(needs_cached_value, true),
            require_minimum_cache, 
            std::move(oracle)
        )
    {
        serialize([&]() -> void {
            h5comp.reset(new Components);
            h5comp->file.openFile(file_name, H5F_ACC_RDONLY, fapl);
            h5comp->data = h5comp->file.openDataSet(data_name);
            h5comp->index = h5comp->file.openDataSet(index_name);
            h5comp->dataspace = h5comp->data.getSpace();
        });

        // Allocating it once so that each fetch() call doesn't have to check for width.
        // Don't be tempted to resize these vectors over the lifetime of this object;
        // we want to avoid destructing the internal vectors to re-use their capacity.
        transpose_store.value.resize(dim_stats.chunk_length);
        if (needs_cached_value) {
            transpose_store.value.resize(dim_stats.chunk_length);
        }
    }

    ~SecondaryBase() {
        serialize([&]() {
            h5comp.reset();
        });
    }

protected:
    const std::vector<hsize_t>& pointers;

    struct Slab {
        std::vector<CachedValue_> data;
        std::vector<Index_> index; 
        std::vector<size_t> pointers;

        void reset() {
            data.clear();
            index.clear();
            pointers.clear();
            pointers.push_back(0);
        }
    };
    bool needs_cached_value;

    tatami_chunked::TypicalSlabCacheWorkspace<oracle_, false, Index_, Slab> cache_workspace;
    Slab solo;

    std::vector<CachedValue_> data_buffer;
    std::vector<Index_> index_buffer;
    struct TransposedStore {
        std::vector<std::vector<CachedValue_> > data;
        std::vector<std::vector<Index_> > index; 
    };
    TransposedStore transpose_store;

    // All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
    std::unique_ptr<Components> h5comp;

private:
    template<class Extract_>
    std::pair<const Slab*, Index_> fetch(Index_ i, Extract_ extract) {
        if constexpr(oracle_) {
            return cache_workspace.cache.next(
                /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                    return std::pair<Index_, Index_>(current / dim_stats.chunk_length, current % dim_stats.chunk_length);
                }, 
                /* create = */ [&]() -> Slab {
                    return Slab();
                },
                /* populate = */ [&](const std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                    serialize([&]() -> void {
                        for (const auto& c : chunks) {
                            extract(c.first * chunk_length, get_primary_chunkdim(c.first), *(c.second));
                        }
                    });
                }
            );

        } else {
            auto chunk = i / dim_stats.chunk_length;
            auto index = i % dim_stats.chunk_length;
            auto ptr = cache_workspace.cache.find(
                chunk, 
                /* create = */ [&]() -> Slab {
                    return Slab();
                },
                /* populate = */ [&](Index_ id, Slab& contents) -> void {
                    serialize([&]() -> void {
                        extract(id * dim_stats.chunk_length, dim_stats.get_primary_chunkdim(id), contents);
                    });
                }
            );

            return std::make_pair(ptr, index);
        }
    }

public:
    // The store_index_ flag specifies whether we want to store the index of
    // the primary value (useful for dense extraction) or the primary value
    // itself (useful for sparse extraction).
    template<bool store_index_>
    std::pair<const Slab*, Index_> fetch_block(Index_ i, Index_ primary_start, Index_ primary_length) {
        if (cache.num_slabs_in_cache == 0) {
            solo.reset();
            for (Index_ px = 0; px < primary_length; ++px) {
                auto primary = primary_indices[px];
                extract_and_append_single(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
            }
            solo.pointers.push_back(solo.index.size());
            return std::make_pair(&solo, 0);

        } else {
            return fetch(i, [&](Index_ secondary_start, Index_ secondary_length, Slab& contents) {
                for (Index_ px = 0; px < primary_length; ++px) {
                    auto primary = px + primary_start;
                    extract_and_append(primary, secondary, secondary_start, (store_index_ ? px : primary));
                }
                compress_and_move_transpose_info(contents, secondary_length);
            });
        }
    }

    template<bool store_index_>
    std::pair<const Slab*, Index_> fetch_indices(Index_ i, const std::vector<Index_>& primary_indices) {
        if (cache.num_slabs_in_cache == 0) {
            solo.reset();
            for (Index_ px = 0, end = primary_indices.size(); px < end; ++px) {
                auto primary = primary_indices[px];
                extract_and_append_single(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
            }
            solo.pointers.push_back(solo.index.size());
            return std::make_pair(&solo, 0);

        } else {
            return fetch(i, [&](Index_ secondary_start, Index_ secondary_length, Slab& contents) {
                for (Index_ px = 0, end = primary_indices.size(); px < end; ++px) {
                    auto primary = primary_indices[px];
                    extract_and_append(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
                }
                compress_and_move_transpose_info(contents, secondary_length);
            });
        }
    }

private:
    // Serial locks should be applied by the callers before calling this.
    void extract_and_append(Index_ primary, Index_ secondary_start, Index_ secondary_length, Index_ primary_to_add) {
        hsize_t left = pointers[primary], right = pointers[primary + 1];
        hsize_t count = right - left;
        size_t old_size = store.index.size();
        index_buffer.resize(count);

        work.dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        work.memspace.setExtentSimple(1, &count);
        work.memspace.selectAll();
        work.index_dataset.read(index_buffer.data(), define_mem_type<Index_>(), work.memspace, work.dataspace);

        auto start = store.index.begin(), end = store.index.end();
        refine_primary_limits(start, end, dim_stats.dimension_extent, secondary_start, secondary_start + secondary_length);
        for (auto x = start; x != end; ++x) {
            transpose_store.index[*start - secondary_start].push_back(primary_to_add);
        }

        if (start != end && needs_cached_value) {
            hsize_t better_left = left + (start - store.index.begin());
            hsize_t better_count = end - start;
            work.dataspace.selectHyperslab(H5S_SELECT_SET, &better_count, &better_offset);
            work.memspace.setExtentSimple(1, &better_count);
            work.memspace.selectAll();

            data_buffer.resize(better_count);
            work.data.read(data_buffer.data(), define_mem_type<CachedValue_>(), work.memspace, work.dataspace);
            size_t y = 0;
            for (auto x = start; x != end; ++x, ++y) {
                transpose_store.value[*start - secondary_start].push_back(data_buffer[y]);
            }
        }
    }

    void compress_and_move_transpose_info(Slab& slab, Index_ secondary_length) {
        slab.reset();
        for (const auto& x : store.index) {
            slab.pointers.push_back(slab.pointers.back() + x.size());
            slab.index.insert(slab.index.end(), x.begin(), x.end());
            x.clear();
        }

        if (needs_value) {
            for (const auto& x : store.data) {
                slab.data.insert(slab.data.end(), x.begin(), x.end());
                x.clear();
            }
        }
    }

    void extract_and_append_single(Index_ primary, Index_ secondary_start, Index_ secondary_length, Index_ primary_to_add) {
        hsize_t left = pointers[primary], right = pointers[primary + 1];
        hsize_t count = right - left;
        size_t old_size = store.index.size();
        index_buffer.resize(count);

        work.dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        work.memspace.setExtentSimple(1, &count);
        work.memspace.selectAll();
        work.index_dataset.read(index_buffer.data(), define_mem_type<Index_>(), work.memspace, work.dataspace);

        auto start = store.index.begin(), end = store.index.end();
        auto it = std::lower_bound(start, end, secondary);

        if (it != end && *it == secondary) {
            solo.index.push_back(primary_to_add);
            if (needs_value) {
                hsize_t offset = left + (it - start);
                count = 1;
                core.dataspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
                core.memspace.setExtentSimple(1, &count);
                core.memspace.selectAll();

                Value_ dest;
                work.data_dataset.read(&dest, define_mem_type<Value_>(), work.memspace, work.dataspace);
                solo.value.push_back(dest);
            }
        }
    }


public:
    template<typename Value_>
    tatami::SparseRange<Value, Index_> slab_to_sparse(const Slab& chunk, Index_ offset, Value_* vbuffer, Index_* ibuffer, bool needs_index) {
        size_t start = chunk.pointers[offset];
        size_t length = chunk.pointers[offset + 1] - start;

        tatami::SparseRange<Value_, Index_> output;
        if (chunk.value) {
            std::copy(chunk.value.begin() + start, length, vbuffer);
            output.value = vbuffer;
        }

        if (needs_index) {
            std::copy(chunk.index.begin() + start, length, ibuffer);
            output.index = ibuffer;
        }

        return output;
    }

    template<typename Value_>
    void slab_to_dense(const Slab& chunk, Index_ offset, Value_* buffer, Index_ extract_length) {
        size_t start = chunk.pointers[offset];
        size_t end = chunk.pointers[offset + 1];
        std::fill_n(buffer, extract_length, 0);

        auto valptr = chunk.value.begin() + start;
        auto idxptr = chunk.index.begin() + start;
        for (; start != end; ++start, ++valptr) {
            buffer[*idxptr] = *valptr;
        }
        return;
    }
};

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryFullSparse : public SecondaryBase<oracle_, Index_, CachedValue_> {
    SecondaryFullSparse(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        tatami_chunked::ChunkDimensionStats<Index_>& cache_dim_stats,
        Index_ uncached_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        size_t cache_size, 
        bool require_minimum_cache,
        bool needs_value,
        bool needs_index) : 
        SecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(cache_dim_stats),
            std::move(oracle), 
            uncached_dim,
            cache_size, 
            require_minimum_value,
            needs_value 
        ),
        uncached_dim(uncached_dim),
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto cached = this->template fetch_block<false>(i, 0, uncached_dim);
        return slab_to_sparse(*(cached.first), cached.second, vbuffer, ibuffer, needs_index);
    }

private:
    Index_ uncached_dim;
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SecondaryFullDense : public ConditionalSecondaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
   SecondaryFullDense(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        tatami_chunked::ChunkDimensionStats<Index_>& cache_dim_stats,
        Index_ uncached_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        size_t cache_size, 
        bool require_minimum_cache) :
        SecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(cache_dim_stats),
            std::move(oracle), 
            uncached_dim,
            cache_size,
            require_minimum_value,
            true
        ),
        uncached_dim(uncached_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = this->template fetch_block<true>(i, 0, uncached_dim);
        slab_to_dense(*(cached.first), cached.second, buffer, uncached_dim);
        return buffer;
    }

private:
    Index_ uncached_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryBlockSparse : public SecondaryBase<oracle_, Index_, CachedIndex_> {
    SecondaryBlockSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        bool require_minimum_cache,
        bool needs_value, 
        bool needs_index) : 
        SecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(cache_dim_stats),
            std::move(oracle), 
            block_length, 
            cache_size, 
            require_minimum_cache,
            needs_value
        ),
        block_start(block_start),
        block_length(block_length),
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto cached = this->fetch_block<false>(i, block_start, block_length);
        return slab_to_sparse(*(cached.first), cached.second, vbuffer, ibuffer, needs_index);
    }

private:
    Index_ block_start;
    Index_ block_length;
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryBlockDense : public SecondaryBase<oracle_, Index_, CachedValue_> {
   SecondaryBlockDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        size_t max_non_zeros) :
        SecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(cache_dim_stats),
            std::move(oracle), 
            block_length,
            cache_size, 
            require_minimum_cache,
            true
        ),
        block_start(block_start),
        block_length(block_length),
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = this->fetch_block<true>(i, block_start, block_length);
        return slab_to_dense(*(cached.first), cached.second, buffer, block_length);
    }

private:
    Index_ block_start;
    Index_ block_length;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryIndexSparse : public SecondaryBase<oracle_, Index_, CachedValue_> {
    SecondaryIndexSparse(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        tatami::VectorPtr<Index_> idx_ptr,
        size_t cache_size, 
        bool require_minimum_cache,
        bool needs_value,
        bool needs_index) : 
        SecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(cache_dim_stats),
            std::move(oracle), 
            idx_ptr->size(),
            cache_size, 
            require_minimum_value,
            needs_value 
        ),
        indices_ptr(std::move(idx_ptr)),
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto cached = this->fetch_indices<false>(i, *indices_ptr);
        return slab_to_sparse(*(cached.first), cached.second, vbuffer, ibuffer, needs_index);
    }

private:
    tatami::VectorPtr<Index_> indices_ptr;
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class SecondaryIndexDense : public ConditionalSecondaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
   SecondaryIndexDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        size_t cache_size,
        size_t max_non_zeros) :
        SecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(cache_dim_stats),
            std::move(oracle), 
            idx_ptr->size(),
            cache_size, 
            require_minimum_value,
            true
        ),
        indices_ptr(std::move(idx_ptr)),
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = this->fetch_indices<true>(i, *indices_ptr);
        return slab_to_dense(*(cached.first), cached.second, buffer, indices_ptr->size());
    }

private:
    tatami::VectorPtr<Index_> indices_ptr;
};

}

}

#endif
