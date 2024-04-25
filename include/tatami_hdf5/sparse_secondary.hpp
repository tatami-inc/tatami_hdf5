#ifndef TATAMI_HDF5_SPARSE_SECONDARY_HPP
#define TATAMI_HDF5_SPARSE_SECONDARY_HPP

#include <vector>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include "sparse_primary.hpp"
#include "serialize.hpp"
#include "utils.hpp"

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

inline size_t multiply(size_t a, size_t b) {
    // Use size_t to avoid overflow for pointers and such.
    return a * b;
}

template<typename Index_, typename CachedValue_, typename Value_>
tatami::SparseRange<Value_, Index_> slab_to_sparse(const tatami::SparseRange<CachedValue_, Index_>& slab, Value_* vbuffer, Index_* ibuffer) {
    tatami::SparseRange<Value_, Index_> output;
    output.number = slab.number;

    if (slab.value) {
        std::copy_n(slab.value, slab.number, vbuffer);
        output.value = vbuffer;
    }

    if (slab.index) {
        std::copy_n(slab.index, slab.number, ibuffer);
        output.index = ibuffer;
    }

    return output;
}

template<typename Index_, typename CachedValue_, typename Value_>
void slab_to_dense(const tatami::SparseRange<CachedValue_, Index_>& slab, Value_* buffer, Index_ extract_length) {
    std::fill_n(buffer, extract_length, 0);
    auto valptr = slab.value;
    auto idxptr = slab.index;
    for (Index_ i = 0; i < slab.number; ++i, ++idxptr, ++valptr) {
        buffer[*idxptr] = *valptr;
    }
    return;
}

struct SecondaryBase {
    SecondaryBase(const std::string& file_name, const std::string& data_name, const std::string& index_name) {
        serialize([&]() -> void {
            h5comp.reset(new Components);
            h5comp->file.openFile(file_name, H5F_ACC_RDONLY);
            h5comp->data_dataset = h5comp->file.openDataSet(data_name);
            h5comp->index_dataset = h5comp->file.openDataSet(index_name);
            h5comp->dataspace = h5comp->data_dataset.getSpace();
        });
    }

    ~SecondaryBase() {
        serialize([&]() {
            h5comp.reset();
        });
    }

protected:
    // All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
    std::unique_ptr<Components> h5comp;
};

// We don't use CachedIndex_, as this may be too small to store the other
// dimension; use Index_ instead as this is guaranteed.
template<typename Index_, typename CachedValue_>
class MyopicSecondaryBase : public SecondaryBase {
public:
    MyopicSecondaryBase(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        bool, // oracle: for consistency with the oracular constructor.
        Index_ extract_length,
        size_t cache_size,
        bool needs_value,
        bool needs_index) :
        SecondaryBase(file_name, data_name, index_name),
        pointers(ptrs),
        sec_dim_stats(
            secondary_dim,
            std::max(
                static_cast<size_t>(1),
                static_cast<size_t>(
                    // The general strategy here is to allocate a single giant slab based on what the 'cache_size' can afford. 
                    (cache_size / Hdf5CompressedSparseMatrix_internal::size_of_cached_element<Index_, CachedValue_>(needs_value, true)) / extract_length
                )
            ) 
        ),
        extract_length(extract_length),
        needs_value(needs_value),
        needs_index(needs_index)
    {
        cache_count.resize(sec_dim_stats.chunk_length);
        size_t cache_size_in_elements = multiply(sec_dim_stats.chunk_length, extract_length);
        cache_index.resize(cache_size_in_elements);
        if (needs_value) {
            cache_data.resize(cache_size_in_elements);
        }

        // Precomputing the offsets so we don't have to do the multiplication every time.
        cache_offsets.reserve(sec_dim_stats.chunk_length);
        size_t current_offset = 0;
        for (Index_ i = 0; i < sec_dim_stats.chunk_length; ++i, current_offset += extract_length) {
            cache_offsets.push_back(current_offset);
        }
    }

private:
    const std::vector<hsize_t>& pointers;
    tatami_chunked::ChunkDimensionStats<Index_> sec_dim_stats;
    Index_ extract_length;
    bool needs_value;
    bool needs_index;

    std::vector<Index_> index_buffer;
    std::vector<CachedValue_> data_buffer;

    Index_ last_chunk_id = 0;
    std::vector<Index_> cache_index;
    std::vector<CachedValue_> cache_data;
    std::vector<Index_> cache_count;
    std::vector<size_t> cache_offsets;
    bool first = true;

private:
    template<class Extract_>
    tatami::SparseRange<CachedValue_, Index_> fetch_raw(Index_ i, Extract_ extract) {
        Index_ chunk_id = i / sec_dim_stats.chunk_length;
        Index_ chunk_offset = i % sec_dim_stats.chunk_length;

        if (chunk_id != last_chunk_id || first) { 
            Index_ clen = sec_dim_stats.get_chunk_length(chunk_id);
            std::fill_n(cache_count.begin(), clen, 0);

            serialize([&]() {
                extract(chunk_id * sec_dim_stats.chunk_length, clen);
            });
            last_chunk_id = chunk_id;
            first = false;
        }

        tatami::SparseRange<CachedValue_, Index_> output(cache_count[chunk_offset]);
        size_t offset = multiply(chunk_offset, extract_length);
        if (needs_value) {
            output.value = cache_data.data() + offset;
        }
        if (needs_index) {
            output.index = cache_index.data() + offset;
        }
        return output;
    }

    // Serial locks should be applied by the callers before calling this.
    void extract_and_append(Index_ primary, Index_ secondary_start, Index_ secondary_length, Index_ primary_to_add) {
        hsize_t left = pointers[primary], right = pointers[primary + 1];
        hsize_t count = right - left;
        index_buffer.resize(count);

        this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        this->h5comp->memspace.setExtentSimple(1, &count);
        this->h5comp->memspace.selectAll();
        this->h5comp->index_dataset.read(index_buffer.data(), define_mem_type<Index_>(), this->h5comp->memspace, this->h5comp->dataspace);

        auto start = index_buffer.begin(), end = index_buffer.end();
        refine_primary_limits(start, end, sec_dim_stats.dimension_extent, secondary_start, secondary_start + secondary_length);

        if (needs_index) {
            for (auto x = start; x != end; ++x) {
                Index_ current = *x - secondary_start;
                cache_index[cache_offsets[current] + static_cast<size_t>(cache_count[current])] = primary_to_add;
            }
        }

        if (start != end && needs_value) {
            hsize_t better_left = left + (start - index_buffer.begin());
            hsize_t better_count = end - start;
            this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &better_count, &better_left);
            this->h5comp->memspace.setExtentSimple(1, &better_count);
            this->h5comp->memspace.selectAll();

            data_buffer.resize(better_count);
            this->h5comp->data_dataset.read(data_buffer.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);

            size_t y = 0;
            for (auto x = start; x != end; ++x, ++y) {
                Index_ current = *x - secondary_start;
                cache_index[cache_offsets[current] + static_cast<size_t>(cache_count[current])] = data_buffer[y];
            }
        }

        for (auto x = start; x != end; ++x) {
            ++(cache_count[*x - secondary_start]);
        }
    }

public:
    // The store_index_ flag specifies whether we want to store the index of
    // the primary value (useful for dense extraction) or the primary value
    // itself (useful for sparse extraction).
    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_block(Index_ i, Index_ primary_start, Index_ primary_length) {
        return fetch_raw(i, [&](Index_ secondary_start, Index_ secondary_length) {
            for (Index_ px = 0; px < primary_length; ++px) {
                auto primary = px + primary_start;
                extract_and_append(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
            }
        });
    }

    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_indices(Index_ i, const std::vector<Index_>& primary_indices) {
        return fetch_raw(i, [&](Index_ secondary_start, Index_ secondary_length) { 
            for (Index_ px = 0, end = primary_indices.size(); px < end; ++px) {
                auto primary = primary_indices[px];
                extract_and_append(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
            }
        });
    }
};

template<typename Index_, typename CachedValue_>
class OracularSecondaryBase : public SecondaryBase {
public:
    OracularSecondaryBase(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        Index_ secondary_dim,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ extract_length,
        size_t cache_size,
        bool needs_value,
        bool needs_index) :
        SecondaryBase(file_name, data_name, index_name),
        pointers(ptrs),
        secondary_dim(secondary_dim),
        extract_length(extract_length),
        needs_value(needs_value),
        needs_index(needs_index),
        cache(
            std::move(oracle),
            std::max(
                static_cast<size_t>(1),
                static_cast<size_t>((cache_size / Hdf5CompressedSparseMatrix_internal::size_of_cached_element<Index_, CachedValue_>(needs_value, true)) / extract_length)
            )
        )
    {
        size_t alloc = multiply(cache.get_num_slabs(), extract_length);
        if (needs_index) {
            cache_index.resize(alloc);
        }
        if (needs_value) {
            cache_data.resize(alloc);
        }
        slab_ptrs.resize(secondary_dim, NULL);
    }

protected:
    const std::vector<hsize_t>& pointers;
    Index_ secondary_dim;
    Index_ extract_length;
    bool needs_value;
    bool needs_index;

    struct Slab {
        CachedValue_* value;
        Index_* index;
        Index_ number = 0;
    };
    tatami_chunked::OracleSlabCache<Index_, Index_, Slab> cache;
    std::vector<Slab*> slab_ptrs;

    std::vector<Index_> index_buffer;
    std::vector<CachedValue_> data_buffer;

    std::vector<Index_> cache_index;
    std::vector<CachedValue_> cache_data;
    size_t counter = 0;

private:
    template<class Extract_>
    tatami::SparseRange<CachedValue_, Index_> fetch_raw(Extract_ extract) {
        auto out = cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current, 0);
            }, 
            /* create = */ [&]() -> Slab {
                Slab latest;
                size_t offset = multiply(counter, extract_length);
                if (needs_value) {
                    latest.value = data_buffer.data() + offset;
                }
                if (needs_index) {
                    latest.index = index_buffer.data() + offset;
                }
                ++counter;
                return latest;
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                if (!chunks.empty()) {
                    Index_ first = secondary_dim, last = 0;
                    for (auto& c : chunks) {
                        slab_ptrs[c.first] = c.second;
                        first = std::min(first, c.first);
                        last = std::max(last, c.first);
                        c.second->number = 0;
                    }

                    serialize([&]() -> void {
                        extract(first, last + 1);
                    });

                    for (auto& c : chunks) {
                        slab_ptrs[c.first] = NULL;
                    }
                }
            }
        );
        return tatami::SparseRange(out.first->number, out.first->value, out.first->index);
    }

    // Serial locks should be applied by the callers before calling this.
    void extract_and_append(Index_ primary, Index_ secondary_first, Index_ secondary_last_plus_one, Index_ primary_to_add) {
        hsize_t left = pointers[primary], right = pointers[primary + 1];
        hsize_t count = right - left;
        index_buffer.resize(count);

        this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        this->h5comp->memspace.setExtentSimple(1, &count);
        this->h5comp->memspace.selectAll();
        this->h5comp->index_dataset.read(index_buffer.data(), define_mem_type<Index_>(), this->h5comp->memspace, this->h5comp->dataspace);

        auto start = index_buffer.begin(), end = index_buffer.end();
        refine_primary_limits(start, end, secondary_dim, secondary_first, secondary_last_plus_one);

        if (needs_index) {
            for (auto x = start; x != end; ++x) {
                auto slab_ptr = slab_ptrs[*x];
                if (slab_ptr != NULL) {
                    slab_ptr->value[slab_ptr->number] = primary_to_add;
                }
            }
        }

        if (start != end && needs_value) {
            hsize_t better_left = left + (start - index_buffer.begin());
            hsize_t better_count = end - start;
            this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &better_count, &better_left);
            this->h5comp->memspace.setExtentSimple(1, &better_count);
            this->h5comp->memspace.selectAll();

            data_buffer.resize(better_count);
            this->h5comp->data_dataset.read(data_buffer.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);

            size_t y = 0;
            for (auto x = start; x != end; ++x, ++y) {
                auto slab_ptr = slab_ptrs[*x];
                if (slab_ptr != NULL) {
                    slab_ptr->value[slab_ptr->number] = data_buffer[y];
                }
            }
        }

        for (auto x = start; x != end; ++x) {
            auto slab_ptr = slab_ptrs[*x];
            if (slab_ptr != NULL) {
                ++(slab_ptr->number);
            }
        }
    }

public:
    // The store_index_ flag specifies whether we want to store the index of
    // the primary value (useful for dense extraction) or the primary value
    // itself (useful for sparse extraction).
    //
    // The first argument here is for compile-time polymorphism with the
    // myopic counterpart, and is not actually needed itself.
    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_block(Index_, Index_ primary_start, Index_ primary_length) {
        const auto& info = fetch_raw([&](Index_ secondary_first, Index_ secondary_last_plus_one) {
            for (Index_ px = 0; px < primary_length; ++px) {
                auto primary = px + primary_start;
                extract_and_append(primary, secondary_first, secondary_last_plus_one, (store_index_ ? px : primary));
            }
        });
        return tatami::SparseRange<CachedValue_, Index_>(info.number, info.value, info.index);
    }

    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_indices(Index_, const std::vector<Index_>& primary_indices) {
        const auto& info = fetch_raw([&](Index_ secondary_first, Index_ secondary_last_plus_one) {
            for (Index_ px = 0, end = primary_indices.size(); px < end; ++px) {
                auto primary = primary_indices[px];
                extract_and_append(primary, secondary_first, secondary_last_plus_one, (store_index_ ? px : primary));
            }
        });
        return tatami::SparseRange<CachedValue_, Index_>(info.number, info.value, info.index);
    }
};

template<bool oracle_, typename Index_, typename CachedValue_>
using ConditionalSecondaryBase = typename std::conditional<oracle_, OracularSecondaryBase<Index_, CachedValue_>, MyopicSecondaryBase<Index_, CachedValue_> >::type;

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct SecondaryFullSparse : 
    public ConditionalSecondaryBase<oracle_, Index_, CachedValue_>,
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
    SecondaryFullSparse(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        Index_ secondary_dim,
        Index_ primary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        size_t cache_size, 
        bool needs_value,
        bool needs_index) : 
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            secondary_dim,
            std::move(oracle), 
            primary_dim,
            cache_size, 
            needs_value,
            needs_index
        ),
        primary_dim(primary_dim)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto cached = this->template fetch_block<false>(i, 0, primary_dim);
        return slab_to_sparse(cached, vbuffer, ibuffer); 
    }

private:
    Index_ primary_dim;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct SecondaryFullDense : 
    public ConditionalSecondaryBase<oracle_, Index_, CachedValue_>,
    public tatami::DenseExtractor<oracle_, Value_, Index_> 
{
    SecondaryFullDense(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        Index_ secondary_dim,
        Index_ primary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        size_t cache_size) :
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            secondary_dim,
            std::move(oracle), 
            primary_dim,
            cache_size,
            true,
            true
        ),
        primary_dim(primary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = this->template fetch_block<true>(i, 0, primary_dim);
        slab_to_dense(cached, buffer, primary_dim);
        return buffer;
    }

private:
    Index_ primary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct SecondaryBlockSparse : 
    public ConditionalSecondaryBase<oracle_, Index_, CachedValue_>,
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
    SecondaryBlockSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        bool needs_value, 
        bool needs_index) : 
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            secondary_dim,
            std::move(oracle), 
            block_length, 
            cache_size, 
            needs_value,
            needs_index
        ),
        block_start(block_start),
        block_length(block_length)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto cached = this->template fetch_block<false>(i, block_start, block_length);
        return slab_to_sparse(cached, vbuffer, ibuffer);
    }

private:
    Index_ block_start;
    Index_ block_length;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct SecondaryBlockDense : 
    public ConditionalSecondaryBase<oracle_, Index_, CachedValue_>,
    public tatami::DenseExtractor<oracle_, Value_, Index_> 
{
    SecondaryBlockDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size) :
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            secondary_dim,
            std::move(oracle), 
            block_length,
            cache_size, 
            true,
            true
        ),
        block_start(block_start),
        block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = this->template fetch_block<true>(i, block_start, block_length);
        slab_to_dense(cached, buffer, block_length);
        return buffer;
    }

private:
    Index_ block_start;
    Index_ block_length;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct SecondaryIndexSparse : 
    public ConditionalSecondaryBase<oracle_, Index_, CachedValue_>,
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
    SecondaryIndexSparse(
        const std::string& file_name, 
        const std::string& data_name, 
        const std::string& index_name, 
        const std::vector<hsize_t>& ptrs, 
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        tatami::VectorPtr<Index_> idx_ptr,
        size_t cache_size, 
        bool needs_value,
        bool needs_index) : 
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            secondary_dim,
            std::move(oracle), 
            idx_ptr->size(),
            cache_size, 
            needs_value,
            needs_index
        ),
        indices_ptr(std::move(idx_ptr)),
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto cached = this->template fetch_indices<false>(i, *indices_ptr);
        return slab_to_sparse(cached, vbuffer, ibuffer);
    }

private:
    tatami::VectorPtr<Index_> indices_ptr;
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct SecondaryIndexDense :
    public ConditionalSecondaryBase<oracle_, Index_, CachedValue_>,
    public tatami::DenseExtractor<oracle_, Value_, Index_> 
{
    SecondaryIndexDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> idx_ptr,
        size_t cache_size) :
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            secondary_dim,
            std::move(oracle), 
            idx_ptr->size(),
            cache_size, 
            true,
            true
        ),
        indices_ptr(std::move(idx_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = this->template fetch_indices<true>(i, *indices_ptr);
        slab_to_dense(cached, buffer, static_cast<Index_>(indices_ptr->size()));
        return buffer;
    }

private:
    tatami::VectorPtr<Index_> indices_ptr;
};

}

}

#endif
