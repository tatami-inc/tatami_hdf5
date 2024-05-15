#ifndef TATAMI_HDF5_SPARSE_SECONDARY_HPP
#define TATAMI_HDF5_SPARSE_SECONDARY_HPP

#include <vector>
#include <algorithm>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include "sparse_primary.hpp"
#include "serialize.hpp"
#include "utils.hpp"

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

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
void slab_to_dense(const tatami::SparseRange<CachedValue_, Index_>& slab, Value_* buffer, size_t extract_length) {
    std::fill_n(buffer, extract_length, 0);
    auto valptr = slab.value;
    auto idxptr = slab.index;
    for (Index_ i = 0; i < slab.number; ++i, ++idxptr, ++valptr) {
        buffer[*idxptr] = *valptr;
    }
    return;
}

struct SecondaryBase {
    template<typename Index_>
    SecondaryBase(const MatrixDetails<Index_>& details) {
        serialize([&]() -> void {
            h5comp.reset(new Components);

            // Using some kinda-big prime number as the number of slots. This
            // doesn't really matter too much as we only intend to store two
            // chunks at most - see Hdf5CompressedSparseMatrix.hpp for the rationale.
            H5::FileAccPropList fapl(H5::FileAccPropList::DEFAULT.getId());
            fapl.setCache(0, 511, details.h5_chunk_cache_size, 0);

            h5comp->file.openFile(details.file_name, H5F_ACC_RDONLY);
            h5comp->data_dataset = h5comp->file.openDataSet(details.data_name);
            h5comp->index_dataset = h5comp->file.openDataSet(details.index_name);
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
        const MatrixDetails<Index_>& details,
        bool, // oracle: for consistency with the oracular constructor.
        Index_ extract_length,
        bool needs_value,
        bool needs_index) :
        SecondaryBase(details),
        pointers(details.pointers),
        sec_dim_stats(
            details.secondary_dim,
            std::max(
                static_cast<size_t>(1),
                // The general strategy here is to allocate a single giant slab based on what the 'cache_size' can afford. 
                static_cast<size_t>(details.our_cache_size / (Hdf5CompressedSparseMatrix_internal::size_of_cached_element<Index_, CachedValue_>(needs_value, true) * extract_length))
            )
        ),
        extract_length(extract_length),
        needs_value(needs_value),
        needs_index(needs_index)
    {
        cache_count.resize(sec_dim_stats.chunk_length);
        size_t cache_size_in_elements = static_cast<size_t>(sec_dim_stats.chunk_length) * extract_length; // cast to avoid overflow.
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
    size_t extract_length; // store as a size_t to avoid overflow in offset calculations.
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
        size_t offset = static_cast<size_t>(chunk_offset) * extract_length; // cast to avoid overflow.
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
        if (count == 0) {
            return;
        }
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
                cache_data[cache_offsets[current] + static_cast<size_t>(cache_count[current])] = data_buffer[y];
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
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ extract_length,
        bool needs_value,
        bool needs_index) :
        SecondaryBase(details),
        pointers(details.pointers),
        secondary_dim(details.secondary_dim),
        extract_length(extract_length),
        needs_value(needs_value),
        needs_index(needs_index),
        cache(
            std::move(oracle),
            tatami_chunked::SlabCacheStats(
                /* target_length = */ 1,
                /* non_target_length = */ extract_length,
                /* target_num_chunks = */ secondary_dim,
                /* cache_size_in_bytes = */ details.our_cache_size,
                /* element_size = */ size_of_cached_element<Index_, CachedValue_>(needs_value, true),
                /* require_minimum_cache = */ true
            ).num_slabs_in_cache
        )
    {
        size_t alloc = static_cast<size_t>(cache.get_max_slabs()) * extract_length; // cast to avoid overflow.
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
    size_t extract_length; // store this as a size_t to avoid overflow when computing offsets.
    bool needs_value;
    bool needs_index;

    struct Slab {
        CachedValue_* value = NULL;
        Index_* index = NULL;
        Index_ number = 0;
    };
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> cache;

    // Contiguous data stores for the Slabs to point to. This avoids
    // the overhead of allocating a lot of little vectors.
    std::vector<Index_> cache_index;
    std::vector<CachedValue_> cache_data;
    size_t counter = 0;

    // Temporary buffers for the HDF5 library to read in values/indices for each dimension element.
    std::vector<Index_> index_buffer;
    std::vector<CachedValue_> data_buffer;

    // Some account-keeping intermediates to move data from the buffers to the cache.
    std::vector<Slab*> slab_ptrs;
    std::vector<CachedValue_*> value_ptrs;
    std::vector<Index_> found;

private:
    template<class Extract_>
    tatami::SparseRange<CachedValue_, Index_> fetch_raw(Extract_ extract) {
        auto out = cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current, 0);
            }, 
            /* create = */ [&]() -> Slab {
                Slab latest;
                size_t offset = counter * extract_length;
                if (needs_value) {
                    latest.value = cache_data.data() + offset;
                }
                if (needs_index) {
                    latest.index = cache_index.data() + offset;
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
        if (count == 0) {
            return;
        }
        index_buffer.resize(count);

        this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        this->h5comp->memspace.setExtentSimple(1, &count);
        this->h5comp->memspace.selectAll();
        this->h5comp->index_dataset.read(index_buffer.data(), define_mem_type<Index_>(), this->h5comp->memspace, this->h5comp->dataspace);

        auto start = index_buffer.begin(), end = index_buffer.end();
        refine_primary_limits(start, end, secondary_dim, secondary_first, secondary_last_plus_one);

        if (needs_value) {
            value_ptrs.clear();
            found.clear();
        }

        Index_ counter = 0;
        for (auto x = start; x != end; ++x, ++counter) {
            auto slab_ptr = slab_ptrs[*x];
            if (slab_ptr != NULL) {
                if (needs_index) {
                    slab_ptr->index[slab_ptr->number] = primary_to_add;
                }
                if (needs_value) {
                    value_ptrs.push_back(slab_ptr->value + slab_ptr->number);
                    found.push_back(counter);
                }
                ++(slab_ptr->number);
            }
        }

        if (!found.empty()) {
            hsize_t new_start = left + (start - index_buffer.begin());
            this->h5comp->dataspace.selectNone();
            tatami::process_consecutive_indices<Index_>(found.data(), found.size(),
                [&](Index_ start, Index_ length) {
                    hsize_t offset = start + new_start;
                    hsize_t count = length;
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &count, &offset);
                }
            );

            hsize_t new_len = found.size();
            this->h5comp->memspace.setExtentSimple(1, &new_len);
            this->h5comp->memspace.selectAll();

            data_buffer.resize(new_len);
            this->h5comp->data_dataset.read(data_buffer.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
            for (hsize_t i = 0; i < new_len; ++i) {
                *(value_ptrs[i]) = data_buffer[i];
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
    SecondaryFullSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, bool needs_value, bool needs_index) : 
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(details, std::move(oracle), details.primary_dim, needs_value, needs_index),
        primary_dim(details.primary_dim)
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
    SecondaryFullDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle) :
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(details, std::move(oracle), details.primary_dim, true, true),
        primary_dim(details.primary_dim)
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
    SecondaryBlockSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(details, std::move(oracle), block_length, needs_value, needs_index),
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
    SecondaryBlockDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length) :
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(details, std::move(oracle), block_length, true, true),
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
    SecondaryIndexSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> idx_ptr, bool needs_value, bool needs_index) : 
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(details, std::move(oracle), idx_ptr->size(), needs_value, needs_index),
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
    SecondaryIndexDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> idx_ptr) :
        ConditionalSecondaryBase<oracle_, Index_, CachedValue_>(details, std::move(oracle), idx_ptr->size(), true, true),
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
