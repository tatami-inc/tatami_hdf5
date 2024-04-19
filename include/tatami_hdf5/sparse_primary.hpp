#ifndef TATAMI_HDF5_SPARSE_PRIMARY_HPP
#define TATAMI_HDF5_SPARSE_PRIMARY_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>

#include "tatami/tatami.hpp"

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

/*****************************************************************************
 * Code below is vendored from tatami/include/sparse/primary_extraction.hpp.
 * We copied it to avoid an implicit dependency on tatami's internals, given
 * that I'm not ready to promise that those internals are stable.
 *****************************************************************************/

template<class IndexIt_, typename Index_>
void refine_primary_limits(IndexIt_& indices_start, IndexIt_& indices_end, Index_ extent, Index_ smallest, Index_ largest_plus_one) {
    if (smallest) {
        // Using custom comparator to ensure that we cast to Index_ for signedness-safe comparisons.
        indices_start = std::lower_bound(indices_start, indices_end, smallest, [](Index_ a, Index_ b) -> bool { return a < b; });
    }

    if (largest_plus_one != extent) {
        indices_end = std::lower_bound(indices_start, indices_end, largest_plus_one, [](Index_ a, Index_ b) -> bool { return a < b; });
    }
}

template<typename Index_>
struct RetrievePrimarySubsetDense {
    RetrievePrimarySubsetDense(const std::vector<Index_>& subset, Index_ extent) : extent(extent) {
        if (!subset.empty()) {
            offset = subset.front();
            lastp1 = subset.back() + 1;
            size_t alloc = lastp1 - offset;
            present.resize(alloc);

            // Starting off at 1 to ensure that 0 is still a marker for
            // absence. It should be fine as subset.size() should fit inside
            // Index_ (otherwise nrow()/ncol() would give the wrong answer).
            Index_ counter = 1; 

            for (auto s : subset) {
                present[s - offset] = counter;
                ++counter;
            }
        }
    }

    template<class IndexIt_, class Store_>
    void populate(IndexIt_ indices_start, IndexIt_ indices_end, Store_ store) const {
        if (present.empty()) {
            return;
        }

        // Limiting the iteration to its boundaries based on the first and last subset index.
        auto original_start = indices_start;
        refine_primary_limits(indices_start, indices_end, extent, offset, lastp1);

        size_t counter = indices_start - original_start;
        for (; indices_start != indices_end; ++indices_start, ++counter) {
            auto ix = *indices_start;
            auto shift = present[ix - offset];
            if (shift) {
                store(shift - 1, counter);
            }
        }
    }

    Index_ extent;
    std::vector<Index_> present;
    Index_ offset = 0;
    Index_ lastp1 = 0;
};

template<typename Index_>
struct RetrievePrimarySubsetSparse {
    RetrievePrimarySubsetSparse(const std::vector<Index_>& subset, Index_ extent) : extent(extent) {
        if (!subset.empty()) {
            offset = subset.front();
            lastp1 = subset.back() + 1;
            size_t alloc = lastp1 - offset;
            present.resize(alloc);

            // Unlike the dense case, this is a simple present/absent signal,
            // as we don't need to map each structural non-zero back onto its 
            // corresponding location on a dense vector.
            for (auto s : subset) {
                present[s - offset] = 1;
            }
        }
    }

    template<class IndexIt_, class Store_>
    void populate(IndexIt_ indices_start, IndexIt_ indices_end, Store_ store) const {
        if (present.empty()) {
            return;
        }

        // Limiting the iteration to its boundaries based on the first and last subset index.
        auto original_start = indices_start;
        refine_primary_limits(indices_start, indices_end, extent, offset, lastp1);

        size_t counter = indices_start - original_start;
        for (; indices_start != indices_end; ++indices_start, ++counter) {
            auto ix = *indices_start;
            if (present[ix - offset]) {
                store(counter, ix);
            }
        }
    }

    Index_ extent;
    std::vector<unsigned char> present;
    Index_ offset = 0;
    Index_ lastp1 = 0;
};

// Other utilities begin here.
template<typename CachedValue_, typename CachedIndex_ = Index_>
size_t size_of_cached_element(bool needs_cached_value, bool needs_cached_index) {
    return (needs_cached_index ? sizeof(CachedIndex_) : 0) + (needs_cached_value ? sizeof(CachedValue_) : 0);
}

struct Components {
    H5::H5File file;
    H5::DataSet data_dataset;
    H5::DataSet index_dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

template<typename Index_, class Function_>
void sort_by_field(std::vector<Index_>& indices, Function_ field) {
    auto comp = [&field](size_t l, size_t r) -> bool {
        return field(l) < field(r);
    };
    if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
        std::sort(indices.begin(), indices.end(), comp);
    }
}

template<typename Index_, typename CachedValue_, typename CachedIndex_ = Index_>
struct Chunk { 
    const CachedValue_* value = NULL;
    const CachedIndex_* index = NULL;
    Index_ length = 0;
};

// Here, we use a giant contiguous buffer to optimize for near-consecutive
// iteration. This allows the HDF5 library to pull out long strips of data from
// the file.  It also allows us to maximize the use of the cache_size_limit by
// accounting for differences in the non-zeros for each element, rather than
// conservatively assuming they're all at max (as in the LRU case). The
// downside is that we need to do some copying within the cache to make space
// for new reads, but that works out to be no more than one extra copy per
// fetch() call, which is tolerable. I suppose we could do better by
// defragmenting within this buffer but that's probably overkill.
template<typename Index_, typename CachedValue_, typename CachedIndex_>
class ContiguousOracleSlabCache {
private:
    std::vector<CachedValue_> cache_value;
    std::vector<CachedIndex_> cache_index;
    std::unordered_map<Index_, Index_> cache_exists, next_cache_exists;

    struct Element {
        size_t data_offset;
        size_t mem_offset;
        Index_ length;
    };
    std::vector<Element> cache_data, next_cache_data;

    std::vector<Index_> needed;
    std::vector<Index_> present;

    std::shared_ptr<const tatami::Oracle<Index_> > oracle;
    size_t max_cache_elements;
    size_t counter = 0, total = 0, future = 0;
    bool needs_cached_index, needs_cached_value;

public:
    ContiguousOracleSlabCache(std::shared_ptr<const tatami::Oracle<Index_> > ora, size_t cache_size, size_t min_elements, bool needs_cached_value, bool needs_cached_index) : 
        oracle(std::move(ora)), 
        max_cache_elements([&]() -> size_t {
            size_t elsize = size_of_element<CachedIndex_, CachedValue_>(needs_cached_value, needs_cached_index);
            if (elsize == 0) {
                return min_elements;
            } else {
                auto proposed = cache_size / elsize;
                return std::max(min_elements, proposed);
            } 
        }()),
        total(oracle->total()), 
        needs_cached_index(needs_cached_index), 
        needs_cached_value(needs_cached_value)
    {
        if (needs_cached_index) {
            cache_index.resize(max_cache_elements);
        }
        if (needs_value) {
            cache_value.resize(max_cache_elements);
        }
    }

public:
    void populate(const std::vector<hsize_t>& pointers, Components& h5comp) {
        size_t filled_elements = 0;
        needed.clear();
        present.clear();
        next_cache_data.clear();
        next_cache_exists.clear();

        for (; future < total; ++future) {
            Index_ current = oracle->get(future);

            // Seeing if this element already exists somewhere.
            auto nit = next_cache_exists.find(current);
            if (nit != next_cache_exists.end()) {
                continue;
            }

            auto it = cache_exists.find(current);
            if (it != cache_exists.end()) {
                auto& candidate = cache_data[it->second];
                filled_elements += candidate.length;
                if (filled_elements > max_cache_elements) { // note that at least one candidate.length will always fit in max_cache_elements due to the min_elements in the constructor.
                    break;
                }

                Index_ used = next_cache_data.size();
                present.push_back(used);
                next_cache_exists[current] = used;
                next_cache_data.push_back(std::move(candidate));
                continue;
            }

            // Considering extraction from file.
            hsize_t extraction_start = pointers[current];
            hsize_t extraction_len = pointers[current + 1] - extraction_start;
            filled_elements += extraction_len;
            if (filled_elements > max_cache_elements) { // see note above.
                break;
            }

            Index_ used = next_cache_data.size();
            needed.emplace_back(used);
            next_cache_exists[current] = used;

            Element latest;
            latest.data_offset = extraction_start;
            latest.length = extraction_len;
            next_cache_data.push_back(std::move(latest));
        }

        if (!needed.empty() && present.size()) {
            size_t dest_offset = 0;

            if (present.size()) {
                // Shuffling all re-used elements to the start of the buffer,
                // so that we can perform a contiguous extraction of the needed
                // elements in the rest of the buffer. This needs some sorting
                // to ensure that we're not clobbering one re-used element's
                // contents when shifting another element to the start.
                sort_by_field(present, [&next_cache_data](size_t i) -> size_t { return next_cache_data[i].mem_offset; });

                for (auto p : present) {
                    auto& info = next_cache_data[p];
                    if (needs_cached_index) {
                        auto isrc = cache_index.begin() + info.mem_offset;
                        std::copy(isrc, isrc + info.length, cache_index.begin() + dest_offset);
                    }
                    if (needs_cached_value) {
                        auto vsrc = cache_value.begin() + info.mem_offset;
                        std::copy(vsrc, vsrc + info.length, cache_value.begin() + dest_offset); 
                    }
                    info.mem_offset = dest_offset;
                    dest_offset += info.length;
                }
            }

            // Sorting so that we get consecutive accesses in the hyperslab construction.
            // This should improve re-use of partially read chunks inside the HDF5 call.
            sort_by_field(needed, [&next_data_cache](size_t i) -> size_t { return next_cache_data[i].data_offset; });

            serialize([&]() -> void {
                size_t sofar = 0;
                hsize_t combined_len = 0;
                h5comp->dataspace.selectNone();

                while (sofar < needed.size()) {
                    auto& first = next_cache_data[needed[sofar]];
                    first.mem_offset = dest_offset + combined_len;
                    hsize_t src_offset = first.data_offset;
                    hsize_t len = first.length;
                    ++sofar;

                    // Finding the stretch of consecutive extractions, and bundling them into a single hyperslab.
                    for (; sofar < needed.size(); ++sofar) {
                        auto& next = next_cache_data[needed[sofar]];
                        if (src_offset + len < next.data_offset) {
                            break;
                        }
                        next.mem_offset = first.mem_offset + len;
                        len += next.length;
                    }

                    h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &len, &src_offset);
                    combined_len += len;
                }

                h5comp->memspace.setExtentSimple(1, &combined_len);
                h5comp->memspace.selectAll();
                if (needs_cached_index) {
                    h5comp->index.read(cache_index.data() + dest_offset, define_mem_type<CachedIndex_>(), h5comp->memspace, h5comp->dataspace);
                }
                if (needs_value) {
                    h5comp->data.read(cache_value.data() + dest_offset, define_mem_type<CachedValue_>(), h5comp->memspace, h5comp->dataspace);
                }
            });
        }

        cache_data.swap(next_cache_data);
        cache_exists.swap(next_cache_exists);
    }

public:
    Chunk next(Components& h5comp) {
        if (counter == future) {
            populate(h5comp);
        }
        auto current = oracle->get(counter++);
        const auto& info = cache_data[cache_exists.find(current)->second];

        Chunk output;
        output.length = info.length;
        if (needs_cached_value) {
            output.value = cached_value.data() + info.mem_offset;
        }
        if (needs_cached_index) {
            output.index = cached_index.data() + info.mem_offset;
        }
        return output;
    }
};

// All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
class PrimaryBase {
public:
    PrimaryBase(const std::string& file_name, const std::string& data_name, const std::string& index_name, const std::vector<hsize_t>& ptrs) : pointers(ptrs) {
        serialize([&]() -> void {
           h5comp.reset(new Components);

            // TODO: set more suitable chunk cache values here, to avoid re-reading
            // chunks that are only partially consumed.
            h5comp->file.openFile(file_name, H5F_ACC_RDONLY);
            h5comp->data = h5comp->file.openDataSet(data_name);
            h5comp->index = h5comp->file.openDataSet(index_name);
            h5comp->dataspace = h5comp->data.getSpace();
        });
    }

    ~PrimaryBase() {
        serialize([&]() {
            h5comp.reset();
        });
    }

protected:
    const std::vector<hsize_t>& pointers;
    std::unique_ptr<Components> h5comp;
};

/****************************************
 **** Virtual classes for extractors ****
 ****************************************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruBase : public PrimaryBase {
    struct Slab {
        Slab(size_t capacity, bool needs_cached_value, bool needs_cached_index) : 
            value(needs_cached_value ? capacity : 0), index(needs_cached_index ? capacity : 0) {}
        std::vector<CachedValue_> value;
        std::vector<CachedIndex_> index;
        Index_ length;
    };

private:
    tatami_chunked::LruSlabCache<Index_, Slab> cache;
    size_t max_non_zeros;
    bool needs_cached_value, needs_cached_index;

public:
    PrimaryLruBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami::MaybeOracle<false, Index_>,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        PrimaryBase(file_name, data_name, index_name, ptrs),
        cache([&]() -> size_t {
            // Always return at least one slab, so that cache.find() is valid.
            if (max_non_zeros == 0) {
                return 1;
            }

            auto elsize = size_of_element<CachedValue_, CachedIndex_>(needs_cached_value, needs_cached_index);
            if elsize == 0 {
                return 1;
            }

            auto num_slabs = cache_size / (max_non_zeros * elsize);
            if (num_slabs == 0) {
                return 1;
            }

            return num_slabs;
        }()),
        max_non_zeros(max_non_zeros),
        needs_cached_value(needs_cached_value),
        needs_cached_index(needs_cached_index)
    {}

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch(Index_ i) {
        const auto& slab = cache.find(
            i, 
            /* create = */ [&]() -> LruSlab {
                return Slab(max_non_zeros, needs_cached_value, needs_cached_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - pointers[i];
                current_cache.length = extraction_len;

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    if (needs_cached_index) {
                        this->h5comp->index.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                    if (needs_value) {
                        this->h5comp->data.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            }
        );

        Chunk<Index_, CachedValue_, CachedIndex_> output;
        output.length = slab.length;
        if (needs_cached_value) {
            output.value = slab.value.data();
        }
        if (needs_cached_index) {
            output.index = slab.index.data();
        }
        return output;
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracleBase : public PrimaryBase {
private:
    ContiguousOracleSlabCache<Index_, CachedValue_, CachedIndex_> cache;

public:
    PrimaryOracleBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        PrimaryBase(file_name, data_name, index_name, ptrs),
        cache(std::move(oracle), cache_size, max_non_zeros, needs_cached_value, needs_cached_index)
    {}

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch(Index_) {
        return cache.next(*(this->h5comp));
    }
};

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryBase = typename std::conditional<oracle_, PrimaryOracleBase<Index_, CachedValue_, CachedIndex_>, PrimaryLruBase<Index_, CachedValue_, CachedIndex_> >::type;

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullSparse : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
    PrimaryFullSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        [[maybe_unused]] Index_ secondary_dim, // for consistency only.
        tatami::MaybeOracle<oracle_, Index_> oracle,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, needs_value, needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        tatami::SparseRange<Value_, Index_> output;
        auto chunk = this->fetch(i);
        if (chunk.value) {
            std::copy_n(chunk.value, chunk.length, vbuffer);
            output.value = vbuffer;
        }
        if (chunk.index) {
            std::copy_n(chunk.index, chunk.length, ibuffer);
            output.index = ibuffer;
        }
        return output;
    }
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullDense : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
   PrimaryFullDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        size_t cache_size,
        size_t max_non_zeros) :
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, true, true),
        secondary_dim(secondary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        std::fill_n(buffer, secondary_dim, 0);
        auto chunk = this->fetch(i);
        for (Index_ j = 0; j < chunk.length; ++j) {
            buffer[chunk.index[j]] = chunk.value[j];
        }
        return output;
    }

private:
    Index_ secondary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryBlockSparse : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
    PrimaryBlockSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, needs_value, true)
        block_start(block_start),
        block_length(block_length),
        secondary_dim(secondary_dim),
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto chunk = this->fetch(i);
        auto start = chunk.index, end = chunk.index + chunk.length;
        auto original = start;
        tatami::sparse_utils::refine_primary_limits(start, end, secondary_dim, block_start, block_start + block_length); // WARNING: not documented by tatami!
        chunk.length = end - start;

        tatami::SparseRange<Value_, Index_> output;
        if (chunk.value) {
            std::copy_n(chunk.value + (start - original), chunk.length, vbuffer);
            output.value = vbuffer;
        }
        if (needs_index) {
            std::copy(start, end, ibuffer);
            output.index = ibuffer;
        }
        return output;
    }

private:
    Index_ block_start;
    Index_ block_length;
    Index_ secondary_dim;
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryBlockDense : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
   PrimaryBlockDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        size_t max_non_zeros) :
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, true, true),
        block_start(block_start),
        block_length(block_length),
        secondary_dim(secondary_dim),
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto chunk = this->fetch(i);
        auto start = chunk.index, end = chunk.index + chunk.length;
        auto original = start;
        tatami::sparse_utils::refine_primary_limits(start, end, secondary_dim, block_start, block_start + block_length); // WARNING: not documented by tatami!

        std::fill_n(buffer, block_length, 0);
        auto valptr = chunk.value + (start - original);
        for (; start != end; ++start, ++valptr) {
            buffer[*start] = *valptr;
        }
        return output;
    }

private:
    Index_ block_start;
    Index_ block_length;
    Index_ secondary_dim;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryIndexSparse : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
    PrimaryIndexSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, needs_value, true),
        retriever(*indices_ptr, secondary_dim), 
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        Index_ count = 0;
        auto vcopy = vbuffer;
        auto icopy = ibuffer;

        auto chunk = this->fetch(i);
        bool needs_value = chunk.value != NULL;
        retriever.populate(
            chunk.index,
            chunk.index + chunk.length,
            [&](size_t offset, Index_ ix) {
                ++count;
                if (needs_value) {
                    *vcopy = *(vIt + offset);
                    ++vcopy;
                }
                if (needs_index) {
                    *icopy = ix;
                    ++icopy;
                }
            }
        );

        return SparseRange<Value_, Index_>(count, needs_value ? vbuffer : NULL, needs_index ? ibuffer : NULL);
    }

private:
    tatami::VectorPtr<Index_> indices_ptr;
    tatami::sparse_utils::RetrievePrimarySubsetSparse<Index_> retriever; // WARNING: not documented by tatami!
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryIndexDense : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
   PrimaryIndexDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        size_t cache_size,
        size_t max_non_zeros) :
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, true, true),
        retriever(*indices_ptr, secondary_dim),
        num_indices(indices_ptr->size())
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        std::fill_n(buffer, num_indices, 0);

        auto chunk = this->fetch(i);
        retriever.populate(
            chunk.index,
            chunk.index + chunk.length,
            [&](size_t offset, Index_ ix) {
                buffer[ix] = *(vIt + offset);
            }
        );

        return buffer;
    }

private:
    tatami::VectorPtr<Index_> indices_ptr;
    tatami::sparse_utils::RetrievePrimarySubsetDense<Index_> retriever; // WARNING: not documented by tatami!
    size_t num_indices;
};

}

}

#endif
