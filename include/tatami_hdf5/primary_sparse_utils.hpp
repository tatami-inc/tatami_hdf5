#ifndef TATAMI_HDF5_PRIMARY_SPARSE_UTILS_HPP
#define TATAMI_HDF5_PRIMARY_SPARSE_UTILS_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

template<typename Index_, class Function_>
void sort_by_field(std::vector<Index_>& indices, Function_ field) {
    auto comp = [&field](size_t l, size_t r) -> bool {
        return field(l) < field(r);
    };
    if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
        std::sort(indices.begin(), indices.end(), comp);
    }
}

template<typename CachedValue_, typename CachedIndex_ = Index_>
size_t size_of_cached_element(bool needs_cached_value, bool needs_cached_index) {
    return (needs_cached_index ? sizeof(CachedIndex_) : 0) + (needs_cached_value ? sizeof(CachedValue_) : 0);
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
    template<class LoadFun_> 
    void populate(const std::vector<hsize_t>& pointers, LoadFun_ load) {
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

            load(needed, next_cache_data);
        }

        cache_data.swap(next_cache_data);
        cache_exists.swap(next_cache_exists);
    }

public:
    Chunk next() {
        if (counter == future) {
            populate();
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
    PrimaryBase(const std::string& file_name, const std::string& data_name, const std::string& index_name, const std::vector<hsize_t>& ptrs) : pointers(ptrs) {
        serialize([&]() -> void {
           h5comp.reset(new PrimaryComponents);

            // TODO: set more suitable chunk cache values here, to avoid re-reading
            // chunks that are only partially consumed.
            h5comp->file.openFile(file_name, H5F_ACC_RDONLY);
            h5comp->data = h5comp->file.openDataSet(data_name);
            h5comp->index = h5comp->file.openDataSet(index_name);
            h5comp->dataspace = h5comp->data.getSpace();
        });
    }

protected:
    const std::vector<hsize_t>& pointers;

    struct Components {
        H5::H5File file;
        H5::DataSet data_dataset;
        H5::DataSet index_dataset;
        H5::DataSpace dataspace;
        H5::DataSpace memspace;
    };

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
        return cache.next();
    }
};

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryBase = typename std::conditional<oracle_, PrimaryOracleBase<Index_, CachedValue_, CachedIndex_>, PrimaryLruBase<Index_, CachedValue_, CachedIndex_> >::type;

/************************************
 **** Concrete extractor classes ****
 ************************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullSparse : public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_> {
    PrimaryFullSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, needs_cached_value, needs_cached_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        tatami::SparseRange<Value_, Index_> output;
        auto chunk = this->fetch(i);
        if (chunk.value) {
            std::copy_n(chunk.value, chunk.length, vbuffer);
            output.value = vbuffer;
        }
        if (chunk.index) {
            std::copy_n(chunk.index, chunk.length, vbuffer);
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
        tatami::MaybeOracle<oracle_, Index_> oracle,
        size_t cache_size,
        size_t max_non_zeros,
        Index_ extracted_length) :
        ConditionalPrimaryBase(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_non_zeros, true, true),
        extracted_length
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        std::fill(
        auto chunk = this->fetch(i);
        for (Index_ j = 0; j < chunk.length; ++j) {

        }
        if (chunk.value) {
            std::copy_n(chunk.value, chunk.length, vbuffer);
            output.value = vbuffer;
        }
        if (chunk.index) {
            std::copy_n(chunk.index, chunk.length, vbuffer);
            output.index = ibuffer;
        }
        return output;
    }
};


}

}

#endif
