#ifndef TATAMI_HDF5_SPARSE_PRIMARY_HPP
#define TATAMI_HDF5_SPARSE_PRIMARY_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include "serialize.hpp"
#include "utils.hpp"

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

template<typename CachedValue_, typename CachedIndex_>
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

template<bool sparse_, typename Index_>
using SparseRemapVector = typename std::conditional<sparse_, std::vector<uint8_t>, std::vector<Index_> >::type;

template<bool sparse_, typename Index_>
void populate_sparse_remap_vector(SparseRemapVector<sparse_, Index_>& remap, Index_& first_index, Index_& past_last_index) {
    if (indices.empty()) {
        return;
    }

    first_index = indices.front();
    past_last_index = indices.back() + 1;
    remap.resize(past_last_index - first_index);

    if constexpr(sparse_) {
        for (auto i : indices) {
            remap[i - first_index] = 1;
        }
    } else {
        // We start from +1 so that we can still use 0 as an 'absent' counter.
        // This should not overflow as Index_ should be large enough to hold
        // the dimension extent so it should be able to tolerate +1.
        Index_ counter = 1;
        for (auto i : indices) {
            remap[i - first_index] = counter;
            ++counter;
        }
    }
}

// Unfortunately we can't use tatami::SparseRange, as CachedIndex_ might not be
// large enough to represent the number of cached indices, e.g., if there were
// 256 indices, a CachedIndex_=uint8_t type would be large enough to represent
// each index (0 - 255) but not the number of indices.
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
    std::vector<CachedValue_> full_value_buffer;
    std::vector<CachedIndex_> full_index_buffer;
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
    bool needs_cached_value, needs_cached_index;

public:
    ContiguousOracleSlabCache(std::shared_ptr<const tatami::Oracle<Index_> > ora, size_t cache_size, size_t min_elements, bool needs_cached_value, bool needs_cached_index) : 
        oracle(std::move(ora)), 
        max_cache_elements([&]() -> size_t {
            size_t elsize = size_of_cached_element<CachedIndex_, CachedValue_>(needs_cached_value, needs_cached_index);
            if (elsize == 0) {
                return min_elements;
            } else {
                auto proposed = cache_size / elsize;
                return std::max(min_elements, proposed);
            } 
        }()),
        total(oracle->total()), 
        needs_cached_value(needs_cached_value),
        needs_cached_index(needs_cached_index) 
    {
        if (needs_cached_index) {
            full_index_buffer.resize(max_cache_elements);
        }
        if (needs_cached_value) {
            full_value_buffer.resize(max_cache_elements);
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

        if (!needed.empty()) {
            size_t dest_offset = 0;

            if (present.size()) {
                // Shuffling all re-used elements to the start of the buffer,
                // so that we can perform a contiguous extraction of the needed
                // elements in the rest of the buffer. This needs some sorting
                // to ensure that we're not clobbering one re-used element's
                // contents when shifting another element to the start.
                sort_by_field(present, [&](size_t i) -> size_t { return next_cache_data[i].mem_offset; });

                for (auto p : present) {
                    auto& info = next_cache_data[p];
                    if (needs_cached_index) {
                        auto isrc = full_index_buffer.begin() + info.mem_offset;
                        std::copy(isrc, isrc + info.length, full_index_buffer.begin() + dest_offset);
                    }
                    if (needs_cached_value) {
                        auto vsrc = full_value_buffer.begin() + info.mem_offset;
                        std::copy(vsrc, vsrc + info.length, full_value_buffer.begin() + dest_offset); 
                    }
                    info.mem_offset = dest_offset;
                    dest_offset += info.length;
                }
            }

            // Sorting so that we get consecutive accesses in the hyperslab construction.
            // This should improve re-use of partially read chunks inside the HDF5 call.
            sort_by_field(needed, [&](size_t i) -> size_t { return next_cache_data[i].data_offset; });

            serialize([&]() -> void {
                size_t sofar = 0;
                hsize_t combined_len = 0;
                h5comp.dataspace.selectNone();

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

                    h5comp.dataspace.selectHyperslab(H5S_SELECT_OR, &len, &src_offset);
                    combined_len += len;
                }

                h5comp.memspace.setExtentSimple(1, &combined_len);
                h5comp.memspace.selectAll();
                if (needs_cached_index) {
                    h5comp.index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), h5comp.memspace, h5comp.dataspace);
                }
                if (needs_cached_value) {
                    h5comp.data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), h5comp.memspace, h5comp.dataspace);
                }
            });
        }

        cache_data.swap(next_cache_data);
        cache_exists.swap(next_cache_exists);
    }

public:
    template<typename ... Args_>
    Chunk<Index_, CachedValue_, CachedIndex_> next(const std::vector<hsize_t>& pointers, Components& h5comp) {
        if (counter == future) {
            populate(pointers, h5comp);
        }
        auto current = oracle->get(counter++);
        const auto& info = cache_data[cache_exists.find(current)->second];

        Chunk<Index_, CachedValue_, CachedIndex_> output;
        output.length = info.length;
        if (needs_cached_value) {
            output.value = full_value_buffer.data() + info.mem_offset;
        }
        if (needs_cached_index) {
            output.index = full_index_buffer.data() + info.mem_offset;
        }
        return output;
    }
};

/****************************************
 **** Virtual classes for extractors ****
 ****************************************/

// All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
class PrimaryBase {
public:
    PrimaryBase(const std::string& file_name, const std::string& data_name, const std::string& index_name, const std::vector<hsize_t>& ptrs) : pointers(ptrs) {
        serialize([&]() -> void {
           h5comp.reset(new Components);

            // TODO: set more suitable chunk cache values here, to avoid re-reading
            // chunks that are only partially consumed.
            h5comp->file.openFile(file_name, H5F_ACC_RDONLY);
            h5comp->data_dataset = h5comp->file.openDataSet(data_name);
            h5comp->index_dataset = h5comp->file.openDataSet(index_name);
            h5comp->dataspace = h5comp->data_dataset.getSpace();
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

template<typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruBase : public PrimaryBase {
    struct Slab {
        Slab(size_t capacity, bool needs_cached_value, bool needs_cached_index) : 
            value(needs_cached_value ? capacity : 0), index(needs_cached_index ? capacity : 0) {}
        std::vector<CachedValue_> value;
        std::vector<CachedIndex_> index;
        Index_ length;

        Chunk<Index_, CachedValue_, CachedIndex_> as_chunk() const {
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

protected:
    tatami_chunked::LruSlabCache<Index_, Slab> cache;
    size_t max_non_zeros;
    bool needs_cached_value, needs_cached_index;

public:
    PrimaryLruBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami::MaybeOracle<false, Index_>, // for consistency with the oracular constructors.
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

            auto elsize = size_of_cached_element<CachedValue_, CachedIndex_>(needs_cached_value, needs_cached_index);
            if (elsize == 0) {
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
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruFullBase : public PrimaryLruBase<Index_, CachedValue_, CachedIndex_> {
    PrimaryLruFullBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        tatami::MaybeOracle<false, Index_> oracle, 
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        PrimaryLruBase<Index_, CachedValue_, CachedIndex_>(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_nonzeros, needs_cached_value, needs_cached_index)
    {}

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ i) {
        const auto& slab = cache.find(
            i, 
            /* create = */ [&]() -> Slab {
                return Slab(max_non_zeros, needs_cached_value, needs_cached_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - pointers[i];
                current_cache.length = extraction_len;
                if (extraction_len == 0) {
                    return;
                }

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    if (this->needs_cached_index) {
                        this->h5comp->index_dataset.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                    if (this->needs_cached_value) {
                        this->h5comp->data_dataset.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            }
        );

        return this->slab_to_chunk(slab);
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruBlockBase : public PrimaryLruBase<Index_, CachedValue_, CachedIndex_> {
    PrimaryLruBlockBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<false, Index_> oracle, 
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        PrimaryLruBase<Index_, CachedValue_, CachedIndex_>(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_nonzeros, needs_cached_value, needs_cached_index),
        secondary_dim(secondary_dim),
        secondary_start(block_start),
        secondary_past_end(block_start + block_length)
    {}

private:
    Index_ secondary_dim;
    Index_ secondary_start, secondary_past_end;
    std::vector<CachedIndex_> index_buffer;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ i) {
        const auto& slab = cache.find(
            i, 
            /* create = */ [&]() -> Slab {
                return Slab(max_non_zeros, needs_cached_value, needs_cached_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - pointers[i];
                index_buffer.resize(extraction_len);
                if (extraction_len == 0) {
                    return;
                }

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    this->h5comp->index_dataset.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);

                    auto indices_start = index_buffer.begin();
                    auto indices_end = index_end.begin();
                    if (secondary_start) {
                        indices_start = std::lower_bound(indices_start, indices_end, secondary_start, [](Index_ a, Index_ b) -> bool { return a < b; });
                    }
                    if (secondary_past_end != secondary_dim) {
                        indices_end = std::lower_bound(indices_start, indices_end, secondary_past_end, [](Index_ a, Index_ b) -> bool { return a < b; });
                    }

                    extraction_len = indices_end - indices_start;
                    if (extraction_len == 0) {
                        return;
                    }

                    // Shifting the indices to the front.
                    if (this->needs_cached_index) {
                        std::copy(indices_start, indices_end, current_cache.index.begin());
                    }
                    if (this->needs_cached_value) {
                        extraction_start += indices_start - index_buffer.begin();
                        this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                        this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                        this->h5comp->memspace.selectAll();
                        this->h5comp->data_dataset.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });

                current_cache.length = extraction_len;
            }
        );

        return this->slab_to_chunk(slab);
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruIndexBase : public PrimaryruBase<Index_, CachedValue_, CachedIndex_> {
    PrimaryLruIndexBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<false, Index_> oracle, 
        const std::vector<Index_>& indices,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        PrimaryLruBase<Index_, CachedValue_, CachedIndex_>(file_name, data_name, index_name, ptrs, std::move(oracle), cache_size, max_nonzeros, needs_cached_value, needs_cached_index),
        secondary_dim(secondary_dim)
    {
        populate_sparse_remap_vector(remap, first_index, past_last_index);
    }

private:
    Index_ secondary_dim;
    Index_ first_index, past_last_index;
    SparseRemapVector<sparse_, Index_> remap;
    std::vector<CachedIndex_> index_buffer;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ i) {
        const auto& slab = cache.find(
            i, 
            /* create = */ [&]() -> Slab {
                return Slab(max_non_zeros, needs_cached_value, needs_cached_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - pointers[i];
                index_buffer.resize(extraction_len);
                if (extraction_len == 0) {
                    return;
                }

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    this->h5comp->index_dataset.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);

                    auto indices_start = index_buffer.begin();
                    auto indices_end = index_end.begin();
                    if (first_index) {
                        indices_start = std::lower_bound(indices_start, indices_end, first_index, [](Index_ a, Index_ b) -> bool { return a < b; });
                    }
                    if (past_last_index != secondary_dim) {
                        indices_end = std::lower_bound(indices_start, indices_end, past_last_index, [](Index_ a, Index_ b) -> bool { return a < b; });
                    }
                    if (indices_start == indices_end) {
                        return;
                    }

                    extraction_len = 0;
                    auto cacheIIt = current_cache.indices.begin();
                    Index_ last_value = secondary_dim;
                    if (this->needs_cached_value) {
                        this->h5comp->dataspace.selectNone();
                    }

                    for (auto x = indices_start; x != indices_end; ++x) {
                        auto offset = *x - first_index;
                        auto present = remap[offset];
                        if (!present) {
                            continue;
                        }

                        if (this->needs_cached_index) {
                            if constexpr(sparse_) {
                                *cacheIit = *x;
                            } else {
                                *cacheIIt = present;
                            }
                            ++cacheIIt;
                        }

                        if (this->needs_cached_value) {

                        }

                        ++extraction_len;
                    }

                    if (this->needs_cached_value) {
                        this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                        this->h5comp->memspace.selectAll();
                        this->h5comp->data_dataset.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });

                current_cache.length = extraction_len;
            }
        );

        return this->slab_to_chunk(slab);
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
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_) {
        return cache.next(this->pointers, *(this->h5comp));
    }
};

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryBase = typename std::conditional<oracle_, PrimaryOracleBase<Index_, CachedValue_, CachedIndex_>, PrimaryLruBase<Index_, CachedValue_, CachedIndex_> >::type;

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryFullSparse : 
    public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>, 
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
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
        ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            needs_value, 
            needs_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        tatami::SparseRange<Value_, Index_> output;
        auto chunk = this->fetch_raw(i);
        output.number = chunk.length;
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
struct PrimaryFullDense : 
    public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>,
    public tatami::DenseExtractor<oracle_, Value_, Index_> 
{
    PrimaryFullDense(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        size_t cache_size,
        size_t max_non_zeros) :
        ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            file_name, 
            data_name, 
            index_name, 
            ptrs, 
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            true, 
            true
        ),
        secondary_dim(secondary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        std::fill_n(buffer, secondary_dim, 0);
        auto chunk = this->fetch_raw(i);
        for (Index_ j = 0; j < chunk.length; ++j) {
            buffer[chunk.index[j]] = chunk.value[j];
        }
        return buffer;
    }

private:
    Index_ secondary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryBlockSparse : 
    public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>,
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
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
        ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            file_name,
            data_name, 
            index_name, 
            ptrs, 
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            needs_value, 
            true
        ),
        block_start(block_start),
        block_length(block_length),
        secondary_dim(secondary_dim),
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto chunk = this->fetch_raw(i);
        auto start = chunk.index, end = chunk.index + chunk.length;
        auto original = start;
        refine_primary_limits(start, end, secondary_dim, block_start, block_start + block_length);

        tatami::SparseRange<Value_, Index_> output;
        output.number = end - start;
        if (chunk.value) {
            std::copy_n(chunk.value + (start - original), output.number, vbuffer);
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
struct PrimaryBlockDense : 
    public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>,
    public tatami::DenseExtractor<oracle_, Value_, Index_> 
{
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
        ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            file_name,
            data_name,
            index_name,
            ptrs,
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            true, 
            true
        ),
        block_start(block_start),
        block_length(block_length),
        secondary_dim(secondary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto chunk = this->fetch_raw(i);
        auto start = chunk.index, end = chunk.index + chunk.length;
        auto original = start;
        refine_primary_limits(start, end, secondary_dim, block_start, block_start + block_length);

        std::fill_n(buffer, block_length, 0);
        auto valptr = chunk.value + (start - original);
        for (; start != end; ++start, ++valptr) {
            buffer[*start - block_start] = *valptr;
        }
        return buffer;
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
struct PrimaryIndexSparse : 
    public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>,
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
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
        ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            file_name,
            data_name,
            index_name,
            ptrs, 
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            needs_value, 
            true
        ),
        retriever(*indices_ptr, secondary_dim), 
        needs_index(needs_index)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        Index_ count = 0;
        auto vcopy = vbuffer;
        auto icopy = ibuffer;

        auto chunk = this->fetch_raw(i);
        bool needs_value = chunk.value != NULL;
        retriever.populate(
            chunk.index,
            chunk.index + chunk.length,
            [&](size_t offset, Index_ ix) {
                ++count;
                if (needs_value) {
                    *vcopy = *(chunk.value + offset);
                    ++vcopy;
                }
                if (needs_index) {
                    *icopy = ix;
                    ++icopy;
                }
            }
        );

        return tatami::SparseRange<Value_, Index_>(count, needs_value ? vbuffer : NULL, needs_index ? ibuffer : NULL);
    }

private:
    tatami::sparse_utils::RetrievePrimarySubsetSparse<Index_> retriever;
    bool needs_index;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryIndexDense : 
    public ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>,
    public tatami::DenseExtractor<oracle_, Value_, Index_> 
{
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
        ConditionalPrimaryBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            file_name,
            data_name,
            index_name,
            ptrs, 
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            true, 
            true
        ),
        retriever(*indices_ptr, secondary_dim),
        num_indices(indices_ptr->size())
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        std::fill_n(buffer, num_indices, 0);

        auto chunk = this->fetch_raw(i);
        retriever.populate(
            chunk.index,
            chunk.index + chunk.length,
            [&](Index_ ix, Index_ offset) {
                buffer[ix] = *(chunk.value + offset);
            }
        );

        return buffer;
    }

private:
    tatami::sparse_utils::RetrievePrimarySubsetDense<Index_> retriever;
    size_t num_indices;
};

}

}

#endif
