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

/***************************
 **** General utitities ****
 ***************************/

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

template<typename Index_>
struct MatrixDetails {
    MatrixDetails(const std::string& f, const std::string& d, const std::string& i, const std::vector<hsize_t>& p, Index_ s) :
        file_name(f), data_name(d), index_name(i), pointers(p), secondary_dim(s) {}

    const std::string& file_name;
    const std::string& data_name;
    const std::string& index_name;
    const std::vector<hsize_t>& pointers;
    Index_ secondary_dim;
};

// Unfortunately we can't use tatami::SparseRange, as CachedIndex_ might not be
// large enough to represent the number of cached indices, e.g., if there were
// 256 indices, a CachedIndex_=uint8_t type would be large enough to represent
// each index (0 - 255) but not the number of indices.
template<typename Index_, typename CachedValue_, typename CachedIndex_ = Index_>
struct Chunk { 
    const CachedValue_* value = NULL;
    const CachedIndex_* index = NULL;
    Index_ length = 0;

    template<typename Value_>
    tatami::SparseRange<Value_, Index_> to_sparse(Value_* vbuffer, Index_* ibuffer) const {
        tatami::SparseRange<Value_, Index_> output(length);
        if (value) {
            std::copy_n(value, length, vbuffer);
            output.value = vbuffer;
        }
        if (index) {
            std::copy_n(index, length, ibuffer);
            output.index = ibuffer;
        }
        return output;
    }

    template<typename Value_>
    Value_* to_dense(Value_* buffer, Index_ extract_length) const {
        std::fill_n(buffer, extract_length, 0);
        for (Index_ i = 0; i < length; ++i) {
            buffer[index[i]] = value[i];
        }
        return buffer;
    }
};

/***********************************
 **** Oracular cache definition ****
 ***********************************/

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
public:
    struct Element {
        size_t data_offset;
        size_t mem_offset;
        Index_ length;
    };

private:
    const std::vector<hsize_t>& pointers;

    std::vector<CachedValue_> full_value_buffer;
    std::vector<CachedIndex_> full_index_buffer;

    std::unordered_map<Index_, Index_> cache_exists, next_cache_exists;
    std::vector<Element> cache_data, next_cache_data;
    std::vector<Index_> needed;
    std::vector<Index_> present;

    std::shared_ptr<const tatami::Oracle<Index_> > oracle;
    size_t max_cache_elements;
    size_t counter = 0, total = 0, future = 0;

public:
    ContiguousOracleSlabCache(
        const std::vector<hsize_t>& ptrs, 
        std::shared_ptr<const tatami::Oracle<Index_> > ora, 
        size_t cache_size, 
        size_t min_elements, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        pointers(ptrs),
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
        total(oracle->total())
    {
        if (needs_cached_value) {
            full_value_buffer.resize(max_cache_elements);
        }
        if (needs_cached_index) {
            full_index_buffer.resize(max_cache_elements);
        }
    }

private:
    template<class Function_>
    void sort_by_field(std::vector<Index_>& indices, Function_ field) {
        auto comp = [&field](size_t l, size_t r) -> bool {
            return field(l) < field(r);
        };
        if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
            std::sort(indices.begin(), indices.end(), comp);
        }
    }

public:
    // Note that needs_cached_index in the constructor may not always equal
    // needs_index. When doing block/indexed extraction, we need the index to
    // determine the block/index (and thus it needs to be loaded into the cache
    // at some point, and thus we need to consider it in the cache size
    // calculations) even if we don't want to report those indices. In those
    // cases, needs_cached_index=true but we might have needs_index=false,
    // hence the separate arguments here versus the constructor. We just do the
    // same for needs_(cached_)value just for consistency.
    template<class Extract_>
    void populate(Extract_ extract, bool needs_value, bool needs_index) {
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
                    if (needs_index) {
                        auto isrc = full_index_buffer.begin() + info.mem_offset;
                        std::copy(isrc, isrc + info.length, full_index_buffer.begin() + dest_offset);
                    }
                    if (needs_value) {
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

            extract(dest_offset, needed, next_cache_data, full_value_buffer, full_index_buffer);
        }

        cache_data.swap(next_cache_data);
        cache_exists.swap(next_cache_exists);
    }

public:
    template<class Extract_>
    Chunk<Index_, CachedValue_, CachedIndex_> next(Extract_&& extract, bool needs_value, bool needs_index) {
        if (counter == future) {
            populate(std::forward<Extract_>(extract), needs_value, needs_index);
        }
        auto current = oracle->get(counter++);
        const auto& info = cache_data[cache_exists.find(current)->second];

        Chunk<Index_, CachedValue_, CachedIndex_> output;
        output.length = info.length;
        if (needs_value) {
            output.value = full_value_buffer.data() + info.mem_offset;
        }
        if (needs_index) {
            output.index = full_index_buffer.data() + info.mem_offset;
        }
        return output;
    }
};

/*************************************
 **** Subset extraction utilities ****
 *************************************/

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

template<bool sparse_, typename Index_>
using SparseRemapVector = typename std::conditional<sparse_, std::vector<uint8_t>, std::vector<Index_> >::type;

template<bool sparse_, typename Index_>
void populate_sparse_remap_vector(const std::vector<Index_>& indices, SparseRemapVector<sparse_, Index_>& remap, Index_& first_index, Index_& past_last_index) {
    if (indices.empty()) {
        first_index = 0;
        past_last_index = 0;
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

template<bool sparse_, typename In_, typename Index_, typename Output_, typename Map_>
Index_ scan_for_indices_in_remap_vector(
    In_ indices_start, 
    In_ indices_end,
    Index_ first_index,
    Output_ output, 
    std::vector<Index_>& found,
    const std::vector<Map_>& remap,
    bool needs_value,
    bool needs_index) 
{
    Index_ counter = 0;
    found.clear();
    Index_ num_found = 0;

    for (auto x = indices_start; x != indices_end; ++x, ++counter) {
        auto present = remap[*x - first_index];
        if (present) {
            if (needs_index) {
                if constexpr(sparse_) {
                    *output = *x;
                } else {
                    // For dense extraction, we store the position on 'indices', to
                    // make life easier when filling up the output vector.
                    // Remember that we +1'd in 'populate_sparse_remap_vector',
                    // so we have to undo it here.
                    *output = present - 1;
                }
                ++output;
            }
            if (needs_value) {
                found.push_back(counter);
            }
            ++num_found;
        }
    }

    return num_found;
}

/*************************************
 **** Sparse base extractor class ****
 *************************************/

// All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
class PrimaryBase {
public:
    template<typename Index_>
    PrimaryBase(const MatrixDetails<Index_>& details) : pointers(details.pointers) {
        serialize([&]() -> void {
           h5comp.reset(new Components);

            // TODO: set more suitable chunk cache values here, to avoid re-reading
            // chunks that are only partially consumed.
            h5comp->file.openFile(details.file_name, H5F_ACC_RDONLY);
            h5comp->data_dataset = h5comp->file.openDataSet(details.data_name);
            h5comp->index_dataset = h5comp->file.openDataSet(details.index_name);
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

/************************************
 **** LRU base extractor classes ****
 ************************************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruBase : public PrimaryBase {
    struct Slab {
        Slab(size_t capacity, bool needs_value, bool needs_index) : 
            value(needs_value ? capacity : 0), index(needs_index ? capacity : 0) {}
        std::vector<CachedValue_> value;
        std::vector<CachedIndex_> index;
        Index_ length;

        Chunk<Index_, CachedValue_, CachedIndex_> as_chunk(bool needs_value, bool needs_index) const {
            Chunk<Index_, CachedValue_, CachedIndex_> output;
            output.length = length;
            if (needs_value) {
                output.value = value.data();
            }
            if (needs_index) {
                output.index = index.data();
            }
            return output;
        }
    };

protected:
    tatami_chunked::LruSlabCache<Index_, Slab> cache;
    size_t max_non_zeros;
    bool needs_value, needs_index;

public:
    PrimaryLruBase(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_>, size_t cache_size, size_t max_non_zeros, bool needs_value, bool needs_index) : 
        PrimaryBase(details),
        cache([&]() -> size_t {
            // Always return at least one slab, so that cache.find() is valid.
            if (max_non_zeros == 0) {
                return 1;
            }

            auto elsize = size_of_cached_element<CachedValue_, CachedIndex_>(needs_value, needs_index);
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
        needs_value(needs_value),
        needs_index(needs_index)
    {}
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruFullBase : public PrimaryLruBase<Index_, CachedValue_, CachedIndex_> {
    PrimaryLruFullBase(
        const MatrixDetails<Index_>& details, 
        tatami::MaybeOracle<false, Index_> oracle, 
        size_t cache_size, 
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        PrimaryLruBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            needs_value, 
            needs_index
        )
    {}

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ i) {
        typedef typename PrimaryLruBase<Index_, CachedValue_, CachedIndex_>::Slab Slab;
        const auto& slab = this->cache.find(
            i, 
            /* create = */ [&]() -> Slab {
                return Slab(this->max_non_zeros, this->needs_value, this->needs_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - this->pointers[i];
                current_cache.length = extraction_len;
                if (extraction_len == 0) {
                    return;
                }

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    if (this->needs_index) {
                        this->h5comp->index_dataset.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                    if (this->needs_value) {
                        this->h5comp->data_dataset.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            }
        );

        return slab.as_chunk(this->needs_value, this->needs_index);
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruBlockBase : public PrimaryLruBase<Index_, CachedValue_, CachedIndex_> {
    PrimaryLruBlockBase(
        const MatrixDetails<Index_>& details,
        tatami::MaybeOracle<false, Index_> oracle, 
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        PrimaryLruBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            cache_size, 
            std::min(max_non_zeros, static_cast<size_t>(block_length)), // Tighten the bounds if we can.
            needs_value, 
            needs_index
        ),
        secondary_dim(details.secondary_dim),
        block_start(block_start),
        block_past_end(block_start + block_length)
    {}

private:
    Index_ secondary_dim;
    Index_ block_start, block_past_end;
    std::vector<CachedIndex_> index_buffer;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ i) {
        typedef typename PrimaryLruBase<Index_, CachedValue_, CachedIndex_>::Slab Slab;
        const auto& slab = this->cache.find(
            i, 
            /* create = */ [&]() -> Slab {
                return Slab(this->max_non_zeros, this->needs_value, this->needs_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - this->pointers[i];
                if (extraction_len == 0) {
                    current_cache.length = 0;
                    return;
                }
                index_buffer.resize(extraction_len);

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    this->h5comp->index_dataset.read(index_buffer.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);

                    auto indices_start = index_buffer.begin();
                    auto indices_end = index_buffer.end();
                    refine_primary_limits(indices_start, indices_end, secondary_dim, block_start, block_past_end);
                    current_cache.length = indices_end - indices_start;

                    if (current_cache.length) {
                        if (this->needs_index) {
                            std::copy(indices_start, indices_end, current_cache.index.begin());
                            if constexpr(!sparse_) {
                                // For dense extraction, we subtract the block_start 
                                // to make life easier when filling up the output vector.
                                for (Index_ i = 0; i < current_cache.length; ++i) {
                                    current_cache.index[i] -= block_start;
                                }
                            }
                        }
                        if (this->needs_value) {
                            hsize_t new_start = extraction_start + (indices_start - index_buffer.begin());
                            hsize_t new_len = current_cache.length;
                            this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &new_len, &new_start);
                            this->h5comp->memspace.setExtentSimple(1, &new_len);
                            this->h5comp->memspace.selectAll();
                            this->h5comp->data_dataset.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                        }
                    }
                });
            }
        );

        return slab.as_chunk(this->needs_value, this->needs_index);
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryLruIndexBase : public PrimaryLruBase<Index_, CachedValue_, CachedIndex_> {
    PrimaryLruIndexBase(
        const MatrixDetails<Index_>& details,
        tatami::MaybeOracle<false, Index_> oracle, 
        const std::vector<Index_>& indices,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        PrimaryLruBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            cache_size, 
            std::min(max_non_zeros, indices.size()), // Tighten the bounds if we can.
            needs_value, 
            needs_index
        ),
        secondary_dim(details.secondary_dim)
    {
        populate_sparse_remap_vector<sparse_>(indices, remap, first_index, past_last_index);
    }

private:
    Index_ secondary_dim;
    Index_ first_index, past_last_index;
    SparseRemapVector<sparse_, Index_> remap;
    std::vector<CachedIndex_> index_buffer;
    std::vector<Index_> found;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ i) {
        typedef typename PrimaryLruBase<Index_, CachedValue_, CachedIndex_>::Slab Slab;
        const auto& slab = this->cache.find(
            i, 
            /* create = */ [&]() -> Slab {
                return Slab(this->max_non_zeros, this->needs_value, this->needs_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - this->pointers[i];
                if (extraction_len == 0) {
                    current_cache.length = 0;
                    return;
                }
                index_buffer.resize(extraction_len);

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    this->h5comp->index_dataset.read(index_buffer.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);

                    auto indices_start = index_buffer.begin();
                    auto indices_end = index_buffer.end();
                    refine_primary_limits(indices_start, indices_end, secondary_dim, first_index, past_last_index);

                    Index_ num_found = 0;
                    if (indices_start != indices_end) {
                        auto ciIt = current_cache.index.begin();
                        num_found = scan_for_indices_in_remap_vector<sparse_>(indices_start, indices_end, first_index, ciIt, found, remap, this->needs_value, this->needs_index);

                        if (this->needs_value && num_found > 0) {
                            hsize_t new_start = extraction_start + (indices_start - index_buffer.begin());
                            this->h5comp->dataspace.selectNone();
                            tatami::process_consecutive_indices<Index_>(found.data(), found.size(),
                                [&](Index_ start, Index_ length) {
                                    hsize_t offset = start + new_start;
                                    hsize_t count = length;
                                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &count, &offset);
                                }
                            );

                            hsize_t new_len = num_found;
                            this->h5comp->memspace.setExtentSimple(1, &new_len);
                            this->h5comp->memspace.selectAll();
                            this->h5comp->data_dataset.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                        }
                    }

                    current_cache.length = num_found;
                });
            }
        );

        return slab.as_chunk(this->needs_value, this->needs_index);
    }
};

/******************************************
 **** Oraclular base extractor classes ****
 ******************************************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracleBase : public PrimaryBase {
protected:
    ContiguousOracleSlabCache<Index_, CachedValue_, CachedIndex_> cache;
    typedef typename decltype(cache)::Element Element;

public:
    PrimaryOracleBase(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        PrimaryBase(details),
        cache(details.pointers, std::move(oracle), cache_size, max_non_zeros, needs_cached_value, needs_cached_index)
    {}

protected:
    // This helper function finds contiguous slabs of indices for extraction,
    // reducing the number of hyperslab unions and improving performance for
    // consecutive extraction by just pulling out a strip of indices at once.
    // Technically, this just sets up the H5::DataSpaces for extraction; the
    // actual extraction is done in each of the subclasses.
    void prepare_contiguous_index_spaces(size_t dest_offset, std::vector<Index_>& needed, std::vector<Element>& next_cache_data) {
        size_t sofar = 0, num_needed = needed.size();
        hsize_t combined_len = 0;
        this->h5comp->dataspace.selectNone();

        while (sofar < num_needed) {
            auto& first = next_cache_data[needed[sofar]];
            first.mem_offset = dest_offset + combined_len;
            hsize_t src_offset = first.data_offset;
            hsize_t len = first.length;
            ++sofar;

            for (; sofar < num_needed; ++sofar) {
                auto& next = next_cache_data[needed[sofar]];
                if (src_offset + len < next.data_offset) {
                    break;
                }
                next.mem_offset = first.mem_offset + len;
                len += next.length;
            }

            this->h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &len, &src_offset);
            combined_len += len;
        }

        this->h5comp->memspace.setExtentSimple(1, &combined_len);
        this->h5comp->memspace.selectAll();
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracleFullBase : public PrimaryOracleBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracleFullBase(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        PrimaryOracleBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            needs_value, 
            needs_index
        ),
        needs_value(needs_value),
        needs_index(needs_index)
    {}

private:
    bool needs_value, needs_index;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename decltype(this->cache)::Element Element;
        return this->cache.next([&](
            size_t dest_offset,
            std::vector<Index_>& needed, 
            std::vector<Element>& next_cache_data, 
            std::vector<CachedValue_>& full_value_buffer, 
            std::vector<CachedIndex_>& full_index_buffer) -> void {
                serialize([&](){
                    this->prepare_contiguous_index_spaces(dest_offset, needed, next_cache_data);
                    if (needs_index) {
                        this->h5comp->index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                    if (needs_value) {
                        this->h5comp->data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            },
            needs_value,
            needs_index
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracleBlockBase : public PrimaryOracleBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracleBlockBase(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        PrimaryOracleBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            cache_size, 
            max_non_zeros, // do NOT tighten the bounds at this point; we need the full space for the initial index load.
            needs_value,
            true // we always need indices to compute the block boundaries.
        ),
        secondary_dim(details.secondary_dim),
        block_start(block_start),
        block_past_end(block_start + block_length),
        needs_value(needs_value),
        needs_index(needs_index)
    {}

private:
    Index_ secondary_dim;
    Index_ block_start, block_past_end;
    bool needs_value, needs_index;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename decltype(this->cache)::Element Element;
        return this->cache.next([&](
            size_t dest_offset,
            std::vector<Index_>& needed,
            std::vector<Element>& next_cache_data, 
            std::vector<CachedValue_>& full_value_buffer, 
            std::vector<CachedIndex_>& full_index_buffer) -> void {
                serialize([&](){
                    this->prepare_contiguous_index_spaces(dest_offset, needed, next_cache_data);
                    this->h5comp->index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);

                    size_t post_shift_len = 0;
                    if (needs_value) {
                        this->h5comp->dataspace.selectNone();
                    }

                    for (size_t i = 0, num_needed = needed.size(); i < num_needed; ++i) {
                        auto& current = next_cache_data[needed[i]];
                        auto pre_shift_offset = current.mem_offset;
                        current.mem_offset = dest_offset + post_shift_len;

                        auto indices_start = full_index_buffer.begin() + pre_shift_offset;
                        auto original_start = indices_start;
                        auto indices_end = indices_start + current.length;
                        refine_primary_limits(indices_start, indices_end, secondary_dim, block_start, block_past_end);

                        size_t new_len = indices_end - indices_start;
                        if (new_len) {
                            if (needs_index) {
                                // Shifting the desired block of indices backwards in the same full_index_buffer,
                                // to free up some space for the indices outside of the block. This should be valid
                                // for std::copy as long as indices_dest < indices_start. 
                                auto indices_dest = full_index_buffer.begin() + current.mem_offset;
                                if (indices_start != indices_dest) {
                                    std::copy(indices_start, indices_end, indices_dest);
                                }
                                if constexpr(!sparse_) {
                                    // For dense extraction, we remove the block start so that the resulting
                                    // indices can be directly used to index into the output buffer.
                                    for (size_t i = 0; i < new_len; ++i, ++indices_dest) {
                                        *indices_dest -= block_start;
                                    }
                                }
                            }
                            if (needs_value) {
                                hsize_t len = new_len;
                                hsize_t src_offset = current.data_offset + (indices_start - original_start);
                                this->h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &len, &src_offset);
                            }
                        }

                        current.length = new_len;
                        post_shift_len += new_len;
                    }

                    if (needs_value && post_shift_len > 0) {
                        hsize_t new_len = post_shift_len;
                        this->h5comp->memspace.setExtentSimple(1, &new_len);
                        this->h5comp->memspace.selectAll();
                        this->h5comp->data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            },
            needs_value,
            needs_index
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracleIndexBase : public PrimaryOracleBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracleIndexBase(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        const std::vector<Index_>& indices,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        PrimaryOracleBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            cache_size, 
            max_non_zeros, // do NOT tighten the bounds at this point; we need the full space for the initial index load.
            needs_value,
            true // we always need indices to figure out which elements to keep.
        ),
        secondary_dim(details.secondary_dim),
        needs_value(needs_value),
        needs_index(needs_index)
    {
        populate_sparse_remap_vector<sparse_>(indices, remap, first_index, past_last_index);
    }

private:
    Index_ secondary_dim;
    Index_ first_index, past_last_index;
    SparseRemapVector<sparse_, Index_> remap;
    std::vector<Index_> found;
    bool needs_value, needs_index;

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename decltype(this->cache)::Element Element;
        return this->cache.next([&](
            size_t dest_offset,
            std::vector<Index_>& needed,
            std::vector<Element>& next_cache_data, 
            std::vector<CachedValue_>& full_value_buffer, 
            std::vector<CachedIndex_>& full_index_buffer) -> void {
                serialize([&](){
                    this->prepare_contiguous_index_spaces(dest_offset, needed, next_cache_data);
                    this->h5comp->index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);

                    size_t post_shift_len = 0;
                    if (this->needs_value) {
                        this->h5comp->dataspace.selectNone();
                    }

                    for (size_t i = 0, num_needed = needed.size(); i < num_needed; ++i) {
                        auto& current = next_cache_data[needed[i]];
                        auto pre_shift_offset = current.mem_offset;
                        current.mem_offset = dest_offset + post_shift_len;

                        auto indices_start = full_index_buffer.begin() + pre_shift_offset;
                        auto original_start = indices_start;
                        auto indices_end = indices_start + current.length;
                        refine_primary_limits(indices_start, indices_end, secondary_dim, first_index, past_last_index);

                        Index_ num_found = 0;
                        if (indices_start != indices_end) {
                            auto fiIt = full_index_buffer.begin() + current.mem_offset;
                            num_found = scan_for_indices_in_remap_vector<sparse_>(indices_start, indices_end, first_index, fiIt, found, remap, this->needs_value, this->needs_index);

                            if (this->needs_value && !found.empty()) {
                                // We fill up the dataspace on each primary element, rather than accumulating
                                // indices in 'found' across 'needed', to reduce the memory usage of 'found';
                                // otherwise we grossly exceed the cache limits during extraction.
                                hsize_t new_start = current.data_offset + (indices_start - original_start);
                                tatami::process_consecutive_indices<Index_>(found.data(), found.size(),
                                    [&](Index_ start, Index_ length) {
                                        hsize_t offset = start + new_start;
                                        hsize_t count = length;
                                        this->h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &count, &offset);
                                    }
                                );
                            }
                        }

                        current.length = num_found;
                        post_shift_len += num_found;
                    }

                    if (this->needs_value && post_shift_len > 0) {
                        hsize_t new_len = post_shift_len;
                        this->h5comp->memspace.setExtentSimple(1, &new_len);
                        this->h5comp->memspace.selectAll();
                        this->h5comp->data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            },
            needs_value,
            needs_index
        );
    }
};

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryFullBase = typename std::conditional<
    oracle_,
    PrimaryOracleFullBase<Index_, CachedValue_, CachedIndex_>,
    PrimaryLruFullBase<Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryFullSparse : 
    public ConditionalPrimaryFullBase<oracle_, Index_, CachedValue_, CachedIndex_>, 
    public tatami::SparseExtractor<oracle_, Value_, Index_> 
{
    PrimaryFullSparse(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        Index_ secondary_dim,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_value, 
        bool needs_index) : 
        ConditionalPrimaryFullBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            MatrixDetails<Index_>(file_name, data_name, index_name, ptrs, secondary_dim),
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            needs_value, 
            needs_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto chunk = this->fetch_raw(i);
        return chunk.to_sparse(vbuffer, ibuffer);
    }
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryFullDense : 
    public ConditionalPrimaryFullBase<oracle_, Index_, CachedValue_, CachedIndex_>,
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
        ConditionalPrimaryFullBase<oracle_, Index_, CachedValue_, CachedIndex_>(
            MatrixDetails<Index_>(file_name, data_name, index_name, ptrs, secondary_dim),
            std::move(oracle), 
            cache_size, 
            max_non_zeros, 
            true, 
            true
        ),
        secondary_dim(secondary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto chunk = this->fetch_raw(i);
        return chunk.to_dense(buffer, secondary_dim);
    }

private:
    Index_ secondary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool sparse_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryBlockBase = typename std::conditional<
    oracle_,
    PrimaryOracleBlockBase<sparse_, Index_, CachedValue_, CachedIndex_>,
    PrimaryLruBlockBase<sparse_, Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryBlockSparse : 
    public ConditionalPrimaryBlockBase<true, oracle_, Index_, CachedValue_, CachedIndex_>,
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
        ConditionalPrimaryBlockBase<true, oracle_, Index_, CachedValue_, CachedIndex_>(
            MatrixDetails<Index_>(file_name, data_name, index_name, ptrs, secondary_dim),
            std::move(oracle), 
            block_start,
            block_length,
            cache_size, 
            max_non_zeros, 
            needs_value, 
            needs_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto chunk = this->fetch_raw(i);
        return chunk.to_sparse(vbuffer, ibuffer);
    }
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryBlockDense : 
    public ConditionalPrimaryBlockBase<false, oracle_, Index_, CachedValue_, CachedIndex_>,
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
        ConditionalPrimaryBlockBase<false, oracle_, Index_, CachedValue_, CachedIndex_>(
            MatrixDetails<Index_>(file_name, data_name, index_name, ptrs, secondary_dim),
            std::move(oracle), 
            block_start,
            block_length,
            cache_size, 
            max_non_zeros, 
            true, 
            true
        ),
        extract_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto chunk = this->fetch_raw(i);
        return chunk.to_dense(buffer, extract_length);
    }

private:
    Index_ extract_length;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool sparse_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryIndexBase = typename std::conditional<
    oracle_,
    PrimaryOracleIndexBase<sparse_, Index_, CachedValue_, CachedIndex_>,
    PrimaryLruIndexBase<sparse_, Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryIndexSparse : 
    public ConditionalPrimaryIndexBase<true, oracle_, Index_, CachedValue_, CachedIndex_>,
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
        ConditionalPrimaryIndexBase<true, oracle_, Index_, CachedValue_, CachedIndex_>(
            MatrixDetails<Index_>(file_name, data_name, index_name, ptrs, secondary_dim),
            std::move(oracle), 
            *indices_ptr,
            cache_size, 
            max_non_zeros, 
            needs_value, 
            needs_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto chunk = this->fetch_raw(i);
        return chunk.to_sparse(vbuffer, ibuffer);
    }
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
struct PrimaryIndexDense : 
    public ConditionalPrimaryIndexBase<false, oracle_, Index_, CachedValue_, CachedIndex_>,
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
        ConditionalPrimaryIndexBase<false, oracle_, Index_, CachedValue_, CachedIndex_>(
            MatrixDetails<Index_>(file_name, data_name, index_name, ptrs, secondary_dim),
            std::move(oracle), 
            *indices_ptr,
            cache_size, 
            max_non_zeros, 
            true, 
            true
        ),
        extract_length(indices_ptr->size())
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto chunk = this->fetch_raw(i);
        return chunk.to_dense(buffer, extract_length);
    }

private:
    Index_ extract_length;
};

}

}

#endif
