#ifndef TATAMI_HDF5_SPARSE_PRIMARY_HPP
#define TATAMI_HDF5_SPARSE_PRIMARY_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

#include "serialize.hpp"
#include "utils.hpp"

namespace tatami_hdf5 {

namespace CompressedSparseMatrix_internal {

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
    size_t chunk_size;
};

template<typename Index_>
struct MatrixDetails {
    MatrixDetails(
        const std::string& file_name, 
        const std::string& value_name, 
        const std::string& index_name, 
        Index_ primary_dim, 
        Index_ secondary_dim, 
        const std::vector<hsize_t>& pointers, 
        size_t slab_cache_size,
        size_t max_non_zeros,
        size_t chunk_cache_size) :
        file_name(file_name), 
        value_name(value_name), 
        index_name(index_name), 
        primary_dim(primary_dim), 
        secondary_dim(secondary_dim), 
        pointers(pointers), 
        slab_cache_size(slab_cache_size),
        max_non_zeros(max_non_zeros),
        chunk_cache_size(chunk_cache_size) 
    {}

    const std::string& file_name;
    const std::string& value_name;
    const std::string& index_name;

    Index_ primary_dim;
    Index_ secondary_dim;
    const std::vector<hsize_t>& pointers;

    // We distinguish between our own cache of slabs versus HDF5's cache of uncompressed chunks.
    size_t slab_cache_size;
    size_t max_non_zeros;
    size_t chunk_cache_size;
};

// All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
template<typename Index_>
void initialize(const MatrixDetails<Index_>& details, std::unique_ptr<Components>& h5comp) {
    serialize([&]() -> void {
        h5comp.reset(new Components);

        // Using some kinda-big prime number as the number of slots. This
        // doesn't really matter too much as we only intend to store two
        // chunks at most - see CompressedSparseMatrix.hpp for the rationale.
        H5::FileAccPropList fapl(H5::FileAccPropList::DEFAULT.getId());
        fapl.setCache(0, 511, details.chunk_cache_size, 0);

        h5comp->file.openFile(details.file_name, H5F_ACC_RDONLY, fapl);
        h5comp->data_dataset = h5comp->file.openDataSet(details.value_name);
        h5comp->index_dataset = h5comp->file.openDataSet(details.index_name);
        h5comp->dataspace = h5comp->data_dataset.getSpace();
    });
}

inline void destroy(std::unique_ptr<Components>& h5comp) {
    serialize([&]() {
        h5comp.reset();
    });
}

// Unfortunately we can't use tatami::SparseRange, as CachedIndex_ might not be
// large enough to represent the number of cached indices, e.g., if there were
// 256 indices, a CachedIndex_=uint8_t type would be large enough to represent
// each index (0 - 255) but not the number of indices.
template<typename Index_, typename CachedValue_, typename CachedIndex_ = Index_>
struct Slab { 
    CachedValue_* value = NULL;
    CachedIndex_* index = NULL;
    Index_ number = 0;
};

template<typename Slab_, typename Value_, typename Index_>
tatami::SparseRange<Value_, Index_> slab_to_sparse(const Slab_& slab, Value_* value_buffer, Index_* index_buffer) {
    tatami::SparseRange<Value_, Index_> output(slab.number);
    if (slab.value) {
        std::copy_n(slab.value, slab.number, value_buffer);
        output.value = value_buffer;
    }
    if (slab.index) {
        std::copy_n(slab.index, slab.number, index_buffer);
        output.index = index_buffer;
    }
    return output;
}

template<typename Slab_, typename Value_, typename Index_>
Value_* slab_to_dense(const Slab_& slab, Value_* buffer, Index_ extract_length) {
    std::fill_n(buffer, extract_length, 0);
    for (Index_ i = 0; i < slab.number; ++i) {
        buffer[slab.index[i]] = slab.value[i];
    }
    return buffer;
}

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

/************************************
 **** LRU base extractor classes ****
 ************************************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruCoreBase {
public:
    PrimaryLruCoreBase(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_>, size_t max_non_zeros, bool needs_value, bool needs_index) : 
        my_pointers(details.pointers),
        my_cache([&]() -> size_t {
            if (max_non_zeros == 0) {
                return 1; // always return at least one slab, so that cache.find() is valid.
            }

            size_t elsize = size_of_cached_element<CachedValue_, CachedIndex_>(needs_value, needs_index);
            if (elsize == 0) {
                return details.primary_dim; // cache everything if we're not storing anything in each slab.
            }

            size_t num_slabs = details.slab_cache_size / (max_non_zeros * elsize);
            if (num_slabs == 0) {
                return 1; // again, return at least one slab so that cache.find() works.
            }

            return num_slabs;
        }()),
        my_needs_value(needs_value),
        my_needs_index(needs_index),
        my_max_non_zeros(max_non_zeros)
    {
        initialize(details, my_h5comp);

        size_t pool_size = max_non_zeros * my_cache.get_max_slabs();
        if (needs_value) {
            my_value_pool.resize(pool_size);
        }
        if (needs_index) {
            my_index_pool.resize(pool_size);
        }
    }

    ~PrimaryLruCoreBase() {
        destroy(my_h5comp);
    }

protected:
    std::unique_ptr<Components> my_h5comp;
    const std::vector<hsize_t>& my_pointers;
    tatami_chunked::LruSlabCache<Index_, Slab<Index_, CachedValue_, CachedIndex_> > my_cache;
    bool my_needs_value, my_needs_index;

private:
    std::vector<CachedValue_> my_value_pool;
    std::vector<CachedIndex_> my_index_pool;
    size_t my_max_non_zeros;
    size_t my_offset = 0;

public:
    Slab<Index_, CachedValue_, CachedIndex_> create() {
        Slab<Index_, CachedValue_, CachedIndex_> output;
        if (my_needs_value) {
            output.value = my_value_pool.data() + my_offset;
        }
        if (my_needs_index) {
            output.index = my_index_pool.data() + my_offset;
        }
        my_offset += my_max_non_zeros;
        return output;
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruFullCore : private PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryLruFullCore(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_> oracle, bool needs_value, bool needs_index) : 
        PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_>(details, std::move(oracle), details.max_non_zeros, needs_value, needs_index) {}

public:
    const Slab<Index_, CachedValue_, CachedIndex_>& fetch_raw(Index_ i) {
        return this->my_cache.find(
            i, 
            /* create = */ [&]() -> Slab<Index_, CachedValue_, CachedIndex_> {
                return this->create();
            },
            /* populate = */ [&](Index_ i, Slab<Index_, CachedValue_, CachedIndex_>& current_cache) -> void {
                const auto& pointers = this->my_pointers; 
                hsize_t extraction_start = pointers[i];
                hsize_t extraction_len = pointers[i + 1] - pointers[i];
                current_cache.number = extraction_len;

                if (extraction_len == 0) {
                    return; 
                }
                if (!(this->my_needs_index || this->my_needs_value)) {
                    return;
                }

                serialize([&]() -> void {
                    auto& comp = *(this->my_h5comp);
                    comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    comp.memspace.setExtentSimple(1, &extraction_len);
                    comp.memspace.selectAll();
                    if (this->my_needs_index) {
                        comp.index_dataset.read(current_cache.index, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);
                    }
                    if (this->my_needs_value) {
                        comp.data_dataset.read(current_cache.value, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                    }
                });
            }
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruBlockCore : private PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryLruBlockCore(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_> oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            std::min(details.max_non_zeros, static_cast<size_t>(block_length)), // Tighten the bounds if we can, to fit more elements into the cache.
            needs_value, 
            needs_index
        ),
        my_secondary_dim(details.secondary_dim),
        my_block_start(block_start),
        my_block_past_end(block_start + block_length)
    {}

private:
    Index_ my_secondary_dim;
    Index_ my_block_start, my_block_past_end;
    std::vector<CachedIndex_> my_index_buffer;

public:
    const Slab<Index_, CachedValue_, CachedIndex_>& fetch_raw(Index_ i) {
        return this->my_cache.find(
            i, 
            /* create = */ [&]() -> Slab<Index_, CachedValue_, CachedIndex_> {
                return this->create();
            },
            /* populate = */ [&](Index_ i, Slab<Index_, CachedValue_, CachedIndex_>& current_cache) -> void {
                const auto& pointers = this->my_pointers;
                hsize_t extraction_start = pointers[i];
                hsize_t extraction_len = pointers[i + 1] - pointers[i];

                if (extraction_len == 0) {
                    current_cache.number = 0;
                    return;
                }
                my_index_buffer.resize(extraction_len);

                serialize([&]() -> void {
                    auto& comp = *(this->my_h5comp);
                    comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    comp.memspace.setExtentSimple(1, &extraction_len);
                    comp.memspace.selectAll();
                    comp.index_dataset.read(my_index_buffer.data(), define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                    auto indices_start = my_index_buffer.begin();
                    auto indices_end = my_index_buffer.end();
                    refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_block_start, my_block_past_end);
                    current_cache.number = indices_end - indices_start;

                    if (current_cache.number) {
                        if (this->my_needs_index) {
                            std::copy(indices_start, indices_end, current_cache.index);
                            if constexpr(!sparse_) {
                                // For dense extraction, we subtract the block_start 
                                // to make life easier when filling up the output vector.
                                for (Index_ i = 0; i < current_cache.number; ++i) {
                                    current_cache.index[i] -= my_block_start;
                                }
                            }
                        }

                        if (this->my_needs_value) {
                            hsize_t new_start = extraction_start + (indices_start - my_index_buffer.begin());
                            hsize_t new_len = current_cache.number;
                            comp.dataspace.selectHyperslab(H5S_SELECT_SET, &new_len, &new_start);
                            comp.memspace.setExtentSimple(1, &new_len);
                            comp.memspace.selectAll();
                            comp.data_dataset.read(current_cache.value, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                        }
                    }
                });
            }
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruIndexCore : private PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryLruIndexCore(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_> oracle, const std::vector<Index_>& indices, bool needs_value, bool needs_index) : 
        PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            std::min(details.max_non_zeros, indices.size()), // Tighten the bounds to fit more elements into the cache.
            needs_value, 
            needs_index
        ),
        my_secondary_dim(details.secondary_dim)
    {
        populate_sparse_remap_vector<sparse_>(indices, my_remap, my_first_index, my_past_last_index);
    }

private:
    Index_ my_secondary_dim;
    Index_ my_first_index, my_past_last_index;
    SparseRemapVector<sparse_, Index_> my_remap;
    std::vector<CachedIndex_> my_index_buffer;
    std::vector<Index_> my_found;

public:
    const Slab<Index_, CachedValue_, CachedIndex_>& fetch_raw(Index_ i) {
        return this->my_cache.find(
            i, 
            /* create = */ [&]() -> Slab<Index_, CachedValue_, CachedIndex_> {
                return this->create();
            },
            /* populate = */ [&](Index_ i, Slab<Index_, CachedValue_, CachedIndex_>& current_cache) -> void {
                const auto& pointers = this->my_pointers;
                hsize_t extraction_start = pointers[i];
                hsize_t extraction_len = pointers[i + 1] - pointers[i];
                if (extraction_len == 0) {
                    current_cache.number = 0;
                    return;
                }
                my_index_buffer.resize(extraction_len);

                serialize([&]() -> void {
                    auto& comp = *(this->my_h5comp);
                    comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    comp.memspace.setExtentSimple(1, &extraction_len);
                    comp.memspace.selectAll();
                    comp.index_dataset.read(my_index_buffer.data(), define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                    auto indices_start = my_index_buffer.begin();
                    auto indices_end = my_index_buffer.end();
                    refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_first_index, my_past_last_index);

                    Index_ num_found = 0;
                    if (indices_start != indices_end) {
                        num_found = scan_for_indices_in_remap_vector<sparse_>(indices_start, indices_end, my_first_index, current_cache.index, my_found, my_remap, this->my_needs_value, this->my_needs_index);

                        if (this->my_needs_value && num_found > 0) {
                            hsize_t new_start = extraction_start + (indices_start - my_index_buffer.begin());
                            comp.dataspace.selectNone();
                            tatami::process_consecutive_indices<Index_>(my_found.data(), my_found.size(),
                                [&](Index_ start, Index_ length) {
                                    hsize_t offset = start + new_start;
                                    hsize_t count = length;
                                    comp.dataspace.selectHyperslab(H5S_SELECT_OR, &count, &offset);
                                }
                            );

                            hsize_t new_len = num_found;
                            comp.memspace.setExtentSimple(1, &new_len);
                            comp.memspace.selectAll();
                            comp.data_dataset.read(current_cache.value, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                        }
                    }

                    current_cache.number = num_found;
                });
            }
        );
    }
};

/******************************************
 **** Oraclular base extractor classes ****
 ******************************************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularCoreBase {
public:
    PrimaryOracularCoreBase(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        bool needs_cached_value, 
        bool needs_cached_index) : 
        my_pointers(details.pointers),
        my_cache(std::move(oracle), [&]() -> size_t {
            size_t elsize = size_of_cached_element<CachedIndex_, CachedValue_>(needs_cached_value, needs_cached_index);
            if (elsize == 0) {
                return -1; // i.e., there is no limit on the number of slabs.
            } else {
                size_t proposed = details.slab_cache_size / elsize;
                return std::max(details.max_non_zeros, proposed); // make sure we always have enough space to store at least one dimension element.
            } 
        }())
    {
        initialize(details, my_h5comp);

        if (needs_cached_value) {
            my_full_value_buffer.resize(my_cache.get_max_size());
        }
        if (needs_cached_index) {
            my_full_index_buffer.resize(my_cache.get_max_size());
        }
    }

    ~PrimaryOracularCoreBase() {
        destroy(my_h5comp);
    }

protected:
    std::unique_ptr<Components> my_h5comp;
    const std::vector<hsize_t>& my_pointers;

    struct SlabPrecursor {
        size_t mem_offset;
        size_t length;
    };

protected:
    // Here, we use a giant contiguous buffer to optimize for near-consecutive
    // iteration. This allows the HDF5 library to pull out long strips of data from
    // the file.  It also allows us to maximize the use of the cache_size_limit by
    // accounting for differences in the non-zeros for each element, rather than
    // conservatively assuming they're all at max (as in the LRU case). The
    // downside is that we need to do some copying within the cache to make space
    // for new reads, but that works out to be no more than one extra copy per
    // fetch() call, which is tolerable. I suppose we could do better by
    // defragmenting within this buffer but that's probably overkill.
    std::vector<CachedValue_> my_full_value_buffer;
    std::vector<CachedIndex_> my_full_index_buffer;

private:
    tatami_chunked::OracularVariableSlabCache<Index_, Index_, SlabPrecursor, size_t> my_cache;

private:
    template<class Function_>
    static void sort_by_field(std::vector<std::pair<Index_, size_t> >& indices, Function_ field) {
        auto comp = [&field](const std::pair<Index_, size_t>& l, const std::pair<Index_, size_t>& r) -> bool {
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
    Slab<Index_, CachedValue_, CachedIndex_> next(Extract_ extract, bool needs_value, bool needs_index) {
        auto out = my_cache.next(
            /* identify = */ [](Index_ i) -> std::pair<Index_, Index_> { 
                return std::pair<Index_, Index_>(i, 0); 
            },
            /* estimated_size = */ [&](Index_ i) -> size_t {
                return my_pointers[i + 1] - my_pointers[i];
            },
            /* actual_size = */ [&](Index_, const SlabPrecursor& preslab) -> size_t {
                return preslab.length;
            },
            /* create = */ [&]() -> SlabPrecursor {
                return SlabPrecursor();
            },
            /* populate = */ [&](std::vector<std::pair<Index_, size_t> >& to_populate, std::vector<std::pair<Index_, size_t> >& to_reuse, std::vector<SlabPrecursor>& all_preslabs) {
                size_t dest_offset = 0;

                if (to_reuse.size()) {
                    // Shuffling all re-used elements to the start of the buffer,
                    // so that we can perform a contiguous extraction of the needed
                    // elements in the rest of the buffer. This needs some sorting
                    // to ensure that we're not clobbering one re-used element's
                    // contents when shifting another element to the start.
                    sort_by_field(to_reuse, [&](const std::pair<Index_, size_t>& p) -> size_t { return all_preslabs[p.second].mem_offset; });

                    for (const auto& p : to_reuse) {
                        auto& preslab = all_preslabs[p.second];
                        if (needs_index) {
                            auto isrc = my_full_index_buffer.begin() + preslab.mem_offset;
                            std::copy_n(isrc, preslab.length, my_full_index_buffer.begin() + dest_offset);
                        }
                        if (needs_value) {
                            auto vsrc = my_full_value_buffer.begin() + preslab.mem_offset;
                            std::copy_n(vsrc, preslab.length, my_full_value_buffer.begin() + dest_offset); 
                        }
                        preslab.mem_offset = dest_offset;
                        dest_offset += preslab.length;
                    }
                }

                extract(dest_offset, to_populate, all_preslabs, my_full_value_buffer, my_full_index_buffer);
            }
        );

        Slab<Index_, CachedValue_, CachedIndex_> output;
        output.number = out.first->length;
        if (needs_value) {
            output.value = my_full_value_buffer.data() + out.first->mem_offset;
        }
        if (needs_index) {
            output.index = my_full_index_buffer.data() + out.first->mem_offset;
        }
        return output;
    }

protected:
    // This helper function finds contiguous slabs of indices for extraction,
    // reducing the number of hyperslab unions and improving performance for
    // consecutive extraction by just pulling out a strip of indices at once.
    // Technically, this just sets up the H5::DataSpaces for extraction; the
    // actual extraction is done in each of the subclasses.
    void prepare_contiguous_index_spaces(size_t dest_offset, std::vector<std::pair<Index_, size_t> >& to_populate, std::vector<SlabPrecursor>& all_preslabs) {
        // Sorting so that we get consecutive accesses in the hyperslab construction.
        // This should improve re-use of partially read HDF5 chunks inside DataSet::read(). 
        sort_by_field(to_populate, [&](const std::pair<Index_, size_t>& p) -> hsize_t { return my_pointers[p.first]; });

        size_t sofar = 0, num_needed = to_populate.size();
        hsize_t combined_len = 0;
        my_h5comp->dataspace.selectNone();

        while (sofar < num_needed) {
            const auto& pfirst = to_populate[sofar];
            hsize_t src_offset = my_pointers[pfirst.first];
            hsize_t src_len = my_pointers[pfirst.first + 1] - src_offset;

            auto& first_preslab = all_preslabs[pfirst.second];
            first_preslab.mem_offset = dest_offset + combined_len;
            first_preslab.length = src_len;
            ++sofar;

            Index_ previous = pfirst.first;
            for (; sofar < num_needed; ++sofar) {
                const auto& pnext = to_populate[sofar];
                if (previous + 1 < pnext.first) {
                    break;
                }

                auto& next_preslab = all_preslabs[pnext.second];
                next_preslab.mem_offset = first_preslab.mem_offset + src_len;
                hsize_t next_len = my_pointers[pnext.first + 1] - my_pointers[pnext.first];
                next_preslab.length = next_len;
                src_len += next_len;
            }

            my_h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &src_len, &src_offset);
            combined_len += src_len;
        }

        my_h5comp->memspace.setExtentSimple(1, &combined_len);
        my_h5comp->memspace.selectAll();
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularFullCore : private PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracularFullCore(const MatrixDetails<Index_>& details, std::shared_ptr<const tatami::Oracle<Index_> > oracle, bool needs_value, bool needs_index) : 
        PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>(details, std::move(oracle), needs_value, needs_index),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {}

private:
    bool my_needs_value, my_needs_index;

public:
    Slab<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::SlabPrecursor SlabPrecursor;
        return this->next([&](
            size_t dest_offset,
            std::vector<std::pair<Index_, size_t> >& to_populate, 
            std::vector<SlabPrecursor>& all_preslabs, 
            std::vector<CachedValue_>& full_value_buffer, 
            std::vector<CachedIndex_>& full_index_buffer) -> void {
                serialize([&](){
                    this->prepare_contiguous_index_spaces(dest_offset, to_populate, all_preslabs);
                    auto& comp = *(this->my_h5comp);
                    if (my_needs_index) {
                        comp.index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);
                    }
                    if (my_needs_value) {
                        comp.data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                    }
                });
            },
            my_needs_value,
            my_needs_index
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularBlockCore : private PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracularBlockCore(const MatrixDetails<Index_>& details, std::shared_ptr<const tatami::Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        // Don't try to tighten the max_non_zeros like in the LRU case; we need
        // to keep enough space to ensure that every primary dimension element
        // can be extracted in its entirety (as we don't have a separate
        // index_buffer class to perform the initial HDF5 extraction).
        PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            needs_value, 
            true // We always need indices to compute the block boundaries.
        ), 
        my_secondary_dim(details.secondary_dim),
        my_block_start(block_start),
        my_block_past_end(block_start + block_length),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {}

private:
    Index_ my_secondary_dim;
    Index_ my_block_start, my_block_past_end;
    bool my_needs_value, my_needs_index;

public:
    Slab<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::SlabPrecursor SlabPrecursor;
        return this->next([&](
            size_t dest_offset,
            std::vector<std::pair<Index_, size_t> >& to_populate,
            std::vector<SlabPrecursor>& all_preslabs, 
            std::vector<CachedValue_>& full_value_buffer, 
            std::vector<CachedIndex_>& full_index_buffer) -> void {
                serialize([&](){
                    this->prepare_contiguous_index_spaces(dest_offset, to_populate, all_preslabs);
                    auto& comp = *(this->my_h5comp);
                    comp.index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                    size_t post_shift_len = 0;
                    if (my_needs_value) {
                        comp.dataspace.selectNone();
                    }

                    for (const auto& p : to_populate) {
                        auto& current_preslab = all_preslabs[p.second];

                        // Remember that 'mem_offset' was set inside 'prepare_contiguous_index_spaces()'.
                        auto pre_shift_offset = current_preslab.mem_offset;
                        current_preslab.mem_offset = dest_offset + post_shift_len;

                        auto indices_start = full_index_buffer.begin() + pre_shift_offset;
                        auto original_start = indices_start;
                        auto indices_end = indices_start + current_preslab.length;
                        refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_block_start, my_block_past_end);

                        size_t new_len = indices_end - indices_start;
                        if (new_len) {
                            if (my_needs_index) {
                                // Shifting the desired block of indices backwards in the same full_index_buffer,
                                // to free up some space for the indices outside of the block. This should be valid
                                // for std::copy as long as indices_dest < indices_start. 
                                auto indices_dest = full_index_buffer.begin() + current_preslab.mem_offset;
                                if (indices_start != indices_dest) {
                                    std::copy(indices_start, indices_end, indices_dest);
                                }
                                if constexpr(!sparse_) {
                                    // For dense extraction, we remove the block start so that the resulting
                                    // indices can be directly used to index into the output buffer.
                                    for (size_t i = 0; i < new_len; ++i, ++indices_dest) {
                                        *indices_dest -= my_block_start;
                                    }
                                }
                            }
                            if (my_needs_value) {
                                hsize_t len = new_len;
                                hsize_t src_offset = this->my_pointers[p.first] + (indices_start - original_start);
                                comp.dataspace.selectHyperslab(H5S_SELECT_OR, &len, &src_offset);
                            }
                        }

                        current_preslab.length = new_len;
                        post_shift_len += new_len;
                    }

                    if (my_needs_value && post_shift_len > 0) {
                        hsize_t new_len = post_shift_len;
                        comp.memspace.setExtentSimple(1, &new_len);
                        comp.memspace.selectAll();
                        comp.data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                    }
                });
            },
            my_needs_value,
            my_needs_index
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularIndexCore : private PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracularIndexCore(const MatrixDetails<Index_>& details, std::shared_ptr<const tatami::Oracle<Index_> > oracle, const std::vector<Index_>& indices, bool needs_value, bool needs_index) : 
        // Don't try to tighten the max_non_zeros like in the LRU case; we need
        // to keep enough space to ensure that every primary dimension element
        // can be extracted in its entirety (as we don't have a separate
        // index_buffer class to perform the initial HDF5 extraction).
        PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            needs_value, 
            true // We always need indices to figure out what to keep.
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
    Slab<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::SlabPrecursor SlabPrecursor;
        return this->next([&](
            size_t dest_offset,
            std::vector<std::pair<Index_, size_t> >& to_populate,
            std::vector<SlabPrecursor>& all_preslabs, 
            std::vector<CachedValue_>& full_value_buffer, 
            std::vector<CachedIndex_>& full_index_buffer) -> void {
                serialize([&](){
                    this->prepare_contiguous_index_spaces(dest_offset, to_populate, all_preslabs);
                    auto& comp = *(this->my_h5comp);
                    comp.index_dataset.read(full_index_buffer.data() + dest_offset, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                    size_t post_shift_len = 0;
                    if (this->needs_value) {
                        comp.dataspace.selectNone();
                    }

                    for (const auto& p : to_populate) {
                        auto& current_preslab = all_preslabs[p.second];

                        auto pre_shift_offset = current_preslab.mem_offset;
                        current_preslab.mem_offset = dest_offset + post_shift_len;

                        auto indices_start = full_index_buffer.begin() + pre_shift_offset;
                        auto original_start = indices_start;
                        auto indices_end = indices_start + current_preslab.length;
                        refine_primary_limits(indices_start, indices_end, secondary_dim, first_index, past_last_index);

                        Index_ num_found = 0;
                        if (indices_start != indices_end) {
                            auto fiIt = full_index_buffer.begin() + current_preslab.mem_offset;
                            num_found = scan_for_indices_in_remap_vector<sparse_>(indices_start, indices_end, first_index, fiIt, found, remap, this->needs_value, this->needs_index);

                            if (this->needs_value && !found.empty()) {
                                // We fill up the dataspace on each primary element, rather than accumulating
                                // indices in 'found' across 'needed', to reduce the memory usage of 'found';
                                // otherwise we grossly exceed the cache limits during extraction.
                                hsize_t new_start = this->my_pointers[p.first] + (indices_start - original_start);
                                tatami::process_consecutive_indices<Index_>(found.data(), found.size(),
                                    [&](Index_ start, Index_ length) {
                                        hsize_t offset = start + new_start;
                                        hsize_t count = length;
                                        comp.dataspace.selectHyperslab(H5S_SELECT_OR, &count, &offset);
                                    }
                                );
                            }
                        }

                        current_preslab.length = num_found;
                        post_shift_len += num_found;
                    }

                    if (this->needs_value && post_shift_len > 0) {
                        hsize_t new_len = post_shift_len;
                        comp.memspace.setExtentSimple(1, &new_len);
                        comp.memspace.selectAll();
                        comp.data_dataset.read(full_value_buffer.data() + dest_offset, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
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
using ConditionalPrimaryFullCore = typename std::conditional<
    oracle_,
    PrimaryOracularFullCore<Index_, CachedValue_, CachedIndex_>,
    PrimaryLruFullCore<Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullSparse : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryFullSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), needs_value, needs_index) {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_sparse(slab, value_buffer, index_buffer);
    }

private:
    ConditionalPrimaryFullCore<oracle_, Index_, CachedValue_, CachedIndex_> my_core;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullDense : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryFullDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle) : 
        my_core(details, std::move(oracle), true, true),
        my_secondary_dim(details.secondary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_dense(slab, buffer, my_secondary_dim);
    }

private:
    ConditionalPrimaryFullCore<oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_secondary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool sparse_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryBlockCore = typename std::conditional<
    oracle_,
    PrimaryOracularBlockCore<sparse_, Index_, CachedValue_, CachedIndex_>,
    PrimaryLruBlockCore<sparse_, Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryBlockSparse : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryBlockSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), block_start, block_length, needs_value, needs_index) {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_sparse(slab, value_buffer, index_buffer);
    }

private:
    ConditionalPrimaryBlockCore<true, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryBlockDense : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryBlockDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length) :
        my_core(details, std::move(oracle), block_start, block_length, true, true), my_extract_length(block_length) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_dense(slab, buffer, my_extract_length);
    }

private:
    ConditionalPrimaryBlockCore<false, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_extract_length;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool sparse_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryIndexCore = typename std::conditional<
    oracle_,
    PrimaryOracularIndexCore<sparse_, Index_, CachedValue_, CachedIndex_>,
    PrimaryLruIndexCore<sparse_, Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryIndexSparse : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryIndexSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), *indices_ptr, needs_value, needs_index) {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_sparse(slab, value_buffer, index_buffer);
    }

private:
    ConditionalPrimaryIndexCore<true, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryIndexDense : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryIndexDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr) :
        my_core(details, std::move(oracle), *indices_ptr, true, true), my_extract_length(indices_ptr->size()) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_dense(slab, buffer, my_extract_length);
    }

private:
    ConditionalPrimaryIndexCore<false, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_extract_length;
};

}

}

#endif
