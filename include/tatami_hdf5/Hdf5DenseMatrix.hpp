#ifndef TATAMI_HDF5_DENSE_MATRIX_HPP
#define TATAMI_HDF5_DENSE_MATRIX_HPP

#include "H5Cpp.h"

#include <string>
#include <cstdint>
#include <type_traits>
#include <cmath>
#include <list>
#include <vector>

#include "serialize.hpp"
#include "utils.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

/**
 * @file Hdf5DenseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed dense matrix.
 */

namespace tatami_hdf5 {

/**
 * @cond
 */
namespace Hdf5DenseMatrix_internal {

// All HDF5-related members.
struct Components {
    H5::H5File file;
    H5::DataSet dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

template<bool by_h5_row_, typename Index_, typename OutputValue_>
void extract_block(Index_ primary_start, Index_ primary_length, Index_ extract_start, Index_ extract_length, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    constexpr int dimdex = by_h5_row;
    offset[1-dimdex] = primary_start;
    count[1-dimdex] = primary_length;

    offset[dimdex] = extract_value;
    count[dimdex] = extract_length;
    comp.dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // HDF5 is a lot faster when the memspace and dataspace match in dimensionality.
    // Presumably there is some shuffling that happens inside when dimensions don't match.
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

template<bool by_h5_row_, typename Index_, typename OutputValue_>
void extract_indices(Index_ primary_start, Index_ primary_length, const std::vector<Index_>& indices, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    constexpr int dimdex = by_h5_row;
    offset[1-dimdex] = primary_start;
    count[1-dimdex] = primary_length;

    // Take slices across the current chunk for each index. This should be okay if consecutive,
    // but hopefully they've fixed the problem with non-consecutive slices in:
    // https://forum.hdfgroup.org/t/union-of-non-consecutive-hyperslabs-is-very-slow/5062
    comp.dataspace.selectNone();
    tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(),
        [&](Index_ start, Index_ length) {
            offset[dimdex] = start;
            count[dimdex] = length;
            comp.dataspace.selectHyperslab(H5S_SELECT_OR, count, offset);
        }
    );

    // Again, matching the dimensionality.
    count[dimdex] = indices.size();
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(target, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

template<typename CachedValue_, typename Index_>
void transpose(std::vector<CachedValue_>& cache, std::vector<CachedValue_>& buffer, Index_ actual_dim, Index_ extract_length) {
    buffer.resize(cache.size());
    if (actual_dim == 1 || extract_length == 1) {
        std::copy(cache.begin(), cache.end(), buffer.begin());
        return;
    }

    // Using a blockwise strategy to perform the transposition,
    // in order to be more cache-friendly.
    constexpr Index_ block = 16;
    for (Index_ xstart = 0; xstart < actual_dim; xstart += block) {
        Index_ xend = xstart + std::min(block, actual_dim - xstart);

        for (Index_ ystart = 0; ystart < extract_length; ystart += block) {
            Index_ yend = ystart + std::min(block, extract_length - ystart);

            auto in = cache.data() + xstart + ystart * actual_dim;
            auto output = buffer.data() + xstart * extract_length + ystart;
            for (Index_ x = xstart; x < xend; ++x, output += extract_length) {
                for (Index_ y = ystart; y < yend; ++y, in += actual_start) {
                    *(output + y) + *in;
                }
            }
        }
    }

    cache.swap(buffer);
    return;
}

template<bool by_h5_row_, typename oracle_, typename Index_, typename CachedValue_>
struct Base {
private:
    // HDF5 members are stored in a separate pointer so we can serialize construction and destruction.
    std::unique_ptr<Components> h5comp;

    // Various dimension-related members.
    Index_ primary_dim;
    Index_ primary_chunkdim;
    Index_ extract_length;
    Index_ num_primary_chunks = 0, last_primary_chunkdim = 0;

    // Other members.
    typedef std::vector<CachedValue_> Slab;
    tatami_chunked::TypicalSlabCacheWorkspace<oracle_, false, Index_, Slab> cache_workspace;
    typename std::conditional<!by_h5_row_, std::vector<CachedValue_>, bool>::type transposition_buffer;
    typename std::conditional<!by_h5_row_, std::vector<std::pair<Index_, Index_> >, bool>::type cache_transpose_info;

public:
    Base(
        const std::string& file_name, 
        const std::string& dataset_name, 
        Index_ primary_dim, 
        Index_ primary_chunkdim, 
        Index_ extract_length,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        primary_dim(primary_dim), 
        primary_chunkdim(primary_chunkdim),
        extract_length(extract_length),
        cache_workspace(primary_chunkdim, extract_length, cache_size_in_elements, require_minimum_cache)
    {
        serialize([&]() -> void {
            h5comp.reset(new Components);

            // Turn off HDF5's caching, as we'll be handling that. This allows us
            // to parallelize extractions without locking when the data has already
            // been loaded into memory; if we just used HDF5's cache, we would have
            // to lock on every extraction, given the lack of thread safety.
            H5::FileAccPropList fapl(H5::FileAccPropList::DEFAULT.getId());
            fapl.setCache(0, 0, 0, 0);

            h5comp->file.openFile(parent->file_name, H5F_ACC_RDONLY, fapl);
            h5comp->dataset = h5comp->file.openDataSet(parent->dataset_name);
            h5comp->dataspace = h5comp->dataset.getSpace();
        });

        if (primary_chunkdim > 0) {
            num_primary_chunks = primary_dim / primary_chunkdim + (primary_dim % primary_chunkdim > 0); // i.e., integer ceiling.
            last_primary_chunkdim = (primary_dim > 0 ? (primary_dim - (num_primary_chunks - 1) * primary_chunkdim) : 0);
        }
    }

    ~Base() {
        serialize([&]() -> void {
            h5comp.reset();
        });
    }

private:
    // Overload that handles the truncated slab at the bottom/right edges of each matrix.
    Index_ get_primary_chunkdim(Index_ chunk_id) const {
        if (chunk_id + 1 == num_primary_chunks) {
            return last_primary_chunkdim;
        } else {
            return primary_chunkdim;
        }
    }

public:
    template<class Extract_>
    void extract(Index_ i, Value_* buffer, Extract_ extract) {
        const CachedValue_* ptr;

        if constexpr(oracle_) {
            auto info = cache_workspace.oracle_cache->next(
                /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                    return std::pair<Index_, Index_>(current / chunk_mydim, current % chunk_mydim);
                }, 
                /* create = */ [&]() -> Slab {
                    return Slab(cache_workspace.slab_size_in_elements);
                },
                /* populate = */ [&](const std::vector<std::pair<Index_, Slab> >& chunks) -> void {
                    if constexpr(!by_h5_row) {
                        cache_transpose_info.clear();
                    }

                    serialize([&]() -> void {
                        for (const auto& c : chunks) {
                            auto curdim = get_primary_chunkdim(c.first);
                            extract_slab(c.first * primary_chunk, curdim, c.second->data());
                            if constexpr(!by_h5_row) {
                                cache_transpose_info.emplace_back(c.second, curdim);
                            }
                        }
                    });

                    // Applying transpositions to all cached buffers for easier retrieval, but only once the lock is released.
                    if constexpr(!by_h5_row) {
                        for (const auto& x : cache_transpose_info) {
                            transpose(*(x.first), transposition_buffer, x.second, extract_length);
                        }
                    }
                }
            );

            ptr = info.first->data() + extract_length * info.second;

        } else {
            auto chunk = i / chunk_mydim;
            auto index = i % chunk_mydim;

            const auto& info = cache_workspace.lru_cache->find(
                chunk, 
                /* create = */ [&]() -> Slab {
                    return Slab(cache_workspace.slab_size_in_elements);
                },
                /* populate = */ [&](Index_ id, Slab* contents) -> void {
                    auto curdim = get_primary_chunkdim(id);
                    serialize([&]() -> void {
                        extract_slab(id * primary_chunk, curdim, contents->data());
                    });

                    // Applying a transposition for easier retrieval, but only once the lock is released.
                    if constexpr(!by_h5_row) {
                        transpose(chunk_contents, transposition_buffer, curdim, extract_length);
                    }
                }
            );

            ptr = info.data() + index * extract_length;
        }

        std::copy_n(ptr, extract_length, buffer);
    }

public:
    Index_ get_extract_length() const {
        return extract_length;
    }
};

template<bool accrow_, bool oracle_, typename Value_, typename Index_, bool transpose_, typename CachedValue_>
struct Full : public Base<accrow_ == transpose_, oracle_, Index_, CachedValue_>, public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Full(
        const std::string& file_name, 
        const std::string& dataset_name, 
        Index_ primary_dim, 
        Index_ primary_chunkdim, 
        Index_ secondary_dim,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        Base<accrow_ == transpose_, oracle_, Index_, CachedValue_>(
            file_name, 
            dataset_name, 
            primary_dim, 
            primary_chunkdim, 
            secondary_dim, 
            cache_size_in_elements, 
            require_minimum_cache
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        if (cache_workspace.num_slabs_in_cache == 0) {
            this->extract_block(i, 1, 0, this->get_extract_length(), buf, this->h5comp);
        } else {
            this->extract(i, buffer, [&](Index_ start, Index_, length, CachedValue_* buf) {
                this->extract_block(start, length, 0, this->get_extract_length(), buf, this->h5comp);
            });
        }
        return buffer;
    }
};

template<bool accrow_, bool oracle_, typename Value_, typename Index_, bool transpose_, typename CachedValue_>
struct Block : public Base<accrow_ != transpose_, oracle_, Index_, CachedValue_>, public tatami::DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    Block(
        const std::string& file_name, 
        const std::string& dataset_name, 
        Index_ primary_dim, 
        Index_ primary_chunkdim, 
        Index_ block_start,
        Index_ block_length,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        Base<accrow_ == transpose_, oracle_, Index_, CachedValue_>(
            file_name, 
            dataset_name, 
            primary_dim, 
            primary_chunkdim, 
            block_length, 
            cache_size_in_elements, 
            require_minimum_cache
        ),
        block_start(block_start),
        block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        if (cache_workspace.num_slabs_in_cache == 0) {
            this->extract_block(i, 1, block_start, block_length, buffer, this->h5comp);
        } else {
            this->extract(i, buffer, [&](Index_ start, Index_, length, CachedValue_* buf) {
                this->extract_block(start, length, block_start, block_length, buf, this->h5comp);
            });
        }
        return buffer;
    }
};

template<bool accrow_, bool oracle_, typename Value_, typename Index_, bool transpose_, typename CachedValue_>
struct Index : public Base<accrow_ != transpose_, oracle_, Index_, CachedValue_>, public tatami::DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    Index(
        const std::string& file_name, 
        const std::string& dataset_name, 
        Index_ primary_dim, 
        Index_ primary_chunkdim, 
        tatami::VectorPtr<Index_> indices_ptr,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        Base<accrow_ != transpose_, oracle_, Index_, CachedValue_>(
            file_name,
            dataset_name, 
            primary_dim, 
            primary_chunkdim, 
            indices_ptr->size(), 
            cache_size_in_elements, 
            require_minimum_cache
        ),
        indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        if (cache_workspace.num_slabs_in_cache == 0) {
            this->extract_indices(i, 1, *indices_ptr, buffer, this->h5comp);
        } else {
            this->extract(i, buffer, [&](Index_ start, Index_, length, CachedValue_* buf) {
                this->extract_indices(start, length, *indices_ptr, buf, this->h5comp);
            });
        }
        return buffer;
    }

    VectorPtr<Index_> indices_ptr; 
};

}
/**
 * @endcond
 */

/**
 * @brief Dense matrix backed by a DataSet in a HDF5 file.
 *
 * This class retrieves data from the HDF5 file on demand rather than loading it all in at the start.
 * This allows us to handle very large datasets in limited memory at the cost of some speed.
 *
 * We manually handle the chunk caching to speed up access for consecutive rows and columns.
 * The policy is to minimize the number of calls to the HDF5 library by requesting large contiguous slices where possible,
 * where each slice typically consists of multiple rows/columns that belong in the same HDF5 chunk.
 * The size of the slice is determined by the `options` in the constructor.
 *
 * Callers should follow the `prefer_rows()` suggestion when extracting data,
 * as this tries to minimize the number of chunks that need to be read per access request.
 * If they do not, the access pattern on disk may be slightly to highly suboptimal, depending on the chunk dimensions.
 *
 * As the HDF5 library is not generally thread-safe, the HDF5-related operations should only be run in a single thread.
 * This is normally handled automatically but developers can check out `serialize()` to customize the locking scheme.
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam transpose_ Whether the dataset is transposed in its storage order, i.e., rows in HDF5 are columns in this matrix.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store the values (based on their HDF5 type or other knowledge).
 */
template<typename Value_, typename Index_, bool transpose_ = false, typename CachedValue_ = Value_>
class Hdf5DenseMatrix : public tatami::Matrix<Value_, Index_> {
    Index_ firstdim, seconddim;
    std::string file_name, dataset_name;

    size_t cache_size_in_elements;
    bool require_minimum_cache;

    Index_ chunk_firstdim, chunk_seconddim;
    bool prefer_firstdim;

public:
    /**
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     * @param options Further options.
     */
    Hdf5DenseMatrix(std::string file, std::string name, const Hdf5Options& options) :
        file_name(std::move(file)), 
        dataset_name(std::move(name)),
        cache_size_in_elements(static_cast<double>(options.maximum_cache_size) / sizeof(CachedValue_)),
        require_minimum_cache(options.require_minimum_cache)
    {
        serialize([&]() -> void {
            H5::H5File fhandle(file_name, H5F_ACC_RDONLY);
            auto dhandle = open_and_check_dataset<false>(fhandle, dataset_name);
            auto dims = get_array_dimensions<2>(dhandle, dataset_name);
            firstdim = dims[0];
            seconddim = dims[1];

            auto dparms = dhandle.getCreatePlist();
            if (dparms.getLayout() != H5D_CHUNKED) {
                // If contiguous, each firstdim is treated as a chunk.
                chunk_firstdim = 1;
                chunk_seconddim = seconddim;
            } else {
                hsize_t chunk_dims[2];
                dparms.getChunk(2, chunk_dims);
                chunk_firstdim = chunk_dims[0];
                chunk_seconddim = chunk_dims[1];
            }
        });

        // Favoring extraction on the dimension that involves pulling out fewer chunks per dimension element.
        double chunks_per_firstdim = static_cast<double>(seconddim)/static_cast<double>(chunk_seconddim);
        double chunks_per_seconddim = static_cast<double>(firstdim)/static_cast<double>(chunk_firstdim);
        prefer_firstdim = (chunks_per_seconddim > chunks_per_firstdim);
    }

    /**
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     *
     * Unlike its overload, this constructor uses the defaults for `Hdf5Options`.
     */
    Hdf5DenseMatrix(std::string file, std::string name) : Hdf5DenseMatrix(std::move(file), std::move(name), Hdf5Options()) {}

public:
    Index_ nrow() const {
        if constexpr(transpose_) {
            return seconddim;
        } else {
            return firstdim;
        }
    }

    Index_ ncol() const {
        if constexpr(transpose_) {
            return firstdim;
        } else {
            return seconddim;
        }
    }

private:
    bool prefer_rows_internal() const {
        if constexpr(transpose_) {
            return !prefer_firstdim;
        } else {
            return prefer_firstdim;
        }
    }

public:
    /**
     * @return Boolean indicating whether to prefer row extraction.
     *
     * We favor extraction on the first dimension (rows by default, columns when `transpose = true`) as this matches the HDF5 storage order.
     * However, for some chunking scheme and `cache_limit`, this might require repeated reads from file;
     * in such cases, we switch to extraction on the second dimension.
     */
    bool prefer_rows() const {
        return prefer_rows_internal();
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(prefer_rows_internal());
    }
    
    bool uses_oracle(bool) const {
        // A non-zero cache is necessary (but not sufficient) for oracle usage.
        return cache_size_in_elements > 0;
    }

    using tatami::Matrix<Value_, Index_>::dense_row;

    using tatami::Matrix<Value_, Index_>::dense_column;

    using tatami::Matrix<Value_, Index_>::sparse_row;

    using tatami::Matrix<Value_, Index_>::sparse_column;

public:
    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_row(const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::FULL>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_row(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::BLOCK>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_row(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::INDEX>(opt, std::move(indices));
    }

    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_column(const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::FULL>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_column(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::BLOCK>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_column(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::INDEX>(opt, std::move(indices));
    }
};

}

#endif
