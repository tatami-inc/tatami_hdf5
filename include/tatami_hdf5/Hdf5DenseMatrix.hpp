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
void extract_block(Index_ cache_start, Index_ cache_length, Index_ block_start, Index_ block_length, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    constexpr int dimdex = by_h5_row_;
    offset[1-dimdex] = cache_start;
    count[1-dimdex] = cache_length;

    offset[dimdex] = block_start;
    count[dimdex] = block_length;
    comp.dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // HDF5 is a lot faster when the memspace and dataspace match in dimensionality.
    // Presumably there is some shuffling that happens inside when dimensions don't match.
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

template<bool by_h5_row_, typename Index_, typename OutputValue_>
void extract_indices(Index_ cache_start, Index_ cache_length, const std::vector<Index_>& indices, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    constexpr int dimdex = by_h5_row_;
    offset[1-dimdex] = cache_start;
    count[1-dimdex] = cache_length;

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

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

// 'cache' is assumed to be row-major with 'actual_dim' columns and
// 'extract_length' rows. On output, it will be column-major instead.
template<typename CachedValue_>
void transpose(std::vector<CachedValue_>& cache, std::vector<CachedValue_>& buffer, size_t actual_dim, size_t extract_length) {
    if (actual_dim == 1 || extract_length == 1) {
        return;
    }

    buffer.resize(cache.size());

    // Using a blockwise strategy to perform the transposition,
    // in order to be more cache-friendly.
    constexpr size_t block = 16;
    for (size_t xstart = 0; xstart < actual_dim; xstart += block) {
        size_t xend = xstart + std::min(block, actual_dim - xstart);

        for (size_t ystart = 0; ystart < extract_length; ystart += block) {
            size_t yend = ystart + std::min(block, extract_length - ystart);

            auto in = cache.data() + xstart + ystart * actual_dim;
            auto output = buffer.data() + xstart * extract_length + ystart;

            for (size_t x = xstart; x < xend; ++x, ++in, output += extract_length) {
                auto in_copy = in;
                auto output_copy = output;

                for (size_t y = ystart; y < yend; ++y, in_copy += actual_dim, ++output_copy) {
                    *output_copy = *in_copy;
                }
            }
        }
    }

    cache.swap(buffer);
    return;
}

template<bool by_h5_row_, bool oracle_, typename Index_, typename CachedValue_>
struct Base {
protected:
    // HDF5 members are stored in a separate pointer so we can serialize construction and destruction.
    std::unique_ptr<Components> h5comp;

    // Various dimension-related members.
    tatami_chunked::ChunkDimensionStats<Index_> dim_stats;
    Index_ extract_length;

    // Other members.
    typedef std::vector<CachedValue_> Slab;
    tatami_chunked::TypicalSlabCacheWorkspace<oracle_, false, Index_, Slab> cache_workspace;
    typename std::conditional<!by_h5_row_, std::vector<CachedValue_>, bool>::type transposition_buffer;
    typename std::conditional<!by_h5_row_ && oracle_, std::vector<std::pair<Slab*, Index_> >, bool>::type cache_transpose_info;

public:
    Base(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ extract_length,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        dim_stats(std::move(cache_dim_stats)),
        extract_length(extract_length),
        cache_workspace(dim_stats.chunk_length, extract_length, cache_size_in_elements, require_minimum_cache, std::move(oracle))
    {
        serialize([&]() -> void {
            h5comp.reset(new Components);

            // Turn off HDF5's caching, as we'll be handling that. This allows us
            // to parallelize extractions without locking when the data has already
            // been loaded into memory; if we just used HDF5's cache, we would have
            // to lock on every extraction, given the lack of thread safety.
            H5::FileAccPropList fapl(H5::FileAccPropList::DEFAULT.getId());
            fapl.setCache(0, 0, 0, 0);

            h5comp->file.openFile(file_name, H5F_ACC_RDONLY, fapl);
            h5comp->dataset = h5comp->file.openDataSet(dataset_name);
            h5comp->dataspace = h5comp->dataset.getSpace();
        });
    }

    ~Base() {
        serialize([&]() -> void {
            h5comp.reset();
        });
    }

public:
    template<typename Value_, class Extract_>
    void extract(Index_ i, Value_* buffer, Extract_ extract) {
        const CachedValue_* ptr;

        if constexpr(oracle_) {
            auto info = cache_workspace.cache.next(
                /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                    return std::pair<Index_, Index_>(current / dim_stats.chunk_length, current % dim_stats.chunk_length);
                }, 
                /* create = */ [&]() -> Slab {
                    return Slab(cache_workspace.slab_size_in_elements);
                },
                /* populate = */ [&](const std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                    if constexpr(!by_h5_row_) {
                        cache_transpose_info.clear();
                    }

                    serialize([&]() -> void {
                        for (const auto& c : chunks) {
                            auto curdim = dim_stats.get_chunk_length(c.first);
                            extract(c.first * dim_stats.chunk_length, curdim, c.second->data());
                            if constexpr(!by_h5_row_) {
                                cache_transpose_info.emplace_back(c.second, curdim);
                            }
                        }
                    });

                    // Applying transpositions to all cached buffers for easier retrieval, but only once the lock is released.
                    if constexpr(!by_h5_row_) {
                        for (const auto& x : cache_transpose_info) {
                            transpose(*(x.first), transposition_buffer, x.second, extract_length);
                        }
                    }
                }
            );

            ptr = info.first->data() + static_cast<size_t>(extract_length) * static_cast<size_t>(info.second); // cast to size_t to avoid overflow

        } else {
            auto chunk = i / dim_stats.chunk_length;
            auto index = i % dim_stats.chunk_length;

            const auto& info = cache_workspace.cache.find(
                chunk, 
                /* create = */ [&]() -> Slab {
                    return Slab(cache_workspace.slab_size_in_elements);
                },
                /* populate = */ [&](Index_ id, Slab& contents) -> void {
                    auto curdim = dim_stats.get_chunk_length(id);
                    serialize([&]() -> void {
                        extract(id * dim_stats.chunk_length, curdim, contents.data());
                    });

                    // Applying a transposition for easier retrieval, but only once the lock is released.
                    if constexpr(!by_h5_row_) {
                        transpose(contents, transposition_buffer, curdim, extract_length);
                    }
                }
            );

            ptr = info.data() + static_cast<size_t>(extract_length) * static_cast<size_t>(index); // cast to size_t to avoid overflow
        }

        std::copy_n(ptr, extract_length, buffer);
    }

public:
    Index_ reindex(Index_ i) {
        if constexpr(oracle_) {
            return cache_workspace.cache.next();
        } else {
            return i;
        }
    }
};

template<bool accrow_, bool oracle_, typename Value_, typename Index_, bool transpose_, typename CachedValue_, bool use_h5_row_ = (accrow_ != transpose_)>
struct Full : public Base<use_h5_row_, oracle_, Index_, CachedValue_>, public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Full(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ uncached_dim,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        Base<use_h5_row_, oracle_, Index_, CachedValue_>(
            file_name, 
            dataset_name, 
            std::move(cache_dim_stats),
            std::move(oracle),
            uncached_dim, 
            cache_size_in_elements, 
            require_minimum_cache
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        if (this->cache_workspace.num_slabs_in_cache == 0) {
            serialize([&](){
                extract_block<use_h5_row_>(this->reindex(i), static_cast<Index_>(1), static_cast<Index_>(0), this->extract_length, buffer, *(this->h5comp));
            });
        } else {
            this->extract(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
                extract_block<use_h5_row_>(start, length, static_cast<Index_>(0), this->extract_length, buf, *(this->h5comp));
            });
        }
        return buffer;
    }
};

template<bool accrow_, bool oracle_, typename Value_, typename Index_, bool transpose_, typename CachedValue_, bool use_h5_row_ = (accrow_ != transpose_)>
struct Block : public Base<use_h5_row_, oracle_, Index_, CachedValue_>, public tatami::DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    Block(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        Base<use_h5_row_, oracle_, Index_, CachedValue_>(
            file_name, 
            dataset_name, 
            std::move(cache_dim_stats),
            std::move(oracle),
            block_length, 
            cache_size_in_elements, 
            require_minimum_cache
        ),
        block_start(block_start),
        block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        if (this->cache_workspace.num_slabs_in_cache == 0) {
            serialize([&](){
                extract_block<use_h5_row_>(this->reindex(i), static_cast<Index_>(1), block_start, block_length, buffer, *(this->h5comp));
            });
        } else {
            this->extract(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
                extract_block<use_h5_row_>(start, length, block_start, block_length, buf, *(this->h5comp));
            });
        }
        return buffer;
    }

private:
    Index_ block_start, block_length;
};

template<bool accrow_, bool oracle_, typename Value_, typename Index_, bool transpose_, typename CachedValue_, bool use_h5_row_ = (accrow_ != transpose_)>
struct Index : public Base<use_h5_row_, oracle_, Index_, CachedValue_>, public tatami::DenseExtractor<oracle_, Value_, Index_> {
    template<typename ... Args_>
    Index(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> cache_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        size_t cache_size_in_elements,
        bool require_minimum_cache) :
        Base<use_h5_row_, oracle_, Index_, CachedValue_>(
            file_name,
            dataset_name, 
            std::move(cache_dim_stats),
            std::move(oracle),
            indices_ptr->size(), 
            cache_size_in_elements, 
            require_minimum_cache
        ),
        indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        if (this->cache_workspace.num_slabs_in_cache == 0) {
            serialize([&](){
                extract_indices<use_h5_row_>(this->reindex(i), static_cast<Index_>(1), *indices_ptr, buffer, *(this->h5comp));
            });
        } else {
            this->extract(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
                extract_indices<use_h5_row_>(start, length, *indices_ptr, buf, *(this->h5comp));
            });
        }
        return buffer;
    }

private:
    tatami::VectorPtr<Index_> indices_ptr; 
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
    std::string file_name, dataset_name;

    size_t cache_size_in_elements;
    bool require_minimum_cache;

    tatami_chunked::ChunkDimensionStats<Index_> firstdim_stats, seconddim_stats;
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

            hsize_t chunk_dims[2];
            auto dparms = dhandle.getCreatePlist();
            if (dparms.getLayout() != H5D_CHUNKED) {
                // If contiguous, each firstdim is treated as a chunk.
                chunk_dims[0] = 1;
                chunk_dims[1] = dims[1];
            } else {
                dparms.getChunk(2, chunk_dims);
            }

            firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(dims[0], chunk_dims[0]);
            seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(dims[1], chunk_dims[1]);
        });

        // Favoring extraction on the dimension that involves pulling out fewer
        // chunks per dimension element. Remember, 'firstdim_stats.num_chunks'
        // represents the number of chunks along the first dimension, and thus
        // is the number of chunks that need to be loaded to pull out an
        // element of the **second** dimension; and vice versa.
        prefer_firstdim = (firstdim_stats.num_chunks > seconddim_stats.num_chunks);
    }

    /**
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     *
     * Unlike its overload, this constructor uses the defaults for `Hdf5Options`.
     */
    Hdf5DenseMatrix(std::string file, std::string name) : Hdf5DenseMatrix(std::move(file), std::move(name), Hdf5Options()) {}

private:
    bool prefer_rows_internal() const {
        if constexpr(transpose_) {
            return !prefer_firstdim;
        } else {
            return prefer_firstdim;
        }
    }

    const tatami_chunked::ChunkDimensionStats<Index_>& get_row_stats() const {
        if constexpr(transpose_) {
            return seconddim_stats;
        } else {
            return firstdim_stats;
        }
    }

    const tatami_chunked::ChunkDimensionStats<Index_>& get_column_stats() const {
        if constexpr(transpose_) {
            return firstdim_stats;
        } else {
            return seconddim_stats;
        }
    }

    Index_ nrow_internal() const {
        return get_row_stats().dimension_extent;
    }

    Index_ ncol_internal() const {
        return get_column_stats().dimension_extent;
    }


public:
    Index_ nrow() const {
        return nrow_internal();
    }

    Index_ ncol() const {
        return ncol_internal();
    }

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

    bool sparse() const {
        return false;
    }

    double sparse_proportion() const { 
        return 0;
    }

    using tatami::Matrix<Value_, Index_>::dense;

    using tatami::Matrix<Value_, Index_>::sparse;

private:
    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(
        bool row, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const tatami::Options&) 
    const {
        if (row) {
            return std::make_unique<Hdf5DenseMatrix_internal::Full<true, oracle_, Value_, Index_, transpose_, CachedValue_> >(
                file_name, dataset_name, get_row_stats(), std::move(oracle), ncol_internal(), cache_size_in_elements, require_minimum_cache
            );
        } else {
            return std::make_unique<Hdf5DenseMatrix_internal::Full<false, oracle_, Value_, Index_, transpose_, CachedValue_> >(
                file_name, dataset_name, get_column_stats(), std::move(oracle), nrow_internal(), cache_size_in_elements, require_minimum_cache
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(
        bool row, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options&) 
    const {
        if (row) {
            return std::make_unique<Hdf5DenseMatrix_internal::Block<true, oracle_, Value_, Index_, transpose_, CachedValue_> >(
                file_name, dataset_name, get_row_stats(), std::move(oracle), block_start, block_length, cache_size_in_elements, require_minimum_cache
            );
        } else {
            return std::make_unique<Hdf5DenseMatrix_internal::Block<false, oracle_, Value_, Index_, transpose_, CachedValue_> >(
                file_name, dataset_name, get_column_stats(), std::move(oracle), block_start, block_length, cache_size_in_elements, require_minimum_cache
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(
        bool row, 
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options&) 
    const {
        if (row) {
            return std::make_unique<Hdf5DenseMatrix_internal::Index<true, oracle_, Value_, Index_, transpose_, CachedValue_> >(
                file_name, dataset_name, get_row_stats(), std::move(oracle), std::move(indices_ptr), cache_size_in_elements, require_minimum_cache
            );
        } else {
            return std::make_unique<Hdf5DenseMatrix_internal::Index<false, oracle_, Value_, Index_, transpose_, CachedValue_> >(
                file_name, dataset_name, get_column_stats(), std::move(oracle), std::move(indices_ptr), cache_size_in_elements, require_minimum_cache
            );
        }
    }

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        return populate<false>(row, false, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false>(row, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate<false>(row, false, std::move(indices_ptr), opt);
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(dense(row, opt), (row ? ncol_internal() : nrow_internal()), opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<false, Value_, Index_> >(dense(row, block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto ptr = dense(row, indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<false, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        const tatami::Options& opt) 
    const {
        return populate<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return populate<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        return populate<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }

    /***********************
     *** Oracular sparse ***
     ***********************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        const tatami::Options& opt) 
    const {
        return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), opt), (row ? ncol_internal() : nrow_internal()), opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto ptr = dense(row, std::move(oracle), indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }
};

}

#endif
