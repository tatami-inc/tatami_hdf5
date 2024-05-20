#ifndef TATAMI_HDF5_DENSE_MATRIX_HPP
#define TATAMI_HDF5_DENSE_MATRIX_HPP

#include "H5Cpp.h"

#include <string>
#include <type_traits>
#include <cmath>
#include <vector>

#include "serialize.hpp"
#include "utils.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

/**
 * @file DenseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed dense matrix.
 */

namespace tatami_hdf5 {

/**
 * @brief Options for `DenseMatrix` extraction.
 */
struct DenseMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all chunks required to read a row/column in `tatami::MyopicDenseExtractor::fetch()` and related methods.
     * This allows us to re-use the cached chunks when adjacent rows/columns are requested, rather than re-reading them from disk.
     *
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored if `require_minimum_cache` is `true`.
     */
    size_t maximum_cache_size = 100000000;

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that all chunks overlapping one row (or a slice/subset thereof) can be retained in memory,
     * so that the same chunks are not repeatedly re-read from disk when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

/**
 * @cond
 */
namespace DenseMatrix_internal {

// All HDF5-related members.
struct Components {
    H5::H5File file;
    H5::DataSet dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

template<typename Index_, typename OutputValue_>
void extract_block(bool h5_row_is_target, Index_ cache_start, Index_ cache_length, Index_ block_start, Index_ block_length, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    int target_dim = 1 - h5_row_is_target;
    offset[target_dim] = cache_start;
    count[target_dim] = cache_length;

    int non_target_dim = h5_row_is_target;
    offset[non_target_dim] = block_start;
    count[non_target_dim] = block_length;
    comp.dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // HDF5 is a lot faster when the memspace and dataspace match in dimensionality.
    // Presumably there is some shuffling that happens inside when dimensions don't match.
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

template<typename Index_, typename OutputValue_>
void extract_indices(bool h5_row_is_target, Index_ cache_start, Index_ cache_length, const std::vector<Index_>& indices, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    int target_dim = 1 - h5_row_is_target;
    offset[target_dim] = cache_start;
    count[target_dim] = cache_length;

    int non_target_dim = h5_row_is_target;

    // Take slices across the current chunk for each index. This should be okay if consecutive,
    // but hopefully they've fixed the problem with non-consecutive slices in:
    // https://forum.hdfgroup.org/t/union-of-non-consecutive-hyperslabs-is-very-slow/5062
    comp.dataspace.selectNone();
    tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(),
        [&](Index_ start, Index_ length) {
            offset[non_target_dim] = start;
            count[non_target_dim] = length;
            comp.dataspace.selectHyperslab(H5S_SELECT_OR, count, offset);
        }
    );

    // Again, matching the dimensionality.
    count[non_target_dim] = indices.size();
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

/********************
 *** Core classes ***
 ********************/

// We store the Components in a pointer so that we can serialize their
// construction and destruction.

inline void initialize(const std::string& file_name, const std::string& dataset_name, std::unique_ptr<Components>& h5comp) {
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

inline void destroy(std::unique_ptr<Components>& h5comp) {
    serialize([&]() -> void {
        h5comp.reset();
    });
}

template<bool oracle_, typename Index_>
class SoloCore {
public:
    SoloCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        bool by_h5_row,
        [[maybe_unused]] tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, // only listed here for compatibility with the other constructors.
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        [[maybe_unused]] Index_ non_target_length, 
        [[maybe_unused]] const tatami_chunked::SlabCacheStats& slab_stats) :
        my_by_h5_row(by_h5_row),
        my_oracle(std::move(oracle))
    {
        initialize(file_name, dataset_name, my_h5comp);
    }

    ~SoloCore() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    bool my_by_h5_row;
    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, size_t, bool>::type my_counter = 0;

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&](){
            extract_block(my_by_h5_row, i, static_cast<Index_>(1), block_start, block_length, buffer, *my_h5comp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&](){
            extract_indices(my_by_h5_row, i, static_cast<Index_>(1), indices, buffer, *my_h5comp);
        });
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
class MyopicCore {
public:
    MyopicCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        bool by_h5_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_>, // for consistency with the oracular version.
        Index_ non_target_length, 
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_by_h5_row(by_h5_row),
        my_dim_stats(std::move(target_dim_stats)),
        my_extract_length(non_target_length),
        my_factory(slab_stats), 
        my_cache(slab_stats.max_slabs_in_cache)
    {
        initialize(file_name, dataset_name, my_h5comp);
        if (!my_by_h5_row) {
            my_transposition_buffer.resize(slab_stats.slab_size_in_elements);
        }
    }

    ~MyopicCore() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    bool my_by_h5_row;

    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;
    Index_ my_extract_length;

    tatami_chunked::DenseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

    std::vector<CachedValue_> my_transposition_buffer;

private:
    template<typename Value_, class Extract_>
    void fetch_raw(Index_ i, Value_* buffer, Extract_ extract) {
        Index_ chunk = i / my_dim_stats.chunk_length;
        Index_ index = i % my_dim_stats.chunk_length;

        const auto& info = my_cache.find(
            chunk, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](Index_ id, Slab& contents) -> void {
                auto curdim = tatami_chunked::get_chunk_length(my_dim_stats, id);

                if (my_by_h5_row) {
                    serialize([&]() -> void {
                        extract(id * my_dim_stats.chunk_length, curdim, contents.data);
                    });

                } else {
                    // Transposing the data for easier retrieval, but only once the lock is released.
                    auto tptr = my_transposition_buffer.data();
                    serialize([&]() -> void {
                        extract(id * my_dim_stats.chunk_length, curdim, tptr);
                    });
                    tatami::transpose(tptr, my_extract_length, curdim, contents.data);
                }
            }
        );

        auto ptr = info.data + static_cast<size_t>(my_extract_length) * static_cast<size_t>(index); // cast to size_t to avoid overflow
        std::copy_n(ptr, my_extract_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_block(my_by_h5_row, start, length, block_start, block_length, buf, *my_h5comp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_indices(my_by_h5_row, start, length, indices, buf, *my_h5comp);
        });
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
struct OracularCore {
    OracularCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        bool by_h5_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        Index_ non_target_length, 
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_by_h5_row(by_h5_row),
        my_dim_stats(std::move(target_dim_stats)),
        my_extract_length(non_target_length),
        my_factory(slab_stats), 
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache)
    {
        initialize(file_name, dataset_name, my_h5comp);
        if (!my_by_h5_row) {
            my_transposition_buffer.resize(slab_stats.slab_size_in_elements);
            my_transposition_buffer_ptr = my_transposition_buffer.data();
            my_cache_transpose_info.reserve(slab_stats.max_slabs_in_cache);
        }
    }

    ~OracularCore() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    bool my_by_h5_row;

    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;
    Index_ my_extract_length;

    tatami_chunked::DenseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

    std::vector<CachedValue_> my_transposition_buffer;
    CachedValue_* my_transposition_buffer_ptr;
    std::vector<std::pair<Slab*, Index_> > my_cache_transpose_info;

public:
    template<typename Value_, class Extract_>
    void fetch_raw([[maybe_unused]] Index_ i, Value_* buffer, Extract_ extract) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_dim_stats.chunk_length, current % my_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                if (!my_by_h5_row) {
                    my_cache_transpose_info.clear();
                }

                serialize([&]() -> void {
                    for (const auto& c : chunks) {
                        auto curdim = tatami_chunked::get_chunk_length(my_dim_stats, c.first);
                        extract(c.first * my_dim_stats.chunk_length, curdim, c.second->data);
                        if (!my_by_h5_row) {
                            my_cache_transpose_info.emplace_back(c.second, curdim);
                        }
                    }
                });

                // Applying transpositions to each cached buffers for easier
                // retrieval. Done outside the serial section to unblock other threads.
                if (!my_by_h5_row) {
                    if (my_extract_length != 1) {
                        for (const auto& c : my_cache_transpose_info) {
                            if (c.second != 1) {
                                tatami::transpose(c.first->data, my_extract_length, c.second, my_transposition_buffer_ptr);

                                // We actually swap the pointers here, so the slab
                                // pointers might not point to the factory pool after this!
                                // Shouldn't matter as long as neither of them leave this class.
                                std::swap(c.first->data, my_transposition_buffer_ptr);
                            }
                        }
                    }
                }
            }
        );

        auto ptr = info.first->data + static_cast<size_t>(my_extract_length) * static_cast<size_t>(info.second); // cast to size_t to avoid overflow
        std::copy_n(ptr, my_extract_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_block(my_by_h5_row, start, length, block_start, block_length, buf, *my_h5comp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(i, buffer, [&](Index_ start, Index_ length, CachedValue_* buf) {
            extract_indices(my_by_h5_row, start, length, indices, buf, *my_h5comp);
        });
        return buffer;
    }
};

template<bool solo_, bool oracle_, typename Index_, typename CachedValue_>
using DenseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, Index_>,
      typename std::conditional<oracle_,
          OracularCore<Index_, CachedValue_>,
          MyopicCore<Index_, CachedValue_>
      >::type
>::type;

/***************************
 *** Concrete subclasses ***
 ***************************/

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct Full : public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Full(
        const std::string& file_name, 
        const std::string& dataset_name, 
        bool by_h5_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            file_name, 
            dataset_name, 
            by_h5_row,
            std::move(target_dim_stats),
            std::move(oracle),
            non_target_dim, 
            slab_stats
        ),
        my_non_target_dim(non_target_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, 0, my_non_target_dim, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_> 
struct Block : public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Block(
        const std::string& file_name, 
        const std::string& dataset_name, 
        bool by_h5_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core( 
            file_name, 
            dataset_name, 
            by_h5_row,
            std::move(target_dim_stats),
            std::move(oracle),
            block_length, 
            slab_stats
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, my_block_start, my_block_length, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, typename Value_, typename Index_, typename CachedValue_>
struct Index : public tatami::DenseExtractor<oracle_, Value_, Index_> {
    Index(
        const std::string& file_name, 
        const std::string& dataset_name, 
        bool by_h5_row,
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami_chunked::SlabCacheStats& slab_stats) :
        my_core(
            file_name,
            dataset_name, 
            by_h5_row,
            std::move(target_dim_stats),
            std::move(oracle),
            indices_ptr->size(), 
            slab_stats
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_indices(i, *my_indices_ptr, buffer);
    }

private:
    DenseCore<solo_, oracle_, Index_, CachedValue_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
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
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store the values (based on their HDF5 type or other knowledge).
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_>
class DenseMatrix : public tatami::Matrix<Value_, Index_> {
    std::string my_file_name, my_dataset_name;
    bool my_transpose;

    size_t my_cache_size_in_elements;
    bool my_require_minimum_cache;

    tatami_chunked::ChunkDimensionStats<Index_> my_firstdim_stats, my_seconddim_stats;
    bool my_prefer_firstdim;

public:
    /**
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     * @param transpose Whether the dataset is transposed in its storage order, i.e., rows in the HDF5 dataset correspond to columns of this matrix.
     * This may be true for HDF5 files generated by frameworks that use column-major matrices,
     * where preserving the data layout between memory and disk is more efficient (see, e.g., the **rhdf5** Bioconductor package).
     * @param options Further options for data extraction.
     */
    DenseMatrix(std::string file, std::string name, bool transpose, const DenseMatrixOptions& options) :
        my_file_name(std::move(file)), 
        my_dataset_name(std::move(name)),
        my_transpose(transpose),
        my_cache_size_in_elements(options.maximum_cache_size / sizeof(CachedValue_)),
        my_require_minimum_cache(options.require_minimum_cache)
    {
        serialize([&]() -> void {
            H5::H5File fhandle(my_file_name, H5F_ACC_RDONLY);
            auto dhandle = open_and_check_dataset<false>(fhandle, my_dataset_name);
            auto dims = get_array_dimensions<2>(dhandle, my_dataset_name);

            hsize_t chunk_dims[2];
            auto dparms = dhandle.getCreatePlist();
            if (dparms.getLayout() != H5D_CHUNKED) {
                // If contiguous, each firstdim is treated as a chunk.
                chunk_dims[0] = 1;
                chunk_dims[1] = dims[1];
            } else {
                dparms.getChunk(2, chunk_dims);
            }

            my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(dims[0], chunk_dims[0]);
            my_seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(dims[1], chunk_dims[1]);
        });

        // Favoring extraction on the dimension that involves pulling out fewer
        // chunks per dimension element. Remember, 'firstdim_stats.num_chunks'
        // represents the number of chunks along the first dimension, and thus
        // is the number of chunks that need to be loaded to pull out an
        // element of the **second** dimension; and vice versa.
        my_prefer_firstdim = (my_firstdim_stats.num_chunks > my_seconddim_stats.num_chunks);
    }

    /**
     * Overload that uses the default `DenseMatrixOptions`.
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     * @param transpose Whether the dataset is transposed in its storage order.
     */
    DenseMatrix(std::string file, std::string name, bool transpose) : 
        DenseMatrix(std::move(file), std::move(name), transpose, DenseMatrixOptions()) {}

private:
    bool prefer_rows_internal() const {
        if (my_transpose) {
            return !my_prefer_firstdim;
        } else {
            return my_prefer_firstdim;
        }
    }

    Index_ nrow_internal() const {
        if (my_transpose) {
            return my_seconddim_stats.dimension_extent;
        } else {
            return my_firstdim_stats.dimension_extent;
        }
    }

    Index_ ncol_internal() const {
        if (my_transpose) {
            return my_firstdim_stats.dimension_extent;
        } else {
            return my_seconddim_stats.dimension_extent;
        }
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
        return true;
    }

    bool is_sparse() const {
        return false;
    }

    double is_sparse_proportion() const { 
        return 0;
    }

    using tatami::Matrix<Value_, Index_>::dense;

    using tatami::Matrix<Value_, Index_>::sparse;

private:
    template<bool oracle_, template<bool, bool, typename, typename, typename> class Extractor_, typename ... Args_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(bool row, Index_ non_target_length, tatami::MaybeOracle<oracle_, Index_> oracle, Args_&& ... args) const {
        bool by_h5_row = (row != my_transpose);
        const auto& dim_stats = (by_h5_row ? my_firstdim_stats : my_seconddim_stats);

        tatami_chunked::SlabCacheStats slab_stats(dim_stats.chunk_length, non_target_length, dim_stats.num_chunks, my_cache_size_in_elements, my_require_minimum_cache);
        if (slab_stats.max_slabs_in_cache > 0) {
            return std::make_unique<Extractor_<false, oracle_, Value_, Index_, CachedValue_> >(
                my_file_name, my_dataset_name, by_h5_row, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats
            );
        } else {
            return std::make_unique<Extractor_<true, oracle_, Value_, Index_, CachedValue_> >(
                my_file_name, my_dataset_name, by_h5_row, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats
            );
        }
    }

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options&) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, DenseMatrix_internal::Full>(row, full_non_target, false, full_non_target);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options&) const {
        return populate<false, DenseMatrix_internal::Block>(row, block_length, false, block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options&) const {
        auto nidx = indices_ptr->size();
        return populate<false, DenseMatrix_internal::Index>(row, nidx, false, std::move(indices_ptr));
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(dense(row, opt), full_non_target, opt);
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
        const tatami::Options&) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<true, DenseMatrix_internal::Full>(row, full_non_target, std::move(oracle), full_non_target);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options&) 
    const {
        return populate<true, DenseMatrix_internal::Block>(row, block_length, std::move(oracle), block_start, block_length);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options&) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, DenseMatrix_internal::Index>(row, nidx, std::move(oracle), std::move(indices_ptr));
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
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), opt), full_non_target, opt);
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
