#ifndef TATAMI_HDF5_DENSE_MATRIX_HPP
#define TATAMI_HDF5_DENSE_MATRIX_HPP

#include "H5Cpp.h"

#include <string>
#include <cstdint>
#include <type_traits>
#include <cmath>
#include <list>
#include <vector>

#include "utils.hpp"

/**
 * @file Hdf5DenseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed dense matrix.
 */

namespace tatami_hdf5 {

/**
 * @brief Dense matrix backed by a DataSet in a HDF5 file.
 *
 * This class retrieves data from the HDF5 file on demand rather than loading it all in at the start.
 * This allows us to handle very large datasets in limited memory at the cost of some speed.
 *
 * We manually handle the chunk caching to speed up access for consecutive rows and columns.
 * The policy is to minimize the number of calls to the HDF5 library by requesting large contiguous slices where possible.
 * The size of the slice is determined by the cache limit in the constructor.
 *
 * Callers should follow the `prefer_rows()` suggestion when extracting data,
 * as this tries to minimize the number of chunks that need to be read per access request.
 * If they do not, the access pattern on disk may be slightly to highly suboptimal, depending on the chunk dimensions.
 *
 * As the HDF5 library is not generally thread-safe, the HDF5-related operations should only be run in a single thread.
 * For OpenMP, this is handled automatically by putting all HDF5 operations in a critical region.
 * For other parallelization schemes, callers should define the `TATAMI_HDF5_PARALLEL_LOCK` macro;
 * this should be a function that accepts and executes a no-argument lambda within an appropriate serial region (e.g., based on a global mutex).
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam transpose_ Whether the dataset is transposed in its storage order, i.e., rows in HDF5 are columns in this matrix.
 */
template<typename Value_, typename Index_, bool transpose_ = false>
class Hdf5DenseMatrix : public tatami::VirtualDenseMatrix<Value_, Index_> {
    Index_ firstdim, seconddim;
    std::string file_name, dataset_name;

    Index_ chunk_firstdim, chunk_seconddim;
    size_t total_cache_size;
    bool prefer_firstdim;

public:
    /**
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     * @param cache_limit Limit to the size of the chunk cache, in bytes.
     *
     * The cache size should be large enough to fit all chunks spanned by a row or column, for (near-)consecutive row and column access respectively.
     * Otherwise, performance will degrade as the same chunks may need to be repeatedly read back into memory.
     */
    Hdf5DenseMatrix(std::string file, std::string name, size_t cache_limit = 100000000) : 
        file_name(std::move(file)), 
        dataset_name(std::move(name))
    {
#ifndef TATAMI_HDF5_PARALLEL_LOCK
        #pragma omp critical
        {
#else
        TATAMI_HDF5_PARALLEL_LOCK([&]() -> void {
#endif

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

        // Favoring extraction on the dimension that involves pulling out fewer chunks per dimension element.
        double nchunks_firstdim = static_cast<double>(firstdim)/static_cast<double>(chunk_firstdim);
        double nchunks_seconddim = static_cast<double>(seconddim)/static_cast<double>(chunk_seconddim);
        prefer_firstdim = (nchunks_firstdim > nchunks_seconddim);

        total_cache_size = static_cast<double>(cache_limit) / sizeof(Value_);

#ifndef TATAMI_HDF5_PARALLEL_LOCK
        }
#else
        });
#endif

        return;
    }

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
        return true;
    }

    using tatami::Matrix<Value_, Index_>::dense_row;

    using tatami::Matrix<Value_, Index_>::dense_column;

    using tatami::Matrix<Value_, Index_>::sparse_row;

    using tatami::Matrix<Value_, Index_>::sparse_column;

private:
    template<bool accrow_>
    struct OracleCache {
        template<typename ... Args_>
        OracleCache(Args_&& ... args) : cache(std::forward<Args_>(args)...) {}

        tatami::OracleChunkCache<Index_, Index_, std::vector<Value_> > cache;
        typename std::conditional<accrow_ == transpose_, std::vector<std::pair<Index_, Index_> >, bool>::type cache_transpose_info;
    };

    typedef tatami::LruChunkCache<Index_, std::vector<Value_> > LruCache;

    template<bool accrow_>
    struct Workspace {
        void fill(const Hdf5DenseMatrix* parent, Index_ other_dim) {
            // Turn off HDF5's caching, as we'll be handling that. This allows us
            // to parallelize extractions without locking when the data has already
            // been loaded into memory; if we just used HDF5's cache, we would have
            // to lock on every extraction, given the lack of thread safety.
            H5::FileAccPropList fapl(H5::FileAccPropList::DEFAULT.getId());
            fapl.setCache(0, 0, 0, 0);

            file.openFile(parent->file_name, H5F_ACC_RDONLY, fapl);
            dataset = file.openDataSet(parent->dataset_name);
            dataspace = dataset.getSpace();

            auto chunk_dim = (accrow_ != transpose_ ? parent->chunk_firstdim : parent->chunk_seconddim);
            per_cache_size = static_cast<size_t>(chunk_dim) * static_cast<size_t>(other_dim);
            num_chunks = static_cast<double>(parent->total_cache_size) / per_cache_size;

            historian.reset(new LruCache(num_chunks));
        }

    public:
        // HDF5 members.
        H5::H5File file;
        H5::DataSet dataset;
        H5::DataSpace dataspace;
        H5::DataSpace memspace;

    public:
        // Caching members.
        size_t per_cache_size;
        Index_ num_chunks;
        typename std::conditional<accrow_ == transpose_, std::vector<Value_>, bool>::type transposition_buffer;

        // Cache with an oracle.
        std::unique_ptr<OracleCache<accrow_> > futurist;

        // Cache without an oracle.
        std::unique_ptr<LruCache> historian;
    };

private:
    template<bool accrow_, typename ExtractType_>
    static void extract_base(Index_ primary_start, Index_ primary_length, Value_* target, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) {
        hsize_t offset[2];
        hsize_t count[2];

        constexpr int dimdex = (accrow_ != transpose_);
        offset[1-dimdex] = primary_start;
        count[1-dimdex] = primary_length;

        constexpr bool indexed = std::is_same<ExtractType_, std::vector<Index_> >::value;

        if constexpr(indexed) {
            // Take slices across the current chunk for each index. This should be okay if consecutive,
            // but hopefully they've fixed the problem with non-consecutive slices in:
            // https://forum.hdfgroup.org/t/union-of-non-consecutive-hyperslabs-is-very-slow/5062
            count[dimdex] = 1;
            work.dataspace.selectNone();
            for (auto idx : extract_value) {
                offset[dimdex] = idx;
                work.dataspace.selectHyperslab(H5S_SELECT_OR, count, offset);
            }
            count[dimdex] = extract_length; // for the memspace setter.
        } else {
            offset[dimdex] = extract_value;
            count[dimdex] = extract_length;
            work.dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
        }

        // HDF5 is a lot faster when the memspace and dataspace match in dimensionality.
        // Presumably there is some shuffling that happens inside when dimensions don't match.
        work.memspace.setExtentSimple(2, count);
        work.memspace.selectAll();

        work.dataset.read(target, define_mem_type<Value_>(), work.memspace, work.dataspace);
    }

    template<bool accrow_, typename ExtractType_>
    static Index_ extract_chunk(Index_ chunk_id, Index_ dim, Index_ chunk_dim, Value_* target, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) {
        Index_ chunk_start = chunk_id * chunk_dim;
        Index_ chunk_end = std::min(dim, chunk_start + chunk_dim);
        Index_ chunk_actual = chunk_end - chunk_start;
        extract_base<accrow_>(chunk_start, chunk_actual, target, extract_value, extract_length, work);
        return chunk_actual;
    }

    static void transpose(std::vector<Value_>& cache, std::vector<Value_>& buffer, Index_ actual_dim, Index_ extract_length) {
        buffer.resize(cache.size());
        auto output = buffer.begin();
        for (Index_ x = 0; x < actual_dim; ++x, output += extract_length) {
            auto in = cache.begin() + x;
            for (Index_ y = 0; y < extract_length; ++y, in += actual_dim) {
                *(output + y) = *in;
            }
        }
        cache.swap(buffer);
        return;
    }

private:
    template<bool accrow_, typename ExtractType_>
    const Value_* extract_without_cache(Index_ i, Value_* buffer, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
#ifndef TATAMI_HDF5_PARALLEL_LOCK
        #pragma omp critical
        {
#else
        TATAMI_HDF5_PARALLEL_LOCK([&]() -> void {
#endif

            extract_base<accrow_>(i, 1, buffer, extract_value, extract_length, work);

#ifndef TATAMI_HDF5_PARALLEL_LOCK
        }
#else
        });
#endif

        return buffer;
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract_with_oracle(Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto& to_transpose = work.futurist->cache_transpose_info;
        auto info = work.futurist->cache.next_chunk(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / chunk_mydim, current % chunk_mydim);
            }, 
            /* swap = */ [](std::vector<Value_>& left, std::vector<Value_>& right) -> void {
                left.swap(right);
            },
            /* ready = */ [](const std::vector<Value_>& x) -> bool {
                return !x.empty();
            },
            /* allocate = */ [&](std::vector<Value_>& x) -> void {
                x.resize(work.per_cache_size);
            },
            /* populate = */ [&](const std::vector<std::pair<Index_, Index_> >& chunks_in_need, std::vector<std::vector<Value_> >& chunk_data) -> void {
                if constexpr(accrow_ == transpose_) {
                    to_transpose.clear();
                }

#ifndef TATAMI_HDF5_PARALLEL_LOCK
                #pragma omp critical
                {
#else
                TATAMI_HDF5_PARALLEL_LOCK([&]() -> void {
#endif

                for (const auto& c : chunks_in_need) {
                    auto& cache_target = chunk_data[c.second];
                    auto actual_dim = this->extract_chunk<accrow_>(c.first, mydim, chunk_mydim, cache_target.data(), extract_value, extract_length, work);
                    if constexpr(accrow_ == transpose_) {
                        to_transpose.emplace_back(c.second, actual_dim);
                    }
                }

#ifndef TATAMI_HDF5_PARALLEL_LOCK
                }
#else
                });
#endif

                // Applying transpositions to all cached buffers for easier retrieval, but only once the lock is released.
                if constexpr(accrow_ == transpose_) {
                    for (const auto& x : to_transpose) {
                        transpose(chunk_data[x.first], work.transposition_buffer, x.second, extract_length);
                    }
                }
            }
        );

        return info.first->data() + extract_length * info.second;
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract_without_oracle(Index_ i, Index_ mydim, Index_ chunk_mydim, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        auto chunk = i / chunk_mydim;
        auto index = i % chunk_mydim;

        const auto& cache_target = work.historian->find_chunk(
            chunk, 
            /* create = */ [&]() -> std::vector<Value_> {
                return std::vector<Value_>(work.per_cache_size);
            },
            /* populate = */ [&](Index_ id, std::vector<Value_>& chunk_contents) -> void {
                Index_ actual_dim;

#ifndef TATAMI_HDF5_PARALLEL_LOCK
                #pragma omp critical
                {
#else
                TATAMI_HDF5_PARALLEL_LOCK([&]() -> void {
#endif

                actual_dim = extract_chunk<accrow_>(chunk, mydim, chunk_mydim, chunk_contents.data(), extract_value, extract_length, work);

#ifndef TATAMI_HDF5_PARALLEL_LOCK
                }
#else
                });
#endif

                // Applying a transposition for easier retrieval, but only once the lock is released.
                if constexpr(accrow_ == transpose_) {
                    transpose(chunk_contents, work.transposition_buffer, actual_dim, extract_length);
                }
            }
        );

        return cache_target.data() + index * extract_length;
    }

    template<bool accrow_>
    Index_ get_target_dim() const {
        if constexpr(accrow_ != transpose_) {
            return firstdim;
        } else {
            return seconddim;
        }
    }

    template<bool accrow_>
    Index_ get_target_chunk_dim() const {
        if constexpr(accrow_ != transpose_) {
            return chunk_firstdim;
        } else {
            return chunk_seconddim;
        }
    }

    template<bool accrow_, typename ExtractType_>
    const Value_* extract(Index_ i, Value_* buffer, const ExtractType_& extract_value, Index_ extract_length, Workspace<accrow_>& work) const {
        // If there isn't any space for caching, we just extract directly.
        if (work.num_chunks == 0) {
            return extract_without_cache(i, buffer, extract_value, extract_length, work);
        }

        Index_ mydim = get_target_dim<accrow_>();
        Index_ chunk_mydim = get_target_chunk_dim<accrow_>();

        const Value_* cache;
        if (work.futurist) {
            cache = extract_with_oracle(mydim, chunk_mydim, extract_value, extract_length, work);
        } else {
            cache = extract_without_oracle(i, mydim, chunk_mydim, extract_value, extract_length, work);
        }

        std::copy(cache, cache + extract_length, buffer);
        return buffer;
    }

private:
    template<bool accrow_, tatami::DimensionSelectionType selection_>
    struct Hdf5Extractor : public tatami::Extractor<selection_, false, Value_, Index_> {
        Hdf5Extractor(const Hdf5DenseMatrix* p) : parent(p) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                this->full_length = (accrow_ ? parent->ncol() : parent->nrow());
                base.fill(parent, this->full_length); 
            }
        }

        Hdf5Extractor(const Hdf5DenseMatrix* p, Index_ start, Index_ length) : parent(p) {
            if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                this->block_start = start;
                this->block_length = length;
                base.fill(parent, this->block_length); 
            }
        }

        Hdf5Extractor(const Hdf5DenseMatrix* p, std::vector<Index_> idx) : parent(p) {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                this->index_length = idx.size();
                indices = std::move(idx);
                base.fill(parent, this->index_length); 
            }
        }

    protected:
        const Hdf5DenseMatrix* parent;
        Workspace<accrow_> base;
        typename std::conditional<selection_ == tatami::DimensionSelectionType::INDEX, std::vector<Index_>, bool>::type indices;

    public:
        const Index_* index_start() const {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                return indices.data();
            } else {
                return NULL;
            }
        }

        const Value_* fetch(Index_ i, Value_* buffer) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                return parent->extract<accrow_>(i, buffer, 0, this->full_length, this->base);
            } else if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                return parent->extract<accrow_>(i, buffer, this->block_start, this->block_length, this->base);
            } else {
                return parent->extract<accrow_>(i, buffer, this->indices, this->index_length, this->base);
            }
        }

        void set_oracle(std::unique_ptr<tatami::Oracle<Index_> > o) {
            auto chunk_mydim = parent->get_target_chunk_dim<accrow_>();
            size_t max_predictions = static_cast<size_t>(base.num_chunks) * chunk_mydim * 2; // double the cache size, basically.
            base.futurist.reset(new OracleCache<accrow_>(std::move(o), max_predictions, base.num_chunks));
            base.historian.reset();
        }
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_, typename ... Args_>
    std::unique_ptr<tatami::Extractor<selection_, false, Value_, Index_> > populate(const tatami::Options& opt, Args_&&... args) const {
        std::unique_ptr<tatami::Extractor<selection_, false, Value_, Index_> > output;

#ifndef TATAMI_HDF5_PARALLEL_LOCK
        #pragma omp critical
        {
#else
        TATAMI_HDF5_PARALLEL_LOCK([&]() -> void {
#endif

        output.reset(new Hdf5Extractor<accrow_, selection_>(this, std::forward<Args_>(args)...));

#ifndef TATAMI_HDF5_PARALLEL_LOCK
        }
#else
        });
#endif

        return output;
    }

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
