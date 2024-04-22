#ifndef TATAMI_HDF5_SPARSE_MATRIX_HPP
#define TATAMI_HDF5_SPARSE_MATRIX_HPP

#include "H5Cpp.h"

#include <string>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <cmath>

#include "tatami/tatami.hpp"

#include "sparse_primary.hpp"
#include "sparse_secondary.hpp"
#include "serialize.hpp"
#include "utils.hpp"

/**
 * @file Hdf5CompressedSparseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed compressed sparse matrix.
 */

namespace tatami_hdf5 {

/**
 * @brief Compressed sparse matrix in a HDF5 file.
 *
 * This class retrieves sparse data from the HDF5 file on demand rather than loading it all in at the start.
 * This allows us to handle very large datasets in limited memory at the cost of speed.
 *
 * We manually handle the chunk caching to speed up access for consecutive rows or columns (for compressed sparse row and column matrices, respectively).
 * The policy is to minimize the number of calls to the HDF5 library - and thus expensive file reads - by requesting large contiguous slices where possible, i.e., multiple columns or rows for CSC and CSR matrices, respectively.
 * These are held in memory in the `Extractor` while the relevant column/row is returned to the user by `row()` or `column()`.
 * The size of the slice is determined by the `options` in the constructor.
 *
 * Callers should follow the `prefer_rows()` suggestion when extracting data,
 * as this tries to minimize the number of chunks that need to be read per access request.
 * This recommendation is even stronger than for the `Hdf5DenseMatrix`,
 * as the access pattern on disk for the non-preferred dimension is very suboptimal.
 *
 * As the HDF5 library is not generally thread-safe, the HDF5-related operations should only be run in a single thread.
 * This is normally handled automatically but developers can check out `serialize()` to customize the locking scheme.
 *
 * @tparam row_ Whether the matrix is stored in compressed sparse row format.
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all values (based on their HDF5 type or other knowledge).
 * @tparam CachedIndex_ Type of the index value to store in the cache.
 * This can be set to a narrower type than `Index_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all indices (based on their HDF5 type or other knowledge).
 */
template<bool row_, typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class Hdf5CompressedSparseMatrix : public tatami::Matrix<Value_, Index_> {
    Index_ nrows, ncols;
    std::string file_name;
    std::string data_name, index_name;
    std::vector<hsize_t> pointers;

    size_t cache_size_limit;
    bool require_minimum_cache; 
    Index_ max_non_zeros;
    tatami_chunked::ChunkDimensionStats<Index_> secondary_chunk_stats;

public:
    /**
     * @param nr Number of rows in the matrix.
     * @param nc Number of columns in the matrix.
     * @param file Path to the file.
     * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
     * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
     * If `row_ = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row_ = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row_ = true`) or columns (otherwise).
     * @param options Further options.
     */
    Hdf5CompressedSparseMatrix(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr, const Hdf5Options& options) :
        nrows(nr), 
        ncols(nc), 
        file_name(std::move(file)), 
        data_name(std::move(vals)), 
        index_name(std::move(idx)), 
        cache_size_limit(options.maximum_cache_size),
        require_minimum_cache(options.require_minimum_cache)
    {
        serialize([&]() -> void {
            H5::H5File file_handle(file_name, H5F_ACC_RDONLY);
            auto dhandle = open_and_check_dataset<false>(file_handle, data_name);
            hsize_t nonzeros = get_array_dimensions<1>(dhandle, "vals")[0];

            auto ihandle = open_and_check_dataset<true>(file_handle, index_name);
            if (get_array_dimensions<1>(ihandle, "idx")[0] != nonzeros) {
                throw std::runtime_error("number of non-zero elements is not consistent between 'data' and 'idx'");
            }

            auto phandle = open_and_check_dataset<true>(file_handle, ptr);
            size_t ptr_size = get_array_dimensions<1>(phandle, "ptr")[0];
            size_t dim_p1 = static_cast<size_t>(row_ ? nrows : ncols) + 1;
            if (ptr_size != dim_p1) {
                throw std::runtime_error("'ptr' dataset should have length equal to the number of " + (row_ ? std::string("rows") : std::string("columns")) + " plus 1");
            }

            // Checking the contents of the index pointers.
            pointers.resize(dim_p1);
            phandle.read(pointers.data(), H5::PredType::NATIVE_HSIZE);
            if (pointers[0] != 0) {
                throw std::runtime_error("first index pointer should be zero");
            }
            if (pointers.back() != nonzeros) {
                throw std::runtime_error("last index pointer should be equal to the number of non-zero elements");
            }
        });

        max_non_zeros = 0;
        for (size_t i = 1; i < pointers.size(); ++i) {
            Index_ diff = pointers[i] - pointers[i-1];
            if (diff > max_non_zeros) {
                max_non_zeros = diff;
            }
        }

        Index_ secondary_dim = (row_ ? nc : nr);
        Index_ secondary_chunkdim = std::max(10, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(secondary_dim))))); // arbitrary choice, but whatever.
        secondary_chunk_stats = tatami_chunked::ChunkDimensionStats<Index_>(secondary_dim, secondary_chunkdim);
    }

    /**
     * @param nr Number of rows in the matrix.
     * @param nc Number of columns in the matrix.
     * @param file Path to the file.
     * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
     * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
     * If `row_ = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row_ = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row_ = true`) or columns (otherwise).
     * 
     * Unlike its overload, this constructor uses the defaults for `Hdf5Options`.
     */
    Hdf5CompressedSparseMatrix(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr) :
        Hdf5CompressedSparseMatrix(nr, nc, std::move(file), std::move(vals), std::move(idx), std::move(ptr), Hdf5Options()) {}

public:
    Index_ nrow() const {
        return nrows;
    }

    Index_ ncol() const {
        return ncols;
    }

    /**
     * @return `true`.
     */
    bool sparse() const {
        return true;
    }

    double sparse_proportion() const { 
        return 1;
    }

    /**
     * @return `true` if this is in compressed sparse row format.
     */
    bool prefer_rows() const {
        return row_;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(row_);
    }

    bool uses_oracle(bool) const {
        return false; // placeholder for proper support.
    }

    using tatami::Matrix<Value_, Index_>::dense_row;

    using tatami::Matrix<Value_, Index_>::dense_column;

    using tatami::Matrix<Value_, Index_>::sparse_row;

    using tatami::Matrix<Value_, Index_>::sparse_column;

    /**************************************
     ************ Myopic dense ************
     **************************************/
private:
    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, const tatami::Options&) const {
        if (row == row_) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryFullDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                file_name, data_name, index_name, pointers, (row ? ncols : nrows), std::move(oracle), cache_size_limit, max_non_zeros
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryFullDense<oracle_, Value_, Index_, CachedValue_> >(
                file_name, data_name, index_name, pointers, secondary_chunk_stats, (row ? ncols : nrows), std::move(oracle), cache_size_limit, require_minimum_cache
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const tatami::Options&) const {
        if (row == row_) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryBlockDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                file_name, data_name, index_name, pointers, (row ? ncols : nrows), std::move(oracle), block_start, block_length, cache_size_limit, max_non_zeros
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryBlockDense<oracle_, Value_, Index_, CachedValue_> >(
                file_name, data_name, index_name, pointers, secondary_chunk_stats, std::move(oracle), block_start, block_length, cache_size_limit, require_minimum_cache
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options&) const {
        if (row == row_) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryIndexDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                file_name, data_name, index_name, pointers, (row ? ncols : nrows), std::move(oracle), std::move(indices_ptr), cache_size_limit, max_non_zeros
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryIndexDense<oracle_, Value_, Index_, CachedValue_> >(
                file_name, data_name, index_name, pointers, secondary_chunk_stats, std::move(oracle), std::move(indices_ptr), cache_size_limit, require_minimum_cache
            );
        }
    }

public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        return populate_dense<false>(row, false, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_dense<false>(row, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_dense<false>(row, false, std::move(indices_ptr), opt);
    }

    /***************************************
     ************ Myopic sparse ************
     ***************************************/
private:
    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, const tatami::Options& opt) const {
        if (row == row_) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryFullSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                file_name,
                data_name, 
                index_name, 
                pointers, 
                (row ? ncols : nrows), 
                std::move(oracle), 
                cache_size_limit, 
                max_non_zeros, 
                opt.sparse_extract_value, 
                opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryFullSparse<oracle_, Value_, Index_, CachedValue_> >(
                file_name, 
                data_name, 
                index_name, 
                pointers, 
                secondary_chunk_stats, 
                (row ? ncols : nrows), 
                std::move(oracle), 
                cache_size_limit, 
                require_minimum_cache, 
                opt.sparse_extract_value, 
                opt.sparse_extract_index
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        if (row == row_) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryBlockSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                file_name, 
                data_name, 
                index_name, 
                pointers, 
                (row ? ncols : nrows), 
                std::move(oracle), 
                block_start, 
                block_length, 
                cache_size_limit, 
                max_non_zeros, 
                opt.sparse_extract_value, 
                opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryBlockSparse<oracle_, Value_, Index_, CachedValue_> >(
                file_name, 
                data_name, 
                index_name, 
                pointers, 
                secondary_chunk_stats, 
                std::move(oracle), 
                block_start, 
                block_length, 
                cache_size_limit, 
                require_minimum_cache, 
                opt.sparse_extract_value, 
                opt.sparse_extract_index
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        if (row == row_) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryIndexSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                file_name, 
                data_name, 
                index_name, 
                pointers, 
                (row ? ncols : nrows), 
                std::move(oracle), 
                std::move(indices_ptr), 
                cache_size_limit, 
                max_non_zeros,
                opt.sparse_extract_value, 
                opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryIndexSparse<oracle_, Value_, Index_, CachedValue_> >(
                file_name, 
                data_name, 
                index_name, 
                pointers, 
                secondary_chunk_stats, 
                std::move(oracle), 
                std::move(indices_ptr), 
                cache_size_limit, 
                require_minimum_cache,
                opt.sparse_extract_value, 
                opt.sparse_extract_index
            );
        }
    }

public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        return populate_sparse<false>(row, false, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_sparse<false>(row, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_sparse<false>(row, false, std::move(indices_ptr), opt);
    }

    /****************************************
     ************ Oracular dense ************
     ****************************************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, const tatami::Options& opt) const {
        return populate_dense<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_dense<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_dense<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }

    /*****************************************
     ************ Oracular sparse ************
     *****************************************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, const tatami::Options& opt) const {
        return populate_sparse<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_sparse<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_sparse<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }
};

}

#endif
