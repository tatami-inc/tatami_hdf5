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
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all values (based on their HDF5 type or other knowledge).
 * @tparam CachedIndex_ Type of the index value to store in the cache.
 * This can be set to a narrower type than `Index_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all indices (based on their HDF5 type or other knowledge).
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class Hdf5CompressedSparseMatrix : public tatami::Matrix<Value_, Index_> {
    Index_ nrows, ncols;
    std::string file_name;
    std::string data_name, index_name;
    std::vector<hsize_t> pointers;
    bool csr;

    // We distinguish between our own cache of dimension elements
    // versus HDF5's cache of uncompressed chunks.
    size_t our_cache_size;
    size_t max_non_zeros;
    size_t h5_chunk_cache_size;

public:
    /**
     * @param nr Number of rows in the matrix.
     * @param nc Number of columns in the matrix.
     * @param file Path to the file.
     * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
     * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
     * If `row = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row = true`) or columns (otherwise).
     * @param row Whether the matrix is stored in compressed sparse row format.
     * @param options Further options.
     */
    Hdf5CompressedSparseMatrix(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr, bool row, const Hdf5Options& options) :
        nrows(nr), 
        ncols(nc), 
        file_name(std::move(file)), 
        data_name(std::move(vals)), 
        index_name(std::move(idx)), 
        csr(row),
        our_cache_size(options.maximum_cache_size)
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
            size_t dim_p1 = static_cast<size_t>(csr ? nrows : ncols) + 1;
            if (ptr_size != dim_p1) {
                throw std::runtime_error("'ptr' dataset should have length equal to the number of " + (csr ? std::string("rows") : std::string("columns")) + " plus 1");
            }

            // We aim to store two chunks in HDF5's chunk cache; one
            // overlapping the start of the primary dimension element's range,
            // and one overlapping the end, so that we don't re-read the
            // content for the new primary dimension element. To simplify
            // matters, we just read the chunk sizes (in bytes) for both
            // datasets and use the larger chunk size for both datasets.
            // Hopefully the chunks are not too big...
            hsize_t dchunk_length = 0;
            size_t dchunk_element_size = 0;
            auto dparms = dhandle.getCreatePlist();
            if (dparms.getLayout() == H5D_CHUNKED) {
                dparms.getChunk(1, &dchunk_length);
                dchunk_element_size = dhandle.getDataType().getSize();
            }

            hsize_t ichunk_length = 0;
            size_t ichunk_element_size = 0;
            auto iparms = ihandle.getCreatePlist();
            if (iparms.getLayout() == H5D_CHUNKED) {
                iparms.getChunk(1, &ichunk_length);
                ichunk_element_size = ihandle.getDataType().getSize();
            }

            auto non_overflow_double_min = [nonzeros](hsize_t chunk_length) -> size_t {
                // Basically computes std::min(chunk_length * 2, nonzeros) without
                // overflowing hsize_t, for a potentially silly choice of hsize_t...
                if (chunk_length < nonzeros) {
                    return nonzeros;
                } else {
                    return chunk_length + std::min(chunk_length, nonzeros - chunk_length);
                }
            };

            h5_chunk_cache_size = std::max(
                non_overflow_double_min(ichunk_length) * ichunk_element_size, 
                non_overflow_double_min(dchunk_length) * dchunk_element_size
            );

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
            hsize_t diff = pointers[i] - pointers[i-1];
            if (diff > max_non_zeros) {
                max_non_zeros = diff;
            }
        }
    }

    /**
     * Overload that uses the defaults for `Hdf5Options`.
     * @param nr Number of rows in the matrix.
     * @param nc Number of columns in the matrix.
     * @param file Path to the file.
     * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
     * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
     * If `row = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row = true`) or columns (otherwise).
     * @param row Whether the matrix is stored in compressed sparse row format.
     */
    Hdf5CompressedSparseMatrix(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr, bool row) :
        Hdf5CompressedSparseMatrix(nr, nc, std::move(file), std::move(vals), std::move(idx), std::move(ptr), row, Hdf5Options()) {}

public:
    Index_ nrow() const {
        return nrows;
    }

    Index_ ncol() const {
        return ncols;
    }

    bool is_sparse() const {
        return true;
    }

    double is_sparse_proportion() const { 
        return 1;
    }

    bool prefer_rows() const {
        return csr;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(csr);
    }

    bool uses_oracle(bool) const {
        return true;
    }

    using tatami::Matrix<Value_, Index_>::dense_row;

    using tatami::Matrix<Value_, Index_>::dense_column;

    using tatami::Matrix<Value_, Index_>::sparse_row;

    using tatami::Matrix<Value_, Index_>::sparse_column;

    /**************************************
     ************ Myopic dense ************
     **************************************/
private:
    Hdf5CompressedSparseMatrix_internal::MatrixDetails<Index_> details() const {
        return Hdf5CompressedSparseMatrix_internal::MatrixDetails<Index_>(
            file_name, 
            data_name, 
            index_name, 
            (csr ? nrows : ncols),
            (csr ? ncols : nrows),
            pointers, 
            our_cache_size,
            max_non_zeros,
            h5_chunk_cache_size
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, const tatami::Options&) const {
        if (row == csr) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryFullDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle)
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryFullDense<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle)
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const tatami::Options&) const {
        if (row == csr) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryBlockDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), block_start, block_length
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryBlockDense<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), block_start, block_length
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options&) const {
        if (row == csr) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryIndexDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), std::move(indices_ptr)
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryIndexDense<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), std::move(indices_ptr)
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
        if (row == csr) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryFullSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), opt.sparse_extract_value, opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryFullSparse<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), opt.sparse_extract_value, opt.sparse_extract_index
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        if (row == csr) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryBlockSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), block_start, block_length, opt.sparse_extract_value, opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryBlockSparse<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), block_start, block_length, opt.sparse_extract_value, opt.sparse_extract_index
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        if (row == csr) {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::PrimaryIndexSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), std::move(indices_ptr), opt.sparse_extract_value, opt.sparse_extract_index
            );
        } else {
            return std::make_unique<Hdf5CompressedSparseMatrix_internal::SecondaryIndexSparse<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), std::move(indices_ptr), opt.sparse_extract_value, opt.sparse_extract_index
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
