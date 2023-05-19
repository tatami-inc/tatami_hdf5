#ifndef TATAMI_LOAD_HDF5_MATRIX_HPP
#define TATAMI_LOAD_HDF5_MATRIX_HPP

#include "H5Cpp.h"

#include <string>
#include <cstdint>
#include <type_traits>
#include <cmath>

#include "tatami/tatami.hpp"
#include "utils.hpp"

/**
 * @file load_hdf5_matrix.hpp
 *
 * @brief Load HDF5 data into `Matrix` objects.
 */

namespace tatami_hdf5 {

/**
 * Load a `tatami::CompressedSparseMatrix` from a HDF5 file.
 *
 * @tparam row_ Whether the matrix is stored in compressed sparse row format.
 * @tparam Value_ Type of the matrix values in the `Matrix` interface.
 * @tparam Index_ Type of the row/column indices.
 * @tparam ValueStorage_ Vector type for storing the values of the non-zero elements.
 * Elements of this vector may be of a different type than `T` for more efficient storage.
 * @tparam IndexStorage_ Vector type for storing the indices.
 * Elements of this vector may be of a different type than `IDX` for more efficient storage.
 * @tparam PointerStorage_ Vector type for storing the index pointers.
 *
 * @param nr Number of rows in the matrix.
 * @param nc Number of columns in the matrix.
 * @param file Path to the file.
 * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
 * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
 * If `row_ = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
 * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row_ = true`) or column (otherwise).
 * This should have length equal to the number of rows (if `row_ = true`) or columns (otherwise).
 *
 * @return A `CompressedSparseMatrix` containing all values and indices in memory.
 * This differs from a `Hdf5CompressedSparseMatrix`, where the loading of data is deferred until requested.
 */
template<bool row_, typename Value_, typename Index_ = int, class ValueStorage_ = std::vector<Value_>, class IndexStorage_ = std::vector<Index_>, class PointerStorage_ = std::vector<size_t> >
tatami::CompressedSparseMatrix<row_, Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_> load_hdf5_compressed_sparse_matrix(
    size_t nr, 
    size_t nc, 
    const std::string& file, 
    const std::string& vals, 
    const std::string& idx, 
    const std::string& ptr) 
{
    H5::H5File file_handle(file, H5F_ACC_RDONLY);

    auto dhandle = open_and_check_dataset<false>(file_handle, vals);
    const size_t nonzeros = get_array_dimensions<1>(dhandle, "vals")[0];

    ValueStorage_ x(nonzeros);
    dhandle.read(x.data(), define_mem_type<Stored<ValueStorage_> >());
    
    auto ihandle = open_and_check_dataset<true>(file_handle, idx);
    if (get_array_dimensions<1>(ihandle, "idx")[0] != nonzeros) {
        throw std::runtime_error("number of non-zero elements is not consistent between 'data' and 'idx'");
    }
    IndexStorage_ i(nonzeros);
    ihandle.read(i.data(), define_mem_type<Stored<IndexStorage_> >());

    auto phandle = open_and_check_dataset<true>(file_handle, ptr);
    const size_t ptr_size = get_array_dimensions<1>(phandle, "ptr")[0];
    if (ptr_size != (row_ ? nr : nc) + 1) {
        throw std::runtime_error("'ptr' dataset should have length equal to the number of " + (row_ ? std::string("rows") : std::string("columns")) + " plus 1");
    }

    // Because HDF5 doesn't have a native type for size_t.
    PointerStorage_ p(ptr_size);
    if constexpr(std::is_same<size_t, Stored<PointerStorage_> >::value) {
        if constexpr(std::is_same<size_t, hsize_t>::value) {
            phandle.read(p.data(), H5::PredType::NATIVE_HSIZE);
        } else {
            std::vector<hsize_t> p0(ptr_size);
            phandle.read(p0.data(), H5::PredType::NATIVE_HSIZE);
            std::copy(p0.begin(), p0.end(), p.begin());
        }
    } else {
        phandle.read(p.data(), define_mem_type<Stored<PointerStorage_> >());
    }

    return tatami::CompressedSparseMatrix<row_, Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_>(nr, nc, std::move(x), std::move(i), std::move(p));
}

/**
 * Load a `tatami::DenseMatrix` from a HDF5 DataSet.
 *
 * @tparam Value_ Type of the matrix values in the `Matrix` interface.
 * @tparam Index_Type of the row/column indices.
 * @tparam ValueStorage_ Vector type for storing the matrix values.
 * This may be different from `T` for more efficient storage.
 * @tparam transpose_ Whether the dataset is transposed in its storage order, i.e., rows in HDF5 are columns in the matrix.
 *
 * @param file Path to the HDF5 file.
 * @param name Name of the dataset inside the file.
 * This should refer to a 2-dimensional dataset of integer or floating-point type.
 *
 * @return A `DenseMatrix` where all values are in memory.
 * This differs from a `Hdf5DenseMatrix`, where the loading of data is deferred until requested.
 */
template<typename Value_, typename Index_ = int, class ValueStorage_ = std::vector<Value_>, bool transpose_ = false>
tatami::DenseMatrix<!transpose_, Value_, Index_, ValueStorage_> load_hdf5_dense_matrix(const std::string& file, const std::string& name) {
    H5::H5File fhandle(file, H5F_ACC_RDONLY);
    auto dhandle = open_and_check_dataset<false>(fhandle, name);

    auto dims = get_array_dimensions<2>(dhandle, name);
    ValueStorage_ values(dims[0] * dims[1]);
    dhandle.read(values.data(), define_mem_type<Stored<ValueStorage_> >());

    if constexpr(transpose_) {
        return tatami::DenseMatrix<false, Value_, Index_, ValueStorage_>(dims[1], dims[0], std::move(values));
    } else {
        return tatami::DenseMatrix<true, Value_, Index_, ValueStorage_>(dims[0], dims[1], std::move(values));
    }
}

}

#endif
