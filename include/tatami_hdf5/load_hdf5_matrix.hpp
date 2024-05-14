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
 * If `row = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
 * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row = true`) or column (otherwise).
 * This should have length equal to the number of rows (if `row = true`) or columns (otherwise) plus 1.
 * @param row Whether the matrix is stored in compressed sparse row format.
 *
 * @return A `tatami::CompressedSparseMatrix` containing all values and indices in memory.
 * This differs from a `Hdf5CompressedSparseMatrix`, where the loading of data is deferred until requested.
 */
template<typename Value_, typename Index_, class ValueStorage_ = std::vector<Value_>, class IndexStorage_ = std::vector<Index_>, class PointerStorage_ = std::vector<size_t> >
tatami::CompressedSparseMatrix<Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_> load_hdf5_compressed_sparse_matrix(
    size_t nr, 
    size_t nc, 
    const std::string& file, 
    const std::string& vals, 
    const std::string& idx, 
    const std::string& ptr,
    bool row) 
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
    if (ptr_size != (row ? nr : nc) + 1) {
        throw std::runtime_error("'ptr' dataset should have length equal to the number of " + (row ? std::string("rows") : std::string("columns")) + " plus 1");
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

    return tatami::CompressedSparseMatrix<Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_>(nr, nc, std::move(x), std::move(i), std::move(p), row);
}

/**
 * Load a `tatami::DenseMatrix` from a HDF5 DataSet.
 *
 * @tparam Value_ Type of the matrix values in the `Matrix` interface.
 * @tparam Index_Type of the row/column indices.
 * @tparam ValueStorage_ Vector type for storing the matrix values.
 * This may be different from `T` for more efficient storage.
 *
 * @param file Path to the HDF5 file.
 * @param name Name of the dataset inside the file.
 * This should refer to a 2-dimensional dataset of integer or floating-point type.
 * @param transpose Whether the dataset is transposed in its storage order, i.e., rows in HDF5 are columns in the matrix.
 *
 * @return A `tatami::DenseMatrix` where all values are in memory.
 * This differs from a `Hdf5DenseMatrix`, where the loading of data is deferred until requested.
 */
template<typename Value_, typename Index_, class ValueStorage_ = std::vector<Value_> >
tatami::DenseMatrix<Value_, Index_, ValueStorage_> load_hdf5_dense_matrix(const std::string& file, const std::string& name, bool transpose) {
    H5::H5File fhandle(file, H5F_ACC_RDONLY);
    auto dhandle = open_and_check_dataset<false>(fhandle, name);

    auto dims = get_array_dimensions<2>(dhandle, name);
    ValueStorage_ values(static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1])); // cast just in case hsize_t is something silly...
    dhandle.read(values.data(), define_mem_type<Stored<ValueStorage_> >());

    if (transpose) {
        return tatami::DenseMatrix<Value_, Index_, ValueStorage_>(dims[1], dims[0], std::move(values), false);
    } else {
        return tatami::DenseMatrix<Value_, Index_, ValueStorage_>(dims[0], dims[1], std::move(values), true);
    }
}

/**
 * @cond
 */
// Back-compatibility.
template<bool row_, typename Value_, typename Index_ = int, class ValueStorage_ = std::vector<Value_>, class IndexStorage_ = std::vector<Index_>, class PointerStorage_ = std::vector<size_t> >
tatami::CompressedSparseMatrix<Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_> load_hdf5_compressed_sparse_matrix(
    size_t nr, 
    size_t nc, 
    const std::string& file, 
    const std::string& vals, 
    const std::string& idx, 
    const std::string& ptr)
{
    return load_hdf5_compressed_sparse_matrix<Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_>(nr, nc, file, vals, idx, ptr, row_);
}

template<typename Value_, typename Index_ = int, class ValueStorage_ = std::vector<Value_>, bool transpose_ = false>
tatami::DenseMatrix<Value_, Index_, ValueStorage_> load_hdf5_dense_matrix(const std::string& file, const std::string& name) {
    return load_hdf5_dense_matrix<Value_, Index_, ValueStorage_>(file, name, transpose_);
}
/**
 * @endcond
 */

}

#endif
