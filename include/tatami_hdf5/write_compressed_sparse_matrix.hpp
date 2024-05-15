#ifndef TATAMI_WRITE_SPARSE_MATRIX_TO_HDF5_HPP
#define TATAMI_WRITE_SPARSE_MATRIX_TO_HDF5_HPP

#include "tatami/tatami.hpp"
#include "utils.hpp"

#include "H5Cpp.h"

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

/**
 * @file write_compressed_sparse_matrix.hpp
 * @brief Write a compressed sparse matrix into a HDF5 file.
 */

namespace tatami_hdf5 {

/**
 * @brief Parameters for `write_compressed_sparse_matrix()`.
 */
struct WriteCompressedSparseMatrixOptions {
    /**
     * @cond
     */
    WriteCompressedSparseMatrixOptions() : data_name("data"), index_name("indices"), ptr_name("indptr") {}
    /**
     * @endcond
     */

    /**
     * Name of the dataset in which to store the data values for non-zero elements.
     * Defaults to `"data"`.
     */
    std::string data_name;

    /**
     * Name of the dataset in which to store the indices for non-zero elements.
     * Defaults to `"indices"`.
     */
    std::string index_name;

    /**
     * Name of the dataset in which to store the column/row pointers.
     * Defaults to `"indptr"`.
     */
    std::string ptr_name;

    /**
     * Whether to save in the compressed sparse column layout.
     * If `false`, this is determined from the layout of the input matrix.
     */
    WriteStorageLayout columnar = WriteStorageLayout::AUTOMATIC;

    /**
     * Storage type for the data values.
     * If `AUTOMATIC`, it is automatically determined from the range and integralness of the data in the input matrix.
     */
    WriteStorageType data_type = WriteStorageType::AUTOMATIC;

    /**
     * Whether to force non-integer floating point values into an integer storage mode.
     * Only relevant if `data_type` is set to `AUTOMATIC`.
     * If `true` and/or all values are integers, the smallest integer storage mode that fits the (truncated) floats is used.
     * If `false` and any non-integer values are detected, the `DOUBLE` storage mode is used instead.
     */
    bool force_integer = false;

    /**
     * Storage type for the data values.
     * If `AUTOMATIC`, it is automatically determined from the range of the indices in the input matrix.
     */
    WriteStorageType index_type = WriteStorageType::AUTOMATIC;

    /**
     * Compression level.
     */
    int deflate_level = 6;

    /**
     * Size of the chunks used for compression.
     */
    size_t chunk_size = 100000;

    /**
     * Number of threads to use for the first pass through the input matrix.
     * This is only used to determine the number of non-zero elements 
     * (and infer an appropriate storage type, if an `AUTOMATIC` selection is requested).
     */
    int num_threads = 1;
};

/**
 * @cond
 */
inline H5::DataSet create_1d_compressed_hdf5_dataset(H5::Group& location, const H5::DataType& dtype, const std::string& name, hsize_t length, int deflate_level, hsize_t chunk) {
    H5::DataSpace dspace(1, &length);
 	H5::DSetCreatPropList plist;

    if (deflate_level >= 0 && length) {
        plist.setDeflate(deflate_level);
        if (chunk > length) {
            plist.setChunk(1, &length);
        } else {
            plist.setChunk(1, &chunk);
        }
    }

    return location.createDataSet(name, dtype, dspace, plist);
}

inline H5::DataSet create_1d_compressed_hdf5_dataset(H5::Group& location, WriteStorageType type, const std::string& name, hsize_t length, int deflate_level, hsize_t chunk) {
    const H5::PredType* dtype;
    switch (type) {
        case WriteStorageType::INT8:
            dtype = &(H5::PredType::NATIVE_INT8);
            break;
        case WriteStorageType::UINT8:
            dtype = &(H5::PredType::NATIVE_UINT8);
            break;
        case WriteStorageType::INT16:
            dtype = &(H5::PredType::NATIVE_INT16);
            break;
        case WriteStorageType::UINT16:
            dtype = &(H5::PredType::NATIVE_UINT16);
            break;
        case WriteStorageType::INT32:
            dtype = &(H5::PredType::NATIVE_INT32);
            break;
        case WriteStorageType::UINT32:
            dtype = &(H5::PredType::NATIVE_UINT32);
            break;
        case WriteStorageType::DOUBLE:
            dtype = &(H5::PredType::NATIVE_DOUBLE);
            break;
        default:
            throw std::runtime_error("automatic HDF5 output type must be resolved before creating a HDF5 dataset");
    }
    return create_1d_compressed_hdf5_dataset(location, *dtype, name, length, deflate_level, chunk);
}

template<typename Native>
bool fits_upper_limit(int64_t max) {
    int64_t limit = std::numeric_limits<Native>::max();
    return limit >= max;
}

template<typename Native>
bool fits_lower_limit(int64_t min) {
    int64_t limit = std::numeric_limits<Native>::min();
    return limit <= min;
}
/**
 * @endcond
 */

/**
 * @cond
 */
template<typename Value_, typename Index_>
struct WriteSparseHdf5Statistics {
    Value_ lower_data = 0;
    Value_ upper_data = 0;
    Index_ upper_index = 0;
    hsize_t non_zeros = 0;
    bool non_integer = false;

    void add_value(Value_ val) {
        if constexpr(!std::is_integral<Value_>::value) {
            if (std::trunc(val) != val || !std::isfinite(val)) {
                non_integer = true;
            }
        }

        if (val < lower_data) {
            lower_data = val;
        } else if (val > upper_data) {
            upper_data = val;
        }
    }

    void add_index(Index_ idx) {
        if (idx > upper_index) {
            upper_index = idx;
        }
    }
};

template<typename Value_, typename Index_>
void update_hdf5_stats(const tatami::SparseRange<Value_, Index_>& extracted, WriteSparseHdf5Statistics<Value_, Index_>& output, bool infer_value, bool infer_index) {
    output.non_zeros += extracted.number;

    if (infer_value) {
        for (Index_ i = 0; i < extracted.number; ++i) {
            output.add_value(extracted.value[i]);
        }
    }

    if (infer_index) {
        for (Index_ i = 0; i < extracted.number; ++i) {
            output.add_index(extracted.index[i]);
        }
    }
}

template<typename Value_, typename Index_>
void update_hdf5_stats(const Value_* extracted, Index_ n, WriteSparseHdf5Statistics<Value_, Index_>& output) {
    for (Index_ i = 0; i < n; ++i) {
        auto val = extracted[i];
        if (val == 0) {
            continue;
        }
        ++output.non_zeros;
        output.add_value(val);
        output.add_index(i);
    }
}

template<typename Value_, typename Index_>
WriteSparseHdf5Statistics<Value_, Index_> write_sparse_hdf5_statistics(const tatami::Matrix<Value_, Index_>* mat, bool infer_value, bool infer_index, int nthreads) {
    auto NR = mat->nrow(), NC = mat->ncol();
    std::vector<WriteSparseHdf5Statistics<Value_, Index_> > collected(nthreads);

    if (mat->sparse()) {
        tatami::Options opt;
        opt.sparse_extract_index = infer_index;
        opt.sparse_extract_value = infer_value;

        if (mat->prefer_rows()) {
            tatami::parallelize([&](size_t t, Index_ start, Index_ len) -> void {
                auto wrk = tatami::consecutive_extractor<true>(mat, true, start, len, opt);
                std::vector<Value_> xbuffer(NC);
                std::vector<Index_> ibuffer(NC);
                for (size_t r = start, end = start + len; r < end; ++r) {
                    auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                    update_hdf5_stats(extracted, collected[t], infer_value, infer_index);
                }
            }, NR, nthreads);

        } else {
            tatami::parallelize([&](size_t t, Index_ start, Index_ len) -> void {
                auto wrk = tatami::consecutive_extractor<true>(mat, false, start, len, opt);
                std::vector<Value_> xbuffer(NR);
                std::vector<Index_> ibuffer(NR);
                for (size_t c = start, end = start + len; c < end; ++c) {
                    auto extracted = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                    update_hdf5_stats(extracted, collected[t], infer_value, infer_index);
                }
            }, NC, nthreads);
        }

    } else {
        if (mat->prefer_rows()) {
            tatami::parallelize([&](size_t t, Index_ start, Index_ len) -> void {
                auto wrk = tatami::consecutive_extractor<false>(mat, true, start, len);
                std::vector<Value_> xbuffer(NC);
                for (size_t r = start, end = start + len; r < end; ++r) {
                    auto extracted = wrk->fetch(r, xbuffer.data());
                    update_hdf5_stats(extracted, NC, collected[t]);
                }
            }, NR, nthreads);

        } else {
            tatami::parallelize([&](size_t t, Index_ start, Index_ len) -> void {
                auto wrk = tatami::consecutive_extractor<false>(mat, false, start, len);
                std::vector<Value_> xbuffer(NR);
                for (size_t c = start, end = start + len; c < end; ++c) {
                    auto extracted = wrk->fetch(c, xbuffer.data());
                    update_hdf5_stats(extracted, NR, collected[t]);
                }
            }, NC, nthreads);
        }
    }

    auto& first = collected.front();
    for (int i = 1; i < nthreads; ++i) {
        auto& current = collected[i];
        first.lower_data = std::min(first.lower_data, current.lower_data);
        first.upper_data = std::max(first.upper_data, current.upper_data);
        first.upper_index = std::max(first.upper_index, current.upper_index);
        first.non_zeros += current.non_zeros;
        first.non_integer = first.non_integer || current.non_integer;
    }

    return std::move(first); // better be at least one thread.
}
/**
 * @endcond
 */

/**
 * Write a sparse matrix inside a HDF5 group.
 * On return, `location` will be populated with three datasets containing the matrix contents in a compressed sparse format.
 * Storage of dimensions and other metadata (e.g., related to column versus row layout) is left to the caller. 
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 *
 * @param mat Pointer to the (presumably sparse) matrix to be written.
 * If a dense matrix is supplied, only the non-zero elements will be saved.
 * @param location Handle to a HDF5 group in which to write the matrix contents.
 * @param params Parameters to use when writing the matrix.
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>* mat, H5::Group& location, const WriteCompressedSparseMatrixOptions& params) {
    auto data_type = params.data_type;
    auto index_type = params.index_type;
    auto use_auto_data_type = (data_type == WriteStorageType::AUTOMATIC);
    auto use_auto_index_type = (index_type == WriteStorageType::AUTOMATIC);
    auto stats = write_sparse_hdf5_statistics(mat, use_auto_data_type, use_auto_index_type, params.num_threads);

    // Choosing the types.
    if (use_auto_data_type) {
        if (stats.non_integer && !params.force_integer) {
            data_type = WriteStorageType::DOUBLE;
        } else {
            auto lower_data = stats.lower_data;
            auto upper_data = stats.upper_data;
            if (lower_data < 0) {
                if (fits_lower_limit<int8_t>(lower_data) && fits_upper_limit<int8_t>(upper_data)) {
                    data_type = WriteStorageType::INT8;
                } else if (fits_lower_limit<int16_t>(lower_data) && fits_upper_limit<int16_t>(upper_data)) {
                    data_type = WriteStorageType::INT16;
                } else {
                    data_type = WriteStorageType::INT32;
                }
            } else {
                if (fits_upper_limit<uint8_t>(upper_data)) {
                    data_type = WriteStorageType::UINT8;
                } else if (fits_upper_limit<uint16_t>(upper_data)) {
                    data_type = WriteStorageType::UINT16;
                } else {
                    data_type = WriteStorageType::UINT32;
                }
            }
        }
    }

    if (use_auto_index_type) {
        auto upper_index = stats.upper_index;
        if (fits_upper_limit<uint8_t>(upper_index)) {
            index_type = WriteStorageType::UINT8;
        } else if (fits_upper_limit<uint16_t>(upper_index)) {
            index_type = WriteStorageType::UINT16;
        } else {
            index_type = WriteStorageType::UINT32;
        }
    }

    // Choosing the layout.
    auto layout = params.columnar;
    if (layout == WriteStorageLayout::AUTOMATIC) {
        if (mat->prefer_rows()) {
            layout = WriteStorageLayout::ROW;
        } else {
            layout = WriteStorageLayout::COLUMN;
        }
    }

    // And then saving it. This time we have no choice but to iterate by the desired dimension.
    auto non_zeros = stats.non_zeros;
    H5::DataSet data_ds = create_1d_compressed_hdf5_dataset(location, data_type, params.data_name, non_zeros, params.deflate_level, params.chunk_size);
    H5::DataSet index_ds = create_1d_compressed_hdf5_dataset(location, index_type, params.index_name, non_zeros, params.deflate_level, params.chunk_size);
    hsize_t offset = 0;
    H5::DataSpace inspace(1, &non_zeros);
    H5::DataSpace outspace(1, &non_zeros);
    const auto& dstype = define_mem_type<Value_>();
    const auto& ixtype = define_mem_type<Index_>();

    size_t NR = mat->nrow(), NC = mat->ncol(); // use size_t to avoid overflow on +1.
    std::vector<hsize_t> ptrs;

    auto fill_datasets = [&](const Value_* vptr, const Index_* iptr, hsize_t count) -> void {
        if (count) {
            inspace.setExtentSimple(1, &count);
            outspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
            data_ds.write(vptr, dstype, inspace, outspace);
            index_ds.write(iptr, ixtype, inspace, outspace);
            offset += count;
        }
    };

    if (mat->sparse()) {
        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(NR + 1);
            std::vector<Value_> xbuffer(NC);
            std::vector<Index_> ibuffer(NC);

            auto wrk = tatami::consecutive_extractor<true>(mat, true, static_cast<Index_>(0), static_cast<Index_>(NR));
            for (size_t r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                fill_datasets(extracted.value, extracted.index, extracted.number);
                ptrs[r+1] = ptrs[r] + extracted.number;
            }

        } else {
            ptrs.resize(NC + 1);
            std::vector<Value_> xbuffer(NR);
            std::vector<Index_> ibuffer(NR);

            auto wrk = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), static_cast<Index_>(NC));
            for (size_t c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                fill_datasets(extracted.value, extracted.index, extracted.number);
                ptrs[c+1] = ptrs[c] + extracted.number;
            }
        }

    } else {
        std::vector<Value_> sparse_xbuffer;
        std::vector<Index_> sparse_ibuffer;
        auto fill_datasets_from_dense = [&](const Value_* extracted, Index_ n) -> hsize_t {
            sparse_xbuffer.clear();
            sparse_ibuffer.clear();
            for (Index_ i = 0; i < n; ++i) {
                if (extracted[i]) {
                    sparse_xbuffer.push_back(extracted[i]);
                    sparse_ibuffer.push_back(i);
                }
            }

            hsize_t count = sparse_xbuffer.size();
            fill_datasets(sparse_xbuffer.data(), sparse_ibuffer.data(), count);
            return count;
        };

        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(NR + 1);
            std::vector<Value_> dbuffer(NC);
            auto wrk = tatami::consecutive_extractor<false>(mat, true, static_cast<Index_>(0), static_cast<Index_>(NR));
            for (size_t r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, dbuffer.data());
                auto count = fill_datasets_from_dense(extracted, NC);
                ptrs[r+1] = ptrs[r] + count;
            }

        } else {
            ptrs.resize(NC + 1);
            std::vector<Value_> dbuffer(NR);
            auto wrk = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), static_cast<Index_>(NC));
            for (size_t c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, dbuffer.data());
                auto count = fill_datasets_from_dense(extracted, NR);
                ptrs[c+1] = ptrs[c] + count;
            }
        }
    }

    // Saving the pointers.
    hsize_t ptr_len = ptrs.size();
    H5::DataSet ptr_ds = create_1d_compressed_hdf5_dataset(location, H5::PredType::NATIVE_HSIZE, params.ptr_name, ptr_len, params.deflate_level, params.chunk_size);
    H5::DataSpace ptr_space(1, &ptr_len);
    ptr_ds.write(ptrs.data(), H5::PredType::NATIVE_HSIZE, ptr_space);

    return;
}

/**
 * Write a sparse matrix inside a HDF5 group.
 * On return, `location` will be populated with three datasets containing the matrix contents in a compressed sparse format.
 * Storage of dimensions and other metadata (e.g., related to column versus row layout) is left to the caller. 
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 *
 * @param mat Pointer to the (presumably sparse) matrix to be written.
 * @param location Handle to a HDF5 group in which to write the matrix contents.
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>* mat, H5::Group& location) {
    WriteCompressedSparseMatrixOptions params;
    write_compressed_sparse_matrix(mat, location, params);
    return;
}

}

#endif
