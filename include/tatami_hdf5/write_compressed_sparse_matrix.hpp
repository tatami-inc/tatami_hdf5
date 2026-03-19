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
#include <optional>

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
     * Name of the dataset in which to store the data values for non-zero elements.
     * If unset, defaults to `"data"`.
     */
    std::optional<std::string> data_name;

    /**
     * Name of the dataset in which to store the indices for non-zero elements.
     * If unset, defaults to `"indices"`.
     */
    std::optional<std::string> index_name;

    /**
     * Name of the dataset in which to store the column/row pointers.
     * If unset, defaults to `"indptr"`.
     */
    std::optional<std::string> ptr_name;

    /**
     * Whether to save in the compressed sparse column layout.
     * If unset, this is determined from the layout of the input matrix.
     */
    std::optional<WriteStorageLayout> columnar;

    /**
     * Storage type for the matrix data.
     * If unset, it is automatically determined from the range and integralness of the data in the input matrix.
     */
    std::optional<WriteStorageType> data_type;

    /**
     * Whether to force non-integer floating point values into an integer storage mode.
     * Only relevant if `data_type` is unset. 
     * If `true` and/or all values are integers, the smallest integer storage mode that fits the (truncated) floats is used.
     * If `false` and any non-integer values are detected, the `DOUBLE` storage mode is used instead.
     */
    bool force_integer = false;

    /**
     * Storage type for the row/column indices.
     * If unset, it is automatically determined from the range of the indices in the input matrix.
     */
    std::optional<WriteStorageType> index_type;

    /**
     * Storage type for the index pointers.
     * If unset, it is automatically determined from the number of structural non-zero elements.
     */
    std::optional<WriteStorageType> ptr_type;

    /**
     * Compression level of DEFLATE.
     * This should be any integer between 0 and 9 inclusive.
     * At zero, no compression is performed.
     */
    int deflate_level = 6;

    /**
     * Size of the chunks used for compression.
     */
    hsize_t chunk_size = sanisizer::cap<hsize_t>(10000);

    /**
     * Number of threads to use for the first pass through the input matrix.
     * This is only used to determine the number of non-zero elements and check storage types.
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
        case WriteStorageType::INT64:
            dtype = &(H5::PredType::NATIVE_INT64);
            break;
        case WriteStorageType::UINT64:
            dtype = &(H5::PredType::NATIVE_UINT64);
            break;
        case WriteStorageType::FLOAT:
            dtype = &(H5::PredType::NATIVE_FLOAT);
            break;
        case WriteStorageType::DOUBLE:
            dtype = &(H5::PredType::NATIVE_DOUBLE);
            break;
        default:
            throw std::runtime_error("automatic HDF5 output type must be resolved before creating a HDF5 dataset");
    }
    return create_1d_compressed_hdf5_dataset(location, *dtype, name, length, deflate_level, chunk);
}

template<typename Left_, typename Right_>
bool is_less_than_or_equal(Left_ l, Right_ r) {
    constexpr bool lsigned = std::is_signed<Left_>::value;
    constexpr bool rsigned = std::is_signed<Right_>::value;
    if constexpr(lsigned == rsigned) {
        return l <= r;
    } else if constexpr(lsigned) {
        return l <= 0 || static_cast<typename std::make_unsigned<Left_>::type>(l) <= r;
    } else {
        return r >= 0 && l <= static_cast<typename std::make_unsigned<Right_>::type>(r);
    }
}

template<typename Native_, typename Max_>
bool fits_upper_limit(Max_ max) {
    constexpr auto native_max = std::numeric_limits<Native_>::max();
    if constexpr(std::is_integral<Max_>::value) { // Native_ is already integral, so no need to check that.
        return is_less_than_or_equal(max, native_max);
    } else {
        return max <= static_cast<double>(native_max);
    }
}

template<typename Native_, typename Min_>
bool fits_lower_limit(Min_ min) {
    constexpr auto native_min = std::numeric_limits<Native_>::min();
    if constexpr(std::is_integral<Min_>::value) {
        return is_less_than_or_equal(native_min, min);
    } else {
        return min >= static_cast<double>(native_min);
    }
}

template<typename Value_>
WriteStorageType choose_data_type(
    const std::optional<WriteStorageType>& data_type,
    Value_ lower_data,
    Value_ upper_data,
    bool has_decimal,
    bool force_integer,
    bool has_nonfinite
) {
    const bool is_lower_negative = [&](){
        if constexpr(std::is_integral<Value_>::value && std::is_unsigned<Value_>::value) {
            return false;
        } else {
            return lower_data < 0;
        }
    }();

    if (!data_type.has_value()) {
        if ((has_decimal && !force_integer) || has_nonfinite) {
            if constexpr(std::is_same<Value_, float>::value) {
                return WriteStorageType::FLOAT;
            } else {
                return WriteStorageType::DOUBLE;
            }
        }

        if (is_lower_negative) {
            if (fits_lower_limit<std::int8_t>(lower_data) && fits_upper_limit<std::int8_t>(upper_data)) {
                return WriteStorageType::INT8;
            } else if (fits_lower_limit<std::int16_t>(lower_data) && fits_upper_limit<std::int16_t>(upper_data)) {
                return WriteStorageType::INT16;
            } else if (fits_lower_limit<std::int32_t>(lower_data) && fits_upper_limit<std::int32_t>(upper_data)) {
                return WriteStorageType::INT32;
            } else if (fits_lower_limit<std::int64_t>(lower_data) && fits_upper_limit<std::int64_t>(upper_data)) {
                return WriteStorageType::INT64;
            }

        } else {
            if (fits_upper_limit<std::uint8_t>(upper_data)) {
                return WriteStorageType::UINT8;
            } else if (fits_upper_limit<std::uint16_t>(upper_data)) {
                return WriteStorageType::UINT16;
            } else if (fits_upper_limit<std::uint32_t>(upper_data)) {
                return WriteStorageType::UINT32;
            } else if (fits_upper_limit<std::uint64_t>(upper_data)) {
                return WriteStorageType::UINT64;
            }
        }

        throw std::runtime_error("no storage type can store the matrix values");
    }

    const auto dtype = *data_type;
    if ((has_decimal && !force_integer) || has_nonfinite) {
        if (dtype != WriteStorageType::DOUBLE && dtype != WriteStorageType::FLOAT) {
            if (has_nonfinite) {
                throw std::runtime_error("cannot store non-finite floating-point values as integers");
            } else {
                throw std::runtime_error("cannot store floating-point values as integers without 'force_integer = true'");
            }
        }

    } else {
        if (
            (dtype == WriteStorageType::INT8   && !(fits_lower_limit<std::int8_t >(lower_data) && fits_upper_limit<std::int8_t  >(upper_data))) ||
            (dtype == WriteStorageType::UINT8  && !(!is_lower_negative                         && fits_upper_limit<std::uint8_t >(upper_data))) ||
            (dtype == WriteStorageType::INT16  && !(fits_lower_limit<std::int16_t>(lower_data) && fits_upper_limit<std::int16_t >(upper_data))) ||
            (dtype == WriteStorageType::UINT16 && !(!is_lower_negative                         && fits_upper_limit<std::uint16_t>(upper_data))) ||
            (dtype == WriteStorageType::INT32  && !(fits_lower_limit<std::int32_t>(lower_data) && fits_upper_limit<std::int32_t >(upper_data))) ||
            (dtype == WriteStorageType::UINT32 && !(!is_lower_negative                         && fits_upper_limit<std::uint32_t>(upper_data))) ||
            (dtype == WriteStorageType::INT64  && !(fits_lower_limit<std::int64_t>(lower_data) && fits_upper_limit<std::int64_t >(upper_data))) ||
            (dtype == WriteStorageType::UINT64 && !(!is_lower_negative                         && fits_upper_limit<std::uint64_t>(upper_data)))
        ) {
            throw std::runtime_error("specified integer type cannot store all matrix values");
        }
    }

    return dtype;
}

template<typename Index_>
WriteStorageType choose_index_type(const std::optional<WriteStorageType>& index_type, Index_ upper_index) {
    if (!index_type.has_value()) {
        if (fits_upper_limit<std::uint8_t>(upper_index)) {
            return WriteStorageType::UINT8;
        } else if (fits_upper_limit<std::uint16_t>(upper_index)) {
            return WriteStorageType::UINT16;
        } else if (fits_upper_limit<std::uint32_t>(upper_index)) {
            return WriteStorageType::UINT32;
        } else if (fits_upper_limit<std::uint64_t>(upper_index)) {
            return WriteStorageType::UINT64;
        }
        throw std::runtime_error("no type can store the largest index");
    }

    const auto itype = *index_type;
    if (
        (itype == WriteStorageType::INT8   && !fits_upper_limit<std::int8_t  >(upper_index)) ||
        (itype == WriteStorageType::UINT8  && !fits_upper_limit<std::uint8_t >(upper_index)) ||
        (itype == WriteStorageType::INT16  && !fits_upper_limit<std::int16_t >(upper_index)) ||
        (itype == WriteStorageType::UINT16 && !fits_upper_limit<std::uint16_t>(upper_index)) ||
        (itype == WriteStorageType::INT32  && !fits_upper_limit<std::int32_t >(upper_index)) ||
        (itype == WriteStorageType::UINT32 && !fits_upper_limit<std::uint32_t>(upper_index)) ||
        (itype == WriteStorageType::INT64  && !fits_upper_limit<std::int64_t >(upper_index)) ||
        (itype == WriteStorageType::UINT64 && !fits_upper_limit<std::uint64_t>(upper_index)) ||
        (itype == WriteStorageType::DOUBLE /* must be integer */                           ) ||
        (itype == WriteStorageType::FLOAT  /* must be integer */                           )
    ) {
        throw std::runtime_error("specified type cannot store the largest index");
    }

    return itype;
}

inline WriteStorageType choose_ptr_type(const std::optional<WriteStorageType>& ptr_type, hsize_t nnzero) {
    if (!ptr_type.has_value()) {
        if (fits_upper_limit<std::uint32_t>(nnzero)) {
            return WriteStorageType::UINT32;
        } else if (fits_upper_limit<std::uint64_t>(nnzero)) {
            return WriteStorageType::UINT64;
        }

        throw std::runtime_error("no type can store the number of non-zero elements");
    }

    const auto ptype = *ptr_type;
    if (
        (ptype == WriteStorageType::INT8   && !fits_upper_limit<std::int8_t  >(nnzero)) ||
        (ptype == WriteStorageType::UINT8  && !fits_upper_limit<std::uint8_t >(nnzero)) ||
        (ptype == WriteStorageType::INT16  && !fits_upper_limit<std::int16_t >(nnzero)) ||
        (ptype == WriteStorageType::UINT16 && !fits_upper_limit<std::uint16_t>(nnzero)) ||
        (ptype == WriteStorageType::INT32  && !fits_upper_limit<std::int32_t >(nnzero)) ||
        (ptype == WriteStorageType::UINT32 && !fits_upper_limit<std::uint32_t>(nnzero)) ||
        (ptype == WriteStorageType::INT64  && !fits_upper_limit<std::int64_t >(nnzero)) ||
        (ptype == WriteStorageType::UINT64 && !fits_upper_limit<std::uint64_t>(nnzero)) ||
        (ptype == WriteStorageType::DOUBLE /* must be integer */                      ) ||
        (ptype == WriteStorageType::FLOAT  /* must be integer */                      )
    ) {
        throw std::runtime_error("specified type cannot store the number of non-zero elements");
    }

    return ptype;
}

template<typename Value_, typename Index_>
struct WriteSparseHdf5Statistics {
    Value_ lower_data = 0;
    Value_ upper_data = 0;
    Index_ upper_index = 0;
    hsize_t non_zeros = 0;
    bool has_decimal = false;
    bool has_nonfinite = false;

    void add_value(Value_ val) {
        if constexpr(!std::is_integral<Value_>::value) {
            if (std::trunc(val) != val) {
                has_decimal = true;
            }
            if (!std::isfinite(val)) {
                has_nonfinite = true;
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
void update_hdf5_stats(const tatami::SparseRange<Value_, Index_>& extracted, WriteSparseHdf5Statistics<Value_, Index_>& output) {
    // We need to protect the addition just in case it overflows from having too many non-zero elements.
    output.non_zeros = sanisizer::sum<hsize_t>(output.non_zeros, extracted.number);

    for (Index_ i = 0; i < extracted.number; ++i) {
        output.add_value(extracted.value[i]);
    }

    for (Index_ i = 0; i < extracted.number; ++i) {
        output.add_index(extracted.index[i]);
    }
}

template<typename Value_, typename Index_>
void update_hdf5_stats(const Value_* extracted, Index_ n, WriteSparseHdf5Statistics<Value_, Index_>& output) {
    Index_ local_nonzero = 0;
    for (Index_ i = 0; i < n; ++i) {
        auto val = extracted[i];
        if (val == 0) {
            continue;
        }
        ++local_nonzero;
        output.add_value(val);
        output.add_index(i);
    }

    // Checking that there aren't overflows, but doing so outside of the hot loop for perf.
    output.non_zeros = sanisizer::sum<hsize_t>(output.non_zeros, local_nonzero);
}

template<typename Value_, typename Index_>
WriteSparseHdf5Statistics<Value_, Index_> write_sparse_hdf5_statistics(const tatami::Matrix<Value_, Index_>& mat, int nthreads) {
    const auto NR = mat.nrow(), NC = mat.ncol();

    WriteSparseHdf5Statistics<Value_, Index_> output;
    auto collected = sanisizer::create<std::vector<WriteSparseHdf5Statistics<Value_, Index_> > >(nthreads - 1); // nthreads had better be >= 1.
    int num_used;

    if (mat.sparse()) {
        if (mat.prefer_rows()) {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<true>(mat, true, start, len);
                std::vector<Value_> xbuffer(NC);
                std::vector<Index_> ibuffer(NC);
                for (Index_ r = start, end = start + len; r < end; ++r) {
                    auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                    update_hdf5_stats(extracted, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NR, nthreads);

        } else {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<true>(mat, false, start, len);
                std::vector<Value_> xbuffer(NR);
                std::vector<Index_> ibuffer(NR);
                for (Index_ c = start, end = start + len; c < end; ++c) {
                    auto extracted = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                    update_hdf5_stats(extracted, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NC, nthreads);
        }

    } else {
        if (mat.prefer_rows()) {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<false>(mat, true, start, len);
                std::vector<Value_> xbuffer(NC);
                for (Index_ r = start, end = start + len; r < end; ++r) {
                    auto extracted = wrk->fetch(r, xbuffer.data());
                    update_hdf5_stats(extracted, NC, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NR, nthreads);

        } else {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<false>(mat, false, start, len);
                std::vector<Value_> xbuffer(NR);
                for (Index_ c = start, end = start + len; c < end; ++c) {
                    auto extracted = wrk->fetch(c, xbuffer.data());
                    update_hdf5_stats(extracted, NR, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NC, nthreads);
        }
    }

    for (int i = 1; i < num_used; ++i) {
        auto& current = collected[i - 1];
        output.lower_data = std::min(output.lower_data, current.lower_data);
        output.upper_data = std::max(output.upper_data, current.upper_data);
        output.upper_index = std::max(output.upper_index, current.upper_index);
        output.non_zeros = sanisizer::sum<hsize_t>(output.non_zeros, current.non_zeros);
        output.has_decimal = output.has_decimal || current.has_decimal;
        output.has_nonfinite = output.has_nonfinite || current.has_nonfinite;
    }

    return output;
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
 * @param mat Matrix to be written to disk, presumably sparse.
 * If a dense matrix is supplied, only the non-zero elements will be written.
 * @param location Handle to a HDF5 group in which to write the matrix contents.
 * @param params Parameters to use when writing the matrix.
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>& mat, H5::Group& location, const WriteCompressedSparseMatrixOptions& params) {
    auto stats = write_sparse_hdf5_statistics(mat, params.num_threads);
    const auto data_type = choose_data_type(params.data_type, stats.lower_data, stats.upper_data, stats.has_decimal, params.force_integer, stats.has_nonfinite);
    const auto index_type = choose_index_type(params.index_type, stats.upper_index);

    // Choosing the layout.
    WriteStorageLayout layout;
    if (params.columnar.has_value()) {
        layout = *(params.columnar);
    } else {
        if (mat.prefer_rows()) {
            layout = WriteStorageLayout::ROW;
        } else {
            layout = WriteStorageLayout::COLUMN;
        }
    }

    // Choosing the names.
    std::string data_name;
    if (params.data_name.has_value()) {
        data_name = *(params.data_name);
    } else {
        data_name = "data";
    }

    std::string index_name;
    if (params.index_name.has_value()) {
        index_name = *(params.index_name);
    } else {
        index_name = "indices";
    }

    std::string ptr_name;
    if (params.ptr_name.has_value()) {
        ptr_name = *(params.ptr_name);
    } else {
        ptr_name = "indptr";
    }

    // And then saving it. This time we have no choice but to iterate by the desired dimension.
    auto non_zeros = stats.non_zeros;
    H5::DataSet data_ds = create_1d_compressed_hdf5_dataset(location, data_type, data_name, non_zeros, params.deflate_level, params.chunk_size);
    H5::DataSet index_ds = create_1d_compressed_hdf5_dataset(location, index_type, index_name, non_zeros, params.deflate_level, params.chunk_size);
    hsize_t offset = 0;
    H5::DataSpace inspace(1, &non_zeros);
    H5::DataSpace outspace(1, &non_zeros);
    const auto& dstype = define_mem_type<Value_>();
    const auto& ixtype = define_mem_type<Index_>();

    Index_ NR = mat.nrow(), NC = mat.ncol();
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

    if (mat.sparse()) {
        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NR, 1));
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            auto wrk = tatami::consecutive_extractor<true>(mat, true, static_cast<Index_>(0), NR);
            for (Index_ r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                fill_datasets(extracted.value, extracted.index, extracted.number);
                ptrs[r+1] = ptrs[r] + extracted.number;
            }

        } else {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NC, 1));
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

            auto wrk = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), NC);
            for (Index_ c = 0; c < NC; ++c) {
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
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NR, 1));
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto wrk = tatami::consecutive_extractor<false>(mat, true, static_cast<Index_>(0), NR);
            for (Index_ r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, dbuffer.data());
                auto count = fill_datasets_from_dense(extracted, NC);
                ptrs[r+1] = ptrs[r] + count;
            }

        } else {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NC, 1));
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            auto wrk = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), NC);
            for (Index_ c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, dbuffer.data());
                auto count = fill_datasets_from_dense(extracted, NR);
                ptrs[c+1] = ptrs[c] + count;
            }
        }
    }

    // Saving the pointers.
    auto ptr_len = sanisizer::cast<hsize_t>(ptrs.size());
    H5::DataSet ptr_ds = create_1d_compressed_hdf5_dataset(
        location,
        choose_ptr_type(params.ptr_type, ptrs.back()),
        ptr_name,
        ptr_len,
        params.deflate_level,
        params.chunk_size
    );
    H5::DataSpace ptr_space(1, &ptr_len);
    ptr_ds.write(ptrs.data(), H5::PredType::NATIVE_HSIZE, ptr_space);

    return;
}

/**
 * Overload of `write_compressed_sparse_matrix()` with default parameters.
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 *
 * @param mat Matrix to be written to disk, presumably sparse.
 * @param location Handle to a HDF5 group in which to write the matrix contents.
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>& mat, H5::Group& location) {
    WriteCompressedSparseMatrixOptions params;
    write_compressed_sparse_matrix(mat, location, params);
    return;
}

/**
 * @cond
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>* mat, H5::Group& location, const WriteCompressedSparseMatrixOptions& params) {
    return write_compressed_sparse_matrix(*mat, location, params);
}

template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>* mat, H5::Group& location) {
    return write_compressed_sparse_matrix(*mat, location);
}
/**
 * @endcond
 */

}

#endif
