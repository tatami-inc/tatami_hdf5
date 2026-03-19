#ifndef TATAMI_HDF5_UTILS_HPP
#define TATAMI_HDF5_UTILS_HPP

#include "H5Cpp.h"

#include "tatami/tatami.hpp"

#include <cstdint>
#include <array>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <cassert>
#include <cmath>

/**
 * @file utils.hpp
 * @brief Utilities for HDF5 extraction.
 */

namespace tatami_hdf5 {

/**
 * Layout of the matrix inside the HDF5 file.
 */
enum class WriteStorageLayout { COLUMN, ROW };

/**
 * Numeric type of the HDF5 dataset in which to store matrix contents.
 */
enum class WriteStorageType { INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, DOUBLE };

/**
 * @cond
 */
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

template<typename Float_>
int required_bits_for_float_safe(Float_ y) {
    int exp;
    std::frexp(y, &exp); 
    // frexp guarantees that 2^(exp - 1) <= y < 2^exp.
    // So, we return exp - 1, as we want the lower bound.
    return exp;
}

template<typename Float_>
int required_bits_for_float(Float_ y) {
    static_assert(std::is_floating_point<Float_>::value);
    assert(y == std::trunc(y));
    assert(y > 0);

    if constexpr(std::numeric_limits<Float_>::radix == 2) {
        // ilogb returns an 'exp' such that 2^exp <= y < 2^(exp + 1).
        // Note that it doesn't work for zero but we can assume that y > 0 in all calls.
        return std::ilogb(y) + 1;
    } else {
        // Ensure we're covered for weird float types where the radix is not 2.
        // This is pretty unusual so we need to use a macro to force test coverage.
        return required_bits_for_float_safe(y);
    }
}

template<typename Native_, typename Max_>
bool fits_upper_limit(Max_ max) {
    if constexpr(std::is_integral<Max_>::value) { // Native_ is already integral, so no need to check that.
        constexpr auto native_max = std::numeric_limits<Native_>::max();
        return is_less_than_or_equal(max, native_max);
    } else {
        // We don't compare values directly as the Native_-to-float conversion might not be exact;
        // if native_max gets rounded up during the conversion, we might end up with a situation where 'native_max < max <= FLOAT(native_max)'. 
        // This would result in undefined behavior when casting values equal to 'max' to Native_. 
        //
        // So instead, we compare the number of bits in Native_ with that required to store our (truncated) 'max'.
        // We ignore negative or zero values of 'max' as required_bits_for_float() expects positive values.
        // (Non-positive values would always be less than any 'native_max', so we can always return true in such cases.)
        constexpr auto digits = std::numeric_limits<Native_>::digits;
        max = std::trunc(max);
        return (max <= 0 || required_bits_for_float(max) <= digits);
    }
}

template<typename Native_, typename Min_>
bool fits_lower_limit(Min_ min) {
    if constexpr(std::is_integral<Min_>::value) {
        constexpr auto native_min = std::numeric_limits<Native_>::min();
        return is_less_than_or_equal(native_min, min);
    } else {
        // This function should never be called for an unsigned Native_ integer,
        // but we'll just implement some protection for the sake of completeness.
        if constexpr(std::is_unsigned<Native_>::value) {
            if (min < 0) {
                return false; 
            }
        }

        // Pretty much the same logic as fits_upper_limit() but we reverse the sign.
        // We add 1 before reversing to account for the sign bit, i.e., -128 becomes 127 for 7 bits.
        constexpr auto digits = std::numeric_limits<Native_>::digits;
        min = std::trunc(min);
        return (min >= -1 || required_bits_for_float(-(min + 1)) <= digits); 
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

inline const H5::PredType* choose_pred_type(WriteStorageType type) {
    const H5::PredType* dtype = NULL;

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

    return dtype;
}

template<typename T>
const H5::PredType& define_mem_type() {
    if constexpr(std::is_same<int, T>::value) {
        return H5::PredType::NATIVE_INT;
    } else if (std::is_same<unsigned int, T>::value) {
        return H5::PredType::NATIVE_UINT;
    } else if (std::is_same<long, T>::value) {
        return H5::PredType::NATIVE_LONG;
    } else if (std::is_same<unsigned long, T>::value) {
        return H5::PredType::NATIVE_ULONG;
    } else if (std::is_same<long long, T>::value) {
        return H5::PredType::NATIVE_LLONG;
    } else if (std::is_same<unsigned long long, T>::value) {
        return H5::PredType::NATIVE_ULLONG;
    } else if (std::is_same<short, T>::value) {
        return H5::PredType::NATIVE_SHORT;
    } else if (std::is_same<unsigned short, T>::value) {
        return H5::PredType::NATIVE_USHORT;
    } else if (std::is_same<char, T>::value) {
        return H5::PredType::NATIVE_CHAR;
    } else if (std::is_same<unsigned char, T>::value) {
        return H5::PredType::NATIVE_UCHAR;
    } else if (std::is_same<double, T>::value) {
        return H5::PredType::NATIVE_DOUBLE;
    } else if (std::is_same<float, T>::value) {
        return H5::PredType::NATIVE_FLOAT;
    } else if (std::is_same<long double, T>::value) {
        return H5::PredType::NATIVE_LDOUBLE;
    } else if (std::is_same<uint8_t, T>::value) {
        return H5::PredType::NATIVE_UINT8;
    } else if (std::is_same<int8_t, T>::value) {
        return H5::PredType::NATIVE_INT8;
    } else if (std::is_same<uint16_t, T>::value) {
        return H5::PredType::NATIVE_UINT16;
    } else if (std::is_same<int16_t, T>::value) {
        return H5::PredType::NATIVE_INT16;
    } else if (std::is_same<uint32_t, T>::value) {
        return H5::PredType::NATIVE_UINT32;
    } else if (std::is_same<int32_t, T>::value) {
        return H5::PredType::NATIVE_INT32;
    } else if (std::is_same<uint64_t, T>::value) {
        return H5::PredType::NATIVE_UINT64;
    } else if (std::is_same<int64_t, T>::value) {
        return H5::PredType::NATIVE_INT64;
    }
    static_assert("unsupported HDF5 type for template parameter 'T'");
}

template<bool integer_only, class GroupLike>
H5::DataSet open_and_check_dataset(const GroupLike& handle, const std::string& name) {
    // Avoid throwing H5 exceptions.
    if (!H5Lexists(handle.getId(), name.c_str(), H5P_DEFAULT) || handle.childObjType(name) != H5O_TYPE_DATASET) {
        throw std::runtime_error("no child dataset named '" + name + "'");
    }

    auto dhandle = handle.openDataSet(name);
    auto type = dhandle.getTypeClass();
    if constexpr(integer_only) {
        if (type != H5T_INTEGER) {
            throw std::runtime_error(std::string("expected integer values in the '") + name + "' dataset");
        }
    } else {
        if (type != H5T_INTEGER && type != H5T_FLOAT) { 
            throw std::runtime_error(std::string("expected numeric values in the '") + name + "' dataset");
        }
    }

    return dhandle;
}

template<int N>
std::array<hsize_t, N> get_array_dimensions(const H5::DataSet& handle, const std::string& name) {
    auto space = handle.getSpace();

    int ndim = space.getSimpleExtentNdims();
    if (ndim != N) {
        throw std::runtime_error(std::string("'") + name + "' should be a " + std::to_string(N) + "-dimensional array");
    }

    std::array<hsize_t, N> dims_out;
    space.getSimpleExtentDims(dims_out.data(), NULL);
    return dims_out;
}
/**
 * @endcond
 */

}

#endif
