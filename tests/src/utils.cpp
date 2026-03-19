#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "tatami_hdf5/utils.hpp"

#include <cstdint>

TEST(Utils, IsLessThanOrEqual) {
    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(1), 2));
    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(-1), 2));
    EXPECT_FALSE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(3), 2));

    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::uint8_t>(1), 2u));
    EXPECT_FALSE(tatami_hdf5::is_less_than_or_equal(static_cast<std::uint8_t>(3), 2u));

    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(1), 2u));
    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(-1), 0u));
    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(0), 0u));
    EXPECT_FALSE(tatami_hdf5::is_less_than_or_equal(static_cast<std::int8_t>(10), 0u));

    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::uint8_t>(1), 1));
    EXPECT_FALSE(tatami_hdf5::is_less_than_or_equal(static_cast<std::uint8_t>(1), -1));
    EXPECT_TRUE(tatami_hdf5::is_less_than_or_equal(static_cast<std::uint8_t>(0), 0));
    EXPECT_FALSE(tatami_hdf5::is_less_than_or_equal(static_cast<std::uint8_t>(10), 0));
}

TEST(Utils, RequiredBitsForFloat) {
    for (int i = 0; i < 2; ++i) {
        auto fun = [&](auto x) -> int {
            if (i == 0) {
                return tatami_hdf5::required_bits_for_float(x);
            } else {
                return tatami_hdf5::required_bits_for_float_safe(x);
            }
        };

        EXPECT_EQ(fun(1.), 1);
        EXPECT_EQ(fun(127.), 7);
        EXPECT_EQ(fun(128.), 8);
        EXPECT_EQ(fun(255.), 8);
        EXPECT_EQ(fun(256.), 9);
        EXPECT_EQ(fun(32767.), 15);
        EXPECT_EQ(fun(32768.), 16);
        EXPECT_EQ(fun(65535.), 16);
        EXPECT_EQ(fun(65536.), 17);
        EXPECT_EQ(fun(2147483647.), 31);
        EXPECT_EQ(fun(2147483648.), 32);
        EXPECT_EQ(fun(4294967295.), 32);
        EXPECT_EQ(fun(4294967296.), 33);
    }
}

TEST(Utils, FitsLimit) {
    EXPECT_FALSE(tatami_hdf5::fits_lower_limit<std::int8_t>(-1000));
    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-10));
    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(10));

    EXPECT_FALSE(tatami_hdf5::fits_upper_limit<std::int8_t>(1000));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(10));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(-10));

    // Now the floats are where it gets interesting:
    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-10.0));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(10.0));

    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(1000.0)); // ignores values of the other sign.
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(-1000.0));

    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(0.0)); // protects against zeros.
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(0.5)); // ... even with truncation.

    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-1.)); // some special behavior at -1 for lower_limit.
    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-1.8)); // ... again, with truncation.

    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-128.1f));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(127.8f));
    EXPECT_FALSE(tatami_hdf5::fits_lower_limit<std::int8_t>(-129.f));
    EXPECT_FALSE(tatami_hdf5::fits_upper_limit<std::int8_t>(128.f));

    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::uint32_t>(0.f));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::uint32_t>(4294967295.6));
    EXPECT_FALSE(tatami_hdf5::fits_lower_limit<std::uint32_t>(-1.f)); // unsigned integers can't handle negative values.
    EXPECT_FALSE(tatami_hdf5::fits_upper_limit<std::uint32_t>(4294967296.6));

    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int32_t>(-2147483648.44));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int32_t>(2147483647.88));
    EXPECT_FALSE(tatami_hdf5::fits_lower_limit<std::int32_t>(-2147483649.1));
    EXPECT_FALSE(tatami_hdf5::fits_upper_limit<std::int32_t>(2147483648.0));
}

template<typename Value_>
static tatami_hdf5::WriteStorageType choose_integer_type(const std::optional<tatami_hdf5::WriteStorageType>& x, Value_ lower, Value_ upper) {
    return tatami_hdf5::choose_data_type(
        x,
        lower,
        upper,
        /* has_decimal = */ false,
        /* force_integer = */ false,
        /* has_nonfinite = */ false
    );
}

TEST(Utils, ChooseDataType) {
    // Seeing if auto-inference works correctly.
    EXPECT_EQ(choose_integer_type({}, 0, 5), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_EQ(choose_integer_type({}, 0, 500), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_EQ(choose_integer_type({}, 0, 500000), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_EQ(choose_integer_type({}, 0ll, 5000000000ll), tatami_hdf5::WriteStorageType::UINT64);

    EXPECT_EQ(choose_integer_type({}, -5, 0), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_EQ(choose_integer_type({}, -500, 0), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_EQ(choose_integer_type({}, -500000, 0), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_EQ(choose_integer_type({}, -5000000000ll, 0ll), tatami_hdf5::WriteStorageType::INT64);

    EXPECT_EQ(choose_integer_type({}, -1, 5), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_EQ(choose_integer_type({}, -1, 500), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_EQ(choose_integer_type({}, -1, 500000), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_EQ(choose_integer_type({}, -1ll, 50000000000ll), tatami_hdf5::WriteStorageType::INT64);

    // Checking for correct auto-inference with floating-point inputs.
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            {},
            -5.0,
            5.0,
            /* has_decimal = */ true,
            /* force_integer = */ false,
            /* has_nonfinite = */ false
        ),
        tatami_hdf5::WriteStorageType::DOUBLE
    );
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            {},
            -5.0f,
            5.0f,
            /* has_decimal = */ true,
            /* force_integer = */ false,
            /* has_nonfinite = */ false
        ),
        tatami_hdf5::WriteStorageType::FLOAT
    );
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            {},
            -5.0,
            5.0,
            /* has_decimal = */ true,
            /* force_integer = */ true,
            /* has_nonfinite = */ false
        ),
        tatami_hdf5::WriteStorageType::INT8
    );
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            {},
            -5.0,
            5.0,
            /* has_decimal = */ true,
            /* force_integer = */ true,
            /* has_nonfinite = */ true
        ),
        tatami_hdf5::WriteStorageType::DOUBLE
    );

    // Checking the verification of a user-supplied type.
    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::UINT8, 0, 5), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT8, -1, 0));
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT8, 0, 300));

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::INT8, -5, 5), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT8, -300, 0));
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT8, 0, 300));

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::UINT16, 0, 500), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT16, -1, 0));
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT16, 0, 100000));

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::INT16, -500, 500), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT16, -100000, 0));
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT16, 0, 100000));

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::UINT32, 0, 100000), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT32, -1, 0));
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT32, 0ll, 5000000000ll));

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::INT32, -100000, 100000), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT32, -5000000000ll, 0ll));
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT32, 0ll, 5000000000ll));

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::INT64, -5000000000ll, 5000000000ll), tatami_hdf5::WriteStorageType::INT64);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::INT64, 0ull, 9223372036854775809ull));
    // Not sure how to do the test for underflow in negative values.

    EXPECT_EQ(choose_integer_type(tatami_hdf5::WriteStorageType::UINT64, 0ll, 5000000000ll), tatami_hdf5::WriteStorageType::UINT64);
    EXPECT_ANY_THROW(choose_integer_type(tatami_hdf5::WriteStorageType::UINT64, -1, 0));
    // Not sure how to do the test for overflow in positive values.

    // Checking for correct handling of floating-point inputs.
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            tatami_hdf5::WriteStorageType::FLOAT,
            0,
            0,
            /* has_decimal = */ true,
            /* force_integer = */ false,
            /* has_nonfinite = */ false
        ),
        tatami_hdf5::WriteStorageType::FLOAT
    );
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            tatami_hdf5::WriteStorageType::FLOAT,
            0,
            0,
            /* has_decimal = */ false,
            /* force_integer = */ false,
            /* has_nonfinite = */ true 
        ),
        tatami_hdf5::WriteStorageType::FLOAT
    );
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            tatami_hdf5::WriteStorageType::DOUBLE,
            0,
            0,
            /* has_decimal = */ true,
            /* force_integer = */ true,
            /* has_nonfinite = */ true 
        ),
        tatami_hdf5::WriteStorageType::DOUBLE
    );
    EXPECT_ANY_THROW(
        tatami_hdf5::choose_data_type(
            tatami_hdf5::WriteStorageType::UINT8,
            0,
            0,
            /* has_decimal = */ true,
            /* force_integer = */ false,
            /* has_nonfinite = */ false 
        )
    );
    EXPECT_ANY_THROW(
        tatami_hdf5::choose_data_type(
            tatami_hdf5::WriteStorageType::UINT8,
            0,
            0,
            /* has_decimal = */ false,
            /* force_integer = */ false,
            /* has_nonfinite = */ true 
        )
    );
    EXPECT_EQ(
        tatami_hdf5::choose_data_type(
            tatami_hdf5::WriteStorageType::UINT8,
            0,
            0,
            /* has_decimal = */ true,
            /* force_integer = */ true,
            /* has_nonfinite = */ false
        ),
        tatami_hdf5::WriteStorageType::UINT8
    );
}
