#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/load_compressed_sparse_matrix.hpp"
#include "tatami_hdf5/write_compressed_sparse_matrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "temp_file_path.h"

#include <vector>
#include <random>
#include <cstdint>

TEST(WriteCompressedSparseMatrix, IsLessThanOrEqual) {
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

TEST(WriteCompressedSparseMatrix, FitsLimit) {
    EXPECT_FALSE(tatami_hdf5::fits_lower_limit<std::int8_t>(-1000));
    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-10));
    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(10));

    EXPECT_FALSE(tatami_hdf5::fits_upper_limit<std::int8_t>(1000));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(10));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(-10));

    EXPECT_TRUE(tatami_hdf5::fits_lower_limit<std::int8_t>(-10.0));
    EXPECT_TRUE(tatami_hdf5::fits_upper_limit<std::int8_t>(10.0));
    EXPECT_FALSE(tatami_hdf5::fits_lower_limit<std::int8_t>(-1000.0f));
    EXPECT_FALSE(tatami_hdf5::fits_upper_limit<std::int8_t>(1000.0f));
}

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixBasicTest : public ::testing::TestWithParam<int> {};

TEST_P(WriteCompressedSparseMatrixBasicTest, SparseColumn) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);

        auto phandle = fhandle.openDataSet("matrix/indptr");
        auto pspace = phandle.getSpace();
        hsize_t dim;
        pspace.getSimpleExtentDims(&dim);
        EXPECT_EQ(dim, NC + 1);
    }

    // Roundtripping.
    {
        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }

    // Forcing it to be integer.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        auto params = param_core;
        params.force_integer = true;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);
    }

    {
        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded->dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = tatami_test::fetch(*mwrk, static_cast<int>(r), NC);
            for (auto& x : matrow) {
                x = static_cast<int>(x);
            }
            auto relrow = tatami_test::fetch(*rwrk, static_cast<int>(r), NC);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

TEST_P(WriteCompressedSparseMatrixBasicTest, SparseRow) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NR, NC, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());
    tatami::CompressedSparseRowMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto phandle = fhandle.openDataSet("matrix/indptr");
        auto pspace = phandle.getSpace();
        hsize_t dim;
        pspace.getSimpleExtentDims(&dim);
        EXPECT_EQ(dim, NR + 1);
    }

    // Roundtripping.
    {
        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", true);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }

    // Forcing it to be columnar.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        auto params = param_core;
        params.columnar = tatami_hdf5::WriteStorageLayout::COLUMN;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto phandle = fhandle.openDataSet("matrix/indptr");
        auto pspace = phandle.getSpace();
        hsize_t dim;
        pspace.getSimpleExtentDims(&dim);
        EXPECT_EQ(dim, NC + 1);
    }

    {
        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }
}

TEST_P(WriteCompressedSparseMatrixBasicTest, DenseColumn) {
    const size_t NR = 190, NC = 210;
    auto vec = tatami_test::simulate_vector<double>(NR * NC, []{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());
    tatami::DenseColumnMatrix<double, int> mat(NR, NC, std::move(vec));

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

TEST_P(WriteCompressedSparseMatrixBasicTest, DenseRow) {
    const size_t NR = 90, NC = 300;
    auto vec = tatami_test::simulate_vector<double>(NR * NC, []{
        tatami_test::SimulateVectorOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());
    tatami::DenseRowMatrix<double, int> mat(NR, NC, std::move(vec));

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", true);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixBasicTest,
    ::testing::Values(1, 3) // Number of threads
);

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixUnsignedDataTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixUnsignedDataTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    double cap = 10;
    if (type == tatami_hdf5::WriteStorageType::UINT16) {
        cap = 1000;
    } else if (type == tatami_hdf5::WriteStorageType::UINT32) {
        cap = 100000;
    } else if (type == tatami_hdf5::WriteStorageType::UINT64) {
        cap = 5000000000;
    }

    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, [&]{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = cap;
        return opt;
    }());

    for (auto& x : triplets.data) {
        x = std::round(x);
    }

    // Make sure at least the largest value is large enough to exceed the immediately preceding type.
    ASSERT_FALSE(triplets.data.empty());
    auto maxIt = std::min_element(triplets.data.begin(), triplets.data.end());
    *maxIt = cap;

    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = nthreads;

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType dtype(dhandle);
        EXPECT_EQ(dtype.getSign(), H5T_SGN_NONE);
        if (type == tatami_hdf5::WriteStorageType::UINT8) {
            EXPECT_EQ(dtype.getSize(), 1);
        } else if (type == tatami_hdf5::WriteStorageType::UINT16) {
            EXPECT_EQ(dtype.getSize(), 2);
        } else if (type == tatami_hdf5::WriteStorageType::UINT32) {
            EXPECT_EQ(dtype.getSize(), 4);
        } else {
            EXPECT_EQ(dtype.getSize(), 8);
        }
    }

    // Roundtripping.
    {
        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }

    // But we can always force it to a float.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        auto params = param_core;
        params.data_type = tatami_hdf5::WriteStorageType::DOUBLE;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
    }

    {
        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixUnsignedDataTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteStorageType::UINT8,
            tatami_hdf5::WriteStorageType::UINT16,
            tatami_hdf5::WriteStorageType::UINT32,
            tatami_hdf5::WriteStorageType::UINT64
        ),
        ::testing::Values(1, 3)
    )
);

TEST_F(WriteCompressedSparseMatrixUnsignedDataTypeTest, Choices) {
    const bool def_non_integer = false;
    const bool def_force_integer = false;

    EXPECT_EQ(tatami_hdf5::choose_data_type({}, 0, 5, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, 0, 500, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, 0, 500000, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, 0ll, 5000000000ll, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT64);

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT8, 0, 5, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT8, -1, 0, def_non_integer, def_force_integer));
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT8, 0, 300, def_non_integer, def_force_integer));

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT16, 0, 500, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT16, -1, 0, def_non_integer, def_force_integer));
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT16, 0, 100000, def_non_integer, def_force_integer));

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT32, 0, 100000, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT32, -1, 0, def_non_integer, def_force_integer));
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT32, 0ll, 5000000000ll, def_non_integer, def_force_integer));

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT64, 0ll, 5000000000ll, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::UINT64);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT64, -1, 0, def_non_integer, def_force_integer));
    // Not sure how to do the test for overflow in positive values.

    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT8, 0, 0, /* non_integer = */ true, /* force_integer = */ false));
    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::UINT8, 0, 0, /* non_integer = */ true, /* force_integer = */ true), tatami_hdf5::WriteStorageType::UINT8);
}

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixSignedDataTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixSignedDataTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    double cap = 10;
    if (type == tatami_hdf5::WriteStorageType::INT16) {
        cap = 1000;
    } else if (type == tatami_hdf5::WriteStorageType::INT32) {
        cap = 100000;
    } else if (type == tatami_hdf5::WriteStorageType::INT64) {
        cap = 5000000000;
    }

    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, [&]{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = -cap;
        opt.upper = cap;
        return opt;
    }());

    for (auto& x : triplets.data) {
        x = std::round(x);
    }

    // Make sure at least the smallest value is small enough to underflow the immediately preceding type.
    ASSERT_FALSE(triplets.data.empty());
    auto maxIt = std::min_element(triplets.data.begin(), triplets.data.end());
    *maxIt = -cap;

    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType dtype(dhandle);
        EXPECT_EQ(dtype.getSign(), H5T_SGN_2);
        if (type == tatami_hdf5::WriteStorageType::INT8) {
            EXPECT_EQ(dtype.getSize(), 1);
        } else if (type == tatami_hdf5::WriteStorageType::INT16) {
            EXPECT_EQ(dtype.getSize(), 2);
        } else if (type == tatami_hdf5::WriteStorageType::INT32) {
            EXPECT_EQ(dtype.getSize(), 4);
        } else {
            EXPECT_EQ(dtype.getSize(), 8);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixSignedDataTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteStorageType::INT8,
            tatami_hdf5::WriteStorageType::INT16,
            tatami_hdf5::WriteStorageType::INT32,
            tatami_hdf5::WriteStorageType::INT64
        ),
        ::testing::Values(1,3)
    )
);

TEST_F(WriteCompressedSparseMatrixSignedDataTypeTest, Choices) {
    const bool def_non_integer = false;
    const bool def_force_integer = false;

    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -5, 0, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -500, 0, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -500000, 0, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -5000000000ll, 0ll, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT64);

    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -1, 5, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -1, 500, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -1, 500000, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -1ll, 50000000000ll, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT64);

    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -5.0, 5.0, /* non_integer = */ true, /* force_integer = */ false), tatami_hdf5::WriteStorageType::DOUBLE);
    EXPECT_EQ(tatami_hdf5::choose_data_type({}, -5.0, 5.0, /* non_integer = */ true, /* force_integer = */ true), tatami_hdf5::WriteStorageType::INT8);

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT8, -5, 5, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT8, -300, 0, def_non_integer, def_force_integer));
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT8, 0, 300, def_non_integer, def_force_integer));

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT16, -500, 500, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT16, -100000, 0, def_non_integer, def_force_integer));
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT16, 0, 100000, def_non_integer, def_force_integer));

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT32, -100000, 100000, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT32, -5000000000ll, 0ll, def_non_integer, def_force_integer));
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT32, 0ll, 5000000000ll, def_non_integer, def_force_integer));

    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT64, -5000000000ll, 5000000000ll, def_non_integer, def_force_integer), tatami_hdf5::WriteStorageType::INT64);
    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT64, 0ull, 9223372036854775809ull, def_non_integer, def_force_integer));
    // Not sure how to do the test for underflow in negative values.

    EXPECT_ANY_THROW(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT8, 0, 0, /* non_integer = */ true, /* force_integer = */ false));
    EXPECT_EQ(tatami_hdf5::choose_data_type(tatami_hdf5::WriteStorageType::INT8, 0, 0, /* non_integer = */ true, /* force_integer = */ true), tatami_hdf5::WriteStorageType::INT8);
}

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixIndexTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixIndexTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    const size_t NC = 10;
    size_t NR = 200;
    if (type == tatami_hdf5::WriteStorageType::UINT16) {
        NR = 2000;
    } else if (type == tatami_hdf5::WriteStorageType::UINT32) {
        NR = 100000;
    }

    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = -100;
        opt.upper = 100;
        return opt;
    }());

    // Make sure at least the largest index is large enough to exceed the immediately preceding type.
    ASSERT_FALSE(triplets.index.empty());
    auto maxIt = std::max_element(triplets.index.begin(), triplets.index.end());
    *maxIt = NR - 1;

    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto ihandle = fhandle.openDataSet("matrix/indices");
        EXPECT_EQ(ihandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType itype(ihandle);
        EXPECT_EQ(itype.getSign(), H5T_SGN_NONE);
        if (type == tatami_hdf5::WriteStorageType::UINT8) {
            EXPECT_EQ(itype.getSize(), 1);
        } else if (type == tatami_hdf5::WriteStorageType::UINT16) {
            EXPECT_EQ(itype.getSize(), 2);
        } else {
            EXPECT_EQ(itype.getSize(), 4);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

// We can't check auto-detection of UINT64 easily as we'd need to allocate a lot of memory to deal with a matrix with >= 4 billion rows.
// Even if the matrix itself is sparse, the buffers would end up being huge, so we're not going to bother with that. 
INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixIndexTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteStorageType::UINT8,
            tatami_hdf5::WriteStorageType::UINT16,
            tatami_hdf5::WriteStorageType::UINT32
        ),
        ::testing::Values(1,3)
    )
);

TEST_F(WriteCompressedSparseMatrixIndexTypeTest, Choices) {
    EXPECT_EQ(tatami_hdf5::choose_index_type({}, 5), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_EQ(tatami_hdf5::choose_index_type({}, 500), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_EQ(tatami_hdf5::choose_index_type({}, 500000), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_EQ(tatami_hdf5::choose_index_type({}, 5000000000ull), tatami_hdf5::WriteStorageType::UINT64);

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT8, 5), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT8, 200));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT8, 5), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT8, 500));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT16, 500), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT16, 50000));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT16, 500), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT16, 100000));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT32, 100000), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT32, 3000000000ull));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT32, 100000), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT32, 5000000000ull));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT64, 5000000000ull), tatami_hdf5::WriteStorageType::INT64);
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::INT64, 9223372036854775808ull));

    EXPECT_EQ(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::UINT64, 5000000000ull), tatami_hdf5::WriteStorageType::UINT64);
    // Don't know how to test for boundaries here, there isn't anything guaranteed to be bigger. 

    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::DOUBLE, 10));
}

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixPtrTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixPtrTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    const size_t NC = 10;
    size_t NR = 200;

    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = -100;
        opt.upper = 100;
        return opt;
    }());
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        params.ptr_type = type; // forcibly setting the type as I don't want to simulate 2^32 non-zero elements.
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto phandle = fhandle.openDataSet("matrix/indptr");
        EXPECT_EQ(phandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType ptype(phandle);
        EXPECT_EQ(ptype.getSign(), H5T_SGN_NONE);
        if (type == tatami_hdf5::WriteStorageType::UINT32) {
            EXPECT_EQ(ptype.getSize(), 4);
        } else {
            EXPECT_EQ(ptype.getSize(), 8);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixPtrTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteStorageType::UINT32,
            tatami_hdf5::WriteStorageType::UINT64
        ),
        ::testing::Values(1,3)
    )
);

TEST_F(WriteCompressedSparseMatrixPtrTypeTest, Choices) {
    EXPECT_EQ(tatami_hdf5::choose_ptr_type({}, 5), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_EQ(tatami_hdf5::choose_ptr_type({}, 5000000000ull), tatami_hdf5::WriteStorageType::UINT64);

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT8, 5), tatami_hdf5::WriteStorageType::INT8);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT8, 200));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT8, 5), tatami_hdf5::WriteStorageType::UINT8);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT8, 500));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT16, 500), tatami_hdf5::WriteStorageType::INT16);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT16, 50000));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT16, 500), tatami_hdf5::WriteStorageType::UINT16);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT16, 100000));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT32, 100000), tatami_hdf5::WriteStorageType::INT32);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT32, 3000000000ull));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT32, 100000), tatami_hdf5::WriteStorageType::UINT32);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT32, 5000000000ull));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT64, 5000000000ull), tatami_hdf5::WriteStorageType::INT64);
    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::INT64, 9223372036854775808ull));

    EXPECT_EQ(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::UINT64, 5000000000ull), tatami_hdf5::WriteStorageType::UINT64);
    // Don't know how to test for boundaries here, there isn't anything guaranteed to be bigger. 

    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::DOUBLE, 10));
}

/*****************************************
 *****************************************/

TEST(WriteCompressedSparseMatrix, Defaults) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    // Dumping it with default parameters, just to check the overload.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle);
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}
