#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/load_hdf5_matrix.hpp"
#include "tatami_hdf5/write_compressed_sparse_matrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixBasicTest : public ::testing::TestWithParam<int> {};

TEST_P(WriteCompressedSparseMatrixBasicTest, SparseColumn) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, 0, 100);
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    tatami_hdf5::WriteCompressedSparseMatrixParameters param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
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
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
        tatami_test::test_simple_row_access(&reloaded, &mat);
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
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded.dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = tatami_test::fetch(mwrk.get(), static_cast<int>(r), NC);
            for (auto& x : matrow) {
                x = static_cast<int>(x);
            }
            auto relrow = tatami_test::fetch(rwrk.get(), static_cast<int>(r), NC);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

TEST_P(WriteCompressedSparseMatrixBasicTest, SparseRow) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.05, 0, 100);
    tatami::CompressedSparseRowMatrix<double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    tatami_hdf5::WriteCompressedSparseMatrixParameters param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
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
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<true, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
        tatami_test::test_simple_row_access(&reloaded, &mat);
    }

    // Forcing it to be columnar.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        auto params = param_core;
        params.columnar = tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageLayout::COLUMN;
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
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
        tatami_test::test_simple_row_access(&reloaded, &mat);
    }
}

TEST_P(WriteCompressedSparseMatrixBasicTest, DenseColumn) {
    const size_t NR = 190, NC = 210;
    auto vec = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.05, 0, 100);
    tatami::DenseColumnMatrix<double, int> mat(NR, NC, std::move(vec));

    tatami_hdf5::WriteCompressedSparseMatrixParameters param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
    tatami_test::test_simple_row_access(&reloaded, &mat);
}

TEST_P(WriteCompressedSparseMatrixBasicTest, DenseRow) {
    const size_t NR = 90, NC = 300;
    auto vec = tatami_test::simulate_sparse_vector<double>(NR * NC, 0.05, 0, 100);
    tatami::DenseRowMatrix<double, int> mat(NR, NC, std::move(vec));

    tatami_hdf5::WriteCompressedSparseMatrixParameters param_core;
    param_core.num_threads = GetParam();

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<true, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
    tatami_test::test_simple_row_access(&reloaded, &mat);
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixBasicTest,
    ::testing::Values(1, 3) // Number of threads
);

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixUnsignedDataTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixUnsignedDataTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, 0, 100);
    for (auto& x : triplets.value) {
        x = std::round(x);
        if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT16) {
            x *= 10;
        }  else if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT32) {
            x *= 1000;
        }
    }
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    tatami_hdf5::WriteCompressedSparseMatrixParameters param_core;
    param_core.num_threads = nthreads;

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
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
        if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT8) {
            EXPECT_TRUE(dtype.getSize() <= 8);
        } else if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT16) {
            EXPECT_TRUE(dtype.getSize() <= 16);
        } else {
            EXPECT_TRUE(dtype.getSize() <= 32);
        }
    }

    // Roundtripping.
    {
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
        tatami_test::test_simple_row_access(&reloaded, &mat);
    }

    // But we can always force it to a float.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        auto params = param_core;
        params.data_type = tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::DOUBLE;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
    }

    {
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
        tatami_test::test_simple_row_access(&reloaded, &mat);
    }
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixUnsignedDataTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT8,
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT16,
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT32
        ),
        ::testing::Values(1, 3)
    )
);

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixSignedDataTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixSignedDataTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, -100, 100);
    for (auto& x : triplets.value) {
        x = std::round(x);
        if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT16) {
            x *= 10;
        }  else if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT32) {
            x *= 1000;
        }
    }
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixParameters params;
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
        if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT8) {
            EXPECT_TRUE(dtype.getSize() <= 8);
        } else if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT16) {
            EXPECT_TRUE(dtype.getSize() <= 16);
        } else {
            EXPECT_TRUE(dtype.getSize() <= 32);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
    tatami_test::test_simple_row_access(&reloaded, &mat);
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixSignedDataTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT8,
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT16,
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::INT32
        ),
        ::testing::Values(1,3)
    )
);

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixIndexTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType, int> > {};

TEST_P(WriteCompressedSparseMatrixIndexTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    const size_t NC = 10;
    size_t NR = 200;
    if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT16) {
        NR = 2000;
    } else if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT32) {
        NR = 100000;
    }

    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, -100, 100);
    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixParameters params;
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
        if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT8) {
            EXPECT_TRUE(itype.getSize() <= 8);
        } else if (type == tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT16) {
            EXPECT_TRUE(itype.getSize() <= 16);
        } else {
            EXPECT_TRUE(itype.getSize() <= 32);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");
    tatami_test::test_simple_row_access(&reloaded, &mat);
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixIndexTypeTest,
    ::testing::Combine(
        ::testing::Values(
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT8,
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT16,
            tatami_hdf5::WriteCompressedSparseMatrixParameters::StorageType::UINT32
        ),
        ::testing::Values(1,3)
    )
);
