#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL // make sure this is included before tatami::apply.
#include "../../stats/custom_parallel.h"
#include "hdf5_custom_lock.h"
#endif

#include "H5Cpp.h"
#include "tatami/base/sparse/CompressedSparseMatrix.hpp"
#include "tatami_hdf5/load_hdf5_matrix.hpp"
#include "tatami_hdf5/write_sparse_matrix_to_hdf5.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>

/*****************************************
 *****************************************/

TEST(WriteSparseMatrixToHdf5Test, SparseColumn) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, 0, 100);
    tatami::CompressedSparseMatrix<false, double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle);
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

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded.dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = mwrk->fetch(r);
            auto relrow = rwrk->fetch(r);
            EXPECT_EQ(matrow, relrow);
        }
    }

    // Forcing it to be integer.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters params;
        params.force_integer = true;
        params.columnar = tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageLayout::COLUMN;
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle, params);
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
            auto matrow = mwrk->fetch(r);
            for (auto& x : matrow) {
                x = static_cast<int>(x);
            }
            auto relrow = rwrk->fetch(r);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

TEST(WriteSparseMatrixToHdf5Test, SparseRow) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.05, 0, 100);
    tatami::CompressedSparseMatrix<true, double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle);
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

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded.dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = mwrk->fetch(r);
            auto relrow = rwrk->fetch(r);
            EXPECT_EQ(matrow, relrow);
        }
    }

    // Forcing it to be columnar.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters params;
        params.columnar = tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageLayout::COLUMN;
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle, params);
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

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded.dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = mwrk->fetch(r);
            auto relrow = rwrk->fetch(r);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

/*****************************************
 *****************************************/

class WriteSparseMatrixToHdf5UnsignedDataTypeTest : public ::testing::TestWithParam<tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType> {};

TEST_P(WriteSparseMatrixToHdf5UnsignedDataTypeTest, Check) {
    const size_t NR = 200, NC = 100;
    auto type = GetParam();

    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, 0, 100);
    for (auto& x : triplets.value) {
        x = std::round(x);
        if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT16) {
            x *= 10;
        }  else if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT32) {
            x *= 1000;
        }
    }
    tatami::CompressedSparseMatrix<false, double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType dtype(dhandle);
        EXPECT_EQ(dtype.getSign(), H5T_SGN_NONE);
        if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT8) {
            EXPECT_TRUE(dtype.getSize() <= 8);
        } else if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT16) {
            EXPECT_TRUE(dtype.getSize() <= 16);
        } else {
            EXPECT_TRUE(dtype.getSize() <= 32);
        }
    }

    // Roundtripping.
    {
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded.dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = mwrk->fetch(r);
            auto relrow = rwrk->fetch(r);
            EXPECT_EQ(matrow, relrow);
        }
    }

    // But we can always force it to a float.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters params;
        params.data_type = tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::DOUBLE;
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
    }

    {
        auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");

        auto mwrk = mat.dense_row();
        auto rwrk = reloaded.dense_row();
        for (size_t r = 0; r < NR; ++r) {
            auto matrow = mwrk->fetch(r);
            auto relrow = rwrk->fetch(r);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    WriteSparseMatrixToHdf5,
    WriteSparseMatrixToHdf5UnsignedDataTypeTest,
    ::testing::Values(
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT8,
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT16,
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT32
    )
);

/*****************************************
 *****************************************/

class WriteSparseMatrixToHdf5SignedDataTypeTest : public ::testing::TestWithParam<tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType> {};

TEST_P(WriteSparseMatrixToHdf5SignedDataTypeTest, Check) {
    const size_t NR = 200, NC = 100;
    auto type = GetParam();

    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, -100, 100);
    for (auto& x : triplets.value) {
        x = std::round(x);
        if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT16) {
            x *= 10;
        }  else if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT32) {
            x *= 1000;
        }
    }
    tatami::CompressedSparseMatrix<false, double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType dtype(dhandle);
        EXPECT_EQ(dtype.getSign(), H5T_SGN_2);
        if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT8) {
            EXPECT_TRUE(dtype.getSize() <= 8);
        } else if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT16) {
            EXPECT_TRUE(dtype.getSize() <= 16);
        } else {
            EXPECT_TRUE(dtype.getSize() <= 32);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");

    auto mwrk = mat.dense_row();
    auto rwrk = reloaded.dense_row();
    for (size_t r = 0; r < NR; ++r) {
        auto matrow = mwrk->fetch(r);
        auto relrow = rwrk->fetch(r);
        EXPECT_EQ(matrow, relrow);
    }
}

INSTANTIATE_TEST_SUITE_P(
    WriteSparseMatrixToHdf5,
    WriteSparseMatrixToHdf5SignedDataTypeTest,
    ::testing::Values(
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT8,
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT16,
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::INT32
    )
);

/*****************************************
 *****************************************/

class WriteSparseMatrixToHdf5IndexTypeTest : public ::testing::TestWithParam<tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType> {};

TEST_P(WriteSparseMatrixToHdf5IndexTypeTest, Check) {
    const size_t NC = 10;
    auto type = GetParam();

    size_t NR = 200;
    if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT16) {
        NR = 2000;
    } else if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT32) {
        NR = 100000;
    }

    auto triplets = tatami_test::simulate_sparse_compressed<double>(NC, NR, 0.05, -100, 100);
    tatami::CompressedSparseMatrix<false, double, int> mat(NR, NC, std::move(triplets.value), std::move(triplets.index), std::move(triplets.ptr));

    // Dumping it.
    auto fpath = tatami_test::temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_sparse_matrix_to_hdf5(&mat, mhandle);
    }

    // Checking the dumped contents.
    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto ihandle = fhandle.openDataSet("matrix/indices");
        EXPECT_EQ(ihandle.getDataType().getClass(), H5T_INTEGER);

        H5::IntType itype(ihandle);
        EXPECT_EQ(itype.getSign(), H5T_SGN_NONE);
        if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT8) {
            EXPECT_TRUE(itype.getSize() <= 8);
        } else if (type == tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT16) {
            EXPECT_TRUE(itype.getSize() <= 16);
        } else {
            EXPECT_TRUE(itype.getSize() <= 32);
        }
    }

    // Roundtripping.
    auto reloaded = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr");

    auto mwrk = mat.dense_column();
    auto rwrk = reloaded.dense_column();
    for (size_t c = 0; c < NC; ++c) {
        auto matrow = mwrk->fetch(c);
        auto relrow = rwrk->fetch(c);
        EXPECT_EQ(matrow, relrow);
    }
}

INSTANTIATE_TEST_SUITE_P(
    WriteSparseMatrixToHdf5,
    WriteSparseMatrixToHdf5IndexTypeTest,
    ::testing::Values(
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT8,
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT16,
        tatami_hdf5::WriteSparseMatrixToHdf5Parameters::StorageType::UINT32
    )
);
