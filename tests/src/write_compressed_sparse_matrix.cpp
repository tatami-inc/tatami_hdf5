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

class WriteCompressedSparseMatrixBasicTest : public ::testing::TestWithParam<std::tuple<bool, int, bool, int> > {
protected:
    static auto simulate(bool dense, bool columnar_input, unsigned long long seed) {
        const std::size_t NR = 200, NC = 100;

        std::unique_ptr<tatami::Matrix<double, int> > mat;
        if (dense) {
            auto sim = tatami_test::simulate_vector<double>(NR * NC, [&]{
                tatami_test::SimulateVectorOptions opt;
                opt.density = 0.05;
                opt.lower = 0;
                opt.upper = 100;
                opt.seed = seed;
                return opt;
            }());

            mat.reset(new tatami::DenseMatrix<double, int, std::vector<double> >(
                NR,
                NC,
                std::move(sim),
                !columnar_input
            ));

        } else {
            auto triplets = tatami_test::simulate_compressed_sparse<double, int>(
                (columnar_input ? NC : NR),
                (columnar_input ? NR : NC),
                [&]{
                    tatami_test::SimulateCompressedSparseOptions opt;
                    opt.density = 0.05;
                    opt.lower = 0;
                    opt.upper = 100;
                    opt.seed = seed;
                    return opt;
                }()
            );

            mat.reset(new tatami::CompressedSparseMatrix<double, int, decltype(triplets.data), decltype(triplets.index), decltype(triplets.indptr)>(
                NR,
                NC,
                std::move(triplets.data),
                std::move(triplets.index),
                std::move(triplets.indptr),
                !columnar_input
            ));
        }

        return mat;
    }

    static bool is_row_output(int layout, const tatami::Matrix<double, int>& mat, std::optional<tatami_hdf5::WriteStorageLayout>& columnar) {
        if (layout > 0) {
            columnar = tatami_hdf5::WriteStorageLayout::ROW;
            return true;
        } else if (layout < 0) {
            columnar = tatami_hdf5::WriteStorageLayout::COLUMN;
            return false;
        } else {
            return mat.prefer_rows();
        }
    }
};

TEST_P(WriteCompressedSparseMatrixBasicTest, TwoPass) {
    auto config = GetParam();
    auto columnar_input = std::get<0>(config);
    auto layout = std::get<1>(config);
    auto dense = std::get<2>(config);
    auto nthreads = std::get<3>(config);

    auto mat = simulate(
        dense,
        columnar_input,
        /* seed = */ columnar_input + 2 * (layout + 2 * (dense + 2 * nthreads))
    );
    const auto NR = mat->nrow(), NC = mat->ncol();

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = nthreads;
    const bool is_row = is_row_output(layout, *mat, param_core.columnar);

    // Standard write/read roundtrip test.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::write_compressed_sparse_matrix(*mat, mhandle, param_core);

            auto dhandle = fhandle.openDataSet("matrix/data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
            H5::FloatType dtype = dhandle.getFloatType();
            EXPECT_EQ(dtype.getSize(), 8); // should be double-precision.

            auto dspace = dhandle.getSpace();
            hsize_t dims, maxdims;
            dspace.getSimpleExtentDims(&dims, &maxdims);
            EXPECT_EQ(maxdims, dims);

            auto ihandle = fhandle.openDataSet("matrix/indices");
            EXPECT_EQ(ihandle.getDataType().getClass(), H5T_INTEGER);
            H5::IntType itype = ihandle.getIntType();
            EXPECT_EQ(itype.getSize(), 1); // should be uint8.

            auto phandle = fhandle.openDataSet("matrix/indptr");
            auto pspace = phandle.getSpace();
            hsize_t dim;
            pspace.getSimpleExtentDims(&dim);
            EXPECT_EQ(dim, (is_row ? NR : NC) + 1);
        }

        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", is_row);
        tatami_test::test_simple_row_access(*reloaded, *mat);
    }

    // Testing what happens when we force the floating-point data to integer.
    { 
        auto iparam = param_core;
        iparam.force_integer = true;
        iparam.data_name = "int_data"; // We also change the dataset names to provide some more test coverage.
        iparam.index_name = "int_indices";
        iparam.ptr_name = "int_indptr";

        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::write_compressed_sparse_matrix(*mat, mhandle, iparam);

            auto dhandle = fhandle.openDataSet("matrix/int_data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);
            H5::IntType dtype = dhandle.getIntType();
            EXPECT_EQ(dtype.getSize(), 1); // should be a 8-bit integer.
        }

        auto freloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/int_data", "matrix/int_indices", "matrix/int_indptr", is_row);
        auto mwrk = mat->dense_row();
        auto fwrk = freloaded->dense_row();
        for (int r = 0; r < NR; ++r) {
            auto matrow = tatami_test::fetch(*mwrk, static_cast<int>(r), NC);
            for (auto& x : matrow) {
                x = static_cast<int>(x);
            }
            auto relrow = tatami_test::fetch(*fwrk, static_cast<int>(r), NC);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

TEST_P(WriteCompressedSparseMatrixBasicTest, OnePass) {
    auto config = GetParam();
    auto columnar_input = std::get<0>(config);
    auto layout = std::get<1>(config);
    auto dense = std::get<2>(config);
    auto nthreads = std::get<3>(config);

    auto mat = simulate(
        dense,
        columnar_input,
        /* seed = */ columnar_input + 2 * (layout + 2 * (dense + 2 * nthreads)) + 1
    );
    const auto NR = mat->nrow(), NC = mat->ncol();

    tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
    param_core.num_threads = nthreads;
    param_core.data_type = tatami_hdf5::WriteStorageType::DOUBLE; // force it it to use a single-pass algorithm.
    param_core.index_type = tatami_hdf5::WriteStorageType::INT32;
    const bool is_row = is_row_output(layout, *mat, param_core.columnar);

    // Standard write/read roundtrip test.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::write_compressed_sparse_matrix(*mat, mhandle, param_core);

            auto dhandle = fhandle.openDataSet("matrix/data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
            H5::FloatType dtype = dhandle.getFloatType();
            EXPECT_EQ(dtype.getSize(), 8); // should be double-precision.

            auto ihandle = fhandle.openDataSet("matrix/indices");
            EXPECT_EQ(ihandle.getDataType().getClass(), H5T_INTEGER);
            H5::IntType itype = ihandle.getIntType();
            EXPECT_EQ(itype.getSize(), 4); // should be int32.

            auto ispace = ihandle.getSpace();
            hsize_t dims, maxdims;
            ispace.getSimpleExtentDims(&dims, &maxdims);
            EXPECT_EQ(maxdims, H5S_UNLIMITED);

            auto phandle = fhandle.openDataSet("matrix/indptr");
            auto pspace = phandle.getSpace();
            hsize_t dim;
            pspace.getSimpleExtentDims(&dim);
            EXPECT_EQ(dim, (is_row ? NR : NC) + 1);
        }

        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", is_row);
        tatami_test::test_simple_row_access(*reloaded, *mat);
    }

    // Testing what happens when we force the floating-point data to integer.
    { 
        auto iparam = param_core;
        iparam.data_type = tatami_hdf5::WriteStorageType::INT8;
        iparam.force_integer = true;

        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::write_compressed_sparse_matrix(*mat, mhandle, iparam);

            auto dhandle = fhandle.openDataSet("matrix/data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);
            H5::IntType dtype = dhandle.getIntType();
            EXPECT_EQ(dtype.getSize(), 1); // should be a 8-bit integer.
        }

        auto freloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", is_row);
        auto mwrk = mat->dense_row();
        auto fwrk = freloaded->dense_row();
        for (int r = 0; r < NR; ++r) {
            auto matrow = tatami_test::fetch(*mwrk, static_cast<int>(r), NC);
            for (auto& x : matrow) {
                x = static_cast<int>(x);
            }
            auto relrow = tatami_test::fetch(*fwrk, static_cast<int>(r), NC);
            EXPECT_EQ(matrow, relrow);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    WriteCompressedSparseMatrix,
    WriteCompressedSparseMatrixBasicTest,
    ::testing::Combine(
        ::testing::Values(false, true), // whether the input matrix is column-major.
        ::testing::Values(0, 1, -1), // whether to use a row-major (1), column-major(-1) or automatic layout for the output
        ::testing::Values(false, true), // dense or not.
        ::testing::Values(1, 3) // number of threads
    )
);

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixUnsignedDataTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {
protected:
    static auto simulate(tatami_hdf5::WriteStorageType type, unsigned long long seed) {
        double cap = 10;
        if (type == tatami_hdf5::WriteStorageType::UINT16) {
            cap = 500;
        } else if (type == tatami_hdf5::WriteStorageType::UINT32) {
            cap = 100000;
        } else if (type == tatami_hdf5::WriteStorageType::UINT64) {
            cap = 5000000000;
        }

        const std::size_t NR = 200, NC = 100;
        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.05;
            opt.lower = 0;
            opt.upper = cap;
            opt.seed = seed;
            return opt;
        }());

        for (auto& x : triplets.data) {
            x = std::round(x);
        }

        // Make sure at least the largest value is large enough to exceed the immediately preceding type.
        EXPECT_FALSE(triplets.data.empty());
        auto maxIt = std::min_element(triplets.data.begin(), triplets.data.end());
        *maxIt = cap;

        return tatami::CompressedSparseMatrix<
            double,
            int,
            decltype(triplets.data),
            decltype(triplets.index),
            decltype(triplets.indptr)
        >(
            NR,
            NC,
            std::move(triplets.data),
            std::move(triplets.index),
            std::move(triplets.indptr),
            /* row = */ false
        );
    }
};

TEST_P(WriteCompressedSparseMatrixUnsignedDataTypeTest, TwoPass) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    auto mat = simulate(type, /* seed = */ int(type) + nthreads * 10);
    const auto NR = mat.nrow(), NC = mat.ncol();

    // Standard write/read roundtrip test.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
            param_core.num_threads = nthreads;
            tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
        }

        {
            H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
            auto dhandle = fhandle.openDataSet("matrix/data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

            auto dspace = dhandle.getSpace();
            hsize_t dims, maxdims;
            dspace.getSimpleExtentDims(&dims, &maxdims);
            EXPECT_EQ(maxdims, dims);

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

        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }

    // But we can always force it to a float.
    {
        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::WriteCompressedSparseMatrixOptions fparam;
            fparam.num_threads = nthreads;
            fparam.data_name = "fp_data"; // We also change the dataset names to provide some more test coverage.
            fparam.index_name = "fp_indices";
            fparam.ptr_name = "fp_indptr";
            fparam.data_type = tatami_hdf5::WriteStorageType::DOUBLE;
            tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, fparam);
        }

        {
            H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
            auto dhandle = fhandle.openDataSet("matrix/fp_data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
        }

        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/fp_data", "matrix/fp_indices", "matrix/fp_indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }
}

TEST_P(WriteCompressedSparseMatrixUnsignedDataTypeTest, OnePass) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    auto mat = simulate(type, /* seed = */ int(type) + nthreads * 11);
    const auto NR = mat.nrow(), NC = mat.ncol();

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions param_core;
        param_core.num_threads = nthreads;
        param_core.data_type = type;
        param_core.index_type = tatami_hdf5::WriteStorageType::UINT32;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, param_core);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

        auto dspace = dhandle.getSpace();
        hsize_t dims, maxdims;
        dspace.getSimpleExtentDims(&dims, &maxdims);
        EXPECT_EQ(maxdims, H5S_UNLIMITED);

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

    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
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

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixSignedDataTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {
protected:
    static auto simulate(tatami_hdf5::WriteStorageType type, unsigned long long seed) {
        double cap = 10;
        if (type == tatami_hdf5::WriteStorageType::INT16) {
            cap = 1000;
        } else if (type == tatami_hdf5::WriteStorageType::INT32) {
            cap = 100000;
        } else if (type == tatami_hdf5::WriteStorageType::INT64) {
            cap = 5000000000;
        }

        const std::size_t NR = 200, NC = 100;
        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.05;
            opt.lower = -cap;
            opt.upper = cap;
            opt.seed = seed;
            return opt;
        }());

        for (auto& x : triplets.data) {
            x = std::round(x);
        }

        // Make sure at least the smallest value is small enough to underflow the immediately preceding type.
        EXPECT_FALSE(triplets.data.empty());
        auto maxIt = std::min_element(triplets.data.begin(), triplets.data.end());
        *maxIt = -cap;

        return tatami::CompressedSparseMatrix<
            double,
            int,
            decltype(triplets.data),
            decltype(triplets.index),
            decltype(triplets.indptr)
        >(
            NR,
            NC,
            std::move(triplets.data),
            std::move(triplets.index),
            std::move(triplets.indptr),
            /* row = */ false
        );
    }
};

TEST_P(WriteCompressedSparseMatrixSignedDataTypeTest, TwoPass) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    auto mat = simulate(type, /* seed = */ int(type) + nthreads * 100);
    const auto NR = mat.nrow(), NC = mat.ncol();

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::WriteCompressedSparseMatrixOptions params;
            params.num_threads = nthreads;
            tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
        }

        {
            H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
            auto dhandle = fhandle.openDataSet("matrix/data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

            auto dspace = dhandle.getSpace();
            hsize_t dims, maxdims;
            dspace.getSimpleExtentDims(&dims, &maxdims);
            EXPECT_EQ(maxdims, dims);

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

        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }

    // But we can always force it to a float.
    {
        {
            H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
            auto mhandle = fhandle.createGroup("matrix");
            tatami_hdf5::WriteCompressedSparseMatrixOptions fparam;
            fparam.num_threads = nthreads;
            fparam.data_name = "fp_data"; // We also change the dataset names to provide some more test coverage.
            fparam.index_name = "fp_indices";
            fparam.ptr_name = "fp_indptr";
            fparam.data_type = tatami_hdf5::WriteStorageType::DOUBLE;
            tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, fparam);
        }

        {
            H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
            auto dhandle = fhandle.openDataSet("matrix/fp_data");
            EXPECT_EQ(dhandle.getDataType().getClass(), H5T_FLOAT);
        }

        auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/fp_data", "matrix/fp_indices", "matrix/fp_indptr", false);
        tatami_test::test_simple_row_access(*reloaded, mat);
    }
}

TEST_P(WriteCompressedSparseMatrixSignedDataTypeTest, OnePass) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    auto mat = simulate(type, /* seed = */ int(type) + nthreads * 101);
    const auto NR = mat.nrow(), NC = mat.ncol();

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        params.two_pass = false;
        params.data_type = type;
        params.index_type = tatami_hdf5::WriteStorageType::INT32;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto dhandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(dhandle.getDataType().getClass(), H5T_INTEGER);

        auto dspace = dhandle.getSpace();
        hsize_t dims, maxdims;
        dspace.getSimpleExtentDims(&dims, &maxdims);
        EXPECT_EQ(maxdims, H5S_UNLIMITED);

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

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixIndexTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, int> > {
protected:
    static auto simulate(tatami_hdf5::WriteStorageType type, unsigned long long seed) {
        const std::size_t NC = 10;
        std::size_t NR = 200;
        if (type == tatami_hdf5::WriteStorageType::UINT16) {
            NR = 2000;
        } else if (type == tatami_hdf5::WriteStorageType::UINT32) {
            NR = 100000;
        }

        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.05;
            opt.lower = -100;
            opt.upper = 100;
            opt.seed = seed;
            return opt;
        }());

        // Make sure at least the largest index is large enough to exceed the immediately preceding type.
        EXPECT_FALSE(triplets.index.empty());
        auto maxIt = std::max_element(triplets.index.begin(), triplets.index.end());
        *maxIt = NR - 1;

        return tatami::CompressedSparseMatrix<
            double,
            int,
            decltype(triplets.data),
            decltype(triplets.index),
            decltype(triplets.indptr)
        >(
            NR,
            NC,
            std::move(triplets.data),
            std::move(triplets.index),
            std::move(triplets.indptr),
            /* row = */ false
        );
    }
};

TEST_P(WriteCompressedSparseMatrixIndexTypeTest, TwoPass) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    auto mat = simulate(type, /* seed = */ int(type) + 11 * nthreads);
    const auto NR = mat.nrow(), NC = mat.ncol();

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto ihandle = fhandle.openDataSet("matrix/indices");
        EXPECT_EQ(ihandle.getDataType().getClass(), H5T_INTEGER);

        auto ispace = ihandle.getSpace();
        hsize_t dims, maxdims;
        ispace.getSimpleExtentDims(&dims, &maxdims);
        EXPECT_EQ(maxdims, dims);

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

    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

TEST_P(WriteCompressedSparseMatrixIndexTypeTest, OnePass) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto nthreads = std::get<1>(params);

    auto mat = simulate(type, /* seed = */ int(type) + 22 * nthreads);
    const auto NR = mat.nrow(), NC = mat.ncol();

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        params.data_type = tatami_hdf5::WriteStorageType::DOUBLE;
        params.index_type = type;
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle, params);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto ihandle = fhandle.openDataSet("matrix/indices");
        EXPECT_EQ(ihandle.getDataType().getClass(), H5T_INTEGER);

        auto ispace = ihandle.getSpace();
        hsize_t dims, maxdims;
        ispace.getSimpleExtentDims(&dims, &maxdims);
        EXPECT_EQ(maxdims, H5S_UNLIMITED);

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

    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::FLOAT, 10));
    EXPECT_ANY_THROW(tatami_hdf5::choose_index_type(tatami_hdf5::WriteStorageType::DOUBLE, 10));
}

/*****************************************
 *****************************************/

class WriteCompressedSparseMatrixPtrTypeTest : public ::testing::TestWithParam<std::tuple<tatami_hdf5::WriteStorageType, bool, int> > {
protected:
    static auto simulate(unsigned long long seed) {
        const std::size_t NC = 10;
        std::size_t NR = 200;

        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.05;
            opt.lower = -100;
            opt.upper = 100;
            opt.seed = seed;
            return opt;
        }());

        return tatami::CompressedSparseMatrix<
            double,
            int,
            decltype(triplets.data),
            decltype(triplets.index),
            decltype(triplets.indptr)
        >(
            NR,
            NC,
            std::move(triplets.data),
            std::move(triplets.index),
            std::move(triplets.indptr),
            /* row = */ false
        );
    }
};

TEST_P(WriteCompressedSparseMatrixPtrTypeTest, Check) {
    auto params = GetParam();
    auto type = std::get<0>(params);
    auto two_pass = std::get<1>(params);
    auto nthreads = std::get<2>(params);

    auto mat = simulate(/* seed = */ int(type) + 10 * two_pass + 100 * nthreads);
    const auto NR = mat.nrow(), NC = mat.ncol();

    // Dumping it.
    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::WriteCompressedSparseMatrixOptions params;
        params.num_threads = nthreads;
        params.two_pass = two_pass;
        params.data_type = tatami_hdf5::WriteStorageType::DOUBLE;
        params.index_type = tatami_hdf5::WriteStorageType::INT32;
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

        // Also checking whether we did a two pass or not.
        auto dhandle = fhandle.openDataSet("matrix/data");
        auto dspace = dhandle.getSpace();
        hsize_t dims, maxdims;
        dspace.getSimpleExtentDims(&dims, &maxdims);
        if (two_pass) {
            EXPECT_EQ(maxdims, dims);
        } else {
            EXPECT_EQ(maxdims, H5S_UNLIMITED);
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
        ::testing::Values(false, true), // one or two pass.
        ::testing::Values(1, 3)
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

    EXPECT_ANY_THROW(tatami_hdf5::choose_ptr_type(tatami_hdf5::WriteStorageType::FLOAT, 10));
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

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle); // Dumping it with the overload that assumes default parameters.
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto phandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(phandle.getDataType().getClass(), H5T_FLOAT);
        H5::FloatType ptype(phandle);
        EXPECT_EQ(ptype.getSize(), 8);
    }

    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

TEST(WriteCompressedSparseMatrix, SinglePrecision) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<float, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());
    tatami::CompressedSparseColumnMatrix<float, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto phandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(phandle.getDataType().getClass(), H5T_FLOAT);
        H5::FloatType ptype(phandle);
        EXPECT_EQ(ptype.getSize(), 4);
    }

    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<float, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

/*****************************************
 *****************************************/

TEST(WriteCompressedSparseMatrix, NonFiniteTwoPass) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());

    // Rounding so that it would ordinarily be detected as integer, but also adding an Inf.
    for (auto& x : triplets.data) {
        x = std::round(x);
    }
    ASSERT_FALSE(triplets.data.empty());
    triplets.data.front() = std::numeric_limits<double>::infinity();

    tatami::CompressedSparseColumnMatrix<double, int> mat(NR, NC, std::move(triplets.data), std::move(triplets.index), std::move(triplets.indptr));

    auto fpath = temp_file_path("tatami-write-test.h5");
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(&mat, mhandle);
    }

    {
        H5::H5File fhandle(fpath, H5F_ACC_RDONLY);
        auto phandle = fhandle.openDataSet("matrix/data");
        EXPECT_EQ(phandle.getDataType().getClass(), H5T_FLOAT);
        H5::FloatType ptype(phandle);
        EXPECT_EQ(ptype.getSize(), 8);
    }

    auto reloaded = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, "matrix/data", "matrix/indices", "matrix/indptr", false);
    tatami_test::test_simple_row_access(*reloaded, mat);
}

TEST(WriteCompressedSparseMatrix, NonFiniteOnePass) {
    const size_t NR = 200, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());

    ASSERT_FALSE(triplets.data.empty());
    triplets.data.front() = std::numeric_limits<double>::infinity();

    tatami::CompressedSparseMatrix<
        double,
        int,
        decltype(triplets.data),
        decltype(triplets.index),
        decltype(triplets.indptr)
    > mat(
        NR,
        NC,
        std::move(triplets.data),
        std::move(triplets.index),
        std::move(triplets.indptr),
        /* by_row = */ false
    );

    auto fpath = temp_file_path("tatami-write-test.h5");

    tatami_hdf5::WriteCompressedSparseMatrixOptions opt;
    opt.data_type = tatami_hdf5::WriteStorageType::INT32;
    opt.index_type = tatami_hdf5::WriteStorageType::UINT8;

    // Sparse fails.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        EXPECT_ANY_THROW(tatami_hdf5::write_compressed_sparse_matrix(mat, mhandle, opt));
    }

    // Dense fails.
    auto dmat = tatami::convert_to_dense<double, int>(mat, true, {});
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        EXPECT_ANY_THROW(tatami_hdf5::write_compressed_sparse_matrix(*dmat, mhandle, opt));
    }
}

TEST(WriteCompressedSparseMatrix, TooLargeIndexOnePass) {
    const size_t NR = 300, NC = 100;
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NC, NR, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());

    ASSERT_FALSE(triplets.index.empty());
    triplets.index.back() = NR - 1;

    tatami::CompressedSparseMatrix<
        double,
        int,
        decltype(triplets.data),
        decltype(triplets.index),
        decltype(triplets.indptr)
    > mat(
        NR,
        NC,
        std::move(triplets.data),
        std::move(triplets.index),
        std::move(triplets.indptr),
        /* by_row = */ false
    );

    auto fpath = temp_file_path("tatami-write-test.h5");

    tatami_hdf5::WriteCompressedSparseMatrixOptions opt;
    opt.data_type = tatami_hdf5::WriteStorageType::INT32;
    opt.index_type = tatami_hdf5::WriteStorageType::UINT8;

    // Sparse fails.
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(mat, mhandle, opt);
    }

    // Dense fails.
    auto dmat = tatami::convert_to_dense<double, int>(mat, true, {});
    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto mhandle = fhandle.createGroup("matrix");
        tatami_hdf5::write_compressed_sparse_matrix(*dmat, mhandle, opt);
    }
}
