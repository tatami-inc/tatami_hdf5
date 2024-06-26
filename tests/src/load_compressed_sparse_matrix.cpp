#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami_hdf5/load_compressed_sparse_matrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>

TEST(LoadCompressedSparseMatrixTest, Basic) {
    const size_t NR = 200, NC = 100;

    // Dumping a sparse matrix.
    auto fpath = tatami_test::temp_file_path("tatami-sparse-test.h5");
    std::string name = "stuff";
    auto triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.05, 0, 100);

    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto ghandle = fhandle.createGroup(name);

        hsize_t dims = triplets.value.size();
        H5::DataSpace dspace(1, &dims);
        {
            H5::DataType dtype(H5::PredType::NATIVE_DOUBLE);
            auto dhandle = ghandle.createDataSet("data", dtype, dspace);
            dhandle.write(triplets.value.data(), H5::PredType::NATIVE_DOUBLE);
        }

        {
            H5::DataType dtype(H5::PredType::NATIVE_UINT16);
            auto dhandle = ghandle.createDataSet("index", dtype, dspace);
            dhandle.write(triplets.index.data(), H5::PredType::NATIVE_INT);
        }

        {
            hsize_t ncp1 = triplets.ptr.size();
            H5::DataSpace dspace(1, &ncp1);
            H5::DataType dtype(H5::PredType::NATIVE_UINT64);
            auto dhandle = ghandle.createDataSet("indptr", dtype, dspace);
            dhandle.write(triplets.ptr.data(), H5::PredType::NATIVE_LONG);
        }
    }

    // Basic load as a CSR matrix (as rows are the primary dimension in this simulation)
    {
        auto mat = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true);
        tatami::CompressedSparseRowMatrix<
            double, 
            int, 
            decltype(triplets.value), 
            decltype(triplets.index), 
            decltype(triplets.ptr)
        > ref(NR, NC, triplets.value, triplets.index, triplets.ptr);

        tatami_test::test_simple_row_access(mat.get(), &ref);
    }

    // Pretending it's a CSC matrix.
    {
        auto mat = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NC, NR, fpath, name + "/data", name + "/index", name + "/indptr", false);
        tatami::CompressedSparseColumnMatrix<
            double, 
            int, 
            decltype(triplets.value), 
            decltype(triplets.index), 
            decltype(triplets.ptr)
        > ref(NC, NR, triplets.value, triplets.index, triplets.ptr);

        tatami_test::test_simple_column_access(mat.get(), &ref);
    }

    // Trying a variety of storage types.
    {
        auto mat = tatami_hdf5::load_compressed_sparse_matrix<
            double, 
            int,
            std::vector<uint16_t>,
            std::vector<uint32_t>,
            std::vector<uint64_t>
        >(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true);

        std::vector<double> truncated = triplets.value;
        for (auto& x : truncated) {
            x = std::trunc(x);
        }

        tatami::CompressedSparseRowMatrix<
            double, 
            int, 
            decltype(truncated), 
            decltype(triplets.index), 
            decltype(triplets.ptr)
        > ref(NR, NC, std::move(truncated), triplets.index, triplets.ptr);

        tatami_test::test_simple_column_access(mat.get(), &ref);
    }
}
