#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami_hdf5/load_compressed_sparse_matrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "temp_file_path.h"

#include <vector>
#include <random>

TEST(LoadCompressedSparseMatrixTest, Basic) {
    const size_t NR = 200, NC = 100;

    // Dumping a sparse matrix.
    auto fpath = temp_file_path("tatami-sparse-test.h5");
    std::string name = "stuff";
    auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NR, NC, []{
        tatami_test::SimulateCompressedSparseOptions opt;
        opt.density = 0.05;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());

    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        auto ghandle = fhandle.createGroup(name);

        hsize_t dims = triplets.data.size();
        H5::DataSpace dspace(1, &dims);
        {
            H5::DataType dtype(H5::PredType::NATIVE_DOUBLE);
            auto dhandle = ghandle.createDataSet("data", dtype, dspace);
            dhandle.write(triplets.data.data(), H5::PredType::NATIVE_DOUBLE);
        }

        {
            H5::DataType dtype(H5::PredType::NATIVE_UINT16);
            auto dhandle = ghandle.createDataSet("index", dtype, dspace);
            dhandle.write(triplets.index.data(), H5::PredType::NATIVE_INT);
        }

        {
            hsize_t ncp1 = triplets.indptr.size();
            H5::DataSpace dspace(1, &ncp1);
            H5::DataType dtype(H5::PredType::NATIVE_UINT64);
            auto dhandle = ghandle.createDataSet("indptr", dtype, dspace);
            std::vector<uint64_t> copy(triplets.indptr.begin(), triplets.indptr.end()); // making a copy as size_t doesn't have a HDF5 datatype.
            dhandle.write(copy.data(), H5::PredType::NATIVE_UINT64);
        }
    }

    // Basic load as a CSR matrix (as rows are the primary dimension in this simulation)
    {
        auto mat = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true);
        tatami::CompressedSparseRowMatrix<
            double, 
            int, 
            decltype(triplets.data), 
            decltype(triplets.index), 
            decltype(triplets.indptr)
        > ref(NR, NC, triplets.data, triplets.index, triplets.indptr);

        tatami_test::test_simple_row_access(*mat, ref);
    }

    // Pretending it's a CSC matrix.
    {
        auto mat = tatami_hdf5::load_compressed_sparse_matrix<double, int>(NC, NR, fpath, name + "/data", name + "/index", name + "/indptr", false);
        tatami::CompressedSparseColumnMatrix<
            double, 
            int, 
            decltype(triplets.data), 
            decltype(triplets.index), 
            decltype(triplets.indptr)
        > ref(NC, NR, triplets.data, triplets.index, triplets.indptr);

        tatami_test::test_simple_column_access(*mat, ref);
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

        std::vector<double> truncated = triplets.data;
        for (auto& x : truncated) {
            x = std::trunc(x);
        }

        tatami::CompressedSparseRowMatrix<
            double, 
            int, 
            decltype(truncated), 
            decltype(triplets.index), 
            decltype(triplets.indptr)
        > ref(NR, NC, std::move(truncated), triplets.index, triplets.indptr);

        tatami_test::test_simple_column_access(*mat, ref);
    }
}
