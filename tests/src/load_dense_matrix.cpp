#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami_hdf5/load_dense_matrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "temp_file_path.h"

#include <vector>
#include <random>

TEST(LoadDenseMatrixTest, Basic) {
    size_t NR = 200, NC = 100;
    auto fpath = temp_file_path("tatami-dense-test.h5");
    std::string name = "stuff";

    std::vector<double> values = tatami_test::simulate_vector<double>(NR * NC, []{
        tatami_test::SimulateVectorOptions opt;
        opt.lower = 0;
        opt.upper = 100;
        return opt;
    }());

    {
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        hsize_t dims[2];
        dims[0] = NR;
        dims[1] = NC;
        H5::DataSpace dspace(2, dims);
        H5::DataType dtype(H5::PredType::NATIVE_DOUBLE);
        auto dhandle = fhandle.createDataSet(name, dtype, dspace);
        dhandle.write(values.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Basic load as a row-major matrix. 
    {
        auto mat = tatami_hdf5::load_dense_matrix<double, int>(fpath, name, false);
        tatami::DenseRowMatrix<double, int> ref(NR, NC, values);
        tatami_test::test_simple_row_access(*mat, ref);
    }

    // Pretending it's a column-major matrix.
    {
        auto mat = tatami_hdf5::load_dense_matrix<double, int, std::vector<double> >(fpath, name, true);
        tatami::DenseColumnMatrix<double, int> ref(NC, NR, values);
        tatami_test::test_simple_column_access(*mat, ref);
    }

    // Trying a different storage type.
    {
        auto mat = tatami_hdf5::load_dense_matrix<double, int, std::vector<int32_t> >(fpath, name, false);

        std::vector<double> truncated = values;
        for (auto& x : truncated) {
            x = std::trunc(x);
        }
        tatami::DenseRowMatrix<double, int> ref(NR, NC, std::move(truncated));

        tatami_test::test_simple_column_access(*mat, ref);
    }
}
