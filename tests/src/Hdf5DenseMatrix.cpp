#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/Hdf5DenseMatrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>
#include <algorithm>

class Hdf5DenseMatrixTestCore {
public:
    static constexpr size_t NR = 200, NC = 100;

protected:
    typedef std::tuple<std::pair<int, int>, int> SimulationParameters;

    inline static SimulationParameters last_params;

public:
    static auto create_combinations() {
        return ::testing::Combine(
            ::testing::Values(
                std::pair<int, int>(NR, 1),
                std::pair<int, int>(1, NC),
                std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
                std::make_pair(19, 7),
                std::make_pair(11, 11),
                std::make_pair(0, 0)
            ),
            ::testing::Values(0, 1, 1000, 10000) // cache size, in elements.
        );
    }

protected:
    inline static std::shared_ptr<tatami::Matrix<double, int> > ref, mat, tref, tmat;

    inline static std::string fpath, name;

    static void assemble(const SimulationParameters& params) {
        if (ref && params == last_params) {
            return;
        }
        last_params = params;

        fpath = tatami_test::temp_file_path("tatami-dense-test.h5");
        tatami_test::remove_file_path(fpath);
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);

        hsize_t dims[2];
        dims[0] = NR;
        dims[1] = NC;
        H5::DataSpace dspace(2, dims);
        H5::DataType dtype(H5::PredType::NATIVE_UINT8);

        H5::DSetCreatPropList plist(H5::DSetCreatPropList::DEFAULT.getId());
        const auto& chunk_sizes = std::get<0>(params);
        if (chunk_sizes.first == 0) {
            plist.setLayout(H5D_CONTIGUOUS);
        } else {
            plist.setLayout(H5D_CHUNKED);
            hsize_t chunkdims[2];
            chunkdims[0] = chunk_sizes.first;
            chunkdims[1] = chunk_sizes.second;
            plist.setChunk(2, chunkdims);
        }

        name = "stuff";
        auto dhandle = fhandle.createDataSet(name, dtype, dspace, plist);
        auto values = tatami_test::simulate_dense_vector<double>(NR * NC, 0, 100);
        dhandle.write(values.data(), H5::PredType::NATIVE_DOUBLE);

        ref.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, std::move(values)));
        tref.reset(new tatami::DelayedTranspose<double, int>(ref));

        auto cache_size = std::get<1>(params);
        tatami_hdf5::Hdf5Options opt;
        opt.maximum_cache_size = sizeof(double) * cache_size;
        opt.require_minimum_cache = (cache_size > 0);

        mat.reset(new tatami_hdf5::Hdf5DenseMatrix<double, int, false>(fpath, name));
        tmat.reset(new tatami_hdf5::Hdf5DenseMatrix<double, int, true>(fpath, name));
        return;
    }
};

/*************************************
 *************************************/

class Hdf5DenseUtilsTest : public ::testing::Test, public Hdf5DenseMatrixTestCore {};

TEST_F(Hdf5DenseUtilsTest, Basic) {
    {
        assemble(SimulationParameters(std::pair(10, 10), 10));
        EXPECT_EQ(mat->nrow(), NR);
        EXPECT_EQ(mat->ncol(), NC);
        EXPECT_FALSE(mat->sparse());
        EXPECT_EQ(mat->sparse_proportion(), 0);
        EXPECT_TRUE(mat->prefer_rows());
        EXPECT_EQ(mat->prefer_rows_proportion(), 1);

        EXPECT_EQ(tmat->nrow(), NC);
        EXPECT_EQ(tmat->ncol(), NR);
        EXPECT_FALSE(tmat->prefer_rows());
        EXPECT_EQ(tmat->prefer_rows_proportion(), 0);
    }

    {
        // First dimension is compromised, switching to the second dimension.
        assemble(SimulationParameters(std::pair<int, int>(NR, 1), 10));
        EXPECT_FALSE(mat->prefer_rows());
        EXPECT_TRUE(tmat->prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        assemble(SimulationParameters(std::pair<int, int>(1, NC), 10));
        EXPECT_TRUE(mat->prefer_rows());
        EXPECT_FALSE(tmat->prefer_rows());
    }
}

/*************************************
 *************************************/

class Hdf5DenseAccessFullTest :
    public ::testing::TestWithParam<std::tuple<Hdf5DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters> >,
    public Hdf5DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(Hdf5DenseAccessFullTest, Basic) {
    auto params = tatami_test::convert_access_parameters(std::get<1>(GetParam()));
    tatami_test::test_full_access(params, mat.get(), ref.get());
    tatami_test::test_full_access(params, tmat.get(), tref.get());
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5DenseMatrix,
    Hdf5DenseAccessFullTest,
    ::testing::Combine(
        Hdf5DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_parameter_combinations()
    )
);

/*************************************
 *************************************/

class Hdf5DenseSlicedTest : 
    public ::testing::TestWithParam<std::tuple<Hdf5DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, double> > >,
    public Hdf5DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(Hdf5DenseSlicedTest, Basic) {
    auto tparam = GetParam();
    auto params = tatami_test::convert_access_parameters(std::get<1>(tparam));
    auto block = std::get<2>(tparam);

    {
        auto len = params.use_row ? ref->ncol() : ref->nrow();
        size_t FIRST = block.first * len, LAST = block.second * len;
        tatami_test::test_block_access(params, mat.get(), ref.get(), FIRST, LAST);
    }

    {
        auto len = params.use_row ? tref->ncol() : tref->nrow();
        size_t FIRST = block.first * len, LAST = block.second * len;
        tatami_test::test_block_access(params, tmat.get(), tref.get(), FIRST, LAST);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5DenseMatrix,
    Hdf5DenseSlicedTest,
    ::testing::Combine(
        Hdf5DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.5), 
            std::make_pair(0.25, 0.75), 
            std::make_pair(0.51, 1.0)
        )
    )
);

/*************************************
 *************************************/

class Hdf5DenseIndexedTest : 
    public ::testing::TestWithParam<std::tuple<Hdf5DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, int> > >,
    public Hdf5DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(Hdf5DenseIndexedTest, Basic) {
    auto tparam = GetParam();
    auto params = tatami_test::convert_access_parameters(std::get<1>(tparam));
    auto index = std::get<2>(tparam);

    {
        auto len = params.use_row ? ref->ncol() : ref->nrow();
        size_t FIRST = index.first * len, STEP = index.second;
        tatami_test::test_indexed_access(params, mat.get(), ref.get(), FIRST, STEP);
    }

    {
        auto len = params.use_row ? tref->ncol() : tref->nrow();
        size_t FIRST = index.first * len, STEP = index.second;
        tatami_test::test_indexed_access(params, tmat.get(), tref.get(), FIRST, STEP);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5DenseMatrix,
    Hdf5DenseIndexedTest,
    ::testing::Combine(
        Hdf5DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 5), 
            std::make_pair(0.11, 9),
            std::make_pair(0.4, 7)
        )
    )
);

/*************************************
 *************************************/

class Hdf5DenseCacheTypeTest : public ::testing::TestWithParam<std::tuple<bool, bool> >, public Hdf5DenseMatrixTestCore {};

TEST_P(Hdf5DenseCacheTypeTest, CastToInt) {
    assemble(SimulationParameters(std::make_pair(9, 13), 1));

    tatami_hdf5::Hdf5DenseMatrix<double, int, false, int> mat(fpath, name);
    auto altref = tatami::convert_to_dense<true, double, int, int>(ref.get());

    tatami_test::TestAccessParameters params;
    auto tparam = GetParam();
    params.use_row = std::get<0>(tparam);
    params.use_oracle = std::get<1>(tparam);

    tatami_test::test_full_access(params, &mat, altref.get());
    tatami_test::test_block_access(params, &mat, altref.get(), 5, 20);
    tatami_test::test_indexed_access(params, &mat, altref.get(), 3, 5);
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5DenseMatrix,
    Hdf5DenseCacheTypeTest,
    ::testing::Combine(
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);
