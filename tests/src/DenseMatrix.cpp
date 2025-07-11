#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/DenseMatrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "temp_file_path.h"

#include <vector>
#include <random>
#include <algorithm>

class DenseMatrixTestCore {
public:
    static constexpr size_t NR = 200, NC = 100;

    typedef std::tuple<std::pair<int, int>, double> SimulationParameters;

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
            ::testing::Values(0, 0.01, 0.1) // cache fraction multiplier
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

        auto chunk_sizes = std::get<0>(params);
        auto cache_fraction = std::get<1>(params);

        fpath = temp_file_path("tatami-dense-test.h5");
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);

        hsize_t dims[2];
        dims[0] = NR;
        dims[1] = NC;
        H5::DataSpace dspace(2, dims);
        H5::DataType dtype(H5::PredType::NATIVE_DOUBLE);

        H5::DSetCreatPropList plist(H5::DSetCreatPropList::DEFAULT.getId());
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
        auto values = tatami_test::simulate_vector<double>(NR * NC, [&]{
            tatami_test::SimulateVectorOptions opt;
            opt.lower = 0;
            opt.upper = 100;
            opt.seed = chunk_sizes.first * chunk_sizes.second + 100 * cache_fraction;
            return opt;
        }());
        dhandle.write(values.data(), H5::PredType::NATIVE_DOUBLE);

        ref.reset(new tatami::DenseRowMatrix<double, int>(NR, NC, std::move(values)));
        tref.reset(new tatami::DelayedTranspose<double, int>(ref));

        tatami_hdf5::DenseMatrixOptions opt;
        opt.maximum_cache_size = static_cast<double>(NR * NC) * cache_fraction * static_cast<double>(sizeof(double));
        opt.require_minimum_cache = (cache_fraction > 0);

        mat.reset(new tatami_hdf5::DenseMatrix<double, int>(fpath, name, false, opt));
        tmat.reset(new tatami_hdf5::DenseMatrix<double, int>(fpath, name, true, opt));
        return;
    }
};

#ifndef TATAMI_HDF5_TEST_PARALLEL_ONLY
/*************************************
 *************************************/

class DenseMatrixUtilsTest : public ::testing::Test, public DenseMatrixTestCore {};

TEST_F(DenseMatrixUtilsTest, Basic) {
    {
        assemble(SimulationParameters(std::pair(10, 10), 10));
        EXPECT_EQ(mat->nrow(), NR);
        EXPECT_EQ(mat->ncol(), NC);
        EXPECT_FALSE(mat->is_sparse());
        EXPECT_EQ(mat->is_sparse_proportion(), 0);
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

class DenseMatrixAccessFullTest :
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions> >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseMatrixAccessFullTest, Basic) {
    auto opts = tatami_test::convert_test_access_options(std::get<1>(GetParam()));
    tatami_test::test_full_access(*mat, *ref, opts);
    tatami_test::test_full_access(*tmat, *tref, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixAccessFullTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations()
    )
);

/*************************************
 *************************************/

class DenseSlicedTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseSlicedTest, Basic) {
    auto tparam = GetParam();
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto block = std::get<2>(tparam);
    tatami_test::test_block_access(*mat, *ref, block.first, block.second, opts);
    tatami_test::test_block_access(*tmat, *tref, block.first, block.second, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseSlicedTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.5), 
            std::make_pair(0.25, 0.5), 
            std::make_pair(0.51, 0.4)
        )
    )
);

/*************************************
 *************************************/

class DenseIndexedTest : 
    public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >,
    public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(DenseIndexedTest, Basic) {
    auto tparam = GetParam();
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto index = std::get<2>(tparam);
    tatami_test::test_indexed_access(*mat, *ref, index.first, index.second, opts);
    tatami_test::test_indexed_access(*tmat, *tref, index.first, index.second, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseIndexedTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(), 
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 0.2), 
            std::make_pair(0.11, 0.11),
            std::make_pair(0.4, 0.15)
        )
    )
);

/*************************************
 *************************************/

class DenseMatrixCacheTypeTest : 
    public ::testing::TestWithParam<std::tuple<bool, bool> >, 
    public DenseMatrixTestCore {};

TEST_P(DenseMatrixCacheTypeTest, CastToInt) {
    assemble(SimulationParameters(std::make_pair(9, 13), 1));

    tatami_hdf5::DenseMatrix<double, int, int> mat(fpath, name, false);
    auto altref = tatami::convert_to_dense<double, int, int>(*ref, true, {});

    auto tparam = GetParam();
    tatami_test::TestAccessOptions opts;
    opts.use_row = std::get<0>(tparam);
    opts.use_oracle = std::get<1>(tparam);

    tatami_test::test_full_access(mat, *altref, opts);
    tatami_test::test_block_access(mat, *altref, 0.2, 0.55, opts);
    tatami_test::test_indexed_access(mat, *altref, 0.3, 0.2, opts);
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixCacheTypeTest,
    ::testing::Combine(
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);

/*************************************
 *************************************/
#endif

class DenseMatrixParallelTest : public ::testing::TestWithParam<std::tuple<DenseMatrixTestCore::SimulationParameters, bool, bool> >, public DenseMatrixTestCore {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }

    template<bool oracle_>
    static void compare_sums(bool row, const tatami::Matrix<double, int>* testmat, const tatami::Matrix<double, int>* refmat) {
        size_t dim = (row ? refmat->nrow() : refmat->ncol());
        size_t otherdim = (row ? refmat->ncol() : refmat->nrow());
        std::vector<double> computed(dim), expected(dim);

        tatami::parallelize([&](size_t, int start, int len) -> void {
            auto ext = [&]() {
                if constexpr(oracle_) {
                    return tatami::consecutive_extractor<false>(testmat, row, start, len);
                } else {
                    return testmat->dense(row, tatami::Options());
                }
            }();

            auto rext = [&]() {
                if constexpr(oracle_) {
                    return tatami::consecutive_extractor<false>(refmat, row, start, len);
                } else {
                    return refmat->dense(row, tatami::Options());
                }
            }();

            std::vector<double> buffer(otherdim), rbuffer(otherdim);
            for (int i = start; i < start + len; ++i) {
                auto ptr = ext->fetch(i, buffer.data());
                auto rptr = rext->fetch(i, rbuffer.data());
                computed[i] = std::accumulate(ptr, ptr + otherdim, 0.0);
                expected[i] = std::accumulate(rptr, rptr + otherdim, 0.0);
            }
        }, dim, 3); // throw it over three threads.

        EXPECT_EQ(computed, expected);
    }
};

TEST_P(DenseMatrixParallelTest, Simple) {
    auto param = GetParam();
    bool row = std::get<1>(param);
    bool oracle = std::get<2>(param);

    if (oracle) {
        compare_sums<true>(row, mat.get(), ref.get());
        compare_sums<true>(row, tmat.get(), tref.get());
    } else {
        compare_sums<false>(row, mat.get(), ref.get());
        compare_sums<false>(row, tmat.get(), tref.get());
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseMatrix,
    DenseMatrixParallelTest,
    ::testing::Combine(
        DenseMatrixTestCore::create_combinations(),
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);
