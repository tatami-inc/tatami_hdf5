#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/Hdf5CompressedSparseMatrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>

class Hdf5SparseMatrixTestCore {
public:
    typedef std::tuple<int, double> SimulationParameters;

    static auto create_combinations() {
        return ::testing::Combine(
            ::testing::Values(0, 100), // chunk size
            ::testing::Values(0, 0.001, 0.01, 0.1) // cache size
        );
    }

protected:
    typedef std::tuple<std::pair<int, int>, SimulationParameters> FullSimulationParameters;

    inline static FullSimulationParameters last_params;

    inline static std::shared_ptr<tatami::Matrix<double, int> > ref, mat, tref, tmat;

    inline static std::string fpath, name;

    inline static tatami_test::CompressedSparseDetails<double> triplets;

    static void assemble(const FullSimulationParameters& params) {
        if (ref && params == last_params) {
            return;
        }
        last_params = params;

        auto dimensions = std::get<0>(params);
        auto sim_params = std::get<1>(params);

        size_t NR = dimensions.first;
        size_t NC = dimensions.second;
        auto chunk_size = std::get<0>(sim_params);
        auto cache_fraction = std::get<1>(sim_params);

        triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.05, 0, 100, /* seed = */ NR * NC + chunk_size + 100 * cache_fraction);

        // Generating the file.
        fpath = tatami_test::temp_file_path("tatami-sparse-test.h5");
        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
        name = "stuff";
        auto ghandle = fhandle.createGroup(name);

        H5::DSetCreatPropList plist(H5::DSetCreatPropList::DEFAULT.getId());
        if (chunk_size == 0) {
            plist.setLayout(H5D_CONTIGUOUS);
        } else {
            plist.setLayout(H5D_CHUNKED);
            hsize_t chunkdim = std::min(triplets.value.size(), static_cast<size_t>(chunk_size));
            plist.setChunk(1, &chunkdim);
        }

        hsize_t dims = triplets.value.size();
        H5::DataSpace dspace(1, &dims);
        {
            H5::DataType dtype(H5::PredType::NATIVE_DOUBLE);
            auto dhandle = ghandle.createDataSet("data", dtype, dspace, plist);
            dhandle.write(triplets.value.data(), H5::PredType::NATIVE_DOUBLE);
        }

        {
            H5::DataType dtype(H5::PredType::NATIVE_UINT16);
            auto dhandle = ghandle.createDataSet("index", dtype, dspace, plist);
            dhandle.write(triplets.index.data(), H5::PredType::NATIVE_INT);
        }

        {
            hsize_t ncp1 = triplets.ptr.size();
            H5::DataSpace dspace(1, &ncp1);
            H5::DataType dtype(H5::PredType::NATIVE_UINT64);
            auto dhandle = ghandle.createDataSet("indptr", dtype, dspace);
            dhandle.write(triplets.ptr.data(), H5::PredType::NATIVE_LONG);
        }

        // We limit the cache size to ensure that chunk management is not trivial.
        size_t actual_cache_size = static_cast<double>(NR * NC) * cache_fraction * static_cast<double>(sizeof(double) + sizeof(int));
        tatami_hdf5::Hdf5Options hopt;
        hopt.maximum_cache_size = actual_cache_size;
        hopt.require_minimum_cache = actual_cache_size > 0;

        mat.reset(new tatami_hdf5::Hdf5CompressedSparseMatrix<true, double, int>(
            NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", hopt
        )); 
        ref.reset(new tatami::CompressedSparseMatrix<true, double, int, decltype(triplets.value), decltype(triplets.index), decltype(triplets.ptr)>(
            NR, NC, triplets.value, triplets.index, triplets.ptr
        ));

        // Creating the transposed versions as well.
        tmat.reset(new tatami_hdf5::Hdf5CompressedSparseMatrix<false, double, int>(
            NC, NR, fpath, name + "/data", name + "/index", name + "/indptr", hopt
        )); 
        tref.reset(new tatami::CompressedSparseMatrix<false, double, int, decltype(triplets.value), decltype(triplets.index), decltype(triplets.ptr)>(
            NC, NR, triplets.value, triplets.index, triplets.ptr
        ));
    }
};

/*************************************
 *************************************/

class Hdf5SparseMatrixUtilsTest : public ::testing::Test, public Hdf5SparseMatrixTestCore {
    void SetUp() {
        assemble({ {200, 100}, { 50, 0.1 } });
    }
};

TEST_F(Hdf5SparseMatrixUtilsTest, Basic) {
    auto dimensions = std::get<0>(last_params);
    size_t NR = dimensions.first;
    size_t NC = dimensions.second;

    EXPECT_EQ(mat->nrow(), NR);
    EXPECT_EQ(mat->ncol(), NC);
    EXPECT_TRUE(mat->sparse());
    EXPECT_EQ(mat->sparse_proportion(), 1);
    EXPECT_TRUE(mat->prefer_rows());
    EXPECT_EQ(mat->prefer_rows_proportion(), 1);

    EXPECT_EQ(tmat->nrow(), NC);
    EXPECT_EQ(tmat->ncol(), NR);
    EXPECT_TRUE(tmat->sparse());
    EXPECT_EQ(tmat->sparse_proportion(), 1);
    EXPECT_FALSE(tmat->prefer_rows());
    EXPECT_EQ(tmat->prefer_rows_proportion(), 0);
}

/*************************************
 *************************************/

class Hdf5SparseMatrixFullAccessTest : 
    public ::testing::TestWithParam<std::tuple<Hdf5SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters> >, 
    public Hdf5SparseMatrixTestCore
{};

TEST_P(Hdf5SparseMatrixFullAccessTest, Primary) {
    auto param = GetParam(); 
    assemble({ {200, 100}, std::get<0>(param) });
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));

    if (tparams.use_row) {
        tatami_test::test_full_access(tparams, mat.get(), ref.get());
    } else {
        tatami_test::test_full_access(tparams, tmat.get(), tref.get());
    }
}

TEST_P(Hdf5SparseMatrixFullAccessTest, Secondary) {
    auto param = GetParam(); 
    assemble({ {50, 10}, std::get<0>(param) }); // much smaller for the secondary dimension.
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));

    if (tparams.use_row) {
        tatami_test::test_full_access(tparams, tmat.get(), tref.get());
    } else {
        tatami_test::test_full_access(tparams, mat.get(), ref.get());
    }
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5SparseMatrix,
    Hdf5SparseMatrixFullAccessTest,
    ::testing::Combine(
        Hdf5SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_parameter_combinations()
    )
);

/*************************************
 *************************************/

class Hdf5SparseMatrixBlockAccessTest : 
    public ::testing::TestWithParam<std::tuple<Hdf5SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, double> > >, 
    public Hdf5SparseMatrixTestCore
{};

TEST_P(Hdf5SparseMatrixBlockAccessTest, Primary) {
    auto param = GetParam(); 
    assemble({ {128, 256}, std::get<0>(param) });
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));
    auto block = std::get<2>(param);

    if (tparams.use_row) {
        size_t FIRST = block.first * ref->ncol(), LAST = block.second * ref->ncol();
        tatami_test::test_block_access(tparams, mat.get(), ref.get(), FIRST, LAST);
    } else {
        size_t FIRST = block.first * tref->nrow(), LAST = block.second * tref->nrow();
        tatami_test::test_block_access(tparams, tmat.get(), tref.get(), FIRST, LAST);
    }
}

TEST_P(Hdf5SparseMatrixBlockAccessTest, Secondary) {
    auto param = GetParam(); 
    assemble({ {10, 50}, std::get<0>(param) }); // much smaller for the secondary dimension.
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));
    auto block = std::get<2>(param);

    if (tparams.use_row) {
        size_t FIRST = block.first * tref->ncol(), LAST = block.second * tref->ncol();
        tatami_test::test_block_access(tparams, tmat.get(), tref.get(), FIRST, LAST);
    } else {
        size_t FIRST = block.first * ref->nrow(), LAST = block.second * ref->nrow();
        tatami_test::test_block_access(tparams, mat.get(), ref.get(), FIRST, LAST);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5SparseMatrix,
    Hdf5SparseMatrixBlockAccessTest,
    ::testing::Combine(
        Hdf5SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.333), 
            std::make_pair(0.222, 0.888), 
            std::make_pair(0.555, 1.0)
        )
    )
);

/*************************************
 *************************************/

class Hdf5SparseMatrixIndexedAccessTest :
    public ::testing::TestWithParam<std::tuple<Hdf5SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, int> > >, 
    public Hdf5SparseMatrixTestCore
{};

TEST_P(Hdf5SparseMatrixIndexedAccessTest, Primary) {
    auto param = GetParam(); 
    assemble({ {197, 125}, std::get<0>(param) });
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));
    auto index_info = std::get<2>(param);

    if (tparams.use_row) {
        size_t FIRST = index_info.first * ref->ncol(), STEP = index_info.second;
        tatami_test::test_indexed_access(tparams, mat.get(), ref.get(), FIRST, STEP);
    } else {
        size_t FIRST = index_info.first * tref->nrow(), STEP = index_info.second;
        tatami_test::test_indexed_access(tparams, tmat.get(), tref.get(), FIRST, STEP);
    }
}

TEST_P(Hdf5SparseMatrixIndexedAccessTest, Secondary) {
    auto param = GetParam(); 
    assemble({ {20, 30}, std::get<0>(param) }); // much smaller for the secondary dimension.
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));
    auto index_info = std::get<2>(param);

    if (tparams.use_row) {
        size_t FIRST = index_info.first * tref->ncol(), STEP = index_info.second; 
        tatami_test::test_indexed_access(tparams, tmat.get(), tref.get(), FIRST, STEP);
    } else {
        size_t FIRST = index_info.first * ref->nrow(), STEP = index_info.second;
        tatami_test::test_indexed_access(tparams, mat.get(), ref.get(), FIRST, STEP);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5SparseMatrix,
    Hdf5SparseMatrixIndexedAccessTest,
    ::testing::Combine(
        Hdf5SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 5), 
            std::make_pair(0.322, 8), 
            std::make_pair(0.455, 9)
        )
    )
);

/*************************************
 *************************************/

class Hdf5SparseMatrixReusePrimaryCacheTest : public ::testing::TestWithParam<std::tuple<double, int, int> >, public Hdf5SparseMatrixTestCore {
protected:
    std::vector<int> predictions;

    void SetUp() {
        auto params = GetParam();
        auto cache_fraction = std::get<0>(params);
        int NR = 150, NC = 200;
        assemble({ { NR, NC }, { 100, cache_fraction } });

        auto interval_jump = std::get<1>(params);
        auto interval_size = std::get<2>(params);

        // Repeated scans over the same area, to check for correct re-use of
        // cache elements. We scramble the interval to check that the
        // reordering of elements is done correctly in oracle mode.
        std::vector<int> interval(interval_size);
        std::iota(interval.begin(), interval.end(), 0);
        std::mt19937_64 rng(cache_fraction * 1000 + interval_size);

        for (int r0 = 0; r0 < NR; r0 += interval_jump) {
            std::shuffle(interval.begin(), interval.end(), rng);
            for (auto i : interval) {
                if (i + r0 < NR) {
                    predictions.push_back(i + r0);
                }
            }
        }
    }
};

TEST_P(Hdf5SparseMatrixReusePrimaryCacheTest, FullExtent) {
    auto rwork = ref->dense_row();
    auto mwork = mat->dense_row();
    auto mwork2 = mat->dense_row(std::make_unique<tatami::FixedViewOracle<int> >(predictions.data(), predictions.size()));
    auto full = ref->ncol();

    for (auto i : predictions) {
        auto expected = tatami_test::fetch(rwork.get(), i, full);
        auto observed = tatami_test::fetch(mwork.get(), i, full);
        EXPECT_EQ(observed, expected);
        auto observed2 = tatami_test::fetch(mwork2.get(), full);
        EXPECT_EQ(observed2, expected);
    }
}

TEST_P(Hdf5SparseMatrixReusePrimaryCacheTest, SlicedBounds) {
    auto full = ref->ncol();
    auto cstart = full * 0.25, clen = full * 0.5;

    auto rwork = ref->dense_row(cstart, clen);
    auto mwork = mat->dense_row(cstart, clen);
    auto mwork2 = mat->dense_row(std::make_unique<tatami::FixedViewOracle<int> >(predictions.data(), predictions.size()), cstart, clen);

    for (auto i : predictions) {
        auto expected = tatami_test::fetch(rwork.get(), i, clen);
        auto observed = tatami_test::fetch(mwork.get(), i, clen);
        EXPECT_EQ(observed, expected);
        auto observed2 = tatami_test::fetch(mwork2.get(), clen);
        EXPECT_EQ(observed2, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5SparseMatrix,
    Hdf5SparseMatrixReusePrimaryCacheTest,
    ::testing::Combine(
        ::testing::Values(0, 0.001, 0.01, 0.1), // cache size multiplier
        ::testing::Values(1, 3), // jump between intervals
        ::testing::Values(5, 10, 20) // reuse interval size
    )
);

/*************************************
 *************************************/

class Hdf5SparseMatrixCacheTypeTest : public ::testing::TestWithParam<std::tuple<bool, bool> >, public Hdf5SparseMatrixTestCore {};

TEST_P(Hdf5SparseMatrixCacheTypeTest, CastToInt) {
    int NR = 500;
    int NC = 200;
    double cache_fraction = 0.1;
    assemble({ { NR, NC }, { 100, cache_fraction } });

    tatami_hdf5::Hdf5Options hopt;
    hopt.maximum_cache_size = static_cast<double>(NR * NC) * cache_fraction * static_cast<double>(sizeof(int) + sizeof(uint16_t));
    tatami_hdf5::Hdf5CompressedSparseMatrix<true, double, int, int, uint16_t> mat(
        NR,
        NC,
        fpath, 
        name + "/data", 
        name + "/index", 
        name + "/indptr", 
        hopt
    );

    std::vector<int> vcasted(triplets.value.begin(), triplets.value.end());
    std::vector<uint16_t> icasted(triplets.index.begin(), triplets.index.end());
    tatami::CompressedSparseMatrix<true, double, int, decltype(vcasted), decltype(icasted), decltype(triplets.ptr)> ref(
        NR,
        NC, 
        std::move(vcasted),
        std::move(icasted),
        triplets.ptr
    );

    tatami_test::TestAccessParameters params;
    auto tparam = GetParam();
    params.use_row = std::get<0>(tparam);
    params.use_oracle = std::get<1>(tparam);

    tatami_test::test_full_access(params, &mat, &ref);
    tatami_test::test_block_access(params, &mat, &ref, 5, 20);
    tatami_test::test_indexed_access(params, &mat, &ref, 3, 5);
}

INSTANTIATE_TEST_SUITE_P(
    Hdf5SparseMatrix,
    Hdf5SparseMatrixCacheTypeTest,
    ::testing::Combine(
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);

/*************************************
 *************************************/

class Hdf5SparseMatrixParallelTest : public ::testing::TestWithParam<std::tuple<Hdf5SparseMatrixTestCore::SimulationParameters, bool, bool> >, public Hdf5SparseMatrixTestCore {
protected:
    void SetUp() {
        assemble({ { 100, 200 }, std::get<0>(GetParam()) });
    }

    template<bool oracle_>
    static void compare_sums(bool row, const tatami::Matrix<double, int>* testmat, const tatami::Matrix<double, int>* refmat) {
        size_t dim = (row ? refmat->nrow() : refmat->ncol());
        size_t otherdim = (row ? refmat->ncol() : refmat->nrow());
        std::vector<double> computed(dim), expected(dim);

        tatami::parallelize([&](size_t, int start, int len) -> void {
            auto ext = [&]() {
                if constexpr(oracle_) {
                    return tatami::consecutive_extractor<true>(testmat, row, start, len);
                } else {
                    return testmat->sparse(row, tatami::Options());
                }
            }();

            auto rext = [&]() {
                if constexpr(oracle_) {
                    return tatami::consecutive_extractor<true>(refmat, row, start, len);
                } else {
                    return refmat->sparse(row, tatami::Options());
                }
            }();

            std::vector<double> vbuffer(otherdim), rvbuffer(otherdim);
            std::vector<int> ibuffer(otherdim), ribuffer(otherdim);
            for (int i = start; i < start + len; ++i) {
                auto out = ext->fetch(i, vbuffer.data(), ibuffer.data());
                auto rout = rext->fetch(i, rvbuffer.data(), ribuffer.data());
                computed[i] = std::accumulate(out.value, out.value + out.number, 0.0);
                expected[i] = std::accumulate(rout.value, rout.value + rout.number, 0.0);
            }
        }, dim, 3); // throw it over three threads.

        EXPECT_EQ(computed, expected);
    }
};

TEST_P(Hdf5SparseMatrixParallelTest, Simple) {
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
    Hdf5SparseMatrix,
    Hdf5SparseMatrixParallelTest,
    ::testing::Combine(
        Hdf5SparseMatrixTestCore::create_combinations(),
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);
