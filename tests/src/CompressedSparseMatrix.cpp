#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/CompressedSparseMatrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>

static void dump_to_file(const tatami_test::CompressedSparseDetails<double>& triplets, const std::string& fpath, const std::string& name, int chunk_size = 50) {
    H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
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
}

static tatami_hdf5::CompressedSparseMatrixOptions create_options(size_t NR, size_t NC, double cache_fraction) {
    // We limit the cache size to ensure that chunk management is not trivial.
    size_t actual_cache_size = static_cast<double>(NR * NC) * cache_fraction * static_cast<double>(sizeof(double) + sizeof(int));
    tatami_hdf5::CompressedSparseMatrixOptions hopt;
    hopt.maximum_cache_size = actual_cache_size;
    return hopt;
}

class SparseMatrixTestCore {
public:
    typedef std::tuple<double> SimulationParameters;

    static auto create_combinations() {
        return ::testing::Combine(
            ::testing::Values(0, 0.01, 0.1) // cache fraction multiplier
        );
    }

protected:
    typedef std::tuple<std::pair<int, int>, SimulationParameters> FullSimulationParameters;

    inline static FullSimulationParameters last_params;

    inline static std::shared_ptr<tatami::Matrix<double, int> > ref, mat, tref, tmat;

    static void assemble(const FullSimulationParameters& params) {
        if (ref && params == last_params) {
            return;
        }
        last_params = params;

        auto dimensions = std::get<0>(params);
        auto sim_params = std::get<1>(params);

        size_t NR = dimensions.first;
        size_t NC = dimensions.second;
        auto cache_fraction = std::get<0>(sim_params);

        auto triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.15, 0, 100, /* seed = */ NR * NC + 100 * cache_fraction);
        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file.
        auto fpath = tatami_test::temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(
            NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt
        )); 
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int, decltype(triplets.value), decltype(triplets.index), decltype(triplets.ptr)>(
            NR, NC, triplets.value, triplets.index, triplets.ptr
        ));

        // Creating the transposed versions as well.
        tmat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(
            NC, NR, fpath, name + "/data", name + "/index", name + "/indptr", false, hopt
        )); 
        tref.reset(new tatami::CompressedSparseColumnMatrix<double, int, decltype(triplets.value), decltype(triplets.index), decltype(triplets.ptr)>(
            NC, NR, triplets.value, triplets.index, triplets.ptr, false
        ));
    }
};

#ifndef TATAMI_HDF5_TEST_PARALLEL_ONLY
/*************************************
 *************************************/

class SparseMatrixUtilsTest : public ::testing::Test, public SparseMatrixTestCore {
    void SetUp() {
        assemble({ {200, 100}, { 0.1 } });
    }
};

TEST_F(SparseMatrixUtilsTest, Basic) {
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

class SparseMatrixFullAccessTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters> >, 
    public SparseMatrixTestCore
{};

TEST_P(SparseMatrixFullAccessTest, Primary) {
    auto param = GetParam(); 
    assemble({ {200, 100}, std::get<0>(param) });
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));

    if (tparams.use_row) {
        tatami_test::test_full_access(tparams, mat.get(), ref.get());
    } else {
        tatami_test::test_full_access(tparams, tmat.get(), tref.get());
    }
}

TEST_P(SparseMatrixFullAccessTest, Secondary) {
    auto param = GetParam(); 
    assemble({ {50, 40}, std::get<0>(param) }); // smaller for the secondary dimension.
    auto tparams = tatami_test::convert_access_parameters(std::get<1>(param));

    if (tparams.use_row) {
        tatami_test::test_full_access(tparams, tmat.get(), tref.get());
    } else {
        tatami_test::test_full_access(tparams, mat.get(), ref.get());
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixFullAccessTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_parameter_combinations()
    )
);

/*************************************
 *************************************/

class SparseMatrixBlockAccessTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, double> > >, 
    public SparseMatrixTestCore
{};

TEST_P(SparseMatrixBlockAccessTest, Primary) {
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

TEST_P(SparseMatrixBlockAccessTest, Secondary) {
    auto param = GetParam(); 
    assemble({ {30, 50}, std::get<0>(param) }); // smaller for the secondary dimension.
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
    SparseMatrix,
    SparseMatrixBlockAccessTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
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

class SparseMatrixIndexedAccessTest :
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessParameters, std::pair<double, int> > >, 
    public SparseMatrixTestCore
{};

TEST_P(SparseMatrixIndexedAccessTest, Primary) {
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

TEST_P(SparseMatrixIndexedAccessTest, Secondary) {
    auto param = GetParam(); 
    assemble({ {35, 40}, std::get<0>(param) }); // much smaller for the secondary dimension.
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
    SparseMatrix,
    SparseMatrixIndexedAccessTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_parameter_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 5), 
            std::make_pair(0.322, 4), 
            std::make_pair(0.455, 3)
        )
    )
);

/*************************************
 *************************************/

class SparseMatrixReusePrimaryCacheTest : 
    public ::testing::TestWithParam<std::tuple<double, int, int> >, 
    public SparseMatrixTestCore 
{
protected:
    std::vector<int> predictions;

    void SetUp() {
        auto params = GetParam();
        auto cache_fraction = std::get<0>(params);
        int NR = 150, NC = 200;
        assemble({ { NR, NC }, { cache_fraction } });

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

TEST_P(SparseMatrixReusePrimaryCacheTest, FullExtent) {
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

TEST_P(SparseMatrixReusePrimaryCacheTest, Block) {
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

TEST_P(SparseMatrixReusePrimaryCacheTest, Indexed) {
    auto full = ref->ncol();
    std::vector<int> chosen;
    for (int i = 10; i < full; i += 7) {
        chosen.push_back(i);
    }

    auto rwork = ref->dense_row(chosen);
    auto mwork = mat->dense_row(chosen);
    auto mwork2 = mat->dense_row(std::make_unique<tatami::FixedViewOracle<int> >(predictions.data(), predictions.size()), chosen);

    int clen = chosen.size();
    for (auto i : predictions) {
        auto expected = tatami_test::fetch(rwork.get(), i, clen);
        auto observed = tatami_test::fetch(mwork.get(), i, clen);
        EXPECT_EQ(observed, expected);
        auto observed2 = tatami_test::fetch(mwork2.get(), clen);
        EXPECT_EQ(observed2, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixReusePrimaryCacheTest,
    ::testing::Combine(
        ::testing::Values(0, 0.01, 0.1), // cache fraction
        ::testing::Values(1, 3), // jump between intervals
        ::testing::Values(5, 10, 20) // reuse interval size
    )
);

/*************************************
 *************************************/

class SparseMatrixCacheTypeTest : public ::testing::TestWithParam<std::tuple<double, bool, bool> > {
protected:
    inline static std::shared_ptr<tatami::Matrix<double, int> > ref, mat;

    inline static std::tuple<double, bool> last_params;

    void SetUp() {
        auto params = GetParam();
        double cache_fraction = std::get<0>(params);
        bool use_row = std::get<1>(params);

        std::tuple<double, bool> copy_params(cache_fraction, use_row);
        if (ref && last_params == copy_params) {
            return;
        }
        last_params = copy_params;

        // Using a smaller matrix for the secondary extractors, for faster testing;
        // otherwise this will take a much longer time.
        int NR = (use_row ? 500 : 41);
        int NC = (use_row ? 200 : 58);

        auto triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.15, 0, 100, /* seed = */ NR * NC + 100 * cache_fraction);
        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file.
        auto fpath = tatami_test::temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int, int, uint16_t>(
            NR,
            NC,
            fpath, 
            name + "/data", 
            name + "/index", 
            name + "/indptr", 
            true,
            hopt
        ));

        std::vector<int> vcasted(triplets.value.begin(), triplets.value.end());
        std::vector<uint16_t> icasted(triplets.index.begin(), triplets.index.end());
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int, decltype(vcasted), decltype(icasted), decltype(triplets.ptr)>(
            NR,
            NC, 
            std::move(vcasted),
            std::move(icasted),
            triplets.ptr
        ));
    }
};

TEST_P(SparseMatrixCacheTypeTest, CastToInt) {
    tatami_test::TestAccessParameters params;
    auto tparam = GetParam();
    params.use_row = std::get<1>(tparam);
    params.use_oracle = std::get<2>(tparam);

    tatami_test::test_full_access(params, mat.get(), ref.get());

    auto len = params.use_row ? ref->ncol() : ref->nrow();
    tatami_test::test_block_access(params, mat.get(), ref.get(), len * 0.25, len * 0.7);
    tatami_test::test_indexed_access(params, mat.get(), ref.get(), len * 0.1, 4);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixCacheTypeTest,
    ::testing::Combine(
        ::testing::Values(0, 0.01, 0.1), // cache fraction
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);

/*************************************
 *************************************/

class SparseMatrixUncompressedTest : public ::testing::TestWithParam<std::tuple<double, bool, bool> > {
protected:
    inline static std::shared_ptr<tatami::Matrix<double, int> > ref, mat;

    inline static std::tuple<double, bool> last_params;

    void SetUp() {
        auto params = GetParam();
        double cache_fraction = std::get<0>(params);
        bool use_row = std::get<1>(params);

        std::tuple<double, bool> copy_params(cache_fraction, use_row);
        if (ref && last_params == copy_params) {
            return;
        }
        last_params = copy_params;

        // Using a smaller matrix for the secondary extractors, for faster testing;
        // otherwise this will take a much longer time.
        int NR = (use_row ? 333 : 50);
        int NC = (use_row ? 444 : 40);

        auto triplets = tatami_test::simulate_sparse_compressed<double>(NR, NC, 0.2, 0, 100, /* seed = */ NR * NC + cache_fraction * 100);
        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file; chunk_size = 0 indicates that we want it uncompressed.
        auto fpath = tatami_test::temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name, /* chunk_size = */ 0);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt));
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, std::move(triplets.value), std::move(triplets.index), triplets.ptr));
    }
};

TEST_P(SparseMatrixUncompressedTest, Basic) {
    tatami_test::TestAccessParameters params;
    auto tparam = GetParam();
    params.use_row = std::get<1>(tparam);
    params.use_oracle = std::get<2>(tparam);

    tatami_test::test_full_access(params, mat.get(), ref.get());

    auto len = params.use_row ? ref->ncol() : ref->nrow();
    tatami_test::test_block_access(params, mat.get(), ref.get(), len * 0.12, len * 0.8);
    tatami_test::test_indexed_access(params, mat.get(), ref.get(), len * 0.05, 2);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixUncompressedTest,
    ::testing::Combine(
        ::testing::Values(0, 0.01, 0.1), // cache fraction
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);

/*************************************
 *************************************/

class SparseMatrixNearEmptyTest : public ::testing::TestWithParam<std::tuple<double, bool, bool> > {
protected:
    inline static std::shared_ptr<tatami::Matrix<double, int> > ref, mat;

    inline static std::tuple<double, bool> last_params;

    void SetUp() {
        auto params = GetParam();
        double cache_fraction = std::get<0>(params);
        bool use_row = std::get<1>(params);

        std::tuple<double, bool> copy_params(cache_fraction, use_row);
        if (ref && last_params == copy_params) {
            return;
        }
        last_params = copy_params;

        // Make a diagonal matrix with every second element missing.  This
        // checks all the shortcuts when there are no elements to be extracted
        // for a particular dimension element. If use_row=false, we reduce the
        // matrix size for faster testing with the slow secondary extractors.
        int NC = use_row ? 200 : 20;
        int NR = NC;

        tatami_test::CompressedSparseDetails<double> triplets;
        triplets.ptr.resize(NC + 1);
        std::mt19937_64 rng(NC * NR * 100 * cache_fraction);
        std::normal_distribution ndist;
        for (int i = 1; i < NC; i += 2) {
            triplets.index.push_back(i);
            triplets.value.push_back(ndist(rng));
            ++(triplets.ptr[i]);
        }
        for (int p = 1; p <= NC; ++p) {
            triplets.ptr[p] += triplets.ptr[p - 1];
        }

        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file.
        auto fpath = tatami_test::temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt));
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, std::move(triplets.value), std::move(triplets.index), triplets.ptr));
    }
};

TEST_P(SparseMatrixNearEmptyTest, Basic) {
    tatami_test::TestAccessParameters params;
    auto tparam = GetParam();
    params.use_row = std::get<1>(tparam);
    params.use_oracle = std::get<2>(tparam);

    tatami_test::test_full_access(params, mat.get(), ref.get());

    auto len = params.use_row ? ref->ncol() : ref->nrow();
    tatami_test::test_block_access(params, mat.get(), ref.get(), len * 0.17, len * 0.6);
    tatami_test::test_indexed_access(params, mat.get(), ref.get(), len * 0.33, 3);
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixNearEmptyTest,
    ::testing::Combine(
        ::testing::Values(0, 0.01, 0.1), // cache fraction
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);

/*************************************
 *************************************/
#endif

class SparseMatrixParallelTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, bool, bool> >, 
    public SparseMatrixTestCore 
{
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

TEST_P(SparseMatrixParallelTest, Simple) {
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
    SparseMatrix,
    SparseMatrixParallelTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        ::testing::Values(true, false), // row access
        ::testing::Values(true, false)  // oracle usage
    )
);
