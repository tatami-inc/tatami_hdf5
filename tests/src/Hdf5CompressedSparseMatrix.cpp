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

///*************************************
// *************************************/
//
//Class Hdf5SparseBasicCacheTest : public ::testing::TestWithParam<double>, public Hdf5SparseMatrixTestMethods {};
//
//TEST_P(Hdf5SparseBasicCacheTest, LruRandomized) {
//    // Check that the LRU cache works as expected with totally random access.
//    const size_t NR = 100, NC = 150; 
//    dump(NR, NC);
//    int cache_size = compute_cache_size(NR, NC, GetParam());
//    auto hopt = custom_options(cache_size);
//
//    {
//        auto mat = create_matrix<true>(NR, NC, hopt);
//        auto ref = create_reference<true>(NR, NC);
//
//        std::mt19937_64 rng(cache_size * 123);
//        auto m_ext = mat.dense_row();
//        auto r_ext = ref.dense_row();
//        for (size_t r0 = 0, end = NR * 10; r0 < end; ++r0) {
//            auto r = rng() % NR;
//            EXPECT_EQ(m_ext->fetch(r), r_ext->fetch(r));
//        }
//    }
//
//    {
//        auto mat = create_matrix<false>(NR, NC, hopt);
//        auto ref = create_reference<false>(NR, NC);
//
//        std::mt19937_64 rng(cache_size * 456);
//        auto m_ext = mat.dense_column();
//        auto r_ext = ref.dense_column();
//        for (size_t c0 = 0, end = NR * 10; c0 < end; ++c0) {
//            auto c = rng() % NR;
//            EXPECT_EQ(m_ext->fetch(c), r_ext->fetch(c));
//        }
//    }
//}
//
//TEST_P(Hdf5SparseBasicCacheTest, SimpleOracle) {
//    // Checking that access with an oracle behaves as expected.
//    const size_t NR = 189, NC = 123; 
//    dump(NR, NC);
//    int cache_size = compute_cache_size(NR, NC, GetParam());
//    auto hopt = custom_options(cache_size);
//
//    {
//        auto mat = create_matrix<true>(NR, NC, hopt);
//        auto ref = create_reference<true>(NR, NC);
//
//        tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, false); // consecutive
//        tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, false, 0.3 * NR, 0.5 * NR); // consecutive with bounds
//
//        tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, true); // randomized
//        tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, true, 0.2 * NR, 0.6 * NR); // randomized with bounds
//
//        // Oracle-based extraction still works if we turn off value or index extraction.
//        auto rwork = ref.sparse_row();
//
//        tatami::Options opt;
//        opt.sparse_extract_value = false;
//        auto iwork = mat.sparse_row(opt);
//        iwork->set_oracle(std::make_unique<tatami::ConsecutiveOracle<int> >(0, NR));
//
//        opt.sparse_extract_index = false;
//        auto bwork = mat.sparse_row(opt);
//        bwork->set_oracle(std::make_unique<tatami::ConsecutiveOracle<int> >(0, NR));
//
//        opt.sparse_extract_value = true;
//        auto vwork = mat.sparse_row(opt);
//        vwork->set_oracle(std::make_unique<tatami::ConsecutiveOracle<int> >(0, NR));
//
//        for (size_t r = 0; r < NR; ++r) {
//            auto rout = rwork->fetch(r);
//
//            auto iout = iwork->fetch(r);
//            EXPECT_EQ(iout.index, rout.index);
//            EXPECT_EQ(iout.value.size(), 0);
//
//            auto vout = vwork->fetch(r);
//            EXPECT_EQ(vout.value, rout.value);
//            EXPECT_EQ(vout.index.size(), 0);
//
//            auto bout = bwork->fetch(r, NULL, NULL);
//            EXPECT_EQ(bout.number, rout.value.size());
//        }
//    }
//
//    {
//        auto mat = create_matrix<false>(NR, NC, hopt);
//        auto ref = create_reference<false>(NR, NC);
//
//        tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, false); // consecutive
//        tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, false, 0.1 * NR, 0.7 * NR); // consecutive with bounds
//
//        tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, true); // randomized
//        tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, true, 0.25 * NR, 0.6 * NR); // randomized with bounds
//    }
//}
//
//TEST_P(Hdf5SparseBasicCacheTest, Repeated) {
//    size_t NR = 199, NC = 288;
//    dump(NR, NC);
//    int cache_size = compute_cache_size(NR, NC, GetParam());
//    auto hopt = custom_options(cache_size);
//
//    // Check that we re-use the cache effectively when no new elements are
//    // requested; no additional extractions from file should occur.
//    std::vector<int> predictions;
//    for (size_t i = 0; i < NR * 10; ++i) {
//        predictions.push_back(i % 2);
//    }
//
//    auto mat = create_matrix<true>(NR, NC, hopt);
//    auto ref = create_reference<true>(NR, NC);
//
//    auto rwork = ref.dense_row();
//    auto mwork = mat.dense_row();
//    auto mwork_o = mat.dense_row();
//    mwork_o->set_oracle(std::make_unique<tatami::FixedOracle<int> >(predictions.data(), predictions.size()));
//
//    for (auto i : predictions) {
//        auto expected = rwork->fetch(i);
//        EXPECT_EQ(mwork->fetch(i), expected);
//        EXPECT_EQ(mwork_o->fetch(i), expected);
//    }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    Hdf5SparseMatrix,
//    Hdf5SparseBasicCacheTest,
//    ::testing::Values(0, 0.001, 0.01, 0.1) 
//);
//
///*************************************
// *************************************/
//
//Class Hdf5SparseReuseCacheTest : public ::testing::TestWithParam<std::tuple<double, int, int> >, public Hdf5SparseMatrixTestMethods {
//Protected:
//    size_t NR = 150, NC = 200; 
//    std::shared_ptr<tatami::NumericMatrix> mat, ref;
//    std::vector<int> predictions;
//
//    template<class Params_>
//    void assemble(const Params_& params) {
//        dump(NR, NC);
//
//        double cache_multiplier = std::get<0>(params);
//        int interval_jump = std::get<1>(params);
//        int interval_size = std::get<2>(params);
//
//        int cache_size = compute_cache_size(NR, NC, cache_multiplier);
//        auto hopt = custom_options(cache_size);
//
//        mat.reset(new tatami_hdf5::Hdf5CompressedSparseMatrix<true, double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", hopt));
//        ref.reset(new tatami::CompressedSparseMatrix<
//            true, 
//            double, 
//            int, 
//            decltype(triplets.value), 
//            decltype(triplets.index), 
//            decltype(triplets.ptr)
//        >(NR, NC, triplets.value, triplets.index, triplets.ptr));
//
//        // Repeated scans over the same area, to check for correct re-use of
//        // cache elements. We scramble the interval to check that the
//        // reordering of elements is done correctly in oracle mode.
//        std::vector<int> interval(interval_size);
//        std::iota(interval.begin(), interval.end(), 0);
//        std::mt19937_64 rng(cache_size + interval_size);
//
//        for (size_t r0 = 0; r0 < NR; r0 += interval_jump) {
//            std::shuffle(interval.begin(), interval.end(), rng);
//            for (auto i : interval) {
//                if (i + r0 < NR) {
//                    predictions.push_back(i + r0);
//                }
//            }
//        }
//    }
//};
//
//TEST_P(Hdf5SparseReuseCacheTest, FullExtent) {
//    assemble(GetParam());
//
//    auto rwork = ref->dense_row();
//    auto mwork = mat->dense_row();
//    auto mwork_o = mat->dense_row();
//    mwork_o->set_oracle(std::make_unique<tatami::FixedOracle<int> >(predictions.data(), predictions.size()));
//
//    for (auto i : predictions) {
//        auto expected = rwork->fetch(i);
//        EXPECT_EQ(mwork->fetch(i), expected);
//        EXPECT_EQ(mwork_o->fetch(i), expected);
//    }
//}
//
//TEST_P(Hdf5SparseReuseCacheTest, SlicedBounds) {
//    assemble(GetParam());
//
//    // Testing that the extraction bounds are actually re-used during extraction.
//    // This requires that the elements leave the cache and are then requested again;
//    // if it's still in the cache, the non-bounded cache element will just be used.
//    auto cstart = NC * 0.25, clen = NC * 0.5;
//    tatami::Options opt;
//    opt.cache_for_reuse = true;
//
//    auto rwork = ref->dense_row(cstart, clen, opt);
//    auto mwork = mat->dense_row(cstart, clen, opt);
//    auto mwork_o = mat->dense_row(cstart, clen, opt);
//
//    // Doing one final scan across all rows to re-request elements that have
//    // likely left the cache, to force them to be reloaded with bounds.
//    for (size_t r = 0; r < NR; ++r) {
//        predictions.push_back(r);
//    }
//    mwork_o->set_oracle(std::make_unique<tatami::FixedOracle<int> >(predictions.data(), predictions.size()));
//
//    for (auto i : predictions) {
//        auto expected = rwork->fetch(i);
//        EXPECT_EQ(mwork->fetch(i), expected);
//        EXPECT_EQ(mwork_o->fetch(i), expected);
//    }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    Hdf5SparseMatrix,
//    Hdf5SparseReuseCacheTest,
//    ::testing::Combine(
//        ::testing::Values(0, 0.001, 0.01, 0.1), // cache size multiplier
//        ::testing::Values(1, 3), // jump between intervals
//        ::testing::Values(5, 10, 20) // reuse interval size
//    )
//);
//
///*************************************
// *************************************/
//
//Class Hdf5SparseApplyTest : public ::testing::TestWithParam<double>, public Hdf5SparseMatrixTestMethods {};
//
//TEST_P(Hdf5SparseApplyTest, Basic) {
//    // Just putting it through its paces for correct oracular parallelization via apply.
//    size_t NR = 500;
//    size_t NC = 200;
//    dump(NR, NC);
//    int cache_size = compute_cache_size(NR, NC, GetParam());
//    auto hopt = custom_options(cache_size);
//
//    auto mat = create_matrix<true>(NR, NC, hopt);
//    auto ref = create_reference<true>(NR, NC);
//
//    EXPECT_EQ(tatami::row_sums(&mat), tatami::row_sums(&ref));
//    EXPECT_EQ(tatami::column_sums(&mat), tatami::column_sums(&ref));
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    Hdf5SparseMatrix,
//    Hdf5SparseApplyTest,
//    ::testing::Values(0, 0.001, 0.01, 0.1) // cache size multiplier
//);
//
///*************************************
// *************************************/
//
//Class Hdf5SparseCacheTypeTest : public ::testing::Test, public Hdf5SparseMatrixTestMethods {};
//
//TEST_F(Hdf5SparseCacheTypeTest, CastToInt) {
//    size_t NR = 500;
//    size_t NC = 200;
//    dump(NR, NC);
//
//    tatami_hdf5::Hdf5CompressedSparseMatrix<true, double, int, int, uint16_t> mat(
//        NR,
//        NC,
//        fpath, 
//        name + "/data", 
//        name + "/index", 
//        name + "/indptr", 
//        tatami_hdf5::Hdf5Options()
//    );
//
//    std::vector<int> vcasted(triplets.value.begin(), triplets.value.end());
//    std::vector<uint16_t> icasted(triplets.index.begin(), triplets.index.end());
//    tatami::CompressedSparseMatrix<true, double, int, decltype(vcasted), decltype(icasted), decltype(triplets.ptr)> ref(
//        NR,
//        NC, 
//        std::move(vcasted),
//        std::move(icasted),
//        triplets.ptr
//    );
//
//    EXPECT_EQ(tatami::row_sums(&mat), tatami::row_sums(&ref));
//    EXPECT_EQ(tatami::column_sums(&mat), tatami::column_sums(&ref));
//}
