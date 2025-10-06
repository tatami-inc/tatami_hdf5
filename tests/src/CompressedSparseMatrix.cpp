#include <gtest/gtest.h>
#include "custom_parallel.h" // make sure this is included before tatami libs.

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "tatami_hdf5/CompressedSparseMatrix.hpp"

#include "tatami_test/tatami_test.hpp"
#include "temp_file_path.h"

#include <vector>
#include <random>

static void dump_to_file(const tatami_test::SimulateCompressedSparseResult<double, int>& triplets, const std::string& fpath, const std::string& name, hsize_t chunk_size = 50) {
    H5::H5File fhandle(fpath, H5F_ACC_TRUNC);
    auto ghandle = fhandle.createGroup(name);

    auto create_plist = [&](hsize_t max_size) -> H5::DSetCreatPropList {
        if (chunk_size > max_size) {
            chunk_size = max_size;
        }
        H5::DSetCreatPropList plist(H5::DSetCreatPropList::DEFAULT.getId());
        if (chunk_size == 0) {
            plist.setLayout(H5D_CONTIGUOUS);
        } else {
            plist.setLayout(H5D_CHUNKED);
            plist.setChunk(1, &chunk_size);
        }
        return plist;
    };

    {
        hsize_t dims = triplets.data.size();
        H5::DataSpace dspace(1, &dims);
        H5::DataType dtype(H5::PredType::NATIVE_DOUBLE);
        auto dhandle = ghandle.createDataSet("data", dtype, dspace, create_plist(dims));
        dhandle.write(triplets.data.data(), H5::PredType::NATIVE_DOUBLE);
    }

    {
        hsize_t dims = triplets.index.size();
        H5::DataSpace dspace(1, &dims);
        H5::DataType dtype(H5::PredType::NATIVE_UINT16);
        auto dhandle = ghandle.createDataSet("index", dtype, dspace, create_plist(dims));
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

        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NR, NC, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.15;
            opt.lower = 0;
            opt.upper = 100;
            opt.seed = NR * NC + 100 * cache_fraction;
            return opt;
        }());
        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file.
        auto fpath = temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(
            NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt
        )); 
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int, decltype(triplets.data), decltype(triplets.index), decltype(triplets.indptr)>(
            NR, NC, triplets.data, triplets.index, triplets.indptr, false
        ));

        // Creating the transposed versions as well.
        tmat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(
            NC, NR, fpath, name + "/data", name + "/index", name + "/indptr", false, hopt
        )); 
        tref.reset(new tatami::CompressedSparseColumnMatrix<double, int, decltype(triplets.data), decltype(triplets.index), decltype(triplets.indptr)>(
            NC, NR, triplets.data, triplets.index, triplets.indptr, false
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
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions> >, 
    public SparseMatrixTestCore
{};

TEST_P(SparseMatrixFullAccessTest, Primary) {
    auto tparam = GetParam(); 
    assemble({ {200, 100}, std::get<0>(tparam) });
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));

    if (opts.use_row) {
        tatami_test::test_full_access(*mat, *ref, opts);
    } else {
        tatami_test::test_full_access(*tmat, *tref, opts);
    }
}

TEST_P(SparseMatrixFullAccessTest, Secondary) {
    auto tparam = GetParam(); 
    assemble({ {50, 40}, std::get<0>(tparam) }); // smaller for the secondary dimension.
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));

    if (opts.use_row) {
        tatami_test::test_full_access(*tmat, *tref, opts);
    } else {
        tatami_test::test_full_access(*mat, *ref, opts);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixFullAccessTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_options_combinations()
    )
);

/*************************************
 *************************************/

class SparseMatrixBlockAccessTest : 
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >, 
    public SparseMatrixTestCore
{};

TEST_P(SparseMatrixBlockAccessTest, Primary) {
    auto tparam = GetParam(); 
    assemble({ {128, 256}, std::get<0>(tparam) });
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto block = std::get<2>(tparam);

    if (opts.use_row) {
        tatami_test::test_block_access(*mat, *ref, block.first, block.second, opts);
    } else {
        tatami_test::test_block_access(*tmat, *tref, block.first, block.second, opts);
    }
}

TEST_P(SparseMatrixBlockAccessTest, Secondary) {
    auto tparam = GetParam(); 
    assemble({ {30, 50}, std::get<0>(tparam) }); // smaller for the secondary dimension.
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto block = std::get<2>(tparam);

    if (opts.use_row) {
        tatami_test::test_block_access(*tmat, *tref, block.first, block.second, opts);
    } else {
        tatami_test::test_block_access(*mat, *ref, block.first, block.second, opts);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixBlockAccessTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.0, 0.333), 
            std::make_pair(0.222, 0.666), 
            std::make_pair(0.555, 0.444)
        )
    )
);

/*************************************
 *************************************/

class SparseMatrixIndexedAccessTest :
    public ::testing::TestWithParam<std::tuple<SparseMatrixTestCore::SimulationParameters, tatami_test::StandardTestAccessOptions, std::pair<double, double> > >, 
    public SparseMatrixTestCore
{};

TEST_P(SparseMatrixIndexedAccessTest, Primary) {
    auto tparam = GetParam(); 
    assemble({ {197, 125}, std::get<0>(tparam) });
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto index_info = std::get<2>(tparam);

    if (opts.use_row) {
        tatami_test::test_indexed_access(*mat, *ref, index_info.first, index_info.second, opts);
    } else {
        tatami_test::test_indexed_access(*tmat, *tref, index_info.first, index_info.second, opts);
    }
}

TEST_P(SparseMatrixIndexedAccessTest, Secondary) {
    auto tparam = GetParam(); 
    assemble({ {35, 40}, std::get<0>(tparam) }); // much smaller for the secondary dimension.
    auto opts = tatami_test::convert_test_access_options(std::get<1>(tparam));
    auto index_info = std::get<2>(tparam);

    if (opts.use_row) {
        tatami_test::test_indexed_access(*tmat, *tref, index_info.first, index_info.second, opts);
    } else {
        tatami_test::test_indexed_access(*mat, *ref, index_info.first, index_info.second, opts);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseMatrix,
    SparseMatrixIndexedAccessTest,
    ::testing::Combine(
        SparseMatrixTestCore::create_combinations(),
        tatami_test::standard_test_access_options_combinations(),
        ::testing::Values(
            std::make_pair(0.3, 0.2), 
            std::make_pair(0.322, 0.25), 
            std::make_pair(0.455, 0.33)
        )
    )
);

class SparseEmptyIndexedTest : public ::testing::Test, public SparseMatrixTestCore {};

TEST_F(SparseEmptyIndexedTest, Basic) {
    assemble(FullSimulationParameters{ { 200, 100 }, { 0.01 } });

    // Also checking for correct behavior if indices are empty.
    auto empty = std::make_shared<std::vector<int> >();
    auto ext = mat->sparse_row(empty);
    auto res = tatami_test::fetch(*ext, 0, 0);
    EXPECT_TRUE(res.index.empty());

    auto text = tmat->sparse_row(empty);
    auto tres = tatami_test::fetch(*text, 0, 0);
    EXPECT_TRUE(tres.index.empty());
}

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
        auto expected = tatami_test::fetch(*rwork, i, full);
        auto observed = tatami_test::fetch(*mwork, i, full);
        EXPECT_EQ(observed, expected);
        auto observed2 = tatami_test::fetch(*mwork2, full);
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
        auto expected = tatami_test::fetch(*rwork, i, clen);
        auto observed = tatami_test::fetch(*mwork, i, clen);
        EXPECT_EQ(observed, expected);
        auto observed2 = tatami_test::fetch(*mwork2, clen);
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
        auto expected = tatami_test::fetch(*rwork, i, clen);
        auto observed = tatami_test::fetch(*mwork, i, clen);
        EXPECT_EQ(observed, expected);
        auto observed2 = tatami_test::fetch(*mwork2, clen);
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

        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NR, NC, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.15;
            opt.lower = 0;
            opt.upper = 100;
            opt.seed = NR * NC + 100 * cache_fraction;
            return opt;
        }());
        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file.
        auto fpath = temp_file_path("tatami-sparse-test.h5");
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

        std::vector<int> vcasted(triplets.data.begin(), triplets.data.end());
        std::vector<uint16_t> icasted(triplets.index.begin(), triplets.index.end());
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int, decltype(vcasted), decltype(icasted), decltype(triplets.indptr)>(
            NR,
            NC, 
            std::move(vcasted),
            std::move(icasted),
            triplets.indptr
        ));
    }
};

TEST_P(SparseMatrixCacheTypeTest, CastToInt) {
    auto tparam = GetParam();
    tatami_test::TestAccessOptions opts;
    opts.use_row = std::get<1>(tparam);
    opts.use_oracle = std::get<2>(tparam);

    tatami_test::test_full_access(*mat, *ref, opts);
    tatami_test::test_block_access(*mat, *ref, 0.25, 0.7, opts);
    tatami_test::test_indexed_access(*mat, *ref, 0.1, 0.25, opts);
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

        auto triplets = tatami_test::simulate_compressed_sparse<double, int>(NR, NC, [&]{
            tatami_test::SimulateCompressedSparseOptions opt;
            opt.density = 0.2;
            opt.lower = 0;
            opt.upper = 100;
            opt.seed = NR * NC + cache_fraction * 100;
            return opt;
        }());
        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file; chunk_size = 0 indicates that we want it uncompressed.
        auto fpath = temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name, /* chunk_size = */ 0);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt));
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, std::move(triplets.data), std::move(triplets.index), triplets.indptr));
    }
};

TEST_P(SparseMatrixUncompressedTest, Basic) {
    auto tparam = GetParam();
    tatami_test::TestAccessOptions opts;
    opts.use_row = std::get<1>(tparam);
    opts.use_oracle = std::get<2>(tparam);

    tatami_test::test_full_access(*mat, *ref, opts);
    tatami_test::test_block_access(*mat, *ref, 0.12, 0.8, opts);
    tatami_test::test_indexed_access(*mat, *ref, 0.05, 0.25, opts);
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

        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.indptr.resize(NC + 1);
        std::mt19937_64 rng(NC * NR * 100 * cache_fraction);
        std::normal_distribution ndist;
        for (int i = 1; i < NC; i += 2) {
            triplets.index.push_back(i);
            triplets.data.push_back(ndist(rng));
            ++(triplets.indptr[i]);
        }
        for (int p = 1; p <= NC; ++p) {
            triplets.indptr[p] += triplets.indptr[p - 1];
        }

        auto hopt = create_options(NR, NC, cache_fraction);

        // Generating the file.
        auto fpath = temp_file_path("tatami-sparse-test.h5");
        std::string name = "stuff";
        dump_to_file(triplets, fpath, name);

        mat.reset(new tatami_hdf5::CompressedSparseMatrix<double, int>(NR, NC, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt));
        ref.reset(new tatami::CompressedSparseRowMatrix<double, int>(NR, NC, std::move(triplets.data), std::move(triplets.index), triplets.indptr));
    }
};

TEST_P(SparseMatrixNearEmptyTest, Basic) {
    auto tparam = GetParam();
    tatami_test::TestAccessOptions opts;
    opts.use_row = std::get<1>(tparam);
    opts.use_oracle = std::get<2>(tparam);

    tatami_test::test_full_access(*mat, *ref, opts);
    tatami_test::test_block_access(*mat, *ref, 0.17, 0.6, opts);
    tatami_test::test_indexed_access(*mat, *ref, 0.33, 0.3, opts);
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

TEST(SparseMatrix, PrimaryOracularCacheSize) {
    std::string placeholder = "placeholder";

    int NR = 8, NC = 20;
    std::vector<hsize_t> pointers { 0, 10, 11, 20, 22, 30, 33, 40, 44 };
    std::size_t slab_size = 10000;
    int max_non_zeros = 4;

    tatami_hdf5::CompressedSparseMatrix_internal::MatrixDetails<int> deets(placeholder, placeholder, placeholder, NR, NC, pointers, slab_size, max_non_zeros, {});
    auto chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_cache_size_for_primary_oracular<double, int>(deets, true, true);
    EXPECT_EQ(chosen, 44);
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_cache_size_for_primary_oracular<double, int>(deets, false, false);
    EXPECT_EQ(chosen, static_cast<std::size_t>(-1));

    deets.slab_cache_size = 100;
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_cache_size_for_primary_oracular<double, int>(deets, true, true);
    EXPECT_EQ(chosen, 8);
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_cache_size_for_primary_oracular<double, int>(deets, true, false);
    EXPECT_EQ(chosen, 12);
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_cache_size_for_primary_oracular<double, int>(deets, false, true);
    EXPECT_EQ(chosen, 25);

    deets.slab_cache_size = 10;
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_cache_size_for_primary_oracular<double, int>(deets, true, true);
    EXPECT_EQ(chosen, max_non_zeros); // must be at least equal to the maximum number of non-zeros in any primary dimension element.
}

TEST(SparseMatrix, MyopicSecondaryChunkLength) {
    std::string placeholder = "placeholder";

    int NR = 10, NC = 20;
    std::vector<hsize_t> pointers { 0, 10, 11, 20, 22, 30, 33, 40, 44, 50, 55 };
    std::size_t slab_size = 10000;
    int max_non_zeros = 4;

    tatami_hdf5::CompressedSparseMatrix_internal::MatrixDetails<int> deets(placeholder, placeholder, placeholder, NR, NC, pointers, slab_size, max_non_zeros, {});
    auto chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_chunk_length_for_myopic_secondary<double, int>(deets, NR, true, true);
    EXPECT_EQ(chosen, 20);
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_chunk_length_for_myopic_secondary<double, int>(deets, NR, false, false);
    EXPECT_EQ(chosen, 20);

    deets.slab_cache_size = 0;
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_chunk_length_for_myopic_secondary<double, int>(deets, NR, true, true);
    EXPECT_EQ(chosen, 1);

    deets.slab_cache_size = 1000;
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_chunk_length_for_myopic_secondary<double, int>(deets, NR, true, true);
    EXPECT_EQ(chosen, 8);
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_chunk_length_for_myopic_secondary<double, int>(deets, NR, true, false);
    EXPECT_EQ(chosen, 12);
    chosen = tatami_hdf5::CompressedSparseMatrix_internal::choose_chunk_length_for_myopic_secondary<double, int>(deets, NR, false, true);
    EXPECT_EQ(chosen, 20);
}

TEST(SparseMatrix, ComputeChunkCacheSize) {
    int elsize = 500;
    EXPECT_EQ(tatami_hdf5::CompressedSparseMatrix_internal::compute_chunk_cache_size(0, 0, elsize), 0);
    EXPECT_EQ(tatami_hdf5::CompressedSparseMatrix_internal::compute_chunk_cache_size(10000, 2000, elsize), 6000 * elsize);
    EXPECT_EQ(tatami_hdf5::CompressedSparseMatrix_internal::compute_chunk_cache_size(10000, 4000, elsize), 12000 * elsize);
    EXPECT_EQ(tatami_hdf5::CompressedSparseMatrix_internal::compute_chunk_cache_size(10000, 7000, elsize), 14000 * elsize);
    EXPECT_EQ(tatami_hdf5::CompressedSparseMatrix_internal::compute_chunk_cache_size(10000, 10, elsize), 1000000);
}

TEST(SparseMatrix, Errors) {
    auto fpath = temp_file_path("tatami-sparse-test.h5");
    std::string name = "stuff";
    tatami_hdf5::CompressedSparseMatrixOptions hopt;

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt);
        }, "not consistent");
    }

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        triplets.index.resize(10);
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt);
        }, "should have length");
    }

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        triplets.index.resize(10);
        triplets.indptr.resize(10);
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", false, hopt);
        }, "should have length");
    }

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        triplets.index.resize(10);
        triplets.indptr.resize(11, 1);
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt);
        }, "should be zero");
    }

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        triplets.index.resize(10);
        triplets.indptr.resize(11);
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt);
        }, "last index pointer");
    }

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        triplets.index.resize(10);
        triplets.indptr.resize(11);
        triplets.indptr[1] = 2;
        triplets.indptr[2] = 1;
        triplets.indptr.back() = 10;
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt);
        }, "should be ordered");
    }

    {
        tatami_test::SimulateCompressedSparseResult<double, int> triplets;
        triplets.data.resize(10);
        triplets.index.resize(10);
        triplets.indptr.resize(11);
        triplets.indptr[1] = 100;
        triplets.indptr.back() = 10;
        dump_to_file(triplets, fpath, name);

        tatami_test::throws_error([&]() -> void {
            tatami_hdf5::CompressedSparseMatrix<double, int>(10, 20, fpath, name + "/data", name + "/index", name + "/indptr", true, hopt);
        }, "should be no greater than");
    }
}

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
