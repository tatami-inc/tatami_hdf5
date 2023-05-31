#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL // make sure this is included before tatami::apply.
#include "custom_parallel.h"
#endif

#include "H5Cpp.h"
#include "tatami/base/dense/DenseMatrix.hpp"
#include "tatami/base/other/DelayedTranspose.hpp"
#include "tatami_hdf5/Hdf5DenseMatrix.hpp"
#include "tatami/stats/sums.hpp"

#include "tatami_test/tatami_test.hpp"
#include "tatami_test/temp_file_path.hpp"

#include <vector>
#include <random>
#include <algorithm>

class Hdf5DenseMatrixTestMethods {
public:
    static constexpr size_t NR = 200, NC = 100;

protected:
    std::vector<double> values;
    std::string fpath;
    std::string name;

    static auto custom_options() {
        tatami_hdf5::Hdf5Options options;

        // Make sure the cache size is smaller than the dataset, but not too much
        // smaller, to make sure we do some caching + evictions. Here, the cache is
        // set at 20% of the size of the entire dataset, i.e., 40 rows or 20 columns.
        options.maximum_cache_size = (NR * NC * 8) / 5;

        return options;
    }

    void dump(const std::pair<int, int>& chunk_sizes) {
        fpath = tatami_test::temp_file_path("tatami-dense-test.h5");
        tatami_test::remove_file_path(fpath);

        H5::H5File fhandle(fpath, H5F_ACC_TRUNC);

        hsize_t dims[2];
        dims[0] = NR;
        dims[1] = NC;
        H5::DataSpace dspace(2, dims);
        H5::DataType dtype(H5::PredType::NATIVE_UINT8);

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
        values = tatami_test::simulate_dense_vector<double>(NR * NC, 0, 100);
        for (auto& v : values) {
            v = std::round(v);
        }

        dhandle.write(values.data(), H5::PredType::NATIVE_DOUBLE);
        return;
    }
};

/*************************************
 *************************************/

class Hdf5DenseUtilsTest : public ::testing::Test, public Hdf5DenseMatrixTestMethods {};

TEST_F(Hdf5DenseUtilsTest, Basic) {
    dump(std::make_pair<int, int>(10, 10));
    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name);
    EXPECT_EQ(mat.nrow(), NR);
    EXPECT_EQ(mat.ncol(), NC);
    EXPECT_FALSE(mat.sparse());
    EXPECT_EQ(mat.sparse_proportion(), 0);
}

TEST_F(Hdf5DenseUtilsTest, Preference) {
    {
        dump(std::make_pair<int, int>(10, 10));
        tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name);
        EXPECT_TRUE(mat.prefer_rows());
        EXPECT_EQ(mat.prefer_rows_proportion(), 1);

        tatami_hdf5::Hdf5DenseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_FALSE(tmat.prefer_rows());
        EXPECT_EQ(tmat.prefer_rows_proportion(), 0);
    }

    {
        // First dimension is compromised, switching to the second dimension.
        dump(std::make_pair<int, int>(NR, 1));
        tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name);
        EXPECT_FALSE(mat.prefer_rows());

        tatami_hdf5::Hdf5DenseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_TRUE(tmat.prefer_rows());
    }

    {
        // Second dimension is compromised, but we just use the first anyway.
        dump(std::make_pair<int, int>(1, NC));
        tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name);
        EXPECT_TRUE(mat.prefer_rows());

        tatami_hdf5::Hdf5DenseMatrix<double, int, true> tmat(fpath, name);
        EXPECT_FALSE(tmat.prefer_rows());
    }
}

/*************************************
 *************************************/

class Hdf5DenseAccessUncachedTest : public ::testing::TestWithParam<std::tuple<bool, int> >, public Hdf5DenseMatrixTestMethods {
protected:
    static auto uncached_options() {
        tatami_hdf5::Hdf5Options opt;
        opt.require_minimum_cache = false;
        opt.maximum_cache_size = 0;
        return opt;
    }
};

TEST_P(Hdf5DenseAccessUncachedTest, Basic) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    // Any chunking will do, we're not using this information anyway.
    dump(std::pair<int, int>(10, 10));

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, uncached_options()); 
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(Hdf5DenseAccessUncachedTest, Transposed) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    // Any chunking will do, we're not using this information anyway.
    dump(std::pair<int, int>(20, 5));

    tatami_hdf5::Hdf5DenseMatrix<double, int, true> mat(fpath, name, uncached_options()); 
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(Hdf5DenseAccessUncachedTest, ForcedCache) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    dump(std::pair<int, int>(15, 20));

    tatami_hdf5::Hdf5Options options;
    options.maximum_cache_size = 0;
    options.require_minimum_cache = true; // force a minimum cache size to avoid redundant reads.

    tatami_hdf5::Hdf5DenseMatrix<double, int, true> mat(fpath, name, options); 
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

INSTANTIATE_TEST_CASE_P(
    Hdf5DenseMatrix,
    Hdf5DenseAccessUncachedTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(1, 3)
    )
);

/*************************************
 *************************************/

class Hdf5DenseAccessTest : public ::testing::TestWithParam<std::tuple<bool, int, std::pair<int, int> > >, public Hdf5DenseMatrixTestMethods {};

TEST_P(Hdf5DenseAccessTest, Basic) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    auto chunk_sizes = std::get<2>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, custom_options()); 
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

TEST_P(Hdf5DenseAccessTest, Transposed) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);

    auto chunk_sizes = std::get<2>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int, true> mat(fpath, name, custom_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    tatami_test::test_simple_column_access(&mat, &ref, FORWARD, JUMP);
    tatami_test::test_simple_row_access(&mat, &ref, FORWARD, JUMP);
}

INSTANTIATE_TEST_CASE_P(
    Hdf5DenseMatrix,
    Hdf5DenseAccessTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(1, 3),
        ::testing::Values(
            std::make_pair(Hdf5DenseMatrixTestMethods::NR, 1),
            std::make_pair(1, Hdf5DenseMatrixTestMethods::NC),
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11),
            std::make_pair(0, 0)
        )
    )
);

/*************************************
 *************************************/

class Hdf5DenseAccessMiscTest : public ::testing::TestWithParam<std::tuple<std::pair<int, int> > >, public Hdf5DenseMatrixTestMethods {};

TEST_P(Hdf5DenseAccessMiscTest, Apply) {
    // Putting it through its paces for correct parallelization via apply.
    auto param = GetParam(); 
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    EXPECT_EQ(tatami::row_sums(&mat), tatami::row_sums(&ref));
    EXPECT_EQ(tatami::column_sums(&mat), tatami::column_sums(&ref));
}

TEST_P(Hdf5DenseAccessMiscTest, LruReuse) {
    // Check that the LRU cache works as expected when cache elements are
    // constantly re-used in a manner that changes the last accessed element.
    auto param = GetParam(); 
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    {
        auto m_ext = mat.dense_row();
        auto r_ext = ref.dense_row();
        for (size_t r0 = 0; r0 < NR; ++r0) {
            auto r = (r0 % 2 ? NR - r0/2 - 1 : r0/2); // alternate between the last and first chunk.
            EXPECT_EQ(m_ext->fetch(r), r_ext->fetch(r));
        }
    }

    {
        auto m_ext = mat.dense_column();
        auto r_ext = ref.dense_column();
        for (size_t c0 = 0; c0 < NC; ++c0) {
            auto c = (c0 % 2 ? NC - c0/2 - 1 : c0/2); // alternate between the last and first.
            EXPECT_EQ(m_ext->fetch(c), r_ext->fetch(c));
        }
    }
}

TEST_P(Hdf5DenseAccessMiscTest, Oracle) {
    auto param = GetParam(); 
    auto chunk_sizes = std::get<0>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, false); // consecutive
    tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, false);

    tatami_test::test_oracle_row_access<tatami::NumericMatrix>(&mat, &ref, true); // randomized
    tatami_test::test_oracle_column_access<tatami::NumericMatrix>(&mat, &ref, true);
}

INSTANTIATE_TEST_CASE_P(
    Hdf5DenseMatrix,
    Hdf5DenseAccessMiscTest,
    ::testing::Combine(
        ::testing::Values(
            std::make_pair(Hdf5DenseMatrixTestMethods::NR, 1),
            std::make_pair(1, Hdf5DenseMatrixTestMethods::NC),
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11),
            std::make_pair(0, 0)
        )
    )
);

/*************************************
 *************************************/

class Hdf5DenseSlicedTest : public ::testing::TestWithParam<std::tuple<bool, size_t, std::vector<double>, std::pair<int, int> > >, public Hdf5DenseMatrixTestMethods {};

TEST_P(Hdf5DenseSlicedTest, Basic) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    {
        size_t FIRST = interval_info[0] * NC, LAST = interval_info[1] * NC;
        tatami_test::test_sliced_row_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST); 
    }

    {
        size_t FIRST = interval_info[0] * NR, LAST = interval_info[1] * NR;
        tatami_test::test_sliced_column_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }
}

TEST_P(Hdf5DenseSlicedTest, Transposed) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int, true> mat(fpath, name, custom_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    {
        size_t FIRST = interval_info[0] * NR, LAST = interval_info[1] * NR; // NR is deliberate here, it's transposed.
        tatami_test::test_sliced_row_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST); 
    }

    {
        size_t FIRST = interval_info[0] * NC, LAST = interval_info[1] * NC; // NC is deliberate here, it's transposed.
        tatami_test::test_sliced_column_access(&mat, &ref, FORWARD, JUMP, FIRST, LAST);
    }
}

INSTANTIATE_TEST_CASE_P(
    Hdf5DenseMatrix,
    Hdf5DenseSlicedTest,
    ::testing::Combine(
        ::testing::Values(true, false), // iterate forward or back, to test the workspace's memory.
        ::testing::Values(1, 3), // jump, to test the workspace's memory.
        ::testing::Values(
            std::vector<double>({ 0, 0.5 }), 
            std::vector<double>({ 0.25, 0.75 }), 
            std::vector<double>({ 0.51, 1 })
        ),
        ::testing::Values(
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11)
        )
    )
);

/*************************************
 *************************************/

class Hdf5DenseIndexedTest : public ::testing::TestWithParam<std::tuple<bool, size_t, std::vector<double>, std::pair<int, int> > >, public Hdf5DenseMatrixTestMethods {};

TEST_P(Hdf5DenseIndexedTest, Basic) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int> mat(fpath, name, custom_options());
    tatami::DenseRowMatrix<double, int> ref(NR, NC, values);

    {
        size_t FIRST = interval_info[0] * NC, STEP = interval_info[1];
        tatami_test::test_indexed_row_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP); 
    }

    {
        size_t FIRST = interval_info[0] * NR, STEP = interval_info[1];
        tatami_test::test_indexed_column_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }
}

TEST_P(Hdf5DenseIndexedTest, Transposed) {
    auto param = GetParam(); 
    bool FORWARD = std::get<0>(param);
    size_t JUMP = std::get<1>(param);
    auto interval_info = std::get<2>(param);

    auto chunk_sizes = std::get<3>(param);
    dump(chunk_sizes);

    tatami_hdf5::Hdf5DenseMatrix<double, int, true> mat(fpath, name, custom_options());
    std::shared_ptr<tatami::Matrix<double, int> > ptr(new tatami::DenseRowMatrix<double, int>(NR, NC, values));
    tatami::DelayedTranspose<double, int> ref(std::move(ptr));

    {
        size_t FIRST = interval_info[0] * NR, STEP = interval_info[1]; // NR is deliberate here, it's transposed.
        tatami_test::test_indexed_row_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP); 
    }

    {
        size_t FIRST = interval_info[0] * NC, STEP = interval_info[1]; // NC is deliberate here, it's transposed.
        tatami_test::test_indexed_column_access(&mat, &ref, FORWARD, JUMP, FIRST, STEP);
    }
}

INSTANTIATE_TEST_CASE_P(
    Hdf5DenseMatrix,
    Hdf5DenseIndexedTest,
    ::testing::Combine(
        ::testing::Values(true, false), // iterate forward or back, to test the workspace's memory.
        ::testing::Values(1, 3), // jump, to test the workspace's memory.
        ::testing::Values(
            std::vector<double>({ 0.3, 5 }), 
            std::vector<double>({ 0.11, 9 }), 
            std::vector<double>({ 0.4, 7 })
        ),
        ::testing::Values(
            std::make_pair(7, 17), // using chunk sizes that are a little odd to check for off-by-one errors.
            std::make_pair(19, 7),
            std::make_pair(11, 11)
        )
    )
);

/*************************************
 *************************************/

class Hdf5DenseCacheTypeTest : public ::testing::Test, public Hdf5DenseMatrixTestMethods {};

TEST_F(Hdf5DenseCacheTypeTest, CastToInt) {
    dump(std::make_pair(9, 13));

    tatami_hdf5::Hdf5DenseMatrix<double, int, false, int> mat(fpath, name, custom_options()); 
    std::vector<int> casted(values.begin(), values.end());
    tatami::DenseRowMatrix<double, int, decltype(casted)> ref(NR, NC, std::move(casted));

    tatami_test::test_simple_column_access(&mat, &ref, true, 1);
    tatami_test::test_simple_row_access(&mat, &ref, true, 1);
}
