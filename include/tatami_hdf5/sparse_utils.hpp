#ifndef TATAMI_HDF5_SPARSE_UTILS_HPP
#define TATAMI_HDF5_SPARSE_UTILS_HPP

#include "H5Cpp.h"

namespace tatami_hdf5 {

namespace Hdf5CompressedSparseMatrix_internal {

struct Components {
    H5::H5File file;
    H5::DataSet data_dataset;
    H5::DataSet index_dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

/*****************************************************************************
 * Code below is vendored from tatami/include/sparse/primary_extraction.hpp.
 * We copied it to avoid an implicit dependency on tatami's internals, given
 * that I'm not ready to promise that those internals are stable.
 *****************************************************************************/

template<class IndexIt_, typename Index_>
void refine_primary_limits(IndexIt_& indices_start, IndexIt_& indices_end, Index_ extent, Index_ smallest, Index_ largest_plus_one) {
    if (smallest) {
        // Using custom comparator to ensure that we cast to Index_ for signedness-safe comparisons.
        indices_start = std::lower_bound(indices_start, indices_end, smallest, [](Index_ a, Index_ b) -> bool { return a < b; });
    }

    if (largest_plus_one != extent) {
        indices_end = std::lower_bound(indices_start, indices_end, largest_plus_one, [](Index_ a, Index_ b) -> bool { return a < b; });
    }
}

template<typename Index_>
struct RetrievePrimarySubsetDense {
    RetrievePrimarySubsetDense(const std::vector<Index_>& subset, Index_ extent) : extent(extent) {
        if (!subset.empty()) {
            offset = subset.front();
            lastp1 = subset.back() + 1;
            size_t alloc = lastp1 - offset;
            present.resize(alloc);

            // Starting off at 1 to ensure that 0 is still a marker for
            // absence. It should be fine as subset.size() should fit inside
            // Index_ (otherwise nrow()/ncol() would give the wrong answer).
            Index_ counter = 1; 

            for (auto s : subset) {
                present[s - offset] = counter;
                ++counter;
            }
        }
    }

    template<class IndexIt_, class Store_>
    void populate(IndexIt_ indices_start, IndexIt_ indices_end, Store_ store) const {
        if (present.empty()) {
            return;
        }

        // Limiting the iteration to its boundaries based on the first and last subset index.
        auto original_start = indices_start;
        refine_primary_limits(indices_start, indices_end, extent, offset, lastp1);

        size_t counter = indices_start - original_start;
        for (; indices_start != indices_end; ++indices_start, ++counter) {
            auto ix = *indices_start;
            auto shift = present[ix - offset];
            if (shift) {
                store(shift - 1, counter);
            }
        }
    }

    Index_ extent;
    std::vector<Index_> present;
    Index_ offset = 0;
    Index_ lastp1 = 0;
};

template<typename Index_>
struct RetrievePrimarySubsetSparse {
    RetrievePrimarySubsetSparse(const std::vector<Index_>& subset, Index_ extent) : extent(extent) {
        if (!subset.empty()) {
            offset = subset.front();
            lastp1 = subset.back() + 1;
            size_t alloc = lastp1 - offset;
            present.resize(alloc);

            // Unlike the dense case, this is a simple present/absent signal,
            // as we don't need to map each structural non-zero back onto its 
            // corresponding location on a dense vector.
            for (auto s : subset) {
                present[s - offset] = 1;
            }
        }
    }

    template<class IndexIt_, class Store_>
    void populate(IndexIt_ indices_start, IndexIt_ indices_end, Store_ store) const {
        if (present.empty()) {
            return;
        }

        // Limiting the iteration to its boundaries based on the first and last subset index.
        auto original_start = indices_start;
        refine_primary_limits(indices_start, indices_end, extent, offset, lastp1);

        size_t counter = indices_start - original_start;
        for (; indices_start != indices_end; ++indices_start, ++counter) {
            auto ix = *indices_start;
            if (present[ix - offset]) {
                store(counter, ix);
            }
        }
    }

    Index_ extent;
    std::vector<unsigned char> present;
    Index_ offset = 0;
    Index_ lastp1 = 0;
};

}

}

#endif
