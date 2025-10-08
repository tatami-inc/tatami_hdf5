#ifndef TATAMI_HDF5_SPARSE_UTILS_HPP
#define TATAMI_HDF5_SPARSE_UTILS_HPP

#include <algorithm>
#include <utility>

namespace tatami_hdf5 {

namespace CompressedSparseMatrix_internal {

// If we're only interested in the block [X, Y), we don't need to extract the entire set of indices for a primary element.
// We only need to extract the first Y elements, as all subsequent elements must have indices greater than or equal to Y.
// Similarly, we only need to extract the last 'dim - X' elements, as all preceding elements must have indices less than X.
// This allows us to narrow the range of indices to be extracted.
template<typename Pointer_, typename Index_>
std::pair<Pointer_, Pointer_> narrow_primary_extraction_range(
    const Pointer_ ptr_start,
    const Pointer_ ptr_end,
    const Index_ block_start,
    const Index_ block_end,
    const Index_ secondary_dim
) {
    const Pointer_ num_nonzeros = ptr_end - ptr_start;
    const Pointer_ new_ptr_end = ptr_start + sanisizer::min(block_end, num_nonzeros);
    const Pointer_ new_ptr_start = ptr_end - sanisizer::min(secondary_dim - block_start, num_nonzeros);

    // It is guaranteed that new_ptr_start <= new_ptr_end.
    // This is because the difference boils down to:
    //
    // -num_nonzeros + num_nonzeros + num_nonzeros, OR
    // -num_nonzeros + block_end + num_nonzeros, OR
    // -num_nonzeros + num_nonzeros + secondary_dim - block_start, OR
    // -num_nonzeros + block_end + secondary_dim - block_start
    // 
    // All of which are guaranteed to be non-negative.
    return std::make_pair(new_ptr_start, new_ptr_end);
}

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

}

}

#endif
