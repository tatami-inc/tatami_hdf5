#ifndef TATAMI_HDF5_SPARSE_MATRIX_HPP
#define TATAMI_HDF5_SPARSE_MATRIX_HPP

#include "H5Cpp.h"

#include <string>
#include <vector>
#include <type_traits>
#include <algorithm>

#include "serialize.hpp"
#include "utils.hpp"
#include "tatami_chunked/tatami_chunked.hpp"

/**
 * @file Hdf5CompressedSparseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed compressed sparse matrix.
 */

namespace tatami_hdf5 {

/**
 * @cond
 */
namespace Hdf5CompressedSparseMatrix_internal {

// All HDF5-related members.
struct PrimaryComponents {
    H5::H5File file;
    H5::DataSet data_dataset;
    H5::DataSet index_dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

class PrimaryBase {
    PrimaryBase(const std::string& file_name, const std::string& data_name, const std::string& index_name, const std::vector<hsize_t>& ptrs) : pointers(ptrs) {
        serialize([&]() -> void {
           h5comp.reset(new PrimaryComponents);

            // TODO: set more suitable chunk cache values here, to avoid re-reading
            // chunks that are only partially consumed.
            h5comp->file.openFile(file_name, H5F_ACC_RDONLY);
            h5comp->data = h5comp->file.openDataSet(data_name);
            h5comp->index = h5comp->file.openDataSet(index_name);
            h5comp->dataspace = h5comp->data.getSpace();
        });
    }

public:
    const std::vector<hsize_t>& pointers;

    // HDF5 members are stored in a separate pointer so we can serialize construction and destruction.
    std::unique_ptr<PrimaryComponents> h5comp;
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class LruBase : public PrimaryBase {
    struct Slab {
        Slab(size_t capacity, bool needs_cached_value, bool needs_cached_index) : 
            value(needs_cached_value ? capacity : 0), index(needs_cached_index ? capacity : 0) {}
        std::vector<CachedValue_> value;
        std::vector<CachedIndex_> index;
        Index_ length;
    };

    tatami_chunked::LruSlabCache<Index_, Slab> cache;
    size_t max_non_zeros;
    bool needs_cached_value, needs_cached_index;

public:
    LruBase(
        const std::string& file_name,
        const std::string& data_name,
        const std::string& index_name,
        const std::vector<hsize_t>& ptrs,
        size_t cache_size,
        size_t max_non_zeros, 
        bool needs_cached_value, 
        bool needs_cached_index) : 
        cache([&]() -> size_t {
            // Always return at least one slab, so that cache.find() is valid.
            if (max_non_zeros == 0) {
                return 1;
            }

            auto elsize = size_of_element<CachedValue_, CachedIndex_>(needs_cached_value, needs_cached_index);
            if elsize == 0 {
                return 1;
            }

            auto num_slabs = cache_size / (max_non_zeros * elsize);
            if (num_slabs == 0) {
                return 1;
            }

            return num_slabs;
        }()),
        max_non_zeros(max_non_zeros),
        needs_cached_value(needs_cached_value),
        needs_cached_index(needs_cached_index)
    {}

public:
    Chunk<Index_, CachedValue_, CachedIndex_> fetch(Index_ i) {
        const auto& slab = cache.find(
            i, 
            /* create = */ [&]() -> LruSlab {
                return Slab(max_non_zeros, needs_cached_value, needs_cached_index);
            },
            /* populate = */ [&](Index_ i, Slab& current_cache) -> void {
                hsize_t extraction_start = this->pointers[i];
                hsize_t extraction_len = this->pointers[i + 1] - pointers[i];
                current_cache.length = extraction_len;

                serialize([&]() -> void {
                    this->h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    this->h5comp->memspace.setExtentSimple(1, &extraction_len);
                    this->h5comp->memspace.selectAll();
                    if (needs_cached_index) {
                        this->h5comp->index.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                    if (needs_value) {
                        this->h5comp->data.read(current_cache.value.data(), define_mem_type<CachedValue_>(), this->h5comp->memspace, this->h5comp->dataspace);
                    }
                });
            }
        );

        Chunk<Index_, CachedValue_, CachedIndex_> output;
        output.length = slab.length;
        if (needs_cached_value) {
            output.value = slab.value.data();
        }
        if (needs_cached_index) {
            output.index = slab.index.data();
        }
        return output;
    }
};




}
/**
 * @endcond
 */

/**
 * @brief Compressed sparse matrix in a HDF5 file.
 *
 * This class retrieves sparse data from the HDF5 file on demand rather than loading it all in at the start.
 * This allows us to handle very large datasets in limited memory at the cost of speed.
 *
 * We manually handle the chunk caching to speed up access for consecutive rows or columns (for compressed sparse row and column matrices, respectively).
 * The policy is to minimize the number of calls to the HDF5 library - and thus expensive file reads - by requesting large contiguous slices where possible, i.e., multiple columns or rows for CSC and CSR matrices, respectively.
 * These are held in memory in the `Extractor` while the relevant column/row is returned to the user by `row()` or `column()`.
 * The size of the slice is determined by the `options` in the constructor.
 *
 * Callers should follow the `prefer_rows()` suggestion when extracting data,
 * as this tries to minimize the number of chunks that need to be read per access request.
 * This recommendation is even stronger than for the `Hdf5DenseMatrix`,
 * as the access pattern on disk for the non-preferred dimension is very suboptimal.
 *
 * As the HDF5 library is not generally thread-safe, the HDF5-related operations should only be run in a single thread.
 * This is normally handled automatically but developers can check out `serialize()` to customize the locking scheme.
 *
 * @tparam row_ Whether the matrix is stored in compressed sparse row format.
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all values (based on their HDF5 type or other knowledge).
 * @tparam CachedIndex_ Type of the index value to store in the cache.
 * This can be set to a narrower type than `Index_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all indices (based on their HDF5 type or other knowledge).
 */
template<bool row_, typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class Hdf5CompressedSparseMatrix : public tatami::Matrix<Value_, Index_> {
    Index_ nrows, ncols;
    std::string file_name;
    std::string data_name, index_name;
    std::vector<hsize_t> pointers;

    size_t cache_size_limit;
    Index_ max_non_zeros;

public:
    /**
     * @param nr Number of rows in the matrix.
     * @param nc Number of columns in the matrix.
     * @param file Path to the file.
     * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
     * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
     * If `row_ = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row_ = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row_ = true`) or columns (otherwise).
     * @param options Further options.
     */
    Hdf5CompressedSparseMatrix(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr, const Hdf5Options& options) :
        nrows(nr), ncols(nc), file_name(file), data_name(std::move(vals)), index_name(std::move(idx)), cache_size_limit(options.maximum_cache_size)
    {
        serialize([&]() -> void {
            H5::H5File file_handle(file_name, H5F_ACC_RDONLY);
            auto dhandle = open_and_check_dataset<false>(file_handle, data_name);
            hsize_t nonzeros = get_array_dimensions<1>(dhandle, "vals")[0];

            auto ihandle = open_and_check_dataset<true>(file_handle, index_name);
            if (get_array_dimensions<1>(ihandle, "idx")[0] != nonzeros) {
                throw std::runtime_error("number of non-zero elements is not consistent between 'data' and 'idx'");
            }

            auto phandle = open_and_check_dataset<true>(file_handle, ptr);
            size_t ptr_size = get_array_dimensions<1>(phandle, "ptr")[0];
            size_t dim_p1 = static_cast<size_t>(row_ ? nrows : ncols) + 1;
            if (ptr_size != dim_p1) {
                throw std::runtime_error("'ptr' dataset should have length equal to the number of " + (row_ ? std::string("rows") : std::string("columns")) + " plus 1");
            }

            // Checking the contents of the index pointers.
            pointers.resize(dim_p1);
            phandle.read(pointers.data(), H5::PredType::NATIVE_HSIZE);
            if (pointers[0] != 0) {
                throw std::runtime_error("first index pointer should be zero");
            }
            if (pointers.back() != nonzeros) {
                throw std::runtime_error("last index pointer should be equal to the number of non-zero elements");
            }
        });

        max_non_zeros = 0;
        for (size_t i = 1; i < pointers.size(); ++i) {
            Index_ diff = pointers[i] - pointers[i-1];
            if (diff > max_non_zeros) {
                max_non_zeros = diff;
            }
        }
    }

    /**
     * @param nr Number of rows in the matrix.
     * @param nc Number of columns in the matrix.
     * @param file Path to the file.
     * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
     * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
     * If `row_ = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row_ = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row_ = true`) or columns (otherwise).
     * 
     * Unlike its overload, this constructor uses the defaults for `Hdf5Options`.
     */
    Hdf5CompressedSparseMatrix(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr) :
        Hdf5CompressedSparseMatrix(nr, nc, std::move(file), std::move(vals), std::move(idx), std::move(ptr), Hdf5Options()) {}

public:
    Index_ nrow() const {
        return nrows;
    }

    Index_ ncol() const {
        return ncols;
    }

    /**
     * @return `true`.
     */
    bool sparse() const {
        return true;
    }

    double sparse_proportion() const { 
        return 1;
    }

    /**
     * @return `true` if this is in compressed sparse row format.
     */
    bool prefer_rows() const {
        return row_;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(row_);
    }

    bool uses_oracle(bool) const {
        return false; // placeholder for proper support.
    }

    using tatami::Matrix<Value_, Index_>::dense_row;

    using tatami::Matrix<Value_, Index_>::dense_column;

    using tatami::Matrix<Value_, Index_>::sparse_row;

    using tatami::Matrix<Value_, Index_>::sparse_column;

    /********************************************
     ************ Primary extraction ************
     ********************************************/
private:
    struct PrimaryWorkspace {
        // HDF5 members.
        H5::H5File file;
        H5::DataSet data, index;
        H5::DataSpace dataspace;
        H5::DataSpace memspace;
    };

    void initialize_lru_cache(std::unique_ptr<tatami_chunked::LruSlabCache<Index_, LruSlab> >& historian, bool needs_value, bool needs_cached_index) const {
        // This function should only be called if at least one of needs_value or needs_cached_index
        // is true, otherwise element_size == 0 and we end up with divide-by-zero errors below.
        size_t element_size = size_of_cached_element(needs_value, needs_cached_index);

        // When we're defining the LRU cache, each slab element is set to the
        // maximum number of non-zeros across all primary elements. This is
        // because the capacity of each recycled vector may be much larger than
        // the reported size of the chunk; this would cause us to overrun the
        // cache_size_limit if we recycled the vector enough times such that
        // each cache element had capacity equal to the maximum number of
        // non-zero elements in any dimension element. By setting every slab
        // element to its maximum, we avoid reallocation and buffer overruns.
        //
        // Alternatives would be to create a new Element on every recycling
        // iteration, or to hope that shrink_to_fit() behaves. Both would allow
        // us to store more cache elements but would involve reallocations,
        // which degrades perf in the most common case where a dimension
        // element is accessed no more than once during iteration.

        size_t max_cache_number = cache_size_limit / (max_non_zeros * element_size);
        if (max_cache_number == 0) {
            max_cache_number = 1; // same effect as always setting 'require_minimum_cache = true'.
        }

        historian.reset(new tatami_chunked::LruSlabCache<Index_, LruSlab>(max_cache_number));
    }

private:
    struct Extracted {
        Extracted() = default;

        Extracted(const LruSlab& cache) {
            value = cache.value.data();
            index = cache.index.data();
            length = cache.length;
            bounded = cache.bounded;
        }

        Extracted(const OracleCache& cache, Index_ i, bool needs_value, bool needs_cached_index) {
            const auto& element = cache.cache_data[i];
            auto offset = element.mem_offset;
            if (needs_value) {
                value = cache.cache_value.data() + offset;
            }
            if (needs_cached_index) {
                index = cache.cache_index.data() + offset;
            }
            length = element.length;
            bounded = element.bounded;
        }

        const CachedValue_* value = NULL;
        const CachedIndex_* index = NULL;
        Index_ length = 0;
        bool bounded = false;
    };

    Extracted extract_primary_without_oracle(Index_ i, PrimaryWorkspace& work, bool needs_value, bool needs_cached_index) const {
        const auto& chosen = work.historian->find(i,
            /* create = */ [&]() -> LruSlab {
                return LruSlab(max_non_zeros, needs_value);
            },
            /* populate = */ [&](Index_ i, LruSlab& current_cache) -> void {
                // Check if bounds already exist from the reusable cache. If so,
                // we can use them to reduce the amount of data I/O.
                hsize_t extraction_start = pointers[i];
                hsize_t extraction_len = pointers[i + 1] - pointers[i];
                bool bounded = false;

                if (work.extraction_bounds.size()) {
                    const auto& current = work.extraction_bounds[i];
                    if (current.first != PrimaryWorkspace::no_extraction_bound) {
                        bounded = true;
                        extraction_start = current.first;
                        extraction_len = current.second;
                    }
                }

                current_cache.length = extraction_len;
                current_cache.bounded = bounded;

                serialize([&]() -> void {
                    work.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    work.memspace.setExtentSimple(1, &extraction_len);
                    work.memspace.selectAll();
                    if (needs_cached_index) {
                        work.index.read(current_cache.index.data(), define_mem_type<CachedIndex_>(), work.memspace, work.dataspace);
                    }
                    if (needs_value) {
                        work.data.read(current_cache.value.data(), define_mem_type<CachedValue_>(), work.memspace, work.dataspace);
                    }
                });
            }
        );

        return Extracted(chosen);
    }

    Extracted extract_primary_with_oracle(PrimaryWorkspace& work, bool needs_value, bool needs_cached_index) const {
        auto& pred = *work.futurist;
        if (pred.predictions_made.size() > pred.predictions_fulfilled) {
            auto chosen = pred.predictions_made[pred.predictions_fulfilled++];
            return Extracted(pred, chosen, needs_value, needs_cached_index);
        }

        // Grow the number of predictions over time, until we get to a point
        // where we consistently fill the cache.
        size_t max_predictions = pred.predictions_made.size() * 2; 
        if (max_predictions < 100) {
            max_predictions = 100;
        } else {
            size_t upper = (row_ ? nrows : ncols);
            if (max_predictions > upper) {
                max_predictions = upper;
            }
        }

        pred.predictions_made.clear();
        pred.needed.clear();
        pred.present.clear();
        pred.next_cache_data.clear();
        pred.next_cache_exists.clear();

        // Here, we use a giant contiguous buffer to optimize for
        // near-consecutive iteration. This allows the HDF5 library to pull out
        // long strips of data from the file.  It also allows us to maximize
        // the use of the cache_size_limit by accounting for differences in the
        // non-zeros for each element, rather than conservatively assuming
        // they're all at max (as in the LRU case). The downside is that we
        // need to do some copying within the cache to make space for new
        // reads, but that works out to be no more than one extra copy per
        // fetch() call, which is tolerable. I suppose we could do better
        // by defragmenting within this buffer but that's probably overkill.

        if (pred.max_cache_elements == static_cast<size_t>(-1)) {
            size_t element_size = size_of_cached_element(needs_value, needs_cached_index); // should be non-zero by the time we get inside this function.

            pred.max_cache_elements = cache_size_limit / element_size;
            if (pred.max_cache_elements < static_cast<size_t>(max_non_zeros)) {
                pred.max_cache_elements = max_non_zeros; // make sure we have enough space to store the largest possible primary dimension element.
            }

            if (needs_cached_index) {
                pred.cache_index.resize(pred.max_cache_elements);
            }
            if (needs_value) {
                pred.cache_value.resize(pred.max_cache_elements);
            }
        }

        size_t filled_elements = 0;

        for (size_t p = 0; p < max_predictions; ++p) {
            Index_ current;
            if (!pred.prediction_stream.next(current)) {
                break;
            }

            // Seeing if this element already exists somewhere.
            auto nit = pred.next_cache_exists.find(current);
            if (nit != pred.next_cache_exists.end()) {
                pred.predictions_made.push_back(nit->second);
                continue;
            }

            auto it = pred.cache_exists.find(current);
            if (it != pred.cache_exists.end()) {
                auto& candidate = pred.cache_data[it->second];
                filled_elements += candidate.length;
                if (filled_elements > pred.max_cache_elements) {
                    pred.prediction_stream.back();
                    break;
                }

                Index_ used = pred.next_cache_data.size();
                pred.predictions_made.push_back(used);
                pred.present.push_back(used);
                pred.next_cache_exists[current] = used;
                pred.next_cache_data.push_back(std::move(candidate));
                continue;
            }

            // Check if bounds already exist from the reusable cache. If so,
            // we can use them to reduce the amount of data I/O.
            hsize_t extraction_start = pointers[current];
            hsize_t extraction_len = pointers[current + 1] - pointers[current];
            bool bounded = false;

            if (work.extraction_bounds.size()) {
                const auto& bounds = work.extraction_bounds[current];
                if (bounds.first != PrimaryWorkspace::no_extraction_bound) {
                    bounded = true;
                    extraction_start = bounds.first;
                    extraction_len = bounds.second;
                }
            }

            filled_elements += extraction_len;
            if (filled_elements > pred.max_cache_elements) {
                pred.prediction_stream.back();
                break;
            }

            Index_ used = pred.next_cache_data.size();
            pred.predictions_made.push_back(used);
            pred.needed.emplace_back(used);
            pred.next_cache_exists[current] = used;

            typename OracleCache::Element latest;
            latest.data_offset = extraction_start;
            latest.length = extraction_len;
            latest.bounded = bounded;
            pred.next_cache_data.push_back(std::move(latest));
        }

        if (pred.needed.size()) {
            size_t dest_offset = 0;

            if (pred.present.size()) {
                // Shuffling all re-used elements to the start of the buffer,
                // so that we can perform a contiguous extraction of the needed
                // elements in the rest of the buffer. This needs some sorting
                // to ensure that we're not clobbering one re-used element's
                // contents when shifting another element to the start.
                sort_by_field(pred.present, [&pred](size_t i) -> size_t { return pred.next_cache_data[i].mem_offset; });

                for (const auto& p : pred.present) {
                    auto& info = pred.next_cache_data[p];

#ifdef DEBUG
                    if (info.mem_offset < dest_offset) {
                        throw std::runtime_error("detected clobbering of memory cache from overlapping offsets");
                    }
#endif

                    if (needs_cached_index) {
                        auto isrc = pred.cache_index.begin() + info.mem_offset;
                        std::copy(isrc, isrc + info.length, pred.cache_index.begin() + dest_offset);
                    }
                    if (needs_value) {
                        auto vsrc = pred.cache_value.begin() + info.mem_offset;
                        std::copy(vsrc, vsrc + info.length, pred.cache_value.begin() + dest_offset); 
                    }

                    info.mem_offset = dest_offset;
                    dest_offset += info.length;
                }
            }

            // Sorting so that we get consecutive accesses in the hyperslab construction.
            // This should improve re-use of partially read chunks inside the HDF5 call.
            sort_by_field(pred.needed, [&pred](size_t i) -> size_t { return pred.next_cache_data[i].data_offset; });

            serialize([&]() -> void {
                size_t sofar = 0;
                hsize_t combined_len = 0;
                work.dataspace.selectNone();

                while (sofar < pred.needed.size()) {
                    auto& first = pred.next_cache_data[pred.needed[sofar]];
                    first.mem_offset = dest_offset + combined_len;
                    hsize_t src_offset = first.data_offset;
                    hsize_t len = first.length;
                    ++sofar;

                    // Finding the stretch of consecutive extractions, and bundling them into a single hyperslab.
                    for (; sofar < pred.needed.size(); ++sofar) {
                        auto& next = pred.next_cache_data[pred.needed[sofar]];
                        if (src_offset + len < next.data_offset) {
                            break;
                        }
                        next.mem_offset = first.mem_offset + len;
                        len += next.length;
                    }

                    work.dataspace.selectHyperslab(H5S_SELECT_OR, &len, &src_offset);
                    combined_len += len;
                }

                work.memspace.setExtentSimple(1, &combined_len);
                work.memspace.selectAll();
                if (needs_cached_index) {
                    work.index.read(pred.cache_index.data() + dest_offset, define_mem_type<Index_>(), work.memspace, work.dataspace);
                }
                if (needs_value) {
                    work.data.read(pred.cache_value.data() + dest_offset, define_mem_type<Value_>(), work.memspace, work.dataspace);
                }
            });
        }

        pred.cache_data.swap(pred.next_cache_data);
        pred.cache_exists.swap(pred.next_cache_exists);
        pred.predictions_fulfilled = 1; // using the first one now.
        return Extracted(pred, pred.predictions_made.front(), needs_value, needs_cached_index);
    }

    /********************************************
     ************ Primary extraction ************
     ********************************************/
private:
    template<class Function_>
    void extract_primary_raw(size_t i, Function_ fill, Index_ start, PrimaryWorkspace& work, bool needs_value, bool needs_cached_index) const {
        Extracted details;
        if (work.futurist) {
            details = extract_primary_with_oracle(work, needs_value, needs_cached_index);
        } else {
            details = extract_primary_without_oracle(i, work, needs_value, needs_cached_index);
        }

        auto istart = details.index; // possibly NULL, if needs_cached_index = false.
        size_t offset = 0;
        size_t len = details.length;

        if (needs_cached_index) {
            // If we used the extraction_bounds during extraction, there's no need
            // to do another search. Similarly, if we didn't use the extraction_bounds
            // (e.g., it was already cached) but we have extraction_bounds available,
            // we can again skip the binary search.
            if (!details.bounded && start) {
                bool hit = false;
                if (work.extraction_bounds.size()) {
                    auto& target = work.extraction_bounds[i];
                    if (target.first != PrimaryWorkspace::no_extraction_bound) {
                        hit = true;
                        offset = target.first - pointers[i];
                        istart += offset;
                    }
                } 

                if (!hit) {
                    auto iend = details.index + details.length;
                    istart = std::lower_bound(details.index, iend, start);
                    offset = istart - details.index;
                }

                len -= offset;
            }
        }

        size_t iterated = fill(len, istart, (needs_value ? details.value + offset : NULL));

        if (needs_cached_index) {
            if (work.extraction_bounds.size()) {
                auto& target = work.extraction_bounds[i];
                if (target.first == PrimaryWorkspace::no_extraction_bound) {
                    target.first = pointers[i] + offset;
                    target.second = iterated;
                }
            }
        }

        return;
    }

    const Value_* extract_primary(size_t i, Value_* buffer, Index_ start, Index_ length, PrimaryWorkspace& work) const {
        if (length) {
            std::fill(buffer, buffer + length, 0);

            extract_primary_raw(i, 

                [&](size_t num, const CachedIndex_* is, const CachedValue_* vs) -> size_t {
                    auto ioriginal = is;
                    Index_ end = start + length;
                    for (size_t i = 0; i < num && *is < end; ++i, ++is, ++vs) {
                        buffer[*is - start] = *vs;
                    }
                    return is - ioriginal;
                },

                start, 
                work,
                /* needs_value = */ true,
                /* needs_cached_index = */ true
            );
        }

        return buffer;
    }

    tatami::SparseRange<Value_, Index_> extract_primary(
        size_t i, 
        Value_* dbuffer, 
        Index_* ibuffer, 
        Index_ start, 
        Index_ length, 
        PrimaryWorkspace& work, 
        bool needs_value, 
        bool needs_index, 
        Index_ full_length) 
    const {
        Index_ counter = 0;
        bool extract_full = (start == 0 && length == full_length);

        if (length) {
            extract_primary_raw(i, 

                [&](size_t num, const CachedIndex_* is, const CachedValue_* vs) -> Index_ {
                    if (extract_full) {
                        counter = num;
                    } else {
                        CachedIndex_ end = start + length;
                        counter = std::lower_bound(is, is + num, end) - is;
                    }

                    if (needs_index) {
                        std::copy(is, is + counter, ibuffer);
                    }
                    if (needs_value) {
                        std::copy(vs, vs + counter, dbuffer);
                    }

                    return counter;
                },

                start, 
                work,
                needs_value,
                needs_index || !extract_full // if we don't need the indices, we still need to load them if we're taking a block instead of the full dimension.
            );
        }

        if (!needs_value) {
            dbuffer = NULL;
        }
        if (!needs_index) {
            ibuffer = NULL;
        }

        return tatami::SparseRange<Value_, Index_>(counter, dbuffer, ibuffer);
    }

private:
    template<class Fill_, class Skip_>
    static size_t indexed_extraction(const CachedIndex_* istart, const CachedIndex_* iend, const CachedValue_* vstart, bool needs_value, const std::vector<Index_>& indices, Fill_ fill, Skip_ skip) {
        auto ioriginal = istart;
        if (needs_value) {
            for (auto idx : indices) {
                while (istart != iend && *istart < idx) {
                    ++istart;
                    ++vstart;
                }
                if (istart == iend) {
                    break;
                }
                if (*istart == idx) {
                    fill(idx, *vstart);
                    ++istart;
                    ++vstart;
                } else {
                    skip();
                }
            }
        } else {
            for (auto idx : indices) {
                while (istart != iend && *istart < idx) {
                    ++istart;
                }
                if (istart == iend) {
                    break;
                }
                if (*istart == idx) {
                    fill(idx, 0);
                    ++istart;
                } else {
                    skip();
                }
            }
        }

        return istart - ioriginal;
    }

    const Value_* extract_primary(size_t i, Value_* buffer, const std::vector<Index_>& indices, PrimaryWorkspace& work) const {
        std::fill(buffer, buffer + indices.size(), 0);
        auto original = buffer;

        if (indices.size()) {
            extract_primary_raw(i, 

                [&](size_t num, const CachedIndex_* is, const CachedValue_* vs) -> size_t {
                    return indexed_extraction(is, is + num, vs, true, indices, 
                        [&](CachedIndex_, CachedValue_ value) -> void {
                            *buffer = value;
                            ++buffer;
                        },
                        [&]() -> void {
                            ++buffer;
                        }
                    );
                },

                indices.front(),
                work,
                /* needs_value = */ true,
                /* needs_cached_index = */ true
            );
        }

        return original;
    }

    tatami::SparseRange<Value_, Index_> extract_primary(size_t i, Value_* dbuffer, Index_* ibuffer, const std::vector<Index_>& indices, PrimaryWorkspace& work, bool needs_value, bool needs_index) const {
        Index_ counter = 0;

        if (indices.size()) {
            extract_primary_raw(i, 

                [&](size_t num, const CachedIndex_* is, const CachedValue_* vs) -> size_t {
                    return indexed_extraction(is, is + num, vs, needs_value, indices,
                        [&](CachedIndex_ pos, CachedValue_ value) -> void {
                            if (needs_value) {
                                dbuffer[counter] = value;
                            }
                            if (needs_index) {
                                ibuffer[counter] = pos;
                            }
                            ++counter;
                        },
                        []() -> void {}
                    );
                },

                indices.front(),
                work,
                needs_value,
                /* needs_cached_index = */ true
            );
        }

        if (!needs_index) {
            ibuffer = NULL;
        }
        if (!needs_value) {
            dbuffer = NULL;
        }

        return tatami::SparseRange<Value_, Index_>(counter, dbuffer, ibuffer);
    }

    /**********************************************
     ************ Secondary extraction ************
     **********************************************/
private:
    // This could be improved by extracting multiple rows at any given call and
    // caching them for subsequent requests. However, even then, we'd require
    // multiple re-reads from file when we exceed the cache. So, any caching
    // would be just turning an extremely bad access pattern into a very bad
    // pattern, when users shouldn't even be calling this at all... 
    struct SecondaryWorkspace {
        H5::H5File file;
        H5::DataSet data, index;
        H5::DataSpace dataspace;
        H5::DataSpace memspace;
        std::vector<Index_> index_cache;
    };

    template<class Function_>
    bool extract_secondary_raw(Index_ primary, Index_ secondary, Function_& fill, SecondaryWorkspace& core, bool needs_value) const {
        hsize_t left = pointers[primary], right = pointers[primary + 1];
        core.index_cache.resize(right - left);

        // Serial locks should be applied by the callers.
        hsize_t offset = left;
        hsize_t count = core.index_cache.size();
        core.dataspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
        core.memspace.setExtentSimple(1, &count);
        core.memspace.selectAll();
        core.index.read(core.index_cache.data(), define_mem_type<Index_>(), core.memspace, core.dataspace);

        auto it = std::lower_bound(core.index_cache.begin(), core.index_cache.end(), secondary);
        if (it != core.index_cache.end() && *it == secondary) {
            if (needs_value) {
                offset = left + (it - core.index_cache.begin());
                count = 1;
                core.dataspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
                core.memspace.setExtentSimple(1, &count);
                core.memspace.selectAll();

                Value_ dest;
                core.data.read(&dest, define_mem_type<Value_>(), core.memspace, core.dataspace);
                fill(primary, dest);
            } else {
                fill(primary, 0);
            }
            return true;
        } else {
            return false;
        }
    }

    template<class Function_>
    void extract_secondary_raw_loop(size_t i, Function_ fill, Index_ start, Index_ length, SecondaryWorkspace& core, bool needs_value) const {
        serialize([&]() -> void {
            Index_ end = start + length;
            for (Index_ j = start; j < end; ++j) {
                extract_secondary_raw(j, i, fill, core, needs_value);
            }
        });
    }

    const Value_* extract_secondary(size_t i, Value_* buffer, Index_ start, Index_ length, SecondaryWorkspace& core) const {
        std::fill(buffer, buffer + length, 0);

        extract_secondary_raw_loop(i, 
            [&](Index_ pos, Value_ value) -> void {
                buffer[pos - start] = value;
            }, 
            start, 
            length, 
            core,
            true
        );

        return buffer;
    }

    tatami::SparseRange<Value_, Index_> extract_secondary(size_t i, Value_* dbuffer, Index_* ibuffer, Index_ start, Index_ length, SecondaryWorkspace& core, bool needs_value, bool needs_index) const {
        Index_ counter = 0;

        extract_secondary_raw_loop(i, 
            [&](Index_ pos, Value_ value) -> void {
                if (needs_value) {
                    dbuffer[counter] = value;
                }
                if (needs_index) {
                    ibuffer[counter] = pos;
                }
                ++counter;
            }, 
            start, 
            length, 
            core,
            needs_value
        );

        if (!needs_value) {
            dbuffer = NULL;
        }
        if (!needs_index) {
            ibuffer = NULL;
        }

        return tatami::SparseRange<Value_, Index_>(counter, dbuffer, ibuffer);
    }

    template<class Function_, class Skip_>
    void extract_secondary_raw_loop(size_t i, Function_ fill, Skip_ skip, const std::vector<Index_>& indices, SecondaryWorkspace& core, bool needs_value) const {
        serialize([&]() -> void {
            for (auto j : indices) {
                if (!extract_secondary_raw(j, i, fill, core, needs_value)) {
                    skip();
                }
            }
        });
    }

    const Value_* extract_secondary(size_t i, Value_* buffer, const std::vector<Index_>& indices, SecondaryWorkspace& core) const {
        std::fill(buffer, buffer + indices.size(), 0);
        auto original = buffer;
        extract_secondary_raw_loop(i, 
            [&](Index_, Value_ value) -> void {
                *buffer = value;
                ++buffer;
            }, 
            [&]() -> void {
                ++buffer;
            },
            indices, 
            core,
            true
        );
        return original;
    }

    tatami::SparseRange<Value_, Index_> extract_secondary(size_t i, Value_* dbuffer, Index_* ibuffer, const std::vector<Index_>& indices, SecondaryWorkspace& core, bool needs_value, bool needs_index) const {
        Index_ counter = 0;

        extract_secondary_raw_loop(i, 
            [&](Index_ pos, Value_ value) -> void {
                if (needs_value) {
                    dbuffer[counter] = value;
                }
                if (needs_index) {
                    ibuffer[counter] = pos;
                }
                ++counter;
            }, 
            []() -> void {},
            indices, 
            core,
            needs_value
        );

        if (!needs_value) {
            dbuffer = NULL;
        }
        if (!needs_index) {
            ibuffer = NULL;
        }

        return tatami::SparseRange<Value_, Index_>(counter, dbuffer, ibuffer);
    }

    /******************************************
     ************ Public overrides ************
     ******************************************/
private:
    template<bool accrow_>
    Index_ full_secondary_length() const {
        if constexpr(accrow_) {
            return ncols;
        } else {
            return nrows;
        }
    }

    template<bool accrow_, tatami::DimensionSelectionType selection_, bool sparse_>
    struct Hdf5SparseExtractor : public tatami::Extractor<selection_, sparse_, Value_, Index_> {
        typedef typename std::conditional<row_ == accrow_, PrimaryWorkspace, SecondaryWorkspace>::type CoreWorkspace;

        Hdf5SparseExtractor(const Hdf5CompressedSparseMatrix* p, const tatami::Options& opt) : parent(p) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                this->full_length = parent->template full_secondary_length<accrow_>();
            }

            serialize([&]() -> void {
                core.reset(new CoreWorkspace);

                // TODO: set more suitable chunk cache values here, to avoid re-reading
                // chunks that are only partially consumed.
                core->file.openFile(parent->file_name, H5F_ACC_RDONLY);
                core->data = core->file.openDataSet(parent->data_name);
                core->index = core->file.openDataSet(parent->index_name);
                core->dataspace = core->data.getSpace();
            });

            if constexpr(row_ == accrow_) {
                if (opt.cache_for_reuse) {
                    auto extraction_cache_size = accrow_ ? p->nrows : p->ncols;
                    core->extraction_bounds.resize(extraction_cache_size, std::pair<size_t, size_t>(-1, 0));
                }
            }
        }

        Hdf5SparseExtractor(const Hdf5CompressedSparseMatrix* p, const tatami::Options& opt, Index_ bs, Index_ bl) : Hdf5SparseExtractor(p, opt) {
            if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                this->block_start = bs;
                this->block_length = bl;
            }
        }

        Hdf5SparseExtractor(const Hdf5CompressedSparseMatrix* p, const tatami::Options& opt, std::vector<Index_> idx) : Hdf5SparseExtractor(p, opt) {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                this->index_length = idx.size();
                indices = std::move(idx);
            }
        }

        ~Hdf5SparseExtractor() {
            // Destructor also needs to be made thread-safe;
            // this is why the workspace is a pointer.
            serialize([&]() -> void {
                core.reset();
            });
        }

    protected:
        const Hdf5CompressedSparseMatrix* parent;
        std::unique_ptr<CoreWorkspace> core;
        typename std::conditional<selection_ == tatami::DimensionSelectionType::INDEX, std::vector<Index_>, bool>::type indices;

    public:
        const Index_* index_start() const {
            if constexpr(selection_ == tatami::DimensionSelectionType::INDEX) {
                return indices.data();
            } else {
                return NULL;
            }
        }

    public:
        void set_oracle(std::unique_ptr<tatami::Oracle<Index_> > o) {
            if constexpr(row_ == accrow_) {
                core->futurist.reset(new OracleCache);
                core->futurist->prediction_stream.set(std::move(o));
                core->historian.reset();
            }
        }
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_>
    struct DenseHdf5SparseExtractor : public Hdf5SparseExtractor<accrow_, selection_, false> {
        template<typename... Args_>
        DenseHdf5SparseExtractor(const Hdf5CompressedSparseMatrix* p, const tatami::Options& opt, Args_&&... args) : 
            Hdf5SparseExtractor<accrow_, selection_, false>(p, opt, std::forward<Args_>(args)...)
        {
            if constexpr(row_ == accrow_) {
                this->parent->initialize_lru_cache(this->core->historian, true, true);
            }
        }

        const Value_* fetch(Index_ i, Value_* buffer) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                if constexpr(row_ == accrow_) {
                    return this->parent->extract_primary(i, buffer, 0, this->full_length, *(this->core));
                } else {
                    return this->parent->extract_secondary(i, buffer, 0, this->full_length, *(this->core));
                }
            } else if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                if constexpr(row_ == accrow_) {
                    return this->parent->extract_primary(i, buffer, this->block_start, this->block_length, *(this->core));
                } else {
                    return this->parent->extract_secondary(i, buffer, this->block_start, this->block_length, *(this->core));
                }
            } else {
                if constexpr(row_ == accrow_) {
                    return this->parent->extract_primary(i, buffer, this->indices, *(this->core));
                } else {
                    return this->parent->extract_secondary(i, buffer, this->indices, *(this->core));
                }
            }
        }
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_>
    struct SparseHdf5SparseExtractor : public Hdf5SparseExtractor<accrow_, selection_, true> {
        template<typename... Args_>
        SparseHdf5SparseExtractor(const Hdf5CompressedSparseMatrix* p, const tatami::Options& opt, Args_&&... args) : 
            Hdf5SparseExtractor<accrow_, selection_, true>(p, opt, std::forward<Args_>(args)...), needs_value(opt.sparse_extract_value), needs_index(opt.sparse_extract_index) 
        {
            if constexpr(row_ == accrow_) {
                if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                    if (needs_value || needs_index) {
                        // If the index isn't requested, we don't need to cache it, because
                        // we can just load the entire set of values for the primary dimension.
                        this->parent->initialize_lru_cache(this->core->historian, needs_value, /* needs_cached_index = */ needs_index);
                    } else {
                        // Otherwise the LRU cache is not needed by fetch(), so
                        // we skip its initialization to avoid divide by zero
                        // problems inside initialize_lru_cache when each
                        // non-zero cache element takes up zero bytes.
                    }
                } else {
                    this->parent->initialize_lru_cache(this->core->historian, needs_value, true);
                }
            }
        }

        tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
            if constexpr(selection_ == tatami::DimensionSelectionType::FULL) {
                if constexpr(row_ == accrow_) {
                    if (needs_index || needs_value) {
                        auto flen = this->parent->template full_secondary_length<accrow_>();
                        return this->parent->extract_primary(i, vbuffer, ibuffer, 0, this->full_length, *(this->core), needs_value, needs_index, flen);
                    } else {
                        // Quick return is possible if we don't need any indices or values.
                        return tatami::SparseRange<Value_, Index_>(this->parent->pointers[i+1] - this->parent->pointers[i], NULL, NULL);
                    }
                } else {
                    return this->parent->extract_secondary(i, vbuffer, ibuffer, 0, this->full_length, *(this->core), needs_value, needs_index);
                }

            } else if constexpr(selection_ == tatami::DimensionSelectionType::BLOCK) {
                if constexpr(row_ == accrow_) {
                    auto flen = this->parent->template full_secondary_length<accrow_>();
                    return this->parent->extract_primary(i, vbuffer, ibuffer, this->block_start, this->block_length, *(this->core), needs_value, needs_index, flen);
                } else {
                    return this->parent->extract_secondary(i, vbuffer, ibuffer, this->block_start, this->block_length, *(this->core), needs_value, needs_index);
                }

            } else {
                if constexpr(row_ == accrow_) {
                    return this->parent->extract_primary(i, vbuffer, ibuffer, this->indices, *(this->core), needs_value, needs_index);
                } else {
                    return this->parent->extract_secondary(i, vbuffer, ibuffer, this->indices, *(this->core), needs_value, needs_index);
                }
            }
        }

    protected:
        bool needs_value;
        bool needs_index;
    };

    template<bool accrow_, tatami::DimensionSelectionType selection_, bool sparse_, typename ... Args_>
    std::unique_ptr<tatami::Extractor<selection_, sparse_, Value_, Index_> > populate(const tatami::Options& opt, Args_&&... args) const {
        std::unique_ptr<tatami::Extractor<selection_, sparse_, Value_, Index_> > output;

        if constexpr(sparse_) {
            output.reset(new SparseHdf5SparseExtractor<accrow_, selection_>(this, opt, std::forward<Args_>(args)...));
        } else {
            output.reset(new DenseHdf5SparseExtractor<accrow_, selection_>(this, opt, std::forward<Args_>(args)...));
        }

        return output;
    }

public:
    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_row(const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::FULL, false>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_row(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::BLOCK, false>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_row(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::INDEX, false>(opt, std::move(indices));
    }

    std::unique_ptr<tatami::FullDenseExtractor<Value_, Index_> > dense_column(const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::FULL, false>(opt);
    }

    std::unique_ptr<tatami::BlockDenseExtractor<Value_, Index_> > dense_column(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::BLOCK, false>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexDenseExtractor<Value_, Index_> > dense_column(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::INDEX, false>(opt, std::move(indices));
    }

public:
    std::unique_ptr<tatami::FullSparseExtractor<Value_, Index_> > sparse_row(const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<tatami::BlockSparseExtractor<Value_, Index_> > sparse_row(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexSparseExtractor<Value_, Index_> > sparse_row(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<true, tatami::DimensionSelectionType::INDEX, true>(opt, std::move(indices));
    }

    std::unique_ptr<tatami::FullSparseExtractor<Value_, Index_> > sparse_column(const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::FULL, true>(opt);
    }

    std::unique_ptr<tatami::BlockSparseExtractor<Value_, Index_> > sparse_column(Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::BLOCK, true>(opt, block_start, block_length);
    }

    std::unique_ptr<tatami::IndexSparseExtractor<Value_, Index_> > sparse_column(std::vector<Index_> indices, const tatami::Options& opt) const {
        return populate<false, tatami::DimensionSelectionType::INDEX, true>(opt, std::move(indices));
    }
};

}

#endif
