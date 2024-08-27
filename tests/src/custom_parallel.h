#ifndef HDF5_CUSTOM_LOCK_H
#define HDF5_CUSTOM_LOCK_H

// Only define these locks if OpenMP is not available. If OpenMP is present,
// tatami::parallelize will switch to using it, and we want to use the in-built
// OpenMP critical regions instead of using this mutex-based lock.

#ifndef _OPENMP
#ifdef TATAMI_HDF5_TEST_CUSTOM_LOCK
#include <thread>
#include <mutex>

inline auto& get_hdf5_lock() {
    static std::mutex hdf5_lock;
    return hdf5_lock;
}

template<class Function>
void hdf5_serialize(Function f) {
    std::lock_guard<std::mutex> thing(get_hdf5_lock());
    f();
}

#define TATAMI_HDF5_PARALLEL_LOCK hdf5_serialize
#endif

#endif
#endif
