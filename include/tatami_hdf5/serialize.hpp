#ifndef TATAMI_HDF5_SERIALIZE_HPP
#define TATAMI_HDF5_SERIALIZE_HPP

/**
 * @file serialize.hpp
 * @brief Default locking for serial access.
 */

namespace tatami_hdf5 {

/**
 * @cond
 */
#ifndef TATAMI_HDF5_PARALLEL_LOCK
#ifndef _OPENMP
#include <thread>
#include <mutex>

inline auto& get_default_hdf5_lock() {
    static std::mutex hdf5_lock;
    return hdf5_lock;
}
#endif
#endif
/**
 * @endcond
 */

/**
 * Serialize a function's execution to avoid simultaneous calls to the (non-thread-safe) HDF5 library.
 *
 * If the `TATAMI_HDF5_PARALLEL_LOCK` macro is defined, it should be a function-like macro that accepts `f` and executes it in a serial section.
 *
 * If OpenMP is available, serialization is achieved by running `f` inside OpenMP critical regions named `"hdf5"`.
 *
 * Otherwise, we use a mutex from the standard `<thread>` library to guard the execution of `f`.
 *
 * @param f Function to be run in a serial section.
 * This accepts no arguments and returns no outputs.
 *
 */
template<class Function_>
void serialize(Function_ f) {
#ifdef TATAMI_HDF5_PARALLEL_LOCK
    TATAMI_HDF5_PARALLEL_LOCK(f);
#else
#ifdef _OPENMP
    #pragma omp critical hdf5
    {
        f();
    }
#else
    static std::mutex hdf5_lock;
    std::lock_guard<std::mutex> thing(get_hdf5_lock());
    f();
#endif
#endif
}

}

#endif
