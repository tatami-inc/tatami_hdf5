#ifndef TATAMI_HDF5_SERIALIZE_HPP
#define TATAMI_HDF5_SERIALIZE_HPP

/**
 * @file serialize.hpp
 * @brief Default locking for serial access.
 */

#ifndef TATAMI_HDF5_PARALLEL_LOCK
#include "subpar/subpar.hpp"
#ifndef SUBPAR_USES_OPENMP
#include <mutex>
#include <thread>
#endif
#endif

namespace tatami_hdf5 {

/**
 * Serialize a function's execution to avoid simultaneous calls to the (non-thread-safe) HDF5 library.
 * This is primarily intended for use inside `tatami::parallelize()` but can also be called anywhere that uses the same parallelization scheme.
 * Also check out the [**subpar**](https://ltla.github.io/subpar) library, which implements the default parallelization scheme for `tatami::parallelize()`.
 *
 * The default serialization mechanism is automatically determined from the definition of the `SUBPAR_USES_OPENMP` macro.
 * If defined (i.e., OpenMP is used), `f` is executed in OpenMP critical regions named `"hdf5"`.
 * Otherwise, a global mutex from `<mutex>` is used to guard the execution of `f`.
 *
 * If a custom parallelization scheme is defined via `TATAMI_CUSTOM_PARALLEL` or `SUBPAR_CUSTOM_PARALLEL`, the default serialization mechanism may not be appropriate.
 * Users should instead define a `TATAMI_HDF5_PARALLEL_LOCK` function-like macro that accepts `f` and executes it in a serial section appropriate to the custom scheme.
 * Once defined, this user-defined lock will be used in all calls to `serialize()`.
 *
 * @param f Function to be run in a serial section.
 * This accepts no arguments and returns no outputs.
 */
template<class Function_>
void serialize(Function_ f) {
#ifdef TATAMI_HDF5_PARALLEL_LOCK
    TATAMI_HDF5_PARALLEL_LOCK(f);
#else
#ifdef SUBPAR_USES_OPENMP
    #pragma omp critical(hdf5)
    {
        f();
    }
#else
    static std::mutex h5lock; // global across all threads.
    std::lock_guard<std::mutex> thing(h5lock);
    f();
#endif
#endif
}

}

#endif
