#ifndef HDF5_CUSTOM_LOCK_H
#define HDF5_CUSTOM_LOCK_H

#include <thread>
#include <mutex>
#include <vector>
#include <cmath>

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
