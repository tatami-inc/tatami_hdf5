# tatami for HDF5 matrices

![Unit tests](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_hdf5/branch/master/graph/badge.svg?token=Z189ORCLLR)](https://codecov.io/gh/tatami-inc/tatami_hdf5)

## Overview

This repository implements [**tatami**](https://github.com/tatami-inc/tatami) bindings for HDF5-backed matrices,
allowing some level of random access without loading the entire dataset into memory.
Matrices can be conventionally stored as 2-dimensional HDF5 datasets,
or in an _ad hoc_ compressed sparse format with a 1-dimensional dataset for each component (data, indices, pointers).

## Quick start

**tatami_hdf5** is a header-only library, so it can be easily used by just `#include`ing the relevant source files:

```cpp
#include "tatami_hdf5/tatami_hdf5.hpp"

// Dense HDF5 datasets.
tatami_hdf5::Hdf5DenseMatrix<double, int> dense_mat("some_file.h5", "dataset_name");

// Compressed sparse data stored in an ad hoc group.
tatami_hdf5::Hdf5CompressedSparseMatrix<false, double, int> sparse_mat(
    nrow,
    ncol,
    "some_file.h5", 
    "group_name/data",
    "group_name/index",
    "group_name/ptrs"
);
```

In cases where performance is more important than memory consumption, we also provide some utilities to quickly create in-memory **tatami** matrices from their HDF5 representations:

```cpp
auto dense_mat_mem = tatami_hdf5::load_hdf5_dense_matrix<double, int>(
    "some_file.h5", 
    "dataset_name"
);

auto sparse_mat_mem = tatami_hdf5::load_hdf5_compressed_sparse_matrix<false, double>(
    nrow,
    ncol,
    "some_file.h5", 
    "group_name/data",
    "group_name/index",
    "group_name/ptrs"
);
```

We can also write a **tatami** sparse matrix into a HDF5 file:

```cpp
H5::H5File fhandle("some_file2.h5", H5F_ACC_TRUNC);
auto ghandle = fhandle.createGroup("group_name");
tatami_hdf5::write_sparse_matrix_to_hdf5(&sparse_mat_mem, ghandle); 
```

Check out the [reference documentation](https://tatami-inc.github.io/tatami_hdf5) for more details.

## Building projects

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  tatami_hdf5
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_hdf5
  GIT_TAG master # or any version of interest 
)

FetchContent_MakeAvailable(tatami_hdf5)
```

Then you can link to **tatami_hdf5** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe tatami_hdf5)

# For libaries
target_link_libraries(mylib INTERFACE tatami_hdf5)
```

You'll also need to link to the HDF5 library yourself (version 1.10 or higher).
In CMake, this is typically done by discovering the system library as shown below.
Specific frameworks may come with their own HDF5 binaries, e.g., [**Rhdf5lib**](https://bioconductor.org/packages/Rhdf5lib) - 
**tatami_hdf5** does not put restrictions on any particular HDF5 installation.

```cmake
find_package(HDF5 COMPONENTS C CXX REQUIRED)
target_link_libraries(myexe hdf5::hdf5 hdf5::hdf5_cpp)
```

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This will also require the core [**tatami**](https://github.com/tatami-inc/tatami) library.
