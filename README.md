# tatami for HDF5 matrices

![Unit tests](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_hdf5/branch/master/graph/badge.svg?token=Z189ORCLLR)](https://codecov.io/gh/tatami-inc/tatami_hdf5)

This repository implements [**tatami**](https://github.com/tatami-inc/tatami) bindings for HDF5-backed matrices,
allowing some level of random access without loading the entire dataset into memory.
Matrices can be conventionally stored as 2-dimensional HDF5 datasets,
or in a compressed sparse format with a 1-dimensional dataset for each component (data, indices, pointers).

```cpp
#include "tatami_hdf5/tatami_hdf5.hpp"

tatami_hdf5::Hdf5DenseMatrix<double, int> dense_mat("some_file.h5", "dataset_name");
tatami_hdf5::Hdf5CompressedSparseMatrix<false, double, int> sparse_mat("some_file.h5", "group_name");
```

In cases where performance is more important than memory consumption, we also provide some utilities to quickly create in-memory **tatami** matrices from their HDF5 representations:

```cpp
auto dense_mat_mem = tatami_hdf5::load_hdf5_dense_matrix("some_file.h5", "dataset_name");
auto sparse_mat_mem = tatami_hdf5::load_hdf5_compressed_sparse_matrix("some_file.h5", "group_name");
```

We can also write a **tatami** sparse matrix into a HDF5 file:

```cpp
H5::H5File fhandle("some_file2.h5", H5F_ACC_TRUNC);
auto ghandle = fhandle.createGroup("group_name");
tatami_hdf5::write_sparse_matrix_to_hdf5(&sparse_mat_mem, ghandle); 
```

Check out the [reference documentation](https://tatami-inc.github.io/tatami_hdf5) for more details.
