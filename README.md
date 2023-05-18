# tatami for HDF5 matrices

![Unit tests](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/run-tests.yaml/badge.svg)
![Gallery](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/run-gallery.yaml/badge.svg)
![Documentation](https://github.com/tatami-inc/tatami_hdf5/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/tatami-inc/tatami_hdf5/branch/master/graph/badge.svg?token=Z189ORCLLR)](https://codecov.io/gh/tatami-inc/tatami_hcdf5)

This repository implements [**tatami**](https://github.com/tatami-inc/tatami) bindings for HDF5-backed matrices,
allowing some level of random access without loading the entire dataset into memory.
Matrices can be conventionally stored as 2-dimensional HDF5 datasets,
or in a compressed sparse format with a 1-dimensional dataset for each component (data, indices, pointers).
