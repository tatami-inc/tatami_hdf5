include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test HDF5 capabilities.
find_package(HDF5 COMPONENTS C CXX REQUIRED)
add_executable(
    h5test 
    src/Hdf5DenseMatrix.cpp
    src/Hdf5CompressedSparseMatrix.cpp
    src/load_hdf5_matrix.cpp
    src/write_sparse_matrix_to_hdf5.cpp
)
target_compile_definitions(h5test PRIVATE DEBUG=1)

# Also checking the custom parallelization in the HDF5 libraries.
add_executable(
    cusparh5test
    src/Hdf5DenseMatrix.cpp
    src/Hdf5CompressedSparseMatrix.cpp
)
target_compile_definitions(cusparh5test PRIVATE TEST_CUSTOM_PARALLEL=1)

target_link_libraries(h5test gtest_main tatami_hdf5 hdf5::hdf5 hdf5::hdf5_cpp)
target_link_libraries(cusparh5test gtest_main tatami_hdf5 hdf5::hdf5 hdf5::hdf5_cpp)

include(CheckIncludeFiles)
check_include_files(filesystem HAVE_CXX_FS)
if (NOT HAVE_CXX_FS) 
    target_link_libraries(h5test stdc++fs) 
    target_link_libraries(cusparh5test stdc++fs) 
endif()

# Checking whether to add OpenMP support. This is turned off
# by default to make it easier to debug test failures.
set(USE_OPENMP OFF CACHE BOOL "Compile with OpenMP support")
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(h5test OpenMP::OpenMP_CXX)
endif()

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(h5test PRIVATE -O0 -g --coverage)
    target_link_options(h5test PRIVATE --coverage)
endif()

# Making the tests discoverable.
include(GoogleTest)
gtest_discover_tests(h5test)
gtest_discover_tests(cusparh5test)
