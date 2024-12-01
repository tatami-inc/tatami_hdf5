include(FetchContent)
FetchContent_Declare(
    tatami_test
    GIT_REPOSITORY https://github.com/tatami-inc/tatami_test
    GIT_TAG master
)
FetchContent_MakeAvailable(tatami_test)

include(CheckIncludeFiles)
check_include_files(filesystem HAVE_CXX_FS)

set(CODE_COVERAGE OFF CACHE BOOL "Enable coverage testing")

include(GoogleTest)

add_executable(
    libtest 
    src/load_dense_matrix.cpp
    src/load_compressed_sparse_matrix.cpp
    src/write_compressed_sparse_matrix.cpp
    src/DenseMatrix.cpp
    src/CompressedSparseMatrix.cpp
)

macro(decorate_test target)
    target_compile_definitions(${target} PRIVATE DEBUG=1)

    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)

    target_link_libraries(${target} tatami_hdf5 tatami_test)

    if (NOT HAVE_CXX_FS) 
        target_link_libraries(${target} stdc++fs) 
    endif()

    if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

decorate_test(libtest)

add_executable(
    cuslocktest
    src/write_compressed_sparse_matrix.cpp
    src/DenseMatrix.cpp
    src/CompressedSparseMatrix.cpp
)
decorate_test(cuslocktest)
target_compile_definitions(cuslocktest PRIVATE TATAMI_HDF5_TEST_PARALLEL_ONLY=1 TATAMI_HDF5_TEST_CUSTOM_LOCK=1)

find_package(OpenMP)
if(OpenMP_FOUND)
    add_executable(
        omptest
        src/write_compressed_sparse_matrix.cpp
        src/DenseMatrix.cpp
        src/CompressedSparseMatrix.cpp
    )
    decorate_test(omptest)
    target_link_libraries(omptest OpenMP::OpenMP_CXX)
    target_compile_definitions(omptest PRIVATE TATAMI_HDF5_TEST_PARALLEL_ONLY=1)
endif()
