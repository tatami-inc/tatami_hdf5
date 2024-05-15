include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

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

    target_link_libraries(${target} gtest_main tatami_hdf5)

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
