cmake_minimum_required(VERSION 3.14)

project(tatami_hdf5
    VERSION 1.0.0
    DESCRIPTION "tatami bindings for HDF5"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(tatami_hdf5 INTERFACE)

add_subdirectory(extern)

target_link_libraries(tatami_hdf5 INTERFACE tatami_chunked)

# Switch between include directories depending on whether the downstream is
# using the build directly or is using the installed package.
target_include_directories(tatami_hdf5 
    INTERFACE 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Building the test-related machinery, if we are compiling this library directly.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(INSTALL_GTEST OFF)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()
