cmake_minimum_required(VERSION 3.24)

project(tatami_hdf5
    VERSION 2.0.3
    DESCRIPTION "tatami bindings for HDF5"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(tatami_hdf5 INTERFACE)
add_library(tatami::tatami_hdf5 ALIAS tatami_hdf5)

option(TATAMI_HDF5_FETCH_EXTERN "Automatically fetch tatami_hdf5's external dependencies." ON)
if(TATAMI_HDF5_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(tatami_tatami_chunked 2.0.0 CONFIG REQUIRED)
    find_package(ltla_subpar 0.4.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.0 CONFIG REQUIRED)
endif()

target_link_libraries(tatami_hdf5 INTERFACE tatami::tatami_chunked ltla::subpar ltla::sanisizer)

option(TATAMI_HDF5_FIND_HDF5 "Try to find and link to HDF5 for tatami_hdf5." ON)
if(TATAMI_HDF5_FIND_HDF5)
    find_package(HDF5 COMPONENTS C CXX)
    if (HDF5_FOUND)
        target_link_libraries(tatami_hdf5 INTERFACE hdf5::hdf5 hdf5::hdf5_cpp)
    endif()
endif()

# Switch between include directories depending on whether the downstream is
# using the build directly or is using the installed package.
include(GNUInstallDirs)
target_include_directories(tatami_hdf5 
    INTERFACE 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_hdf5>"
)

# Building the test-related machinery, if we are compiling this library directly.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(TATAMI_HDF5_TESTS "Build tatami_hdf5's test suite." ON)
else()
    option(TATAMI_HDF5_TESTS "Build tatami_hdf5's test suite." OFF)
endif()

if(TATAMI_HDF5_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Installing for find_package.
include(CMakePackageConfigHelpers)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_hdf5)

install(TARGETS tatami_hdf5
    EXPORT tatami_hdf5Targets)

install(EXPORT tatami_hdf5Targets
    FILE tatami_tatami_hdf5Targets.cmake
    NAMESPACE tatami::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_hdf5)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_hdf5Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_hdf5)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_hdf5ConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_hdf5Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_hdf5ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_hdf5)
