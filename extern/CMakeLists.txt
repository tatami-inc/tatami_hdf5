cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami_chunked
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_chunked
  GIT_TAG master
)

FetchContent_MakeAvailable(tatami_chunked)

