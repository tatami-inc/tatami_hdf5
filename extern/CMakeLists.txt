cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami_chunked
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_chunked
  GIT_TAG 1b79c177ff7bdb19075f8a4c6e6d0a51071ccfd8
)

FetchContent_MakeAvailable(tatami_chunked)

