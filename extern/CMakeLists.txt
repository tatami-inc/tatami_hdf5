cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami_chunked
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_chunked
  GIT_TAG 4153bed77daf9f92785bdde8ec4d09d4c12d6f53
)

FetchContent_MakeAvailable(tatami_chunked)

