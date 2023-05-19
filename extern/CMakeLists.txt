cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG 0b3dfba5b040a0a93b818f955ed4b2a9a10e672d
)

FetchContent_MakeAvailable(tatami)
