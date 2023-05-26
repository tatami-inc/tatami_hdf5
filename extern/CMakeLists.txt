cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG 7b923545f225b75bdee7e801db9ab4edb5aa9d24
)

FetchContent_MakeAvailable(tatami)
