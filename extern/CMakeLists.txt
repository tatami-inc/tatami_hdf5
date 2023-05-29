cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG ce581970b1bbbec4e7a5b34eea411c429fac576d
)

FetchContent_MakeAvailable(tatami)
