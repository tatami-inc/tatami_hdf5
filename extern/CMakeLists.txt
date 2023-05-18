cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/LTLA/tatami
  GIT_TAG eddc3350b290adcd69a226ffb18bcf3bfe0020ef
)

FetchContent_MakeAvailable(tatami)
