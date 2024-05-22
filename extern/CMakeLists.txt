include(FetchContent)

FetchContent_Declare(
  tatami_chunked
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_chunked
  GIT_TAG master # ^v2.0.0
)

FetchContent_MakeAvailable(tatami_chunked)

