# Copyright (C) 2019 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Find or download cJSON using FetchContent

include(FetchContent)

# Prefer local source if available, otherwise download
set(CJSON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/cjson") # Adjust path if needed
if(EXISTS ${CJSON_SOURCE_DIR} AND IS_DIRECTORY ${CJSON_SOURCE_DIR})
  message(STATUS "cJSON: Using existing source at ${CJSON_SOURCE_DIR}")
  FetchContent_Declare(
    cjson
    SOURCE_DIR     ${CJSON_SOURCE_DIR}
  )
else()
  message(STATUS "cJSON: Downloading source to ${CJSON_SOURCE_DIR}")
  FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG        v1.7.18 # Or choose a specific commit/tag
    SOURCE_DIR     ${CJSON_SOURCE_DIR}
    GIT_SHALLOW    TRUE # Optional: Faster download
  )
endif()

# Configure cJSON build (disable tests/uninstall)
set(ENABLE_CJSON_TEST OFF CACHE BOOL "Disable cJSON tests" FORCE)
set(ENABLE_CJSON_UNINSTALL OFF CACHE BOOL "Disable cJSON uninstall target" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build cJSON as a static library" FORCE) # Build static

FetchContent_MakeAvailable(cjson)

message(STATUS "cJSON available as target 'cjson'")
