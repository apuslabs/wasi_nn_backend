# Copyright (C) 2019 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

set(CJSON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/cjson")
if(EXISTS ${CJSON_SOURCE_DIR})
  message("Use existed source code under ${CJSON_SOURCE_DIR}")
  FetchContent_Declare(
    cjson
    SOURCE_DIR     ${CJSON_SOURCE_DIR}
  )
else()
  message("download source code and store it at ${CJSON_SOURCE_DIR}")
  FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG        v1.7.18
    SOURCE_DIR     ${CJSON_SOURCE_DIR}
  )
endif()

set(ENABLE_CJSON_TEST OFF CACHE BOOL "Disable cJSON tests" FORCE)
set(ENABLE_CJSON_UNINSTALL OFF CACHE BOOL "Disable cJSON uninstall target" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build cJSON as a static library" FORCE) # Build static

FetchContent_MakeAvailable(cjson)

message(STATUS "cJSON available as target 'cjson'")
