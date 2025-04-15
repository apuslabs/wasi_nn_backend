cmake_minimum_required(VERSION 3.13) # Increased for FetchContent details
project(llama_runtime_project C CXX CUDA) # Add CUDA language

set(CMAKE_CXX_STANDARD 17) # Use a modern C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_JOB_POOLS compile_job_pool=16 link_job_pool=8)
set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
set(CMAKE_JOB_POOL_LINK link_job_pool)

# Basic settings for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

# --- Visibility Setup (for exporting symbols from shared library) ---
# Check if we are building a shared library target named 'llama_runtime'
# This allows the C++ code to use LLAMA_RUNTIME_BUILD_SHARED
set(LLAMA_RUNTIME_TARGET_NAME "llama_runtime") # Name of our shared library target



# --- Compiler Flags ---
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) # Default to Release for performance
  message(STATUS "Build type not set, defaulting to Release")
endif()

if (NOT WIN32)
  # Common flags for GCC/Clang
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -ffunction-sections -fdata-sections -Wno-unused-parameter -Wno-pedantic -fvisibility=hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wformat -Wformat-security -Wno-unused-parameter -fvisibility=hidden")
  # Remove extra spaces
  string(REGEX REPLACE "  +" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REGEX REPLACE "  +" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
  # Add Windows specific flags if needed
endif()


# --- CMake Modules Path ---
# Ensure Find*.cmake files are here
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# --- Include Library Build Logic ---
# This file contains the add_library() command for llama_runtime
include(build_llama_runtime)



