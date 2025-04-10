# Find dependencies (using the Find*.cmake modules)
find_package(cjson REQUIRED)     # If needed by your implementation
find_package(llamacpp REQUIRED) # This now sets LLAMA_CPP_BINARY_DIR

# --- Define the Shared Library Target ---
add_library(llama_runtime SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llama_runtime.cpp
)

target_compile_definitions(llama_runtime PRIVATE LLAMA_RUNTIME_BUILD_SHARED)

# --- Include Directories ---
target_include_directories(llama_runtime
    PUBLIC
        # Directory containing the public header (llama_runtime.h)
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${cjson_SOURCE_DIR} # If cJSON headers are used internally
)



# --- Link Dependencies ---
target_link_libraries(llama_runtime
    PUBLIC
        cjson     # Link cJSON statically
        common
        llama
        ggml
)

# --- Installation ---
# Install the shared library and the public header file
install(TARGETS llama_runtime
    LIBRARY DESTINATION lib  
)


message(STATUS "Target 'llama_runtime' defined.")
