find_package(cjson REQUIRED)
find_package(llamacpp REQUIRED)

# Build mtmd library from llama.cpp
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/tools/mtmd mtmd)

# Generate the required static assets for server
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

# Create generated directory for assets
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Generate asset headers
set(GENERATED_ASSETS)
foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/tools/server/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/generated/${asset}.hpp")
    list(APPEND GENERATED_ASSETS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

# Create the llama_server library
add_library(
    llama_server
    SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/llama_server.cpp
        ${GENERATED_ASSETS}
)

target_include_directories(
    llama_server
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/tools/server
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp/tools/llava
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${cjson_SOURCE_DIR}
)

# Link libraries for llama_server
target_link_libraries(
    llama_server
    PUBLIC
        cjson
    PRIVATE
        common
        mtmd
        ggml
        llama
        pthread
)

# Set C++ standard
target_compile_features(llama_server PRIVATE cxx_std_17)

# Install library
install(TARGETS llama_server
    LIBRARY DESTINATION lib
)

# Build test executable (optional)
if(BUILD_TESTING)
    add_executable(test_llama_server 
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_llama_server.c
    )
    
    target_link_libraries(test_llama_server
        llama_server
        pthread
    )
    
    # Build example
    add_executable(simple_inference
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_inference.c
    )
    
    target_link_libraries(simple_inference
        llama_server
    )
endif()
