
find_package(cjson REQUIRED)
find_package(llamacpp REQUIRED)

add_library(
    wasi_nn_backend
    SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/src/wasi_nn_llama.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/server/server.cpp
)


target_include_directories(
    wasi_nn_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/server>
        ${cjson_SOURCE_DIR}
)

# Link libraries
target_link_libraries(
    wasi_nn_backend
    PUBLIC
        cjson
        common
        ggml
        llama
)

# Install
install(TARGETS wasi_nn_backend
    LIBRARY DESTINATION lib
)
