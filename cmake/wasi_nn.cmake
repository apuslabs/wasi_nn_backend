
find_package(cjson REQUIRED)
find_package(llamacpp REQUIRED)

add_library(wasi_nn_llamacpp SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/wasi_nn_llamacpp.c
)

target_include_directories(wasi_nn_llamacpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${cjson_SOURCE_DIR}
)

# Link libraries
target_link_libraries(wasi_nn_llamacpp
    PUBLIC
        cjson
        common
        ggml
        llama
)

# Install
install(TARGETS wasi_nn_llamacpp
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)