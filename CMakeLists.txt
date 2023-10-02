cmake_minimum_required(VERSION 3.25)
project(bvh_quantize)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(src/bvh)

add_executable(bvh_quantize src/main.cpp)
target_link_libraries(bvh_quantize PUBLIC bvh)

add_executable(vanilla profiling/vanilla.cpp)
target_link_libraries(vanilla PUBLIC bvh)

add_executable(quantized profiling/quantized.cpp)
target_link_libraries(quantized PUBLIC bvh)
