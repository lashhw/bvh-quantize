cmake_minimum_required(VERSION 3.25)
project(bvh_quantize)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(bvh)

add_executable(bvh_quantize main.cpp)
target_link_libraries(bvh_quantize PUBLIC bvh)
