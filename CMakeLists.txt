cmake_minimum_required(VERSION 3.0)

project(energy-model)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(Tools)

add_executable(Prova Prova.cpp)

target_link_libraries(Prova Tools)