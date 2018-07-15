cmake_minimum_required(VERSION 3.0)

project(energy-model)

include_directories("${PROJECT_SOURCE_DIR}/Tools" "${PROJECT_SOURCE_DIR}/Kernels")

add_executable(Prova Prova.cpp Davisloading.cpp)