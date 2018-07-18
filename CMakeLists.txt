cmake_minimum_required(VERSION 3.0)

project(energy-model)

include_directories("${PROJECT_SOURCE_DIR}/Tools")

add_subdirectory(Tools)

add_executable(Prova Prova.cpp)

target_link_libraries(Prova Tools)