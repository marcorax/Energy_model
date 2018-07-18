cmake_minimum_required(VERSION 3.0)

project(energy-model)

find_package( OpenCV REQUIRED )

include_directories("${PROJECT_SOURCE_DIR}/Tools" "${OpenCV_INCLUDE_DIRS}")

add_subdirectory(Tools)

add_executable(Prova Prova.cpp)

target_link_libraries(Prova Tools ${OpenCV_LIBS})


