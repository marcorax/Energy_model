cmake_minimum_required(VERSION 3.0)

project(energy-model)

find_package( OpenCV REQUIRED )
find_package( OpenCL REQUIRED )

include_directories("${PROJECT_SOURCE_DIR}/Tools" "${OpenCV_INCLUDE_DIRS}" "${OpenCL_INCLUDE_DIRS}")

add_subdirectory(Tools)

add_executable(Host Host.cpp)

target_link_libraries(Host Tools "${OpenCV_LIBS}" "${OpenCL_LIBRARY}")


