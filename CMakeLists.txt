cmake_minimum_required(VERSION 3.16)
project(opencv_project)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_project main.cpp)
target_link_libraries(opencv_project ${OpenCV_LIBS})

