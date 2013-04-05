cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(multisvmpathplanning)

set(OpenCV_DIR "/usr/local/share/OpenCV")
# set(PCL_ROOT "/usr/local")

# find_package(OMPL    REQUIRED )
find_package(Boost REQUIRED)
find_package(CUDA 5.0 REQUIRED)
find_package(OpenCV  REQUIRED )
find_package( Eigen3    REQUIRED )
find_package(PCL REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# add_subdirectory(tutorials)
add_subdirectory(src)

message(${EXECUTABLE_OUTPUT_PATH})