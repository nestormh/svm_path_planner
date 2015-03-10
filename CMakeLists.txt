cmake_minimum_required(VERSION 2.8.3)
project(svm_path_planner)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    costmap_2d
    pluginlib
    nav_core
    base_local_planner
    tf
    angles
    pcl_ros
    cv_bridge
)

set(ROS_BUILD_TYPE Release)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ffast-math -unroll-loops -march=native -fopenmp -msse3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -frounding-math -ffast-math -unroll-loops -march=native -fopenmp -msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -frounding-math -unroll-loops -march=native -fopenmp -msse3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -unroll-loops -march=native -std=c++0x -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -ffast-math -march=native -std=c++0x")
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(catkin)
find_package(cmake_modules REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA 5.0 REQUIRED)
find_package(OpenCV  REQUIRED )
find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)
find_package(PCL 1.6 REQUIRED)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES svm_path_planner
 CATKIN_DEPENDS 
    roscpp
    costmap_2d
    pluginlib
    nav_core
    base_local_planner
    tf
    angles
    cv_bridge
    pcl_ros
    pcl
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

# SVM Path Planner
set(CCFILES
  src/SVMPathPlanner/gpudt/predicates.cpp
  src/SVMPathPlanner/gpudt/gpudt.cpp
  src/SVMPathPlanner/GPULibSVM/svm-train.c
  src/SVMPathPlanner/GPULibSVM/svm.cpp
  src/SVMPathPlanner/MSVMPP/svmpathplanning.cpp
  src/SVMPathPlanner/MSVMPP/svmpathplanningsingle.cpp
  src/SVMPathPlanner/MSVMPP/voronoipathplanning.cpp
  src/SVMPathPlanner/MSVMPP/voronoisvmpathplanning.cpp
#   src/svm_path_planner_ros.cpp
)

set (HFILES
  src/SVMPathPlanner/gpudt/gpudt.h
  src/SVMPathPlanner/GPULibSVM/svm.h
)

set (CUHFILES
  src/SVMPathPlanner/gpudt/Cuda/common.h
  src/SVMPathPlanner/gpudt/Cuda/cudaDecl.h
  src/SVMPathPlanner/gpudt/Cuda/cudaCCW.h
  src/SVMPathPlanner/gpudt/Cuda/kernelVoronoi.h
  src/SVMPathPlanner/gpudt/Cuda/pba2DKernel.h
  src/SVMPathPlanner/gpudt/predicates.h  
)

set(CUFILES
  src/SVMPathPlanner/gpudt/Cuda/cudaMain.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaVoronoi.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaReconstruction.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaShifting.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaMissing.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaBoundary.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaConstraint.cu
  src/SVMPathPlanner/gpudt/Cuda/cudaFlipping.cu
  src/SVMPathPlanner/gpudt/Cuda/pba2DHost.cu
  src/SVMPathPlanner/MSVMPP/cuda/predictSVM.cu
  src/SVMPathPlanner/MSVMPP/cuda/checkEdges.cu
)

set(HFILES_PUBLIC
)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)

set(CUDAGDB_OPTS -g -G)

# set(CUDA_BIND_COMPILER --compiler-bindir /home/nestor/Dropbox/projects/MultiSVMPathPlanning/compiler_bind)

set(CUDA_VERBOSE_PTXAS FALSE)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

set(CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
set (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(CUDA_LIB_TYPE STATIC) 
set(OTHER_FLAGS -frounding-math)

cuda_add_library(SVMPathPlanningCUDA  ${CUDA_LIB_TYPE} 
  ${CUFILES} 
#   ${CCFILES} 
#   ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS} ${CUDAGDB_OPTS}
#   ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS} ${CUDAGDB_OPTS}
)
  
include_directories(SVMPathPlanningCUDA
    src/SVMPathPlanner
#     ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}/pcl-1.7
    ${Boost_INCLUDE_DIR}
    emon.a
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(SVMPathPlanningCUDA
#   ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${CGAL_LIBRARY} 
  ${CGAL_Core_LIBRARY}
  ${GMP_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
  emon
)

# CUDA_ADD_CUBLAS_TO_TARGET(SVMPathPlanningCUDA)

add_library(svm_path_planner 
  ${CCFILES} 
  ${HFILES}
  src/svm_path_planner_ros.cpp)
target_link_libraries(svm_path_planner 
    ${PCL_LIBRARIES}
    SVMPathPlanningCUDA)
include_directories(svm_path_planner
                    src/SVMPathPlanner)
                    
# add_executable(testInflation src/testInflation.cpp)
# include_directories(testInflation ${OpenCV_INCLUDE_DIR})
# 
# target_link_libraries(testInflation ${OpenCV_LIBS})

# testSVMPathPlanning
# add_executable(testSVMPathPlanning src/SVMPathPlanner/testSVMPathPlanning.cpp)
# include_directories(testSVMPathPlanning 
#     ${OpenCV_INCLUDE_DIR}
# #     ${PCL_INCLUDE_DIRS}
#     src/SVMPathPlanner)
# 
# target_link_libraries(testSVMPathPlanning 
# #     ${PCL_LIBRARIES}
#     ${OpenCV_LIBS}
#     ${ROS_LIBRARIES}
#     SVMPathPlanningCUDA)

## Declare a cpp library
# add_library(svm_path_planner
#   src/${PROJECT_NAME}/svm_path_planner.cpp
# )

## Declare a cpp executable
# add_executable(svm_path_planner_node src/svm_path_planner_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(svm_path_planner_node svm_path_planner_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(svm_path_planner_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS svm_path_planner svm_path_planner_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_svm_path_planner.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
