cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(multisvmpathplanning)

# set(OpenCV_DIR "/usr/local/share/OpenCV")
set(PCL_DIR "/usr/local/share/pcl-1.6")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

set (CMAKE_MODULE_PATH "/usr/local/share/pcl-1.6" ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(OMPL    REQUIRED )
find_package(Boost REQUIRED)
find_package(CUDA 5.0 REQUIRED)
find_package(OpenCV  REQUIRED )
find_package( Eigen3    REQUIRED )
# find_package(PCL REQUIRED)
find_package(PCL 1.6 REQUIRED)
find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(CCFILES
  gpudt/predicates.cpp
  gpudt/gpudt.cpp
  GPULibSVM/svm-train.c
  GPULibSVM/svm.cpp
  GPULibSVM/svmpathplanning.cpp
  GPULibSVM/svmpathplanningsingle.cpp
  GPULibSVM/voronoipathplanning.cpp
  GPULibSVM/voronoisvmpathplanning.cpp
  testSVMPathPlanning.cpp
)

set (HFILES
  gpudt/gpudt.h
  GPULibSVM/svm.h
)

set (CUHFILES
  gpudt/Cuda/common.h
  gpudt/Cuda/cudaDecl.h
  gpudt/Cuda/cudaCCW.h
  gpudt/Cuda/kernelVoronoi.h
  gpudt/Cuda/pba2DKernel.h
  gpudt/predicates.h  
)

set(CUFILES
  gpudt/Cuda/cudaShifting.cu
  gpudt/Cuda/pba2DHost.cu
  gpudt/Cuda/cudaVoronoi.cu
  gpudt/Cuda/cudaMain.cu
  gpudt/Cuda/cudaFlipping.cu
  gpudt/Cuda/cudaReconstruction.cu
  gpudt/Cuda/cudaMissing.cu
  gpudt/Cuda/cudaBoundary.cu
  gpudt/Cuda/cudaConstraint.cu
  GPULibSVM/predictSVM.cu
  GPULibSVM/checkEdges.cu
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

set(CUDA_BIND_COMPILER --compiler-bindir /home/nestor/Dropbox/projects/MultiSVMPathPlanning/compiler_bind)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_executable(testSVMPathPlanning 
  ${CUFILES} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS} ${CUDA_BIND_COMPILER}
#   ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS} ${CUDAGDB_OPTS}
)
  
include_directories(testSVMPathPlanning
    GPULibSVM
    .
#     ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    emon.a
    ${CUDA_INCLUDE_DIRS}
)

# set(CUDA_CURAND_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand.so)

target_link_libraries(testSVMPathPlanning
#   ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${CGAL_CGAL_LIBRARY}
  ${GMP_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
  emon
)
CUDA_ADD_CUBLAS_TO_TARGET(testSVMPathPlanning)

message("PCL " ${PCL_LIBRARIES})

add_subdirectory(cuSVM)
add_subdirectory(GPULibSVM)
add_subdirectory(libsvm)
