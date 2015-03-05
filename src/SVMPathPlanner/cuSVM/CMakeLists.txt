set(CCFILES
  testCuSVM.cpp 
)

set (HFILES
  cuSVMutil.h
)

set (CUHFILES
  
)

set(CUFILES
  cuSVMPredictKernel.cu
  cuSVMSolver.cu
  cuSVMUtil.cu
)

set(HFILES_PUBLIC
)

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_executable(testCuSVM
  ${CUFILES} 
  ${CCFILES} 
#   ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
#   OPTIONS ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS}
)
  


include_directories(testCuSVM
#     ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
#     emon.a
    ${CUDA_INCLUDE_DIRS}
)

set(CUDA_CURAND_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand.so)

target_link_libraries(testCuSVM
#   ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
#   emon.a
)
CUDA_ADD_CUBLAS_TO_TARGET(testCuSVM)
