set(CCFILES
  testGPULibSVM.cpp
  svm-train.c
  svm.cpp
  svmpathplanning.cpp
)

set (HFILES
  svm.h
)

set (CUHFILES
  
)

set(CUFILES
    predictSVM.cu
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

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

cuda_add_executable(testGPULibSVM
  ${CUFILES} 
  ${CCFILES} 
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  OPTIONS ${GENCODE_SM30} ${CUDAGDB_OPTS}
#   ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20} ${VERBOSE_PTXAS}
)
  


include_directories(testGPULibSVM
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(testGPULibSVM
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${BOOST_LIBRARIES}
)
CUDA_ADD_CUBLAS_TO_TARGET(testGPULibSVM)
