add_executable(testSVMPathPlanning testSVMPathPlanning.cpp)

include_directories(testSVMPathPlanning
#     ${OMPL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
    
target_link_libraries(testSVMPathPlanning
#   ${OMPL_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  
)

add_subdirectory(cuSVM)
add_subdirectory(GPULibSVM)
add_subdirectory(libsvm)