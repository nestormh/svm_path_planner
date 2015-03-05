set(CCFILES
  testLibSVM.cpp
  svm.cpp
  svmpathplanning.cpp
)

set (HFILES
  svm.h
)

set(HFILES_PUBLIC
)


add_executable(testLibSVM
  ${CCFILES} 
  ${HFILES}
  ${HFILES_PUBLIC}
)

include_directories(testLibSVM
#     ../GPULibSVM
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
)

target_link_libraries(testLibSVM
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)
