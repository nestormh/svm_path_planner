find_package( OpenCV REQUIRED )
find_package( PCL 1.2 REQUIRED )

include_directories(
  ${CMAKE_SOURCE_DIR}/src 
#   ${GOLD_DIR}/src 
#   ${GOLD_CONFIG_HEADERS_DIR} 
#   ${GOLD_INCLUDE_DIR} 
#   ${Boost_INCLUDE_DIR} 
#   ${Loki_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIR}  
  /usr/local/include/eigen3  
#   /home/nestor/gold/apps/cpd/libCPD/distrib/
#   /home/nestor/gold/apps/cpd/libHelloMatlab/distrib/
#   /opt/MATLAB/MATLAB_Compiler_Runtime/v715/extern/include  
)

# add_library(gold_app_cpd cpdApp.cpp Rectification.cpp StereoRectificationTools.cpp)
# 
# target_link_libraries(gold_app_cpd 
#   ${GOLD_LIBRARIES} 
#   ${Boost_SERIALIZATION_LIBRARY} 
#   ${Boost_SIGNALS_LIBRARY} 
#   ${Boost_SYSTEM_LIBRARY} 
#   ${Boost_THREAD_LIBRARY}
#   ${Boost_RANDOM_LIBRARY}
#   ${OpenCV_LIBRARIES_CALIB3D}
#   ${OpenCV_LIBRARIES_CORE}
#   ${OpenCV_LIBRARIES_HIGHGUI}
#   ${OpenCV_LIBRARIES_CONTRIB}
#   ${OpenCV_LIBRARIES_FEATURES2}
#   ${OpenCV_LIBRARIES_FLANN}
#   ${OpenCV_LIBRARIES_GPU}
#   ${OpenCV_LIBRARIES_IMGPROC}
#   ${OpenCV_LIBRARIES_LEGACY}
#   ${OpenCV_LIBRARIES_ML}
#   ${OpenCV_LIBRARIES_OBJECTDETECT}
#   ${OpenCV_LIBRARIES_VIDEO}
#   ${GOLD_RUNTIME_DIR}/../engine/libgold_engine.so
#   /opt/MATLAB/MATLAB_Compiler_Runtime/v715/runtime/glnxa64/libmwmclmcrrt.so
#   /usr/local/MATLAB/R2011a/bin/glnxa64/libmx.so
#   /usr/local/MATLAB/R2011a/bin/glnxa64/libmex.so
#   /usr/local/MATLAB/R2011a/bin/glnxa64/libeng.so
# #   /tmp/libHelloMatlab/distrib/libHelloMatlab.so
#   )

#   add_executable(mainCPD mainCPD.cpp ccpdtracking.cpp)
  add_executable(mainCPD mainCPD.cpp)
  include_directories(mainCPD
#     ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
#     /opt/MATLAB/MATLAB_Compiler_Runtime/v715/extern/include
#     /home/nestor/gold/apps/cpd/libCPD/distrib/    
    /home/nestor/gold/apps/cpd/libHelloMatlab/distrib/
    )
    
 link_directories(
#     ${PCL_LIBRARY_DIRS}
#     /usr/lib/x86_64-linux-gnu
#     /opt/MATLAB/MATLAB_Compiler_Runtime/v715/runtime/glnxa64
#     /usr/local/MATLAB/R2011a/bin/glnxa64
#     /home/nestor/gold/apps/cpd/libCPD/distrib
#     /usr/local/MATLAB/R2011a/bin/glnxa64
#     /opt/MATLAB/MATLAB_Compiler_Runtime/v715/runtime/glnxa64 
#         /opt/MATLAB/MATLAB_Compiler_Runtime/v715/bin/glnxa64/
    )
    
 target_link_libraries(mainCPD
#   ${Boost_SYSTEM_LIBRARY} 
  ${OpenCV_LIBRARIES_CALIB3D}
  ${OpenCV_LIBRARIES_CORE}
  ${OpenCV_LIBRARIES_HIGHGUI}
  ${OpenCV_LIBRARIES_IMGPROC}
  ${PCL_LIBRARIES}
#   /usr/lib/libnetcdf.so
#   /usr/lib/libnetcdff.so
#   /usr/lib/libhdf5.so.6
#   /usr/lib/libhdf5_hl.so.6
#   /home/nestor/gold/apps/cpd/libHelloMatlab/distrib/libHelloMatlab.so
#   /home/nestor/gold/apps/cpd/libCPD/distrib/libCPD.so
  

#   /opt/MATLAB/MATLAB_Compiler_Runtime/v715/bin/glnxa64/libnetcdf.so.6
#   /home/nestor/gold/apps/cpd/libHelloMatlab/distrib/libHelloMatlab.so
)

add_definitions(${PCL_DEFINITIONS})
 
# set_target_properties(mainCPD PROPERTIES OUTPUT_DIRECTORY /tmp)  
SET(EXECUTABLE_OUTPUT_PATH /home/nestor/gold/apps/cpd/build/bin)

# # mette il binario nella cartella dei plugind di gold
# message ("GOLD_PLUGINS_DIR=${GOLD_PLUGINS_DIR}")
# 
# #TODO: qui ci vuole una directory di output diversa se stiamo impacchettando o installando
# set_target_properties(gold_app_cpd PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${GOLD_DEV_PLUGINS_DIR})
# install( TARGETS gold_app_cpd DESTINATION ${GOLD_PLUGINS_DIR} )