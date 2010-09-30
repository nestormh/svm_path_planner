#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
CCADMIN=CCadmin
RANLIB=ranlib
CC=nvcc
CCC=nvcc
CXX=nvcc
FC=
AS=as

# Macros
CND_PLATFORM=NVCC-Linux-x86
CND_CONF=Debug
CND_DISTDIR=dist

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=build/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/devices.o \
	${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/sumaArrays.o \
	${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/matchSURF.o

# C Compiler Flags
CFLAGS=-DUNIX -D_DEBUG --compiler-bindir=/usr/bin/gcc-4.3

# CC Compiler Flags
CCFLAGS=-DUNIX -D_DEBUG --compiler-bindir=/usr/bin/gcc-4.3
CXXFLAGS=-DUNIX -D_DEBUG --compiler-bindir=/usr/bin/gcc-4.3

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Debug.mk dist/Debug/NVCC-Linux-x86/libcudalib.a

dist/Debug/NVCC-Linux-x86/libcudalib.a: ${OBJECTFILES}
	${MKDIR} -p dist/Debug/NVCC-Linux-x86
	${RM} dist/Debug/NVCC-Linux-x86/libcudalib.a
	${AR} rv ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libcudalib.a ${OBJECTFILES} 
	$(RANLIB) dist/Debug/NVCC-Linux-x86/libcudalib.a

${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/devices.o: nbproject/Makefile-${CND_CONF}.mk /home/nestor/NetBeansProjects/CUDALib/devices.cu 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib
	$(COMPILE.cc) -g -D__global__ -D__shared__ -D__constant__ -D__device__ -D__host__ -I/usr/include/c++/4.3 -I/usr/include -I/usr/local/cuda/include -I/home/nestor/NVIDIA_GPU_Computing_SDK/C/common/inc -o ${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/devices.o /home/nestor/NetBeansProjects/CUDALib/devices.cu

${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/sumaArrays.o: nbproject/Makefile-${CND_CONF}.mk /home/nestor/NetBeansProjects/CUDALib/sumaArrays.cu 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib
	$(COMPILE.cc) -g -D__global__ -D__shared__ -D__constant__ -D__device__ -D__host__ -I/usr/include/c++/4.3 -I/usr/include -I/usr/local/cuda/include -I/home/nestor/NVIDIA_GPU_Computing_SDK/C/common/inc -o ${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/sumaArrays.o /home/nestor/NetBeansProjects/CUDALib/sumaArrays.cu

${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/matchSURF.o: nbproject/Makefile-${CND_CONF}.mk /home/nestor/NetBeansProjects/CUDALib/matchSURF.cu 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib
	$(COMPILE.cc) -g -D__global__ -D__shared__ -D__constant__ -D__device__ -D__host__ -I/usr/include/c++/4.3 -I/usr/include -I/usr/local/cuda/include -I/home/nestor/NVIDIA_GPU_Computing_SDK/C/common/inc -o ${OBJECTDIR}/_ext/home/nestor/NetBeansProjects/CUDALib/matchSURF.o /home/nestor/NetBeansProjects/CUDALib/matchSURF.cu

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/Debug
	${RM} dist/Debug/NVCC-Linux-x86/libcudalib.a

# Subprojects
.clean-subprojects:
