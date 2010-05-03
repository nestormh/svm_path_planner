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
CC=gcc
CCC=g++
CXX=g++
FC=
AS=

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_CONF=Debug
CND_DISTDIR=dist

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=build/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/GeographicLib/DMS.o \
	${OBJECTDIR}/NonRigidTransform.o \
	${OBJECTDIR}/GeographicLib/UTMUPS.o \
	${OBJECTDIR}/ImageRegistration.o \
	${OBJECTDIR}/GeographicLib/AzimuthalEquidistant.o \
	${OBJECTDIR}/NavEntorno.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CMRPT_Route.o \
	${OBJECTDIR}/mainCjtosImagenes.o \
	${OBJECTDIR}/GeographicLib/PolarStereographic.o \
	${OBJECTDIR}/GeographicLib/Geodesic.o \
	${OBJECTDIR}/GeographicLib/CassiniSoldner.o \
	${OBJECTDIR}/NavBasadaEntorno.o \
	${OBJECTDIR}/Surf/imload.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/wm.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CStatistics.o \
	${OBJECTDIR}/GeographicLib/Geocentric.o \
	${OBJECTDIR}/GeographicLib/TransverseMercator.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/stereo.o \
	${OBJECTDIR}/ContourMatching.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_10.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_11.o \
	${OBJECTDIR}/Flusser.o \
	${OBJECTDIR}/GeographicLib/GeoCoords.o \
	${OBJECTDIR}/CornerDetectionAndMatching.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cvfast.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CRealMatches.o \
	${OBJECTDIR}/Surf/os_mapping.o \
	${OBJECTDIR}/mser/msertest.o \
	${OBJECTDIR}/pruebasSurf.o \
	${OBJECTDIR}/AffineAndEuclidean.o \
	${OBJECTDIR}/utils.o \
	${OBJECTDIR}/GeographicLib/LocalCartesian.o \
	${OBJECTDIR}/GeographicLib/EllipticFunction.o \
	${OBJECTDIR}/ViewMorphing.o \
	${OBJECTDIR}/stdafx.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_9.o \
	${OBJECTDIR}/GeographicLib/Geoid.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/changeDetection.o \
	${OBJECTDIR}/CRutaDB.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_12.o \
	${OBJECTDIR}/xform.o \
	${OBJECTDIR}/GeographicLib/TransverseMercatorExact.o \
	${OBJECTDIR}/CRuta.o \
	${OBJECTDIR}/GeographicLib/MGRS.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/piecewiseLinear.o \
	${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/pca.o

# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L/usr/lib -lcv -lcvaux -lcxcore -lhighgui -lml -lgsl -lgslcblas -lSurfJni

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Debug.mk navbasadaentorno

navbasadaentorno: ${OBJECTFILES}
	${LINK.cc} -o navbasadaentorno ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/GeographicLib/DMS.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/DMS.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/DMS.o GeographicLib/DMS.cpp

${OBJECTDIR}/NonRigidTransform.o: nbproject/Makefile-${CND_CONF}.mk NonRigidTransform.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/NonRigidTransform.o NonRigidTransform.cpp

${OBJECTDIR}/GeographicLib/UTMUPS.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/UTMUPS.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/UTMUPS.o GeographicLib/UTMUPS.cpp

${OBJECTDIR}/ImageRegistration.o: nbproject/Makefile-${CND_CONF}.mk ImageRegistration.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/ImageRegistration.o ImageRegistration.cpp

${OBJECTDIR}/GeographicLib/AzimuthalEquidistant.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/AzimuthalEquidistant.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/AzimuthalEquidistant.o GeographicLib/AzimuthalEquidistant.cpp

${OBJECTDIR}/NavEntorno.o: nbproject/Makefile-${CND_CONF}.mk NavEntorno.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/NavEntorno.o NavEntorno.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CMRPT_Route.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/CMRPT_Route.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CMRPT_Route.o /home/neztol/NetBeansProjects/NavBasadaEntorno/CMRPT_Route.cpp

${OBJECTDIR}/mainCjtosImagenes.o: nbproject/Makefile-${CND_CONF}.mk mainCjtosImagenes.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/mainCjtosImagenes.o mainCjtosImagenes.cpp

${OBJECTDIR}/GeographicLib/PolarStereographic.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/PolarStereographic.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/PolarStereographic.o GeographicLib/PolarStereographic.cpp

${OBJECTDIR}/GeographicLib/Geodesic.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/Geodesic.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/Geodesic.o GeographicLib/Geodesic.cpp

${OBJECTDIR}/GeographicLib/CassiniSoldner.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/CassiniSoldner.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/CassiniSoldner.o GeographicLib/CassiniSoldner.cpp

${OBJECTDIR}/NavBasadaEntorno.o: nbproject/Makefile-${CND_CONF}.mk NavBasadaEntorno.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/NavBasadaEntorno.o NavBasadaEntorno.cpp

${OBJECTDIR}/Surf/imload.o: nbproject/Makefile-${CND_CONF}.mk Surf/imload.cpp 
	${MKDIR} -p ${OBJECTDIR}/Surf
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/Surf/imload.o Surf/imload.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/wm.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/wm.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/wm.o /home/neztol/NetBeansProjects/NavBasadaEntorno/wm.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CStatistics.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/CStatistics.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CStatistics.o /home/neztol/NetBeansProjects/NavBasadaEntorno/CStatistics.cpp

${OBJECTDIR}/GeographicLib/Geocentric.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/Geocentric.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/Geocentric.o GeographicLib/Geocentric.cpp

${OBJECTDIR}/GeographicLib/TransverseMercator.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/TransverseMercator.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/TransverseMercator.o GeographicLib/TransverseMercator.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/stereo.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/stereo.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/stereo.o /home/neztol/NetBeansProjects/NavBasadaEntorno/stereo.cpp

${OBJECTDIR}/ContourMatching.o: nbproject/Makefile-${CND_CONF}.mk ContourMatching.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/ContourMatching.o ContourMatching.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_10.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_10.cc 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_10.o /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_10.cc

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_11.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_11.cc 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_11.o /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_11.cc

${OBJECTDIR}/Flusser.o: nbproject/Makefile-${CND_CONF}.mk Flusser.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/Flusser.o Flusser.cpp

${OBJECTDIR}/GeographicLib/GeoCoords.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/GeoCoords.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/GeoCoords.o GeographicLib/GeoCoords.cpp

${OBJECTDIR}/CornerDetectionAndMatching.o: nbproject/Makefile-${CND_CONF}.mk CornerDetectionAndMatching.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/CornerDetectionAndMatching.o CornerDetectionAndMatching.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cvfast.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cvfast.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cvfast.o /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cvfast.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CRealMatches.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/CRealMatches.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/CRealMatches.o /home/neztol/NetBeansProjects/NavBasadaEntorno/CRealMatches.cpp

${OBJECTDIR}/Surf/os_mapping.o: nbproject/Makefile-${CND_CONF}.mk Surf/os_mapping.cpp 
	${MKDIR} -p ${OBJECTDIR}/Surf
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/Surf/os_mapping.o Surf/os_mapping.cpp

${OBJECTDIR}/mser/msertest.o: nbproject/Makefile-${CND_CONF}.mk mser/msertest.cpp 
	${MKDIR} -p ${OBJECTDIR}/mser
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/mser/msertest.o mser/msertest.cpp

${OBJECTDIR}/pruebasSurf.o: nbproject/Makefile-${CND_CONF}.mk pruebasSurf.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/pruebasSurf.o pruebasSurf.cpp

${OBJECTDIR}/AffineAndEuclidean.o: nbproject/Makefile-${CND_CONF}.mk AffineAndEuclidean.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/AffineAndEuclidean.o AffineAndEuclidean.cpp

${OBJECTDIR}/utils.o: nbproject/Makefile-${CND_CONF}.mk utils.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/utils.o utils.cpp

${OBJECTDIR}/GeographicLib/LocalCartesian.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/LocalCartesian.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/LocalCartesian.o GeographicLib/LocalCartesian.cpp

${OBJECTDIR}/GeographicLib/EllipticFunction.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/EllipticFunction.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/EllipticFunction.o GeographicLib/EllipticFunction.cpp

${OBJECTDIR}/ViewMorphing.o: nbproject/Makefile-${CND_CONF}.mk ViewMorphing.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/ViewMorphing.o ViewMorphing.cpp

${OBJECTDIR}/stdafx.o: nbproject/Makefile-${CND_CONF}.mk stdafx.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/stdafx.o stdafx.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_9.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_9.cc 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_9.o /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_9.cc

${OBJECTDIR}/GeographicLib/Geoid.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/Geoid.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/Geoid.o GeographicLib/Geoid.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/changeDetection.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/changeDetection.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/changeDetection.o /home/neztol/NetBeansProjects/NavBasadaEntorno/changeDetection.cpp

${OBJECTDIR}/CRutaDB.o: nbproject/Makefile-${CND_CONF}.mk CRutaDB.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/CRutaDB.o CRutaDB.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_12.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_12.cc 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_12.o /home/neztol/NetBeansProjects/NavBasadaEntorno/fast/cv_fast_12.cc

${OBJECTDIR}/xform.o: nbproject/Makefile-${CND_CONF}.mk xform.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/xform.o xform.cpp

${OBJECTDIR}/GeographicLib/TransverseMercatorExact.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/TransverseMercatorExact.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/TransverseMercatorExact.o GeographicLib/TransverseMercatorExact.cpp

${OBJECTDIR}/CRuta.o: nbproject/Makefile-${CND_CONF}.mk CRuta.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/CRuta.o CRuta.cpp

${OBJECTDIR}/GeographicLib/MGRS.o: nbproject/Makefile-${CND_CONF}.mk GeographicLib/MGRS.cpp 
	${MKDIR} -p ${OBJECTDIR}/GeographicLib
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/GeographicLib/MGRS.o GeographicLib/MGRS.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/piecewiseLinear.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/piecewiseLinear.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/piecewiseLinear.o /home/neztol/NetBeansProjects/NavBasadaEntorno/piecewiseLinear.cpp

${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/pca.o: nbproject/Makefile-${CND_CONF}.mk /home/neztol/NetBeansProjects/NavBasadaEntorno/pca.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno
	${RM} $@.d
	$(COMPILE.cc) -g -Isift -ILeastSquares -I/usr/local/include/opencv -I/usr/include/gtk-2.0 -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/neztol/NetBeansProjects/NavBasadaEntorno/pca.o /home/neztol/NetBeansProjects/NavBasadaEntorno/pca.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf:
	${RM} -r build/Debug
	${RM} navbasadaentorno

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc