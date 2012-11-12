# CPack configuration
###############################################################################


# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. Example variables are:
#   CPACK_GENERATOR                     - Generator used to create package
#   CPACK_INSTALL_CMAKE_PROJECTS        - For each project (path, name, component)
#   CPACK_CMAKE_GENERATOR               - CMake Generator used for the projects
#   CPACK_INSTALL_COMMANDS              - Extra commands to install components
#   CPACK_INSTALL_DIRECTORIES           - Extra directories to install
#   CPACK_PACKAGE_DESCRIPTION_FILE      - Description file for the package
#   CPACK_PACKAGE_DESCRIPTION_SUMMARY   - Summary of the package
#   CPACK_PACKAGE_EXECUTABLES           - List of pairs of executables and labels
#   CPACK_PACKAGE_FILE_NAME             - Name of the package generated
#   CPACK_PACKAGE_ICON                  - Icon used for the package
#   CPACK_PACKAGE_INSTALL_DIRECTORY     - Name of directory for the installer
#   CPACK_PACKAGE_NAME                  - Package project name
#   CPACK_PACKAGE_VENDOR                - Package project vendor
#   CPACK_PACKAGE_VERSION               - Package project version
#   CPACK_PACKAGE_VERSION_MAJOR         - Package project version (major)
#   CPACK_PACKAGE_VERSION_MINOR         - Package project version (minor)
#   CPACK_PACKAGE_VERSION_PATCH         - Package project version (patch)

# There are certain generator specific ones

# NSIS Generator:
#   CPACK_PACKAGE_INSTALL_REGISTRY_KEY  - Name of the registry key for the installer
#   CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS - Extra commands used during uninstall
#   CPACK_NSIS_EXTRA_INSTALL_COMMANDS   - Extra commands used during install

# CPack configuration
# For more information see:
# http://www.cmake.org/Wiki/CMake:CPackPackageGenerators

# include(InstallRequiredSystemLibraries)




set(CPACK_BINARY_BUNDLE "")
set(CPACK_BINARY_CYGWIN "")
set(CPACK_BINARY_DEB "OFF")
set(CPACK_BINARY_DRAGNDROP "")
set(CPACK_BINARY_NSIS "OFF")
set(CPACK_BINARY_OSXX11 "")
set(CPACK_BINARY_PACKAGEMAKER "")
set(CPACK_BINARY_RPM "OFF")
set(CPACK_BINARY_STGZ "ON")
set(CPACK_BINARY_TBZ2 "OFF")
set(CPACK_BINARY_TGZ "ON")
set(CPACK_BINARY_TZ "ON")
set(CPACK_BINARY_ZIP "")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENTS_ALL "RunTime;SDK;DDK")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_GENERATOR "TGZ;TBZ2;TZ")
set(CPACK_IGNORE_FILES "/CVS/;/\\.svn/;/\\.bzr/;/\\.hg/;/\\.git/;\\.swp$;\\.#;/#")
# set(CPACK_INSTALLED_DIRECTORIES "/media/sdb2/Development/ParaView;/")
set(CPACK_INSTALL_CMAKE_PROJECTS "${PACKAGE_BUILD_DIR};${PACKAGE_NAME};${COMPONENT};install")
set(CPACK_INSTALL_CMAKE_PROJECTS_DRAGNDROP "${PACKAGE_BUILD_DIR};GOLD Mac Bundle;Bundle;/")
set(CPACK_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_MODULE_PATH "${CMAKE_MODULES_PATH};")
set(CPACK_NSIS_DISPLAY_NAME "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_MODIFY_PATH "OFF")
set(CPACK_NSIS_PACKAGE_NAME "${PACKAGE_NAME} ${PACKAGE_VERSION}")

# set(CPACK_OUTPUT_CONFIG_FILE "/media/sdb2/Development/ParaView/build/Applications/ParaView/CPackParaViewConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION   "${PACKAGE_NAME} long description of the package here")
# TODO: set(CPACK_PACKAGE_DESCRIPTION_FILE description_file_name )
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PACKAGE_NAME} (Generic Obstacle and Lane Detector) is a computer vision framework (summary)")
set(CPACK_PACKAGE_EXECUTABLES "${PACKAGE_NAME};")
set(CPACK_PACKAGE_FILE_NAME "${PACKAGE_NAME}-${COMPONENT_NAME}-${PACKAGE_VERSION}-${PACKAGE_RELEASE}.${PACKAGE_SYSTEM_PROCESSOR}.${PACKAGE_TARGET_OS}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set(CPACK_PACKAGE_NAME "${PACKAGE_NAME}")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR ${PACKAGE_VENDOR} )
set(CPACK_PACKAGE_VERSION_MAJOR "${PACKAGE_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PACKAGE_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PACKAGE_VERSION_PATCH}")

# set(CPACK_PROJECT_CONFIG_FILE "/media/sdb2/Development/ParaView/build/Applications/ParaView/CPackParaViewOptions.cmake")
# set(CPACK_RESOURCE_FILE_LICENSE "/media/sdb2/Development/ParaView/License_v1.2.txt")
# set(CPACK_RESOURCE_FILE_README "/usr/share/cmake/Templates/CPack.GenericDescription.txt")
# set(CPACK_RESOURCE_FILE_WELCOME "/usr/share/cmake/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "ON")
# set(CPACK_SOURCE_CYGWIN "")
# set(CPACK_SOURCE_GENERATOR "TGZ;TBZ2;TZ")
# set(CPACK_SOURCE_IGNORE_FILES "/CVS/;/\\.svn/;/\\.bzr/;/\\.hg/;/\\.git/;\\.swp$;\\.#;/#")
# set(CPACK_SOURCE_INSTALLED_DIRECTORIES "/media/sdb2/Development/ParaView;/")
# set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/media/sdb2/Development/ParaView/build/CPackSourceConfig.cmake")
# set(CPACK_SOURCE_PACKAGE_FILE_NAME "ParaView-3.11.0-Source")
# set(CPACK_SOURCE_STRIP_FILES "OFF")
# set(CPACK_SOURCE_TBZ2 "ON")
# set(CPACK_SOURCE_TGZ "ON")
# set(CPACK_SOURCE_TOPLEVEL_TAG "Linux-x86_64-Source")
# set(CPACK_SOURCE_TZ "ON")
# set(CPACK_SOURCE_ZIP "OFF")
set(CPACK_STRIP_FILES "OFF")
set(CPACK_SYSTEM_NAME "${GOLD_SYSTEM_PROCESSOR}.${PACKAGE_TARGET_OS}")
set(CPACK_TOPLEVEL_TAG "${GOLD_SYSTEM_PROCESSOR}.${PACKAGE_TARGET_OS}-Source")

###############################################################################
# RPM specific settings
# some macro is still unavailable due to pending patches.
set(CPACK_RPM_COMPONENT_INSTALL ON)
# CPACK_RPM_PACKAGE_SUMMARY       The RPM package summary   CPACK_PACKAGE_DESCRIPTION_SUMMARY
# CPACK_RPM_PACKAGE_NAME          The RPM package name  CPACK_PACKAGE_NAME
# CPACK_RPM_PACKAGE_VERSION       The RPM package version   CPACK_PACKAGE_VERSION
# CPACK_RPM_PACKAGE_ARCHITECTURE  The RPM package architecture. This may be set to "noarch" if you know you are building a noarch package.  -
set (CPACK_RPM_PACKAGE_ARCHITECTURE "${PACKAGE_SYSTEM_PROCESSOR}")
# CPACK_RPM_PACKAGE_RELEASE   The RPM package release. This is the numbering of the RPM package itself, i.e. the version of the packaging and not the version of the content (see CPACK_RPM_PACKAGE_VERSION). One may change the default value if the previous packaging was buggy and/or you want to put here a fancy Linux distro specific numbering. 1
set (CPACK_RPM_PACKAGE_RELEASE "1.fc13")
# CPACK_RPM_PACKAGE_LICENSE       The RPM package license policy.   "unknown"
# CPACK_RPM_PACKAGE_GROUP         The RPM package group   "unknown"
# CPACK_RPM_PACKAGE_VENDOR        The RPM package group   CPACK_PACKAGE_VENDOR if set or "unknown" if not set
# CPACK_RPM_PACKAGE_DESCRIPTION   The RPM package description   The content of CPACK_PACKAGE_DESCRIPTION_FILE if set or "no package description available" if not set
# CPACK_RPM_PACKAGE_REQUIRES      May be used to set RPM dependencies. see [RPM dependencies specification]) for precise syntax.  -
# CPACK_RPM_SPEC_INSTALL_POST     May be used to set an RPM post-install command inside the spec file. For example setting it to "/bin/true" may be used to prevent rpmbuild to strip binaries (see [Bug7435])  -
# CPACK_RPM_SPEC_MORE_DEFINE      May be used to add any %define lines to the generated spec file.  -
# CPACK_RPM_USER_BINARY_SPECFILE  May be used to specify a user provided spec file instead of generating one. This is an feature which currently needs a patch see [Bug8988]  -
# CPACK_RPM_<POST/PRE>_<UN>INSTALL_SCRIPT_FILE  The content of the specified files will be embedded in the RPM spec file in the appropriate sections. This is an feature which currently needs a patch see [Bug8988]  -
# CPACK_RPM_PACKAGE_DEBUG
set(CPACK_RPM_PACKAGE_DEBUG "ON")
# set(CPACK_RPM_SPEC_INSTALL_POST "/bin/true")














# ------------------------------------------------------------------------------------------------

# include( BSM_CPackConfig )

# CPack configuration
# For more information see:
# http://www.cmake.org/Wiki/CMake:CPackPackageGenerators

# include(InstallRequiredSystemLibraries)

# se si settano questi le opzioni non sono più disponibili nelle gui
# set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
# set(CPACK_GENERATOR "RPM")

# TODO: questo è importante?
# set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${PACKAGE_NAME_VERSION};ALL;/")

# set(CPACK_OUTPUT_CONFIG_FILE "/home/andy/vtk/CMake-bin/CPackConfig.cmake")

# message("${PACKAGE_NAME}-${PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}.${CPACK_RPM_PACKAGE_ARCHITECTURE}")


# if (CMAKE_SYSTEM_PROCESSOR MATCHES "unknown")
#   set (CMAKE_SYSTEM_PROCESSOR "x86")
# endif (CMAKE_SYSTEM_PROCESSOR MATCHES "unknown")

# set( CPACK_SYSTEM_NAME ${GOLD_SYSTEM_PROCESSOR} CACHE STRING "" )


# if(NOT DEFINED CPACK_SYSTEM_NAME)
  # set( CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}.${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "")
# endif(NOT DEFINED CPACK_SYSTEM_NAME)
# if(${CPACK_SYSTEM_NAME} MATCHES Windows)
#   if(CMAKE_CL_64)
#     set(CPACK_SYSTEM_NAME win64.${CMAKE_SYSTEM_PROCESSOR})
#   else(CMAKE_CL_64)
#     set(CPACK_SYSTEM_NAME win32.${CMAKE_SYSTEM_PROCESSOR})
#   endif(CMAKE_CL_64)
# endif(${CPACK_SYSTEM_NAME} MATCHES Windows)

# RPM specific settings
# some macro is still unavailable due to pending patches.
# NOTE: ldconfig run and other specific stuff for this package can be found in Scripts/build_rpm

# CPACK_RPM_PACKAGE_SUMMARY    The RPM package summary   CPACK_PACKAGE_DESCRIPTION_SUMMARY
# CPACK_RPM_PACKAGE_NAME  The RPM package name  CPACK_PACKAGE_NAME
# CPACK_RPM_PACKAGE_VERSION   The RPM package version   CPACK_PACKAGE_VERSION
# CPACK_RPM_PACKAGE_ARCHITECTURE  The RPM package architecture. This may be set to "noarch" if you know you are building a noarch package.  -
# set (CPACK_RPM_PACKAGE_ARCHITECTURE "${GOLD_SYSTEM_PROCESSOR}")
# # CPACK_RPM_PACKAGE_RELEASE   The RPM package release. This is the numbering of the RPM package itself, i.e. the version of the packaging and not the version of the content (see CPACK_RPM_PACKAGE_VERSION). One may change the default value if the previous packaging was buggy and/or you want to put here a fancy Linux distro specific numbering. 1
# set (CPACK_RPM_PACKAGE_RELEASE "${PACKAGE_RELEASE}")
# if(NOT ${PACKAGE_TARGET_OS} STREQUAL "")
#   set (CPACK_RPM_PACKAGE_RELEASE "${CPACK_RPM_PACKAGE_RELEASE}.${PACKAGE_TARGET_OS}")
# endif()
# CPACK_RPM_PACKAGE_LICENSE   The RPM package license policy.   "unknown"
# CPACK_RPM_PACKAGE_GROUP   The RPM package group   "unknown"
# CPACK_RPM_PACKAGE_VENDOR  The RPM package group   CPACK_PACKAGE_VENDOR if set or "unknown" if not set
# CPACK_RPM_PACKAGE_DESCRIPTION   The RPM package description   The content of CPACK_PACKAGE_DESCRIPTION_FILE if set or "no package description available" if not set
# CPACK_RPM_PACKAGE_REQUIRES  May be used to set RPM dependencies. see [RPM dependencies specification]) for precise syntax.  -
# CPACK_RPM_SPEC_INSTALL_POST   May be used to set an RPM post-install command inside the spec file. For example setting it to "/bin/true" may be used to prevent rpmbuild to strip binaries (see [Bug7435])  -
# CPACK_RPM_SPEC_MORE_DEFINE  May be used to add any %define lines to the generated spec file.  -
# CPACK_RPM_USER_BINARY_SPECFILE  May be used to specify a user provided spec file instead of generating one. This is an feature which currently needs a patch see [Bug8988]  -
# CPACK_RPM_<POST/PRE>_<UN>INSTALL_SCRIPT_FILE  The content of the specified files will be embedded in the RPM spec file in the appropriate sections. This is an feature which currently needs a patch see [Bug8988]  -
# CPACK_RPM_PACKAGE_DEBUG
# set(CPACK_RPM_PACKAGE_DEBUG "ON")
# set(CPACK_RPM_SPEC_INSTALL_POST "/bin/true")
# message("RPM rpath check skipping: export QA_RPATHS=$[0x0001|0x0002|0x0010]")

# generators settings
# if(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
  # set(CPACK_PACKAGE_ICON "${ParaView_SOURCE_DIR}/Applications/Client\\\\ParaViewLogo.png")
  # set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\paraview.exe")
  # set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} a cross-platform, open-source visualization system")
  # set(CPACK_NSIS_HELP_LINK "http://www.paraview.org")
  # set(CPACK_NSIS_URL_INFO_ABOUT "http://www.kitware.com")
  # set(CPACK_NSIS_CONTACT "webmaster@paraview.org")
  #    set(CPACK_NSIS_MODIFY_PATH ON)
# else(WIN32 AND NOT UNIX)
  # set(CPACK_STRIP_FILES "")
  # set(CPACK_SOURCE_STRIP_FILES "")
  # IF (NOT APPLE)
    # set(CPACK_GE#ERATOR TGZ)
  # ENDIF (NOT APPLE)
# endif(WIN32 AND NOT UNIX)

# Adjustmens NOT done automatically
# set(CPACK_PACKAGE_EXECUTABLES "ccmake;CMake")
# set(CPACK_PACKAGE_FILE_NAME  "${CPACK_PACKAGE_NAME}-${PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}.${CPACK_RPM_PACKAGE_ARCHITECTURE}")

# set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PACKAGE_NAME_VERSION}")
# set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${PACKAGE_NAME_VERSION}")
# set(CPACK_TOPLEVEL_TAG "x86_64/RPM")
# set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Description.txt")
# set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
# set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
# set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_CURRENT_SOURCE_DIR}/Welcome.txt"

# set(CPACK_SOURCE_GENERATOR "TGZ;TZ")
# set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/andy/vtk/CMake-bin/CPackSourceConfig.cmake")
# set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PACKAGE_NAME_VERSION}-src" )
# set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PACKAGE_NAME_VERSION}" )
# set(CPACK_SOURCE_STRIP_FILES "")
# set(CPACK_STRIP_FILES "bin/gold;/bin/otherbin")

# if(NOT DEFINED CPACK_PACKAGE_FILE_NAME)
# set(CPACK_PACKAGE_FILE_NAME "${CPACK_SOURCE_PACKAGE_FILE_NAME}-${CPACK_SYSTEM_NAME}" CACHE STRING "")
# endif(NOT DEFINED CPACK_PACKAGE_FILE_NAME)

# set(CPACK_PACKAGE_EXECUTABLES "paraview" "ParaView") ???

# set(CPACK_SET_DESTDIR "ON")


# Debug
# message( "CPack configuration:")
# message( "CPACK_TOPLEVEL_DIRECTORY=${CPACK_TOPLEVEL_DIRECTORY}")
# message( "PACKAGE_NAME=${PACKAGE_NAME}")
# message( "PACKAGE_NAME_VERSION=${PACKAGE_NAME_VERSION}")
# message( "CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")
# message( "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
# message( "CPACK_TOPLEVEL_TAG=${CPACK_TOPLEVEL_TAG}")
# message( "CPACK_SYSTEM_NAME=${CPACK_SYSTEM_NAME}")
# message( "GOLD_SYSTEM_PROCESSOR=${GOLD_SYSTEM_PROCESSOR}" )
# message( "CPACK_PACKAGE_FILE_NAME=${CPACK_PACKAGE_FILE_NAME}" )
# message( "CPACK_PACKAGE_INSTALL_DIRECTORY=${CPACK_PACKAGE_INSTALL_DIRECTORY}" )
# message( FATAL_ERROR "")






