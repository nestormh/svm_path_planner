//
// MATLAB Compiler: 4.15 (R2011a)
// Date: Thu Oct 11 14:08:56 2012
// Arguments: "-B" "macro_default" "-W" "cpplib:libCPD" "-T" "link:lib" "-d"
// "/home/nestor/gold/apps/cpd/cpd_code/libCPD/src" "-w"
// "enable:specified_file_mismatch" "-w" "enable:repeated_file" "-w"
// "enable:switch_ignored" "-w" "enable:missing_lib_sentinel" "-w"
// "enable:demo_license" "-v"
// "/home/nestor/gold/apps/cpd/cpd_code/core/cpd_register.m" 
//

#ifndef __libCPD_h
#define __libCPD_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SUNPRO_CC)
/* Solaris shared libraries use __global, rather than mapfiles
 * to define the API exported from a shared library. __global is
 * only necessary when building the library -- files including
 * this header file to use the library do not need the __global
 * declaration; hence the EXPORTING_<library> logic.
 */

#ifdef EXPORTING_libCPD
#define PUBLIC_libCPD_C_API __global
#else
#define PUBLIC_libCPD_C_API /* No import statement needed. */
#endif

#define LIB_libCPD_C_API PUBLIC_libCPD_C_API

#elif defined(_HPUX_SOURCE)

#ifdef EXPORTING_libCPD
#define PUBLIC_libCPD_C_API __declspec(dllexport)
#else
#define PUBLIC_libCPD_C_API __declspec(dllimport)
#endif

#define LIB_libCPD_C_API PUBLIC_libCPD_C_API


#else

#define LIB_libCPD_C_API

#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libCPD_C_API 
#define LIB_libCPD_C_API /* No special import/export declaration */
#endif

extern LIB_libCPD_C_API 
bool MW_CALL_CONV libCPDInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libCPD_C_API 
bool MW_CALL_CONV libCPDInitialize(void);

extern LIB_libCPD_C_API 
void MW_CALL_CONV libCPDTerminate(void);



extern LIB_libCPD_C_API 
void MW_CALL_CONV libCPDPrintStackTrace(void);

extern LIB_libCPD_C_API 
bool MW_CALL_CONV mlxCpd_register(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_libCPD_C_API 
long MW_CALL_CONV libCPDGetMcrID();


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__BORLANDC__)

#ifdef EXPORTING_libCPD
#define PUBLIC_libCPD_CPP_API __declspec(dllexport)
#else
#define PUBLIC_libCPD_CPP_API __declspec(dllimport)
#endif

#define LIB_libCPD_CPP_API PUBLIC_libCPD_CPP_API

#else

#if !defined(LIB_libCPD_CPP_API)
#if defined(LIB_libCPD_C_API)
#define LIB_libCPD_CPP_API LIB_libCPD_C_API
#else
#define LIB_libCPD_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_libCPD_CPP_API void MW_CALL_CONV cpd_register(int nargout, mwArray& Transform, mwArray& C, const mwArray& X, const mwArray& Y, const mwArray& opt);

#endif
#endif
