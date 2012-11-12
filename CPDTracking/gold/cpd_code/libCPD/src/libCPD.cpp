//
// MATLAB Compiler: 4.15 (R2011a)
// Date: Thu Oct 18 12:13:10 2012
// Arguments: "-B" "macro_default" "-W" "cpplib:libCPD" "-T" "link:lib" "-d"
// "/home/nestor/gold/apps/cpd/cpd_code/libCPD/src" "-w"
// "enable:specified_file_mismatch" "-w" "enable:repeated_file" "-w"
// "enable:switch_ignored" "-w" "enable:missing_lib_sentinel" "-w"
// "enable:demo_license" "-M" "-L/usr/lib" "-v"
// "/home/nestor/gold/apps/cpd/cpd_code/core/cpd_register.m" 
//

#include <stdio.h>
#define EXPORTING_libCPD 1
#include "libCPD.h"

static HMCRINSTANCE _mcr_inst = NULL;


#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultPrintHandler(const char *s)
{
  return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int mclDefaultErrorHandler(const char *s)
{
  int written = 0;
  size_t len = 0;
  len = strlen(s);
  written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
  if (len > 0 && s[ len-1 ] != '\n')
    written += mclWrite(2 /* stderr */, "\n", sizeof(char));
  return written;
}

#ifdef __cplusplus
} /* End extern "C" block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libCPD_C_API
#define LIB_libCPD_C_API /* No special import/export declaration */
#endif

LIB_libCPD_C_API 
bool MW_CALL_CONV libCPDInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
  if (_mcr_inst != NULL)
    return true;
  if (!mclmcrInitialize())
    return false;
    {
        mclCtfStream ctfStream = 
            mclGetEmbeddedCtfStream((void *)(libCPDInitializeWithHandlers), 
                                    79068);
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(   &_mcr_inst,
                                                                error_handler, 
                                                                print_handler,
                                                                ctfStream, 
                                                                79068);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
  return true;
}

LIB_libCPD_C_API 
bool MW_CALL_CONV libCPDInitialize(void)
{
  return libCPDInitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_libCPD_C_API 
void MW_CALL_CONV libCPDTerminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

LIB_libCPD_C_API 
long MW_CALL_CONV libCPDGetMcrID() 
{
  return mclGetID(_mcr_inst);
}

LIB_libCPD_C_API 
void MW_CALL_CONV libCPDPrintStackTrace(void) 
{
  char** stackTrace;
  int stackDepth = mclGetStackTrace(&stackTrace);
  int i;
  for(i=0; i<stackDepth; i++)
  {
    mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
    mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
  }
  mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_libCPD_C_API 
bool MW_CALL_CONV mlxCpd_register(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "cpd_register", nlhs, plhs, nrhs, prhs);
}

LIB_libCPD_CPP_API 
void MW_CALL_CONV cpd_register(int nargout, mwArray& Transform, mwArray& C, const 
                               mwArray& X, const mwArray& Y, const mwArray& opt)
{
  mclcppMlfFeval(_mcr_inst, "cpd_register", nargout, 2, 3, &Transform, &C, &X, &Y, &opt);
}
