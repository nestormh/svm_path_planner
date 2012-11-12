//
// MATLAB Compiler: 4.15 (R2011a)
// Date: Thu Oct 11 14:05:11 2012
// Arguments: "-B" "macro_default" "-W" "cpplib:CPD" "-T" "link:lib" "-d"
// "/home/nestor/gold/apps/cpd/cpd_code/CPD/src" "-w"
// "enable:specified_file_mismatch" "-w" "enable:repeated_file" "-w"
// "enable:switch_ignored" "-w" "enable:missing_lib_sentinel" "-w"
// "enable:demo_license" "-v"
// "/home/nestor/gold/apps/cpd/cpd_code/core/cpd_register.m" 
//

#include <stdio.h>
#define EXPORTING_CPD 1
#include "CPD.h"

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
#ifndef LIB_CPD_C_API
#define LIB_CPD_C_API /* No special import/export declaration */
#endif

LIB_CPD_C_API 
bool MW_CALL_CONV CPDInitializeWithHandlers(
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
            mclGetEmbeddedCtfStream((void *)(CPDInitializeWithHandlers), 
                                    78198);
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(   &_mcr_inst,
                                                                error_handler, 
                                                                print_handler,
                                                                ctfStream, 
                                                                78198);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
  return true;
}

LIB_CPD_C_API 
bool MW_CALL_CONV CPDInitialize(void)
{
  return CPDInitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_CPD_C_API 
void MW_CALL_CONV CPDTerminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

LIB_CPD_C_API 
long MW_CALL_CONV CPDGetMcrID() 
{
  return mclGetID(_mcr_inst);
}

LIB_CPD_C_API 
void MW_CALL_CONV CPDPrintStackTrace(void) 
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


LIB_CPD_C_API 
bool MW_CALL_CONV mlxCpd_register(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "cpd_register", nlhs, plhs, nrhs, prhs);
}

LIB_CPD_CPP_API 
void MW_CALL_CONV cpd_register(int nargout, mwArray& Transform, mwArray& C, const 
                               mwArray& X, const mwArray& Y, const mwArray& opt)
{
  mclcppMlfFeval(_mcr_inst, "cpd_register", nargout, 2, 3, &Transform, &C, &X, &Y, &opt);
}
