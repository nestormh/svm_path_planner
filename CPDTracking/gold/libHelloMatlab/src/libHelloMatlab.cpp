//
// MATLAB Compiler: 4.15 (R2011a)
// Date: Thu Oct 18 14:03:52 2012
// Arguments: "-B" "macro_default" "-W" "cpplib:libHelloMatlab" "-T" "link:lib"
// "-d" "/home/nestor/gold/apps/cpd/libHelloMatlab/src" "-w"
// "enable:specified_file_mismatch" "-w" "enable:repeated_file" "-w"
// "enable:switch_ignored" "-w" "enable:missing_lib_sentinel" "-w"
// "enable:demo_license" "-v" "/home/nestor/hellomatlab.m" 
//

#include <stdio.h>
#define EXPORTING_libHelloMatlab 1
#include "libHelloMatlab.h"

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
#ifndef LIB_libHelloMatlab_C_API
#define LIB_libHelloMatlab_C_API /* No special import/export declaration */
#endif

LIB_libHelloMatlab_C_API 
bool MW_CALL_CONV libHelloMatlabInitializeWithHandlers(
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
            mclGetEmbeddedCtfStream((void *)(libHelloMatlabInitializeWithHandlers), 
                                    45089);
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(   &_mcr_inst,
                                                                error_handler, 
                                                                print_handler,
                                                                ctfStream, 
                                                                45089);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
  return true;
}

LIB_libHelloMatlab_C_API 
bool MW_CALL_CONV libHelloMatlabInitialize(void)
{
  return libHelloMatlabInitializeWithHandlers(mclDefaultErrorHandler, 
                                              mclDefaultPrintHandler);
}

LIB_libHelloMatlab_C_API 
void MW_CALL_CONV libHelloMatlabTerminate(void)
{
  if (_mcr_inst != NULL)
    mclTerminateInstance(&_mcr_inst);
}

LIB_libHelloMatlab_C_API 
long MW_CALL_CONV libHelloMatlabGetMcrID() 
{
  return mclGetID(_mcr_inst);
}

LIB_libHelloMatlab_C_API 
void MW_CALL_CONV libHelloMatlabPrintStackTrace(void) 
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


LIB_libHelloMatlab_C_API 
bool MW_CALL_CONV mlxHellomatlab(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
  return mclFeval(_mcr_inst, "hellomatlab", nlhs, plhs, nrhs, prhs);
}

LIB_libHelloMatlab_CPP_API 
void MW_CALL_CONV hellomatlab()
{
  mclcppMlfFeval(_mcr_inst, "hellomatlab", 0, 0, 0);
}
