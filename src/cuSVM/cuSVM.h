#ifndef CU_SVM_H
#define CU_SVM_H

extern "C"
void GPUPredictWrapper (int m, int n, int k, float kernelwidth, const float *Test, const float *Svs, float * alphas,float *prediction, float beta,float isregression);

extern "C"
// void SVMTrain(float *alpha,float* beta,float*y,float *x ,float C, float kernelwidth, int m, int n, float StoppingCrit);
void SVMTrain(float * elapsed, float *alpha,float *beta, float *y,float *x, float C, float kernelwidth, int m, int n, float StoppingCrit);

extern "C"
void SVRTrain(float *alpha,float* beta,float*y,float *x ,float C, float kernelwidth, float eps, int m, int n, float StoppingCrit);

#endif