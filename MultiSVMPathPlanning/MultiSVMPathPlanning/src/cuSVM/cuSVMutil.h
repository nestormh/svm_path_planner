
#ifndef _CUSVMUTIL_H_
#define _CUSVMUTIL_H_



#define MBtoLeave         (200)

#define CUBIC_ROOT_MAX_OPS         (2000)

#define SAXPY_CTAS_MAX           (80)
#define SAXPY_THREAD_MIN         (32)
#define SAXPY_THREAD_MAX         (128)
#define TRANS_BLOCK_DIM             (16)

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                        \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL_NO_SYNC( call) call
#  define CUDA_SAFE_CALL( call) call
    
// #ifdef _DEBUG
// 
// 	#  define mxCUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
//     cudaError err = call;                                                    \
//     if( cudaSuccess != err) {                                                \
//         printf( "Cuda error in line " );              \
// 		char linebuffer [20];  sprintf(linebuffer,"%d" ,__LINE__ );									\
// 		printf( linebuffer  );              \
// 		printf( ": " );              \
// 		printf( cudaGetErrorString( err) );              \
// 		printf( "." );              \
// 		getchar();                                                           \
//                                               \
//     } } while (0)
// 
// #  define mxCUDA_SAFE_CALL( call) do {                                         \
//     mxCUDA_SAFE_CALL_NO_SYNC(call);                                            \
//     cudaError err = cudaThreadSynchronize();                                 \
//     if( cudaSuccess != err) {                                                \
//         printf( "Cuda error in line " );              \
// 		char linebuffer [20];  sprintf(linebuffer,"%d" ,__LINE__ );									\
// 		printf( linebuffer  );              \
// 		printf( ": " );              \
// 		printf( cudaGetErrorString( err) );              \
// 		printf( "." );              \
// 		getchar();                                                           \
//                                                       \
//     } } while (0)
// #else  // not DEBUG
// 
// #  define mxCUDA_SAFE_CALL_NO_SYNC( call) call
// #  define mxCUDA_SAFE_CALL( call) call
// 
// #endif

// void checkCUDAError(const char *msg) {
//     cudaThreadSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if( cudaSuccess != err) {
//         printf(msg);
//         printf(" "); 
//         printf(cudaGetErrorString( err) );
//         printf(".  "); 
//     }                         
// }


void VectorSplay (int n, int tMin, int tMax, int gridW, int *nbrCtas, 
                        int *elemsPerCta, int *threadsPerCta);

__global__ void transpose(float *odata, float *idata, int width, int height);

#endif