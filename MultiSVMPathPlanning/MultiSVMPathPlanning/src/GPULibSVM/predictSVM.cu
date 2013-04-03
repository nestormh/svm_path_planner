#include "svm.h"
#include <time.h>
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdio.h>
#include <iostream>
#include <complex>

#define BLOCK_SIZE 32
#define MEM_BLOCK 1024

using namespace std;

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

__global__
void predictPixel(const float * d_coeffs, const float2 * d_SVs, 
                  const float gamma, const float rho, const int totalSVs,
                  const float minCornerX, const float minCornerY, 
                  const float intervalX, const float intervalY, 
                  const unsigned int rows, const unsigned int cols, 
                  const float resolution, unsigned char * d_data) {

    const int2 pos2D = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                  blockIdx.y * blockDim.y + threadIdx.y);

    const int posInBlock = threadIdx.y * blockDim.x + threadIdx.x;
                                  
    __shared__ float2 shared_SVs[MEM_BLOCK];
    __shared__ float shared_coeffs[MEM_BLOCK];
    
    if ((posInBlock < totalSVs) && (posInBlock < MEM_BLOCK)) {
        shared_SVs[posInBlock] = d_SVs[posInBlock];
        shared_coeffs[posInBlock] = d_coeffs[posInBlock];
    }
    
    __syncthreads();
    
    if ((pos2D.x > cols) || (pos2D.y > rows))
        return;
        
    const int pos1D = pos2D.y * cols + pos2D.x;
    
    const float2 posRealWorld = make_float2((intervalX * pos2D.x / cols) + minCornerX,
                                              (intervalY * pos2D.y / rows) + minCornerY);
    float sum = 0.0;
    float val;
    for (int i = 0; i < totalSVs; i++) {
        val = -gamma * ((shared_SVs[i].x - posRealWorld.x) * (shared_SVs[i].x - posRealWorld.x) + 
                       (shared_SVs[i].y - posRealWorld.y) * (shared_SVs[i].y - posRealWorld.y));
        sum += shared_coeffs[i] * exp(val);
    }
    sum -=  rho;
    
    if (sum > 0)
        d_data[pos1D] = 255;
}

void chk_error(){

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess){

        const char * err_str = cudaGetErrorString(err);

        cout << "Error: " << err_str << endl;
    }
}

extern "C"
void launchSVMPrediction(const svm_model * &model, 
                         const double & minCornerX, const double & minCornerY, 
                         const double & intervalX, const double & intervalY, 
                         const unsigned int & rows, const unsigned int & cols, 
                         const double & resolution, unsigned char * &h_data) {
                         
    if (model->l > (MEM_BLOCK)) {
        cerr << "Error: The number of support vectors is " << model->l << endl;
        cerr << "Unable to reserve memory for more than " << (BLOCK_SIZE * BLOCK_SIZE) << "SVs" << endl;
        cerr << "Exiting..." << endl;
        exit(-1);
    }
                         
    // Reserving device memory for ouput data
    unsigned char * d_data;
    CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(unsigned char) * rows * cols));
    
    // Data structure is transformed for Support Vectors and passed to the device 
    float2 * h_SVs = new float2[model->l];
    for (int i = 0; i < model->l; i++) {
        h_SVs[i] = make_float2(model->SV[i].values[0], model->SV[i].values[1]);
    }   
    float2 * d_SVs;
    CUDA_SAFE_CALL(cudaMalloc(&d_SVs, sizeof(float2) * model->l));
    CUDA_SAFE_CALL(cudaMemcpy(d_SVs, h_SVs, sizeof(float2) * model->l, cudaMemcpyHostToDevice));
    
    // Support vector coefficients are passed to device memory
    float * h_coeffs = new float[model->l];
    for (int i = 0; i < model->l; i++) {
        h_coeffs[i] = model->sv_coef[0][i];
    } 
    float * d_coeffs;
    CUDA_SAFE_CALL(cudaMalloc(&d_coeffs, sizeof(float) * model->l));
    CUDA_SAFE_CALL(cudaMemcpy(d_coeffs, h_coeffs, sizeof(float) * model->l, cudaMemcpyHostToDevice));
    
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(cols / blockSize.x + 1, rows / blockSize.x + 1, 1);
    
//     cout << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << endl;
//     cout << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << endl;
//     cout << rows << ", " << cols << endl;
//     
//     cout << "minCornerX " << minCornerX << endl;
//     cout << "minCornerY " << minCornerY << endl;
//     cout << "intervalX " << intervalX << endl;
//     cout << "intervalY " << intervalY << endl;
//     cout << "rows " << rows << endl;
//     cout << "cols " << cols << endl;
    
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    predictPixel <<<gridSize, blockSize>>> (d_coeffs, d_SVs, model->param.gamma, model->rho[0], model->l, 
                                            minCornerX, minCornerY, intervalX, intervalY, 
                                            rows, cols, resolution, d_data);
    
    cudaDeviceSynchronize(); CUDA_SAFE_CALL(cudaGetLastError());
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for kernel = " << elapsed << endl;
    
    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost));
    
    CUDA_SAFE_CALL(cudaFree(d_data));
    
    CUDA_SAFE_CALL(cudaGetLastError());
    
//     for (int y = 0; y < rows; y++) {
//         for (int x = 0; x < rows; x++)  {
//         
//             const int2 pos2D = make_int2(x, y);
//             const int pos1D = pos2D.y * cols + pos2D.x;
//         
//             const double2 posRealWorld = make_double2((intervalX * pos2D.x / cols) + minCornerX,
//                                                     (intervalY * pos2D.y / rows) + minCornerY);
// 
//             double sum = 0.0;
//             double term1, term2, val;
//             bool paint = false;
//             for (int i = 0; i < model->l; i++) {
//                 val = -model->param.gamma * ((h_SVs[i].x - posRealWorld.x) * (h_SVs[i].x - posRealWorld.x) + 
//                                (h_SVs[i].y - posRealWorld.y) * (h_SVs[i].y - posRealWorld.y));
//                 sum += model->sv_coef[0][i] * exp(val);
//                 int2 posImg = make_int2(cols * (h_SVs[i].x - minCornerX) / intervalX, 
//                                         rows * (h_SVs[i].y - minCornerY) / intervalY); 
//                 
//                 if ((posImg.x == pos2D.x) && (posImg.y == pos2D.y))
//                     paint = true;
//             }
//             sum -=  model->rho[0];
// 
//             if (paint)
//                 if (sum > 0)
//                     d_data[pos1D] = 255;
//                 else
//                     d_data[pos1D] = 128;
//         }
//     }
}


// extern "C"
// void launchSVMPrediction(const svm_model * &model, 
//                          const double & minCornerX, const double & minCornerY, 
//                          const double & intervalX, const double & intervalY, 
//                          const unsigned int & rows, const unsigned int & cols, 
//                          const double & resolution, unsigned char * &h_data) {
// 
//                         
//     GPUPredictWrapper(int m, int n, int k, float kernelwidth, const float *Test, 
//                        const float *Svs, float * alphas,float *prediction, float beta,
//                        float isregression, float * elapsed);
// }

                         
