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
                  const unsigned int rows, const unsigned int cols, 
                  unsigned char * d_data) {

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
    
//     const float2 * shared_SVs = d_SVs;
//     const float * shared_coeffs = d_coeffs;
    
    if ((pos2D.x > cols) || (pos2D.y > rows))
        return;
        
    const int pos1D = pos2D.y * cols + pos2D.x;
    
    const float2 posRealWorld = make_float2((float)pos2D.x / cols, (float)pos2D.y / rows);
    
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
    else
        d_data[pos1D] = 0;
}

void chk_error(){

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess){

        const char * err_str = cudaGetErrorString(err);

        cout << "Error: " << err_str << endl;
    }
}

// TODO: Modificar para que sea capaz de permitir mapas grandes y tiles en los SVs
extern "C"
void launchSVMPrediction(const svm_model * &model,
                         const unsigned int & rows, const unsigned int & cols, 
                         unsigned char * &h_data) {
    int L = model->l;
    if (model->l > (MEM_BLOCK)) {
        cerr << "Error: The number of support vectors is " << model->l << endl;
        cerr << "Unable to reserve memory for more than " << MEM_BLOCK << "SVs" << endl;
        cerr << "Using just the first " << MEM_BLOCK << "SVs" << endl;
        L = MEM_BLOCK;
    }
                         
    // Reserving device memory for ouput data
    unsigned char * d_data;
    CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(unsigned char) * rows * cols));
    
    // Data structure is transformed for Support Vectors and passed to the device 
    float2 * h_SVs = new float2[L];
    for (int i = 0; i < L; i++) {
        h_SVs[i] = make_float2(model->SV[i].values[0], model->SV[i].values[1]);
    }   
    float2 * d_SVs;
    CUDA_SAFE_CALL(cudaMalloc(&d_SVs, sizeof(float2) * L));
    CUDA_SAFE_CALL(cudaMemcpy(d_SVs, h_SVs, sizeof(float2) * L, cudaMemcpyHostToDevice));
    
    // Support vector coefficients are passed to device memory
    float * h_coeffs = new float[L];
    for (int i = 0; i < L; i++) {
        h_coeffs[i] = model->sv_coef[0][i];
    }
    float * d_coeffs;
    CUDA_SAFE_CALL(cudaMalloc(&d_coeffs, sizeof(float) * L));
    CUDA_SAFE_CALL(cudaMemcpy(d_coeffs, h_coeffs, sizeof(float) * L, cudaMemcpyHostToDevice));
    
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(cols / blockSize.x + 1, rows / blockSize.x + 1, 1);
    
#ifdef DEBUG
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif    
    
    predictPixel <<<gridSize, blockSize>>> (d_coeffs, d_SVs, model->param.gamma, model->rho[0], L,
                                            rows, cols, d_data);
    
    cudaDeviceSynchronize(); CUDA_SAFE_CALL(cudaGetLastError());
    
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for kernel = " << elapsed << endl;
#endif
    
    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost));
    
    CUDA_SAFE_CALL(cudaFree(d_data));
    
    CUDA_SAFE_CALL(cudaGetLastError());
}