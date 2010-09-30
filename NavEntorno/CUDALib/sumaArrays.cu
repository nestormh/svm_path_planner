#include "CUDAlib.h"
#include <cutil_inline.h>

using namespace std;

float * dV1;
float * dV2;
float * dV3;

__global__
void vecAdd(float * v1, float * v2, float * v3, float N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //if (i < N)
        v3[i] = v1[i] + v2[i];
}

void sumaArrays() {

    int N = 640 * 480;
    size_t size = N * sizeof(float);
    float * v1 = (float *)malloc(size);
    float * v2 = (float *)malloc(size);
    float * v3 = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        v1[i] = 1;
        v2[i] = 2;
    }

    cutilSafeCall(cudaMalloc(&dV1, size));
    cutilSafeCall(cudaMalloc(&dV2, size));
    cutilSafeCall(cudaMalloc(&dV3, size));

    cutilSafeCall(cudaMemcpy(dV1, v1, size, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(dV2, v2, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vecAdd<<<blocksPerGrid, threadsPerBlock>>> (dV1, dV2, dV3, N);

    cutilSafeCall(cudaMemcpy(v1, dV1, size, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(v2, dV2, size, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(v3, dV3, size, cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaFree(dV1));
    cutilSafeCall(cudaFree(dV2));
    cutilSafeCall(cudaFree(dV3));

    for (int i = 0; i < N; i++) {
        printf("%d: %f\t%f\t%f\n", i, v1[i], v2[i], v3[i]);
    }

    free(v1);
    free(v2);
    free(v3);
}