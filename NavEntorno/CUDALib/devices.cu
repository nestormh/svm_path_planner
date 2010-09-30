#include "CUDAlib.h"
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

void enumerateDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("%s\n", deviceProp.name);
    }
}