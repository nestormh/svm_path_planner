#include "CUDAlib.h"
#include <cutil_inline.h>
#include <c++/4.4/limits>
#include <c++/4.4/cmath>

__constant__ float * d_desc1;
__constant__ float * d_desc2;

__global__
void calcMean(float * desc1, float * desc2, float * m1, float * m2) {
    __shared__ float partialSum1[MEAN_SDV_THREADS][SURF_DESCRIPTOR_SIZE];
    __shared__ float partialSum2[MEAN_SDV_THREADS][SURF_DESCRIPTOR_SIZE];
    
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    unsigned int descPos = (bx * (MEAN_SDV_THREADS) + ty) * SURF_DESCRIPTOR_SIZE + tx;
    unsigned int mPos = bx * MEAN_SDV_THREADS + ty;

    partialSum1[ty][tx] = desc1[descPos];
    partialSum2[ty][tx] = desc2[descPos];

    for (unsigned int stride = blockDim.x>>1; stride > 0; stride >>= 1) {
        __syncthreads();

        if (tx < stride) {
            partialSum1[ty][tx] += partialSum1[ty][tx + stride];
            partialSum2[ty][tx] += partialSum2[ty][tx + stride];
        }        
    }

    __syncthreads();

    if (tx == 0) {
        m1[mPos] = partialSum1[ty][0] / SURF_DESCRIPTOR_SIZE;
        m2[mPos] = partialSum2[ty][0] / SURF_DESCRIPTOR_SIZE;
    }    
}

__global__
void calcSdv(float * desc1, float * desc2, float * m1, float * m2, float * sdv1, float * sdv2) {
    __shared__ float partialSum1[MEAN_SDV_THREADS][SURF_DESCRIPTOR_SIZE];
    __shared__ float partialSum2[MEAN_SDV_THREADS][SURF_DESCRIPTOR_SIZE];
    __shared__ float mean1[MEAN_SDV_THREADS];
    __shared__ float mean2[MEAN_SDV_THREADS];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    unsigned int mPos = bx * MEAN_SDV_THREADS + ty;
    unsigned int descPos = (bx * (MEAN_SDV_THREADS) + ty) * SURF_DESCRIPTOR_SIZE + tx;

    if (tx == 0) {
        mean1[ty] = m1[mPos];
        mean2[ty] = m2[mPos];
    }
    __syncthreads();

    float sub1 = desc1[descPos] - mean1[ty];
    float sub2 = desc2[descPos] - mean2[ty];

    partialSum1[ty][tx] = sub1 * sub1;
    partialSum2[ty][tx] = sub2 * sub2;
    
    for (unsigned int stride = blockDim.x>>1; stride > 0; stride >>= 1) {
        __syncthreads();

        if (tx < stride) {
            partialSum1[ty][tx] += partialSum1[ty][tx + stride];
            partialSum2[ty][tx] += partialSum2[ty][tx + stride];
        }
    }

    __syncthreads();

    if (tx == 0) {
        sdv1[mPos] = sqrt(partialSum1[ty][0] / SURF_DESCRIPTOR_SIZE);
        sdv2[mPos] = sqrt(partialSum2[ty][0] / SURF_DESCRIPTOR_SIZE);
    }       
}

__global__
void calcCorrelation(float * desc1, float * desc2, float * corr, float * mean1, float * mean2, float * sdv1, float * sdv2, int cols) {

    __shared__ float tmpDesc1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tmpDesc2[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tmpMean[2][TILE_WIDTH];
    __shared__ float tmpSdv[2][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int width = SURF_DESCRIPTOR_SIZE;

    if (ty == 0) {
        tmpMean[ty][tx] = mean1[by * TILE_WIDTH + tx];
        tmpSdv[ty][tx] = sdv1[by * TILE_WIDTH + tx];
    } else if (ty == 1) {
        tmpMean[ty][tx] = mean2[bx * TILE_WIDTH + tx];
        tmpSdv[ty][tx] = sdv2[bx * TILE_WIDTH + tx];
    }
    __syncthreads();

    float pVal = 0;
    for (int m = 0; m < width / TILE_WIDTH; m++) {
        // Collaborative loading of tiles
        tmpDesc1[ty][tx] = desc1[row * width + (m * TILE_WIDTH + tx)];
        tmpDesc2[tx][ty] = desc2[col * width + (m * TILE_WIDTH + ty)];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            pVal += (tmpDesc1[ty][k] - tmpMean[0][ty]) * (tmpDesc2[tx][k] - tmpMean[1][tx]);
        }

        __syncthreads();
    }
    pVal /= (width - 1) * tmpSdv[0][ty] * tmpSdv[1][tx];
        
    corr[row * cols + col] = pVal;
}

__global__
void bestCorrX(float * corr, int * bestCorr2, int size1, int size2) {
    __shared__ float partialComp[BEST_CORR_X][BEST_CORR_Y];
    __shared__ int partialIdx[BEST_CORR_X][BEST_CORR_Y];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    float tmpVal = 0;

    unsigned int partialCorrPos = (bx * BEST_CORR_Y) + tx;
    unsigned int yPos = 0;

    partialComp[ty][tx] = corr[(ty * size2) + partialCorrPos];
    partialIdx[ty][tx] = ty;

    // Obtenemos el mayor de cada "pseudobloque"
    for (unsigned int i = BEST_CORR_X; i < size1; i += BEST_CORR_X) {
        yPos = i + ty;
        tmpVal = corr[(yPos * size2) + partialCorrPos];
        if (partialComp[ty][tx] < tmpVal) {
            partialComp[ty][tx] = tmpVal;
            partialIdx[ty][tx] = yPos;
        }
    }

    // Obtenemos el mayor para el bloque inicial
    for (unsigned int stride = blockDim.y>>1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (ty < stride) {
            if (partialComp[ty + stride][tx] > partialComp[ty][tx]) {
                partialComp[ty][tx] = partialComp[ty + stride][tx];
                partialIdx[ty][tx] = partialIdx[ty + stride][tx];
            }
        }        
    }

    if (ty == 0) {
        bestCorr2[partialCorrPos] = partialIdx[ty][tx];
    }
}

__global__
void bestCorrY(float * corr, int * bestCorr2, int * matches, int size2) {
    __shared__ float partialComp[BEST_CORR_Y][BEST_CORR_X];
    __shared__ int partialIdx[BEST_CORR_Y][BEST_CORR_X];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    float tmpVal = 0;

    unsigned int yPos = (by * BEST_CORR_Y + ty) * size2;
    unsigned int xPos = 0;

    partialComp[ty][tx] = corr[yPos + tx];
    partialIdx[ty][tx] = tx;

    // Obtenemos el mayor de cada "pseudobloque"
    for (unsigned int i = BEST_CORR_X; i < size2; i += BEST_CORR_X) {
        xPos = i + tx;
        tmpVal = corr[yPos + xPos];
        if (partialComp[ty][tx] < tmpVal) {
            partialComp[ty][tx] = tmpVal;
            partialIdx[ty][tx] = xPos;
        }
    }

    // Obtenemos el mayor para el bloque inicial
    for (unsigned int stride = blockDim.x>>1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tx < stride) {
            if (partialComp[ty][tx + stride] > partialComp[ty][tx]) {
                partialComp[ty][tx] = partialComp[ty][tx + stride];
                partialIdx[ty][tx] = partialIdx[ty][tx + stride];
            }
        }
    }

    if (tx == 0) {
        int pos = ty + by * BEST_CORR_Y;
        int bestCorr1 = partialIdx[ty][tx];

        if ((bestCorr2[bestCorr1] == pos) && (partialComp[ty][tx] > CORRELATION_THRESH)) {
            matches[pos] = bestCorr1;
        } else {
            matches[pos] = -1;
        }
    }    
}

void matchSURFGPU(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, vector<int> &matches, t_Timings &timings) {
    clock_t myTime = clock();    

    int size1 = (int(points1.size() / (TILE_WIDTH * 2)) + 1) * (TILE_WIDTH * 2);
    int size2 = (int(points2.size() / (TILE_WIDTH * 2)) + 1) * (TILE_WIDTH * 2);
    int size = max(size1, size2);
    size_t corrSize = size1 * size2 * sizeof(float);

    float * h_desc1 = (float *)malloc(size * SURF_DESCRIPTOR_SIZE * sizeof(float));
    float * h_desc2 = (float *)malloc(size * SURF_DESCRIPTOR_SIZE * sizeof(float));

    for (int i = 0; i < desc1.size(); i++) {        
        h_desc1[i] = (float)desc1.at(i);
    }
    for (int i = 0; i < desc2.size(); i++) {
        h_desc2[i] = (float)desc2.at(i);
    }
    for (int i = desc1.size(); i < size * SURF_DESCRIPTOR_SIZE; i++) {
        h_desc1[i] = 0;
    }
    for (int i = desc2.size(); i < size * SURF_DESCRIPTOR_SIZE; i++) {
        h_desc2[i] = 0;
    }
    
    float * d_m1;
    float * d_m2;
    float * d_sdv1;
    float * d_sdv2;
    float * d_corr;
    int * d_bestCorr2;
    int * d_matches;        

    cutilSafeCall(cudaMalloc(&d_m1, size * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_sdv1, size * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_m2, size * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_sdv2, size * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_desc1, size * SURF_DESCRIPTOR_SIZE * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_desc2, size * SURF_DESCRIPTOR_SIZE * sizeof(float)));

    cutilSafeCall(cudaMemcpy(d_desc1, h_desc1, size * SURF_DESCRIPTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_desc2, h_desc2, size * SURF_DESCRIPTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    timings.tMalloc1 = clock() - myTime;
    myTime = clock();

    dim3 dimBlockMeanSdv(SURF_DESCRIPTOR_SIZE, MEAN_SDV_THREADS);
    dim3 dimGridMeanSdv(size / dimBlockMeanSdv.y, 1);    
    calcMean <<< dimGridMeanSdv, dimBlockMeanSdv >>> (d_desc1, d_desc2, d_m1, d_m2);
    calcSdv <<< dimGridMeanSdv, dimBlockMeanSdv >>> (d_desc1, d_desc2, d_m1, d_m2, d_sdv1, d_sdv2);
    cudaThreadSynchronize();

    timings.tCalcMeanSdv = clock() - myTime;
    myTime = clock();

    cutilSafeCall(cudaMalloc(&d_corr, corrSize));

    timings.tMalloc2 = clock() - myTime;
    timings.tMalloc = timings.tMalloc1 + timings.tMalloc2;
    myTime = clock();

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(size2 / TILE_WIDTH, size1 / TILE_WIDTH);

    calcCorrelation <<< dimGrid, dimBlock >>> (d_desc1, d_desc2, d_corr, d_m1, d_m2, d_sdv1, d_sdv2, size2);
    cudaThreadSynchronize();

    cutilSafeCall(cudaFree(d_m1));
    cutilSafeCall(cudaFree(d_m2));
    cutilSafeCall(cudaFree(d_desc1));
    cutilSafeCall(cudaFree(d_desc2));
    cutilSafeCall(cudaFree(d_sdv1));
    cutilSafeCall(cudaFree(d_sdv2));

    timings.tCalcCorrelation = clock() - myTime;
    myTime = clock();

    cutilSafeCall(cudaMalloc(&d_bestCorr2, size2 * sizeof(int)));
    cutilSafeCall(cudaMalloc(&d_matches, size1 * sizeof(int)));

    dim3 dimBlockBestCorrX(BEST_CORR_Y, BEST_CORR_X);
    dim3 dimGridBestCorrX(size2 / BEST_CORR_Y, 1);
    bestCorrX <<< dimGridBestCorrX, dimBlockBestCorrX >>> (d_corr, d_bestCorr2, size1, size2);
    cudaThreadSynchronize();

    dim3 dimBlockBestCorrY(BEST_CORR_X, BEST_CORR_Y);
    dim3 dimGridBestCorrY(1, size1 / BEST_CORR_Y);
    bestCorrY <<< dimGridBestCorrY, dimBlockBestCorrY >>> (d_corr, d_bestCorr2, d_matches, size2);
    cudaThreadSynchronize();

    timings.tCalcBestCorr = clock() - myTime;
    myTime = clock();

    int * h_matches = (int *)malloc(points1.size() * sizeof(int));
    cutilSafeCall(cudaMemcpy(h_matches, d_matches, points1.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < points1.size(); i++)
        matches.push_back(h_matches[i]);

    timings.tMemCpy = clock() - myTime;
    myTime = clock();
    
    cutilSafeCall(cudaFree(d_corr));
    cutilSafeCall(cudaFree(d_bestCorr2));
    cutilSafeCall(cudaFree(d_matches));

    timings.tFreeMem = clock() - myTime;
    
    free(h_desc1);
    free(h_desc2);
    free(h_matches);
}
