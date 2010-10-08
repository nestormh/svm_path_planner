#include "CUDAlib.h"
#include <cutil_inline.h>

__constant__ float * d_desc1;
__constant__ float * d_desc2;

__global__
void calcMeanSdv(float * desc1, float * desc2, float * m1, float * m2, float * sdv1, float * sdv2, int n1, int n2) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    float * m;
    float * sdv;
    float * desc;

    if (pos < n1) {
        m = m1;
        sdv = sdv1;
        desc = desc1;
    } else if (pos < (n1 + n2)) {
        pos = pos - n1;
        m = m2;
        sdv = sdv2;
        desc = desc2;        
    } else return;
    
    m[pos] = 0.0f;
    sdv[pos] = 0.0f;

    #pragma unroll 64
    for (int i = 0; i < SURF_DESCRIPTOR_SIZE; i++) {
        m[pos] = __fadd_rn(m[pos], desc[pos * SURF_DESCRIPTOR_SIZE + i]);
    }
    m[pos] = __fdiv_rn(m[pos], SURF_DESCRIPTOR_SIZE);


    float tmp;
    for (int i = 0; i < SURF_DESCRIPTOR_SIZE; i++) {
        desc[pos * SURF_DESCRIPTOR_SIZE + i] = __fadd_rn(desc[pos * SURF_DESCRIPTOR_SIZE + i], -m[pos]);
        tmp = __fmul_rn(desc[pos * SURF_DESCRIPTOR_SIZE + i], desc[pos * SURF_DESCRIPTOR_SIZE + i]);

        sdv[pos] = __fadd_rn(tmp, sdv[pos]);
    }
    sdv[pos] = __fdiv_rn(sdv[pos], SURF_DESCRIPTOR_SIZE);
    sdv[pos] = __fsqrt_rn(sdv[pos]);
}

__global__
void calcCorrelation(float * desc1, float * desc2, float * corr, float * mean1, float * mean2, float * sdv1, float * sdv2, bool * resp1, bool * resp2, int rows, int cols) {

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
    
    corr[col * rows + row] = pVal;
}

__global__
void calcCorrelationOrig(float * desc1, float * desc2, float * corr, float * sdv1, float * sdv2, bool * resp1, bool * resp2, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;
    if (col >= cols) return;

    //if (resp1[row] == resp2[col]) {
    int idx = __fmaf_rn(row, cols, col);
    float correl = 0.0f;
#pragma unroll 64
    for (int i = 0; i < SURF_DESCRIPTOR_SIZE; i++) {
        correl = __fadd_rn(correl, __fmul_rn(desc1[row * SURF_DESCRIPTOR_SIZE + i], desc2[col * SURF_DESCRIPTOR_SIZE + i]));
    }
    corr[idx] = __fdiv_rn(__fdiv_rn(__fdiv_rn(correl, SURF_DESCRIPTOR_SIZE - 1), sdv1[row]), sdv2[col]);
    //}
}

__global__
void calcBestCorr(float * corr, int * bestCorr1, int * bestCorr2, int rows, int cols) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    int maxPos;
    
    float * bestCorrelation;

    if (pos < rows) {
        int row = pos;
        int best = -1;
        float bestCorr = 0.0f;
        //#pragma unroll 30
        for (int i = row * cols; i < (row * cols) + cols; i++) {
            //if ((corr[i] > bestCorr) && (corr[i] > CORRELATION_THRESH)) {
            if (corr[i] > bestCorr) {
                best = i - (row * cols);
                bestCorr = corr[i];
            }
        }
        bestCorr1[row] = best;
    } else if (pos < (rows + cols)) {
        int col = pos - rows;

        int best = 0;
        float bestCorr = 0.0f;
        //#pragma unroll 30
        int tmpPos = 0;
        for (int i = col; i < col + (rows * cols); i += cols) {
            tmpPos++;
            //if ((corr[i] > bestCorr) && (corr[i] > CORRELATION_THRESH)) {
            if (corr[i] > bestCorr) {
                best = (i - col) / cols;
                bestCorr = corr[i];
            }
        }
        bestCorr2[col] = best;
    }
}

__global__
void calcMatches(int * bestCorr1, int * bestCorr2, int * matches, int n1) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= n1) return;
    
    if (bestCorr2[bestCorr1[pos]] == pos) {
        matches[pos] = bestCorr1[pos];
    } else {
        matches[pos] = -1;
    }
}

void calcMeanSdvSequential(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, float * d_m1, float * d_m2, float * d_sdv1, float * d_sdv2) {
    float* avg1 = (float*) malloc(sizeof (float) * points1.size());
    float* avg2 = (float*) malloc(sizeof (float) * points2.size());
    float* dev1 = (float*) malloc(sizeof (float) * points1.size());
    float* dev2 = (float*) malloc(sizeof (float) * points2.size());

    int descriptor_size = 64;
    for (int i = 0; i < points1.size(); i++) {
        // find average and standard deviation of each descriptor
        avg1[i] = 0;
        dev1[i] = 0;

        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            avg1[i] += desc1.at(k);
        }
        avg1[i] /= descriptor_size;
        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            dev1[i] += (desc1.at(k) - avg1[i]) * (desc1.at(k) - avg1[i]);
        }
        dev1[i] = sqrt(dev1[i] / descriptor_size);
    }
    for (int i = 0; i < points2.size(); i++) {
        // find average and standard deviation of each descriptor
        avg2[i] = 0;
        dev2[i] = 0;

        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            avg2[i] += desc2.at(k);
        }
        avg2[i] /= descriptor_size;
        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            dev2[i] += (desc2.at(k) - avg2[i]) * (desc2.at(k) - avg2[i]);
        }
        dev2[i] = sqrt(dev2[i] / descriptor_size);
    }

    cutilSafeCall(cudaMemcpy(d_m1, avg1, points1.size() * sizeof(float), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_m2, avg2, points2.size() * sizeof(float), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_sdv1, dev1, points1.size() * sizeof(float), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_sdv2, dev2, points2.size() * sizeof(float), cudaMemcpyHostToDevice));

    delete avg1;
    delete avg2;
    delete dev1;
    delete dev2;
}

void bruteMatchParallel(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, vector<int> &matches, t_Timings &timings) {

    cout << "Points1 " << points1.size() << endl;
    cout << "Points2 " << points2.size() << endl;

    clock_t myTime = clock();    

    float * h_desc1 = (float *)malloc(desc1.size() * sizeof(float));
    float * h_desc2 = (float *)malloc(desc2.size() * sizeof(float));
    float * h_response1 = (float *)malloc(points1.size() * sizeof(float));
    float * h_response2 = (float *)malloc(points2.size() * sizeof(float));

    for (int i = 0; i < desc1.size(); i++) {
        h_desc1[i] = desc1.at(i);
    }
    for (int i = 0; i < desc2.size(); i++) {
        h_desc2[i] = desc2.at(i);
    }
    for (int i = 0; i < points1.size(); i++) {
        h_response1[i] = points1.at(i).response;
    }
    for (int i = 0; i < points2.size(); i++) {
        h_response2[i] = points2.at(i).response;
    }

    float * d_corr;
    float * d_m1;
    float * d_m2;
    float * d_sdv1;
    float * d_sdv2;
    bool * d_response1;
    bool * d_response2;
    int * d_bestCorr1;
    int * d_bestCorr2;
    int * d_matches;

    int size1 = (int(points1.size() / TILE_WIDTH) + 1) * TILE_WIDTH;
    int size2 = (int(points2.size() / TILE_WIDTH) + 1) * TILE_WIDTH;
    size_t corrSize = size1 * size2 * sizeof(float);

    //cutilSafeCall(cudaMalloc(&d_m2, size1 + size2 * sizeof(float)));
    //cutilSafeCall(cudaMalloc(&d_sdv2, size2 * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_m1, size1 * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_sdv1, size1 * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_m2, size2 * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_sdv2, size2 * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_desc1, size1 * SURF_DESCRIPTOR_SIZE * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_desc2, size2 * SURF_DESCRIPTOR_SIZE * sizeof(float)));

    cutilSafeCall(cudaMemcpy(d_desc1, h_desc1, desc1.size() * sizeof(float), cudaMemcpyHostToDevice));    
    cutilSafeCall(cudaMemcpy(d_desc2, h_desc2, desc2.size() * sizeof(float), cudaMemcpyHostToDevice));    

    timings.tMalloc1 = clock() - myTime;
    myTime = clock();

    int threadsPerBlock = 512;
    int blocksPerGrid = ((points1.size() + points2.size()) / threadsPerBlock) + 1;
    //calcMeanSdv <<< blocksPerGrid, threadsPerBlock >>> (d_desc1, d_desc2, d_m1, d_m2, d_sdv1, d_sdv2, points1.size(), points2.size());
    //cudaThreadSynchronize();
    calcMeanSdvSequential(points1, points2, desc1, desc2, d_m1, d_m2, d_sdv1, d_sdv2);

    timings.tCalcMeanSdv = clock() - myTime;
    myTime = clock();

    //cutilSafeCall(cudaFree(d_m1));
    //cutilSafeCall(cudaFree(d_m2));

    cutilSafeCall(cudaMalloc(&d_corr, corrSize));
    cutilSafeCall(cudaMalloc(&d_response1, size1 * sizeof(bool)));
    cutilSafeCall(cudaMalloc(&d_response2, size2 * sizeof(bool)));
    cutilSafeCall(cudaMalloc(&d_bestCorr1, size1 * sizeof(int)));
    cutilSafeCall(cudaMalloc(&d_bestCorr2, size2 * sizeof(int)));
    cutilSafeCall(cudaMalloc(&d_matches, size1 * sizeof(int)));

    cutilSafeCall(cudaMemcpy(d_response1, h_response1, points1.size() * sizeof(bool), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_response2, h_response2, points2.size() * sizeof(bool), cudaMemcpyHostToDevice));

    timings.tMalloc2 = clock() - myTime;
    timings.tMalloc = timings.tMalloc1 + timings.tMalloc2;
    myTime = clock();

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(size2 / TILE_WIDTH, size1 / TILE_WIDTH);

    calcCorrelation <<< dimGrid, dimBlock >>> (d_desc1, d_desc2, d_corr, d_m1, d_m2, d_sdv1, d_sdv2, d_response1, d_response2, size1, size2);
    cudaThreadSynchronize();

    timings.tCalcCorrelation = clock() - myTime;
    myTime = clock();

    /*calcBestCorr <<< blocksPerGrid, threadsPerBlock >>> (d_corr, d_bestCorr1, d_bestCorr2, points1.size(), points2.size());
    cudaThreadSynchronize();

    timings.tCalcBestCorr = clock() - myTime;
    myTime = clock();

    blocksPerGrid = ((points1.size() - 1) / threadsPerBlock) + 1;
    calcMatches <<< blocksPerGrid, threadsPerBlock >>> (d_bestCorr1, d_bestCorr2, d_matches, points1.size());
    cudaThreadSynchronize();

    timings.tCalcMatches = clock() - myTime;
    myTime = clock();

    clock_t tmpTime = clock();
    int * h_matches = (int *)malloc(points1.size() * sizeof(int));
    cout << "memCpy1 = " << clock() - tmpTime << endl;
    tmpTime = clock();

    cutilSafeCall(cudaMemcpy(h_matches, d_matches, points1.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cout << "memCpy2 = " << clock() - tmpTime << endl;
    tmpTime = clock();

    for (int i = 0; i < points1.size(); i++)
        matches.push_back(h_matches[i]);
    cout << "memCpy3 = " << clock() - tmpTime << endl;

    timings.tMemCpy = clock() - myTime;
    myTime = clock();*/

    cout << (size1 + size2) * sizeof(float) << endl;
    cutilSafeCall(cudaFree(d_m1));
    cutilSafeCall(cudaFree(d_m2));
    cutilSafeCall(cudaFree(d_desc1));
    cutilSafeCall(cudaFree(d_desc2));
    cutilSafeCall(cudaFree(d_corr));
    cutilSafeCall(cudaFree(d_sdv1));
    cutilSafeCall(cudaFree(d_sdv2));
    cutilSafeCall(cudaFree(d_response1));
    cutilSafeCall(cudaFree(d_response2));
    cutilSafeCall(cudaFree(d_bestCorr1));
    cutilSafeCall(cudaFree(d_bestCorr2));
    cutilSafeCall(cudaFree(d_matches));

    timings.tFreeMem = clock() - myTime;
    
    free(h_desc1);
    free(h_desc2);
    free(h_response1);
    free(h_response2);
    //free(h_matches);

    
}
