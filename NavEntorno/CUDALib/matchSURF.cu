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
void calcCorrelation(float * desc1, float * desc2, float * corr, float * sdv1, float * sdv2, bool * resp1, bool * resp2, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) return;
    if (col >= cols) return;

    if (resp1[row] == resp2[col]) {
        int idx = __fmaf_rn(row, cols, col);
        float correl = 0.0f;
        #pragma unroll 64
        for (int i = 0; i < SURF_DESCRIPTOR_SIZE; i++) {
            correl = __fadd_rn(correl, __fmul_rn(desc1[row * SURF_DESCRIPTOR_SIZE + i], desc2[col * SURF_DESCRIPTOR_SIZE + i]));
        }
        corr[idx] = __fdiv_rn(__fdiv_rn(__fdiv_rn(correl, SURF_DESCRIPTOR_SIZE - 1), sdv1[row]), sdv2[col]);
    }
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

void bruteMatchParallel(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, vector<int> &matches) {

    cout << "Points1 " << points1.size() << endl;
    cout << "Points2 " << points2.size() << endl;

    size_t corrSize = points1.size() * points2.size() * sizeof(float);    

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

    cutilSafeCall(cudaMalloc(&d_m2, points2.size() * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_sdv2, points2.size() * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_m1, points1.size() * sizeof(float)));    
    cutilSafeCall(cudaMalloc(&d_sdv1, points1.size() * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_desc1, desc1.size() * sizeof(float)));
    cutilSafeCall(cudaMalloc(&d_desc2, desc2.size() * sizeof(float)));

    cutilSafeCall(cudaMemcpy(d_desc1, h_desc1, desc1.size() * sizeof(float), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_desc2, h_desc2, desc2.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int blocksPerGrid = ((points1.size() + points2.size()) / threadsPerBlock) + 1;
    calcMeanSdv <<< blocksPerGrid, threadsPerBlock >>> (d_desc1, d_desc2, d_m1, d_m2, d_sdv1, d_sdv2, points1.size(), points2.size());

    cutilSafeCall(cudaFree(d_m1));
    cutilSafeCall(cudaFree(d_m2));

    cutilSafeCall(cudaMalloc(&d_corr, corrSize));
    cutilSafeCall(cudaMalloc(&d_response1, points1.size() * sizeof(bool)));
    cutilSafeCall(cudaMalloc(&d_response2, points2.size() * sizeof(bool)));
    cutilSafeCall(cudaMalloc(&d_bestCorr1, points1.size() * sizeof(int)));
    cutilSafeCall(cudaMalloc(&d_bestCorr2, points2.size() * sizeof(int)));
    cutilSafeCall(cudaMalloc(&d_matches, points1.size() * sizeof(int)));

    cutilSafeCall(cudaMemcpy(d_response1, h_response1, points1.size() * sizeof(bool), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_response2, h_response2, points2.size() * sizeof(bool), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((points2.size() / dimBlock.x) + 1, (points1.size() / dimBlock.y) + 1);

    calcCorrelation <<< dimGrid, dimBlock >>> (d_desc1, d_desc2, d_corr, d_sdv1, d_sdv2, d_response1, d_response2, points1.size(), points2.size());

    calcBestCorr <<< blocksPerGrid, threadsPerBlock >>> (d_corr, d_bestCorr1, d_bestCorr2, points1.size(), points2.size());

    blocksPerGrid = ((points1.size() - 1) / threadsPerBlock) + 1;
    calcMatches <<< blocksPerGrid, threadsPerBlock >>> (d_bestCorr1, d_bestCorr2, d_matches, points1.size());

    int * h_matches = (int *)malloc(points1.size() * sizeof(int));

    cutilSafeCall(cudaMemcpy(h_matches, d_matches, points1.size() * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < points1.size(); i++)
        matches.push_back(h_matches[i]);

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

    free(h_desc1);
    free(h_desc2);
    free(h_response1);
    free(h_response2);
    free(h_matches);
}
