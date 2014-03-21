/*
    Copyright (c) 2013, Néstor Morales Hernández <nestor@isaatc.ull.es>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <time.h>
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdio.h>
#include <iostream>
#include <complex>

#define BLOCK_SIZE 1024
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
void checkEdge(const float2 * d_pointsInMap, const unsigned int nPointsInMap,
               const float2 * d_edgeU, const float2 * d_edgeV, const unsigned int nEdges,
               float minDist, bool * d_validEdges) {
               
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > nEdges)
        return;
    
    __syncthreads();

    float2 v = d_edgeU[idx];
    float2 w = d_edgeV[idx];
    float2 p;
    
    float lineLenghtSqr;
    float t;
    float tmpDist;
    
    float2 tmpPoint;
    bool valid = true;
    for (unsigned int i = 0; i < nPointsInMap; i++) {
        p = d_pointsInMap[i];
        
        lineLenghtSqr = (v.x - w.x) * (v.x - w.x) + (v.y - w.y) * (v.y - w.y);
        
        if (lineLenghtSqr == 0) {
            tmpDist = (p.x - v.x) * (p.x - v.x) + (p.y - v.y) * (p.y - v.y);
            if (tmpDist < minDist) {
                valid = false;
                break;
            }
            continue;
        }
        
        t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / lineLenghtSqr;

        if (t < 0) {
            tmpDist = (p.x - v.x) * (p.x - v.x) + (p.y - v.y) * (p.y - v.y);
            if (tmpDist < minDist) {
                valid = false;
                break;
            }
            continue;
        }
        if (t > 1) {
            tmpDist = (p.x - w.x) * (p.x - w.x) + (p.y - w.y) * (p.y - w.y);
            if (tmpDist < minDist) {
                valid = false;
                break;
            }
            continue;
        }
        
        tmpPoint = make_float2(v.x + t * (w.x - v.x), v.y + t * (w.y - v.y));
    
        tmpDist = (p.x - tmpPoint.x) * (p.x - tmpPoint.x) + (p.y - tmpPoint.y) * (p.y - tmpPoint.y);
        
        
        if (tmpDist < minDist) {
            valid = false;
            break;
        }
    }
    
    __syncthreads();
    
    d_validEdges[idx] = valid;
}

extern "C"
void launchCheckEdges(const float2 * &h_pointsInMap, const unsigned int &nPointsInMap,
                      const float2 * &h_edgeU, const float2 * &h_edgeV, const unsigned int &nEdges,
                      const float &minDist, bool * &h_validEdges) {
                      
    float2 *d_pointsInMap, *d_edgeU, *d_edgeV;
    bool *d_validEdges;
    CUDA_SAFE_CALL(cudaMalloc(&d_pointsInMap, (sizeof(float2) * nPointsInMap)));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeU, (sizeof(float2) * nEdges)));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeV, (int)(sizeof(float2) * nEdges)));
    CUDA_SAFE_CALL(cudaMalloc(&d_validEdges, (int)(sizeof(bool) * nEdges)));
    
    CUDA_SAFE_CALL(cudaMemcpy(d_pointsInMap, h_pointsInMap, sizeof(float2) * nPointsInMap, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeU, h_edgeU, sizeof(float2) * nEdges, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeV, h_edgeV, sizeof(float2) * nEdges, cudaMemcpyHostToDevice));
    
    const dim3 blockSize(BLOCK_SIZE, 1, 1);
    const dim3 gridSize((nEdges / blockSize.x) + 1, 1, 1);
    
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    checkEdge <<<gridSize, blockSize>>> (d_pointsInMap, nPointsInMap, d_edgeU, d_edgeV, nEdges, minDist, d_validEdges);
    cudaDeviceSynchronize(); CUDA_SAFE_CALL(cudaGetLastError());
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for cleaning kernel = " << elapsed << endl;
    
    CUDA_SAFE_CALL(cudaMemcpy(h_validEdges, d_validEdges, sizeof(bool) * nEdges, cudaMemcpyDeviceToHost));
    
    CUDA_SAFE_CALL(cudaFree(d_pointsInMap));
    CUDA_SAFE_CALL(cudaFree(d_edgeU));
    CUDA_SAFE_CALL(cudaFree(d_edgeV));
    CUDA_SAFE_CALL(cudaFree(d_validEdges));
    
    CUDA_SAFE_CALL(cudaGetLastError());

}