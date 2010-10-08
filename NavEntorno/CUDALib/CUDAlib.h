/* 
 * File:   testCUDA.h
 * Author: nestor
 *
 * Created on 5 de julio de 2010, 10:19
 */

#ifndef _TESTCUDA_H
#define	_TESTCUDA_H

#include <vector>
#include <stdio.h>
#include <iostream>

#define SURF_DESCRIPTOR_SIZE 64
#define CORRELATION_THRESH 0.99
#define TILE_WIDTH 16

using namespace std;

typedef struct {
    float x;
    float y;
    bool response;
} t_Point;

typedef struct {
    long tTotal;
    long tCalcMeanSdv;
    long tCalcCorrelation;
    long tCalcBestCorr;
    long tCalcMatches;
    long tMalloc1;
    long tMalloc2;
    long tMalloc;
    long tMemCpy;
    long tFreeMem;
    long tRANSAC;
    long tPrevRANSAC;

    long tSurf1;
    long tSurf2;

    int nPoints1;
    int nPoints2;
    int nPairs;
    int nPairsClean;

    int threadsPerBlock;
    int blocksPerGrid;

    t_Point dimBlock;
    t_Point dimGrid;

} t_Timings;

    void sumaArrays();
    void enumerateDevices();
    void bruteMatchParallel(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, vector<int> &matches, t_Timings &timings);

#endif	/* _TESTCUDA_H */

