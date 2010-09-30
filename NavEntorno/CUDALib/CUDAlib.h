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

using namespace std;

typedef struct {
    float x;
    float y;
    bool response;
} t_Point;

    void sumaArrays();
    void enumerateDevices();
    void bruteMatchParallel(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, vector<int> &matches);
    void bruteMatchParallel2(vector<t_Point> points1, vector<t_Point> points2, vector<float> desc1, vector<float> desc2, vector<int> &matches);

#endif	/* _TESTCUDA_H */

