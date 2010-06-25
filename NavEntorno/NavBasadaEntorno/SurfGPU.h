/* 
 * File:   SurfGPU.h
 * Author: nestor
 *
 * Created on 25 de junio de 2010, 12:39
 */

#ifndef _SURFGPU_H
#define	_SURFGPU_H

#include <gpusurf/GpuSurfDetector.hpp>
#include <vector>
#include <string>
#include "stdafx.h"

using namespace cv;
using namespace asrl;
using namespace std;

class SurfGPU {
public:
    SurfGPU();
    SurfGPU(const SurfGPU& orig);
    virtual ~SurfGPU();
    void testSurf(string file1, string file2);

private:
    void testSurfGPU(cv::Mat img, vector<KeyPoint> &points);
    bool isLastBitSet(const int * f);
    bool isLastBitSet(const float & f);
    void drawKeypoints(std::vector<cv::KeyPoint> const & keypoints, IplImage * imgGrayscale, string name);
};

#endif	/* _SURFGPU_H */

