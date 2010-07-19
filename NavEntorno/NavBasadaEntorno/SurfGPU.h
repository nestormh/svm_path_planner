/* 
 * File:   SurfGPU.h
 * Author: nestor
 *
 * Created on 25 de junio de 2010, 12:39
 */

#ifndef _SURFGPU_H
#define	_SURFGPU_H

#include <gpusurf/GpuSurfDetector.hpp>
#include <gpu_globals.h>
#include <vector>
#include <string>
#include "stdafx.h"
#include "ImageRegistration.h"
#include "CUDAlib.h"

using namespace cv;
using namespace asrl;
using namespace std;

typedef struct {
    KeyPoint kp1;
    KeyPoint kp2;

    float size;
    float angle;

    vector<float> desc1;
    vector<float> desc2;
} t_SURF_Pair;

class SurfGPU {
public:
    SurfGPU();
    SurfGPU(const SurfGPU& orig);
    virtual ~SurfGPU();
    void testSurf(string file1, string file2);
    void testSurf(IplImage * img1, IplImage * img2, vector<t_SURF_Pair> &pairs);

private:
    void testSurfGPU(cv::Mat img, vector<KeyPoint> &points, vector<float> &descriptors, GpuSurfConfiguration config);
    bool isLastBitSet(const int * f);
    bool isLastBitSet(const float & f);
    void drawKeypoints(std::vector<cv::KeyPoint> const & keypoints, IplImage * imgGrayscale, string name);
    void drawPairs(vector<t_SURF_Pair> const & pairs, IplImage * imgGrayscale1, IplImage * imgGrayscale2);

    float distSquare(vector <float> v1, vector <float> v2);
    void matchSequential(vector<KeyPoint> points1, vector<KeyPoint> points2,  vector<float> desc1, vector<float> desc2, vector <t_SURF_Pair> &pairs);

    void bruteMatch(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<float> desc1, vector<float> desc2, vector<t_SURF_Pair> &pairs);

    void removeOutliers(CvMat **points1, CvMat **points2, CvMat *status);
    void cleanRANSAC(int method, vector<t_SURF_Pair> &pairs);

    void setMaskFromPoints(IplImage * &mask, vector<t_SURF_Pair> pairs, int index);
    void cleanByPosition(vector<t_SURF_Pair> pairs, CvSize size);

    CvPoint2D32f * getSquare(KeyPoint k);
    double calcCCorr(IplImage * img1, IplImage * img2, t_SURF_Pair pair, bool show);
    double getCorrelation(vector <float> data1, vector<float> data2);
    void cleanByCorrelation(vector<t_SURF_Pair> &pairs, IplImage * img1, IplImage * img2);

    void cleanDistances(IplImage * img1, IplImage * img2, vector<t_SURF_Pair> &pairs);

    GpuSurfConfiguration config;
    GpuSurfDetector detector;
};

#endif	/* _SURFGPU_H */

