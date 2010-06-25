/* 
 * File:   SurfGPU.cpp
 * Author: nestor
 * 
 * Created on 25 de junio de 2010, 12:39
 */

#include "SurfGPU.h"

SurfGPU::SurfGPU() {
}

SurfGPU::SurfGPU(const SurfGPU& orig) {
}

SurfGPU::~SurfGPU() {
}

void SurfGPU::testSurfGPU(cv::Mat img, vector<KeyPoint> &points) {
    GpuSurfDetector detector;

    detector.buildIntegralImage(img);
    detector.detectKeypoints();
    detector.findOrientationFast();
    detector.computeDescriptors();
	          
    detector.getKeypoints(points);
}

inline bool SurfGPU::isLastBitSet(const int * f) {
    return (*f & 0x1);
}

inline bool SurfGPU::isLastBitSet(const float & f) {
    return isLastBitSet((const int*)&f);
}

void SurfGPU::drawKeypoints(std::vector<cv::KeyPoint> const & keypoints, IplImage * imgGrayscale, string name) {

    IplImage * imgColor = cvCreateImage(cvGetSize(imgGrayscale), IPL_DEPTH_8U, 3);
    cvCvtColor(imgGrayscale, imgColor, CV_GRAY2BGR);

    vector<KeyPoint>::const_iterator k = keypoints.begin();
    KeyPoint k2;
    Scalar red(255, 0, 0);
    Scalar blue(0, 0, 255);

    for (; k != keypoints.end(); k++) {
        Scalar * choice = NULL;
        if (isLastBitSet(k->response))
            choice = &red;
        else
            choice = &blue;

        Point2f dir(k->pt);
        float st = k->size * sin(k->angle);
        float ct = k->size * cos(k->angle);
        dir.x += ct;
        dir.y += st;
        cvCircle(imgColor, k->pt, (int) k->size, *choice, 1, CV_AA);
        cvLine(imgColor, k->pt, dir, *choice, 1, CV_AA);
        //cvCircle(imgColor, k->pt, 2, cvScalar(0, 0, 255), -1);
    }

    cvShowImage(name.c_str(), imgColor);

    cvReleaseImage(&imgColor);
}

void SurfGPU::testSurf(string file1, string file2) {    
    IplImage * img1In = cvLoadImage(file1.c_str(), 0);
    IplImage * img2In = cvLoadImage(file2.c_str(), 0);

    IplImage * img1 = cvCreateImage(cvSize(800, 600), IPL_DEPTH_8U, 1);
    IplImage * img2 = cvCreateImage(cvSize(800, 600), IPL_DEPTH_8U, 1);

    cvResize(img1In, img1);
    cvResize(img2In, img2);

    cvReleaseImage(&img1In);
    cvReleaseImage(&img2In);

    vector<KeyPoint> points1;
    vector<KeyPoint> points2;

    for (int i = 0; i < 10; i++) {
        clock_t myTime = clock();
        testSurfGPU(img1, points1);
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo img1 = " << time << endl;
        myTime = clock();
        testSurfGPU(img2, points2);
        time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo img2 = " << time << endl;
    }

    drawKeypoints(points1, img1, "surfImg1");
    drawKeypoints(points2, img2, "surfImg2");

    cvWaitKey(0);

    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
}