/* 
 * File:   ImageRegistration.cpp
 * Author: neztol
 * 
 * Created on 3 de noviembre de 2009, 11:29
 */

#include "ImageRegistration.h"

CImageRegistration::CImageRegistration(CvSize size) {
    this->size = size;

    // Temporary images and matrix
    img32fc1_a = cvCreateImage(size, IPL_DEPTH_32F, 1);
    img32fc1_b = cvCreateImage(size, IPL_DEPTH_32F, 1);

    // Corner detection
    eigen = img32fc1_a;
    tmp = img32fc1_b;

    numberOfMeshedFeatures = ((size.width / MESH_DISTANCE) - 1) * ((size.height / MESH_DISTANCE) - 1);
    meshedFeatures = new CvPoint2D32f[numberOfMeshedFeatures];
    for (int index = 0, i = MESH_DISTANCE; i < size.width; i += MESH_DISTANCE) {
        for (int j = MESH_DISTANCE; j < size.height; j += MESH_DISTANCE, index++) {
            meshedFeatures[index] = cvPoint2D32f(i, j);
        }
    }

    // Optical flow
    CvSize pyramidImageSize = cvSize(size.width + 8, size.height / 3);
    pyramidImage1 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
    pyramidImage2 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);

    remapX = cvCreateMat(size.height, size.width, CV_32FC1);
    remapY = cvCreateMat(size.height, size.width, CV_32FC1);
    tps = cvCreateImage(size, IPL_DEPTH_8U, 1);

    oldNumberOfFeatures = 0;
    oldImgDB = cvCreateImage(size, IPL_DEPTH_8U, 1);
    oldImgRT = cvCreateImage(size, IPL_DEPTH_8U, 1);
    CvPoint2D32f * oldPointsDB;
    CvPoint2D32f * oldPointsRT;

}

CImageRegistration::~CImageRegistration() {
    /*cvReleaseImage(&img32fc1_a);
    cvReleaseImage(&img32fc1_b);
    cvReleaseImage(&pyramidImage1);
    cvReleaseImage(&pyramidImage2);
    cvReleaseImage(&tps);

    cvReleaseMat(&remapX);
    cvReleaseMat(&remapY);

    /*delete pointsBD;
    delete pointsRT;
    delete meshedFeatures;
    delete status0;
    delete status1;*/
}

// This is just for testing. In the final version, old features and images were obtained in previous iterations
void CImageRegistration::registration(IplImage * imgDB1, IplImage * imgDB2, IplImage * imgRT) {
    clock_t myTime = clock();
    findPairs(imgDB1, imgRT, pointsDB, pointsRT, numberOfFeatures, false, USE_REGIONS);
    if (numberOfFeatures == 0) {
        cerr << "No hay características suficientes" << endl;
        return;
    }
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en features = " << time << endl;

    /*if (oldNumberOfFeatures != 0) {
        CvPoint2D32f * points1, * points2;
        int nPoints;
        getPairsFromPreviousImg(imgDB1, imgRT, points1, points2, nPoints);
    }//*/

    showPairs("Clean", imgDB1, imgRT, pointsDB, pointsRT, numberOfFeatures);
    /*if (imgDB2 != NULL) {
        findPairs(imgDB2, imgRT, pointsDB, pointsRT, numberOfFeatures, true, USE_REGIONS);
        showPairs("DB2", imgDB2, imgRT, pointsDB, pointsRT, numberOfFeatures);
    }//*/
    
    myTime = clock();
    TPS(imgDB1, pointsRT, pointsDB, numberOfFeatures);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en transformacion = " << time << endl;

    cvNamedWindow("TPS", 1);
    cvShowImage("TPS", imgDB1);

    cvAbsDiff(imgDB1, imgRT, tps);
    cvNamedWindow("Diff", 1);
    cvShowImage("Diff", tps);

    /*oldNumberOfFeatures = numberOfFeatures;
    cvCopyImage(imgDB1, oldImgDB);
    cvCopyImage(imgRT, oldImgRT);    
    oldPointsDB = pointsDB;
    oldPointsRT = pointsRT;//*/

}

void CImageRegistration::getPairsOnBigImg(IplImage * imgDB, IplImage * imgRT, IplImage * imgDBC, IplImage * imgRTC) {
    clock_t myTime = clock();
    //findPairs(imgDB, imgRT, pointsDB, pointsRT, numberOfFeatures, false, USE_REGIONS);
    CvPoint2D32f * meshed = NULL, * corners1 = NULL;
    CvPoint2D32f * flow1 = NULL, * flow2 = NULL;
    int oFlowN;
    int nMeshed, nCorners1;

    findInitialPoints(imgDB, corners1, nCorners1);
    time_t timeInit = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en initial = " << timeInit << endl;
    showFeat("InitialP", imgDB, corners1, nCorners1);
    findOFlowPairs(imgDB, imgRT, corners1, nCorners1, flow1, flow2, oFlowN);
    //findPairsWithCorrelation(imgDB, imgRT, flow1, flow2, oFlowN);

    double kW = 320.0 / size.width;
    double kH = 240.0 / size.height;
    kW = kH = 1;
    pointsDB = new CvPoint2D32f[oFlowN];
    pointsRT = new CvPoint2D32f[oFlowN];
    for (int i = 0; i < oFlowN; i++) {
        //pointsDB[i] = cvPoint2D32f(flow1[i].x * kW, flow1[i].y * kH);
        //pointsRT[i] = cvPoint2D32f(flow2[i].x * kW, flow2[i].y * kH);
        pointsDB[i] = flow1[i];
        pointsRT[i] = flow2[i];
        //cout << "DB: " << pointsDB[i].x << ", " << pointsDB[i].y << endl;
    }

    numberOfFeatures = oFlowN;

    if (meshed != NULL) delete meshed;
    if (corners1 != NULL) delete corners1;
    if (flow1 != NULL) delete flow1;
    if (flow2 != NULL) delete flow2;

    if (numberOfFeatures == 0) {
        cerr << "No hay características suficientes" << endl;
        return;
    }


    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en features = " << time << endl;

    /*CvSize oldSize = size;
    size = cvSize(320, 240);
    IplImage * smallImgDB = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * smallImgRT = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvResize(imgDB, smallImgDB);
    cvResize(imgRT, smallImgRT);

    showPairs("myPairs", smallImgDB, smallImgRT, pointsDB, pointsRT, numberOfFeatures);//*/
    //showPairs("myPairs", imgDB, imgRT, pointsDB, pointsRT, numberOfFeatures);
    if (cvWaitKey(0) == 27)
        exit(0);

    /*cvReleaseImage(&smallImgDB);
    cvReleaseImage(&smallImgRT);

    size = oldSize;//*/
}