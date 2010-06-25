/* 
 * File:   CImageNavigation.cpp
 * Author: neztol
 * 
 * Created on 21 de junio de 2010, 16:39
 */
#include "CImageNavigation.h"
#include <dirent.h>

CImageNavigation::CImageNavigation(string route, string ext_in) {
    this->route = route;

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(route.c_str())) == NULL) {
        cout << "Error: La carpeta no existe" << endl;
        return;
    }

    size = cvSize(640, 480);
    
    while ((dirp = readdir(dp)) != NULL) {
        string fileName = string(dirp->d_name);
        if (fileName.length() < 3)
            continue;
        string ext = fileName.substr(fileName.length() - 3, fileName.length() - 1);        
        if (ext != ext_in)
            continue;

        /*if (fileNames.size() == 0) {
            string fullFileName = route + fileName;
            cout << fullFileName << endl;
            IplImage * tmpImg = cvLoadImage(fullFileName.c_str(), 1);
            size = cvGetSize(tmpImg);
            cvReleaseImage(&tmpImg);
        }//*/

        fileNames.push_back(fileName);
    }
    closedir(dp);

    sort( fileNames.begin(), fileNames.end() );
}

CImageNavigation::CImageNavigation(const CImageNavigation& orig) {
}

CImageNavigation::~CImageNavigation() {
}

void CImageNavigation::makePairsOFlow() {
    IplImage * tmpImg1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * tmpImg2 = cvCreateImage(size, IPL_DEPTH_8U, 1);

    vector<CvPoint2D32f> points1;
    vector<CvPoint2D32f> points2;

    cout << fileNames.size() << endl;

    for (int i = 0; i < fileNames.size(); i++) {
        string fileName1 = route + fileNames.at(i);
        cout << "F1 = " << fileName1 << endl;
        IplImage * img1 = cvLoadImage(fileName1.c_str(), 0);
        cvResize(img1, tmpImg1);

        testFast(tmpImg1, points1);
        for (int j = i + 1; j < fileNames.size(); j += 1) {
            string fileName2 = route + fileNames.at(j);
            cout <<  "F2 = " << fileName2 << endl;
            IplImage * img2 = cvLoadImage(fileName2.c_str(), 0);
            cvResize(img2, tmpImg2);

            clock_t myTime = clock();

            testFast(tmpImg2, points2);

            oFlow(points1, pairs, tmpImg1, tmpImg2);

            if (pairs.size() < 8) {
                continue;
            }
            cleanRANSAC(CV_RANSAC, pairs);

            time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
            cout << "Tiempo invertido = " << time << endl;

            cout << pairs.size() << endl;

            IplImage * surf1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
            IplImage * surf2 = cvCreateImage(size, IPL_DEPTH_8U, 3);            
            cvCvtColor(tmpImg1, surf1, CV_GRAY2BGR);
            cvCvtColor(tmpImg2, surf2, CV_GRAY2BGR);

            for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
                CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
                cvCircle(surf1, cvPointFrom32f(it->p1), 2, color, -1);
                cvCircle(surf2, cvPointFrom32f(it->p2), 2, color, -1);
            }//*/

            cvShowImage("surf1", surf1);
            cvShowImage("surf2", surf2);

            cvReleaseImage(&surf1);
            cvReleaseImage(&surf2);//*/

            if (cvWaitKey(0)== 27)
                exit(0);

            cvReleaseImage(&img2);
        }
        cvReleaseImage(&img1);
    }

    cvReleaseImage(&tmpImg1);
    cvReleaseImage(&tmpImg2);
}

inline void CImageNavigation::cleanRANSAC(int method, vector<t_Pair> &pairs) {
    int nPairs = pairs.size();
    CvMat * p1 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * p2 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat *statusM = cvCreateMat(1, nPairs, CV_8UC1);

    for (int i = 0; i < nPairs; i++) {
        cvSet2D(p1, 0, i, cvScalar(pairs.at(i).p1.x, pairs.at(i).p1.y));
        cvSet2D(p2, 0, i, cvScalar(pairs.at(i).p2.x, pairs.at(i).p2.y));
    }

    //cvFindFundamentalMat(p1, p2, F, method, 3., 0.70, statusM);
    cvFindFundamentalMat(p1, p2, F, method, 3., 0.99, statusM);

    removeOutliers(&p1, &p2, statusM);

    nPairs = p1->cols;
    pairs.clear();
    CvScalar pA, pB;
    for (int i = 0; i < nPairs; i++) {
        pA = cvGet2D(p1, 0, i);
        pB = cvGet2D(p2, 0, i);

        t_Pair pair;
        pair.p1 = cvPoint2D32f(pA.val[0], pA.val[1]);
        pair.p2 = cvPoint2D32f(pB.val[0], pB.val[1]);
        pairs.push_back(pair);
    }

    cvReleaseMat(&p1);
    cvReleaseMat(&p2);
    cvReleaseMat(&F);
    cvReleaseMat(&statusM);
}

inline void CImageNavigation::findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures) {
    CvSize pyramidImageSize = cvSize(size.width + 8, size.height / 3);
    IplImage * pyramidImage1 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
    IplImage * pyramidImage2 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);

    numberOfFeatures = nOrigFeat;

    CvPoint2D32f ** oFlowPoints1To2 = new CvPoint2D32f *[2];
    CvPoint2D32f ** oFlowPoints2To1 = new CvPoint2D32f *[2];
    oFlowPoints1To2[0] = new CvPoint2D32f[numberOfFeatures];
    oFlowPoints1To2[1] = new CvPoint2D32f[numberOfFeatures];
    oFlowPoints2To1[0] = new CvPoint2D32f[numberOfFeatures];
    oFlowPoints2To1[1] = new CvPoint2D32f[numberOfFeatures];

    char * status0 = new char[numberOfFeatures];
    char * status1 = new char[numberOfFeatures];

    for (int i = 0; i < numberOfFeatures; i++) {
        oFlowPoints1To2[0][i] = origPoints[i];
    }

    // Optical flow from image 1 to image 2
    cvCalcOpticalFlowPyrLK(img1, img2, pyramidImage1, pyramidImage2,
                            oFlowPoints1To2[0], oFlowPoints1To2[1], numberOfFeatures,
                            cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
                            PYRAMID_DEPTH, status0, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

    // Now features of the second image are the origin of the optical flow
    for (int i = 0; i < numberOfFeatures; i++) {
        oFlowPoints2To1[0][i] = oFlowPoints1To2[1][i];
    }

    // Optical flow from image 2 to image 1
    cvCalcOpticalFlowPyrLK(img2, img1, pyramidImage2, pyramidImage1,
        oFlowPoints2To1[0], oFlowPoints2To1[1], numberOfFeatures,
	cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
	PYRAMID_DEPTH, status1, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

    int maxFeat = numberOfFeatures;
    numberOfFeatures = 0;

    delete points1;
    points1 = new CvPoint2D32f[maxFeat];
    points2 = new CvPoint2D32f[maxFeat];

    // Detected features are compared to obtain the final dataset
    for (int i = 0; i < maxFeat; i++) {
        if (status0[i] && status1[i]) {
            double distance = hypot(oFlowPoints2To1[1][i].x - oFlowPoints1To2[0][i].x,
                                    oFlowPoints2To1[1][i].y - oFlowPoints1To2[0][i].y);
            if (distance < MAX_PIXEL_DISTANCE) {
                points1[numberOfFeatures] = oFlowPoints1To2[0][i];
                points2[numberOfFeatures] = oFlowPoints1To2[1][i];
                numberOfFeatures++;
            }
        }
    }

    delete oFlowPoints1To2[0];
    delete oFlowPoints1To2[1];
    delete [] oFlowPoints1To2;
    delete oFlowPoints2To1[0];
    delete oFlowPoints2To1[1];
    delete [] oFlowPoints2To1;
    delete status0;
    delete status1;
    cvReleaseImage(&pyramidImage1);
    cvReleaseImage(&pyramidImage2);
}

inline void CImageNavigation::oFlow(vector <CvPoint2D32f> &points1, vector <t_Pair> &pairs, IplImage * &img1, IplImage * &img2) {
    CvPoint2D32f * p1 = new CvPoint2D32f[points1.size()];
    for (int i = 0; i < points1.size(); i++) {
        p1[i] = points1.at(i);
    }
    CvPoint2D32f * p2;
    int nFeat = 0;
    findOFlowPairs(img1, img2, p1, points1.size(), p1, p2, nFeat);
    pairs.clear();
    for (int i = 0; i < nFeat; i++) {
        t_Pair pair;
        pair.p1 = p1[i];
        pair.p2 = p2[i];
        pairs.push_back(pair);
    }

    delete p1;
    delete p2;
}

inline void CImageNavigation::testFast(IplImage * img, vector<CvPoint2D32f> &points) {
    int inFASTThreshhold = 10; //30; //80
    int inNpixels = 9;
    int inNonMaxSuppression = 1;

    CvPoint* corners;
    int numCorners;

    cvCornerFast(img, inFASTThreshhold, inNpixels, inNonMaxSuppression, &numCorners, & corners);

    points.clear();
    for (int i = 0; i < numCorners; i++) {
        //if (cvGetReal2D(pointsMask, corners[i].y, corners[i].x) == 255)
            points.push_back(cvPointTo32f(corners[i]));
    }

    delete corners;
}
