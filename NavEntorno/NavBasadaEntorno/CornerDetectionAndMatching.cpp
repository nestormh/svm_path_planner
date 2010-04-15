#include "ImageRegistration.h"

void CImageRegistration::findInitialPoints(IplImage * img, CvPoint2D32f * &corners, int &nCorners, CvPoint2D32f * &meshed, int &nMeshed) {
    nCorners = MAX_FEATURES;
    CvPoint2D32f * tmpPoints = new CvPoint2D32f[MAX_FEATURES];

    // Corners are obtained
    cvGoodFeaturesToTrack(img, eigen, tmp, tmpPoints, &nCorners,
        DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);

    // Add corners to the final points detected
    meshed = new CvPoint2D32f[nCorners + numberOfMeshedFeatures];
    corners = new CvPoint2D32f[nCorners];
    for (int i = 0; i < nCorners; i++) {
        corners[i] = tmpPoints[i];
        meshed[i] = tmpPoints[i];
    }
    // Meshed points are added without additional information
    for (int i = nCorners, j = 0; i < nCorners + numberOfMeshedFeatures; i++, j++) {
        meshed[i] = meshedFeatures[j];
    }
    nMeshed = nCorners + numberOfMeshedFeatures;

    delete tmpPoints;
}

void CImageRegistration::findInitialPoints(IplImage * img, CvPoint2D32f * &corners, int &nCorners) {
    nCorners = MAX_FEATURES;
    CvPoint2D32f * tmpPoints = new CvPoint2D32f[MAX_FEATURES];

    // Corners are obtained
    cvGoodFeaturesToTrack(img, eigen, tmp, tmpPoints, &nCorners,
        DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);

    // Add corners to the final points detected
    corners = new CvPoint2D32f[nCorners];
    for (int i = 0; i < nCorners; i++) {
        corners[i] = tmpPoints[i];
    }

    delete tmpPoints;
}

void CImageRegistration::findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures) {
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
}

void CImageRegistration::findDistanceBasedPairs(const CvPoint2D32f * flow1, const CvPoint2D32f * flow2, int oFlowN,
                                                const CvPoint2D32f * origPoints1, int nOrig1, const CvPoint2D32f * origPoints2, int nOrig2,
                                                CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures) {

        // Get the outer points to calculate a projective transform
        double minX = DBL_MAX, maxX = DBL_MIN, minY = DBL_MAX, maxY = DBL_MIN;
        CvPoint2D32f src[4] = { cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1) };
        CvPoint2D32f dst[4] = { cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1) };

        for (int i = 0; i < oFlowN; i++) {
            if (flow1[i].x < minX) {
                minX = flow1[i].x;
                src[0] = flow1[i];
                dst[0] = flow2[i];
            }
        }
        for (int i = 0; i < oFlowN; i++) {
            if (flow1[i].x > maxX) {
                bool isUsed = false;
                for (int j = 0; j < 1; j++) {
                    if ((src[j].x == flow1[i].x) && (src[j].y == flow1[i].y)) {
                        isUsed = true;
                        break;
                    }
                }
                if (isUsed == false) {
                    maxX = flow1[i].x;
                    src[1] = flow1[i];
                    dst[1] = flow2[i];
                }
            }
        }
        for (int i = 0; i < oFlowN; i++) {
            if (flow1[i].y < minY) {
                bool isUsed = false;
                for (int j = 0; j < 2; j++) {
                    if ((src[j].x == flow1[i].x) && (src[j].y == flow1[i].y)) {
                        isUsed = true;
                        break;
                    }
                }
                if (isUsed == false) {
                    minY = flow1[i].y;
                    src[2] = flow1[i];
                    dst[2] = flow2[i];
                }
            }
        }
        for (int i = 0; i < oFlowN; i++) {
            if (flow1[i].y > maxY) {
                bool isUsed = false;
                for (int j = 0; j < 3; j++) {
                    if ((src[j].x == flow1[i].x) && (src[j].y == flow1[i].y)) {
                        isUsed = true;
                        break;
                    }
                }
                if (isUsed == false) {
                    maxY = flow1[i].y;
                    src[3] = flow1[i];
                    dst[3] = flow2[i];
                }
            }
        }

        CvMat * X = cvCreateMat(3, 3, CV_64FC1);
        cvGetPerspectiveTransform(src, dst, X);

        double * pts = new double[nOrig1 * 2];
        for (int i = 0; i < nOrig1; i++) {
            pts[i * 2] = origPoints1[i].x;
            pts[i * 2 + 1] = origPoints1[i].y;
        }
        CvMat mPts = cvMat(1, nOrig1, CV_64FC2, pts);
	cvPerspectiveTransform(&mPts, &mPts, X);        

        CvPoint2D32f * tmpPoints = new CvPoint2D32f[nOrig1];

        for (int i = 0; i < nOrig1; i++) {
            tmpPoints[i].x = pts[i * 2];
            tmpPoints[i].y = pts[i * 2 + 1];
        }

        numberOfFeatures = 0;
        points1 = new CvPoint2D32f[min(nOrig1, nOrig2)];
        points2 = new CvPoint2D32f[min(nOrig1, nOrig2)];

        if (nOrig1 < nOrig2) {
            for (int i = 0; i < nOrig1; i++) {
                double minDist = MAX_DIST_CLOUD;
                int minIndex = -1;
                for (int j = 0; j < nOrig2; j++) {
                    double dist = hypot(tmpPoints[i].x - origPoints2[j].x, tmpPoints[i].y - origPoints2[j].y);
                    if (dist < minDist) {
                        minDist = dist;
                        minIndex = j;
                    }
                }
                if (minIndex != -1) {
                    points1[numberOfFeatures] = tmpPoints[i];
                    points2[numberOfFeatures] = origPoints2[minIndex];
                    numberOfFeatures++;
                }
            }
        } else {
            for (int i = 0; i < nOrig2; i++) {
                double minDist = MAX_DIST_CLOUD;
                int minIndex = -1;
                for (int j = 0; j < nOrig1; j++) {
                    double dist = hypot(tmpPoints[j].x - origPoints2[i].x, tmpPoints[j].y - origPoints2[i].y);
                    if (dist < minDist) {
                        minDist = dist;
                        minIndex = j;
                    }
                }
                if (minIndex != -1) {
                    points1[numberOfFeatures] = tmpPoints[minIndex];
                    points2[numberOfFeatures] = origPoints2[i];
                    numberOfFeatures++;
                }
            }
        }

        cvReleaseMat(&X);
        delete pts;
        delete tmpPoints;
}

void CImageRegistration::cleanFeat(CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat) {
    CvPoint2D32f * oldPoints1 = points1;
    CvPoint2D32f * oldPoints2 = points2;
    int oldNFeat = nFeat;

    points1 = new CvPoint2D32f[nFeat];
    points2 = new CvPoint2D32f[nFeat];
    nFeat = 0;
    
    for (int i = 0; i < oldNFeat; i++) {
        bool hasNeighbors = false;        
        for (int j = 0; j < oldNFeat; j++) {
            if (i != j) {
                double dist1 = hypot(oldPoints1[i].x - oldPoints1[j].x, oldPoints1[i].y - oldPoints1[j].y);
                double dist2 = hypot(oldPoints2[i].x - oldPoints2[j].x, oldPoints2[i].y - oldPoints2[j].y);

                if ((dist1 < CLEAN_THRESH) && (dist2 < CLEAN_THRESH)) {
                    hasNeighbors = true;
                    break;
                }
            }
        }
        if (hasNeighbors) {
            points1[nFeat] = oldPoints1[i];
            points2[nFeat] = oldPoints2[i];
            nFeat++;
        }
    }   

    delete oldPoints1;
    delete oldPoints2;
}

void CImageRegistration::showPairs(char * name, IplImage * img1, IplImage * img2, CvPoint2D32f * points1, CvPoint2D32f * points2, int numberOfFeatures) {    
    IplImage * imgA = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgB = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgC = cvCreateImage(cvSize(size.width * 2, size.height), IPL_DEPTH_8U, 3);
    
    cvCvtColor(img1, imgA, CV_GRAY2BGR);
    cvCvtColor(img2, imgB, CV_GRAY2BGR);

    cvZero(imgC);
    cvSetImageROI(imgC, cvRect(0, 0, size.width, size.height));
    cvAdd(imgA, imgC, imgC);
    cvCvtColor(img2, imgB, CV_GRAY2BGR);
    cvSetImageROI(imgC, cvRect(size.width, 0, size.width, size.height));
    cvAdd(imgB, imgC, imgC);
    cvResetImageROI(imgC);
    for (int i = 0; i < numberOfFeatures; i++) {
        CvScalar color = cvScalar(rand() % 255, rand() % 255, rand() % 255);
        cvCircle(imgC, cvPointFrom32f(points1[i]), 2, color, -1);
        cvCircle(imgC, cvPoint((int) points2[i].x + size.width, (int) points2[i].y), 2, color, -1);
    }

    cvNamedWindow(name, 1);
    cvShowImage(name, imgC);

    cvReleaseImage(&imgA);
    cvReleaseImage(&imgB);
    cvReleaseImage(&imgC);
}

void CImageRegistration::cleanWithPCACutreishon(IplImage * img1, IplImage * img2, CvPoint2D32f * points1, CvPoint2D32f * points2, int numberOfFeatures) {
    IplImage * localMask = cvCreateImage(cvSize(265, 265), IPL_DEPTH_8U, 1);
    cvZero(localMask);

    IplImage * dilated1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * dilated2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    //cvDilate(img1, dilated1, 0, 2);
    //cvDilate(img2, dilated2, 0, 2);
    cvSmooth(img1, dilated1);
    cvSmooth(img2, dilated2);

    // Calculamos la recta de regresión y la pintamos
    double meanX = 0;
    double meanY = 0;
    for (int i = 0; i < numberOfFeatures; i++) {
        meanX += (int)cvGetReal2D(dilated1, points1[i].y, points1[i].x);
        meanY += (int)cvGetReal2D(dilated2, points2[i].y, points2[i].x);
    }

    double partX = 0;
    double partY = 0;
    for (int i = 0; i < numberOfFeatures; i++) {
        partX += (int)cvGetReal2D(dilated1, points1[i].y, points1[i].x) - meanX;
        partY += (int)cvGetReal2D(dilated2, points2[i].y, points2[i].x) - meanY;
    }

    double a = (partX * partY) / (partX * partX);
    double b = meanY - a * meanX;

    CvPoint p1 = cvPoint(0, b);
    CvPoint p2 = cvPoint(255, a * 255 + b);
    cvLine(localMask, p1, p2, cvScalar(255, 0, 0), 100);

    IplImage * finalPoints = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvCvtColor(img1, finalPoints, CV_GRAY2BGR);
    for (int i = 0; i < numberOfFeatures; i++) {
        cvCircle(finalPoints, cvPointFrom32f(points1[i]), 2, cvScalar(0, 255, 0), -1);
    }
    for (int i = 0; i < numberOfFeatures; i++) {
        int posX = (int)cvGetReal2D(dilated1, points1[i].y, points1[i].x);
        int posY = (int)cvGetReal2D(dilated2, points2[i].y, points2[i].x);

        if (cvGetReal2D(localMask, posY, posX) != 255)
            cvCircle(finalPoints, cvPointFrom32f(points1[i]), 2, cvScalar(0, 0, 255), -1);
    }

    cvNamedWindow("Erased", 1);
    cvShowImage("Erased", finalPoints);


    cvReleaseImage(&localMask);
    cvReleaseImage(&finalPoints);
}

void CImageRegistration::showPairsRelationship(char * name, IplImage * img1, IplImage * img2, CvPoint2D32f * points1, CvPoint2D32f * points2, int numberOfFeatures) {
    IplImage * chart = cvCreateImage(cvSize(265, 265), IPL_DEPTH_8U, 3);
    cvZero(chart);
    
    IplImage * dilated1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * dilated2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    //cvDilate(img1, dilated1, 0, 2);
    //cvDilate(img2, dilated2, 0, 2);
    cvSmooth(img1, dilated1);
    cvSmooth(img2, dilated2);

    // Calculamos la recta de regresión y la pintamos
    double meanX = 0;
    double meanY = 0;
    for (int i = 0; i < numberOfFeatures; i++) {
        meanX += (int)cvGetReal2D(dilated1, points1[i].y, points1[i].x);
        meanY += (int)cvGetReal2D(dilated2, points2[i].y, points2[i].x);
    }

    double partX = 0;
    double partY = 0;
    for (int i = 0; i < numberOfFeatures; i++) {
        partX += (int)cvGetReal2D(dilated1, points1[i].y, points1[i].x) - meanX;
        partY += (int)cvGetReal2D(dilated2, points2[i].y, points2[i].x) - meanY;
    }

    double a = (partX * partY) / (partX * partX);
    double b = meanY - a * meanX;

    CvPoint p1 = cvPoint(0, b);
    CvPoint p2 = cvPoint(255, a * 255 + b);
    cvLine(chart, p1, p2, cvScalar(255, 0, 0), 100);
    cvLine(chart, p1, p2, cvScalar(0, 255, 0));

    IplImage * imgA = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgB = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgC = cvCreateImage(cvSize(size.width * 2, size.height), IPL_DEPTH_8U, 3);

    cvCvtColor(dilated1, imgA, CV_GRAY2BGR);
    cvCvtColor(dilated2, imgB, CV_GRAY2BGR);

    cvZero(imgC);
    cvSetImageROI(imgC, cvRect(0, 0, size.width, size.height));
    cvAdd(imgA, imgC, imgC);
    cvCvtColor(dilated2, imgB, CV_GRAY2BGR);
    cvSetImageROI(imgC, cvRect(size.width, 0, size.width, size.height));
    cvAdd(imgB, imgC, imgC);
    cvResetImageROI(imgC);
    for (int i = 0; i < numberOfFeatures; i++) {
        CvScalar color = cvScalar(rand() % 255, rand() % 255, rand() % 255);
        cvCircle(imgC, cvPointFrom32f(points1[i]), 2, color, -1);
        cvCircle(imgC, cvPoint((int) points2[i].x + size.width, (int) points2[i].y), 2, color, -1);
        int posX = (int)cvGetReal2D(dilated1, points1[i].y, points1[i].x);
        int posY = (int)cvGetReal2D(dilated2, points2[i].y, points2[i].x);

        cvCircle(chart, cvPoint(posX, posY), 2, color, -1);
    }   

    cvNamedWindow(name, 1);
    cvShowImage(name, imgC);

    cvReleaseImage(&imgA);
    cvReleaseImage(&imgB);
    cvReleaseImage(&imgC);

    cvNamedWindow("Chart", 1);
    cvShowImage("Chart", chart);

    cvReleaseImage(&chart);
    cvReleaseImage(&dilated1);
    cvReleaseImage(&dilated2);
}

void CImageRegistration::showFeat(char * name, IplImage * img, CvPoint2D32f * points, int numberOfFeatures) {
    IplImage * imgC = cvCreateImage(size, IPL_DEPTH_8U, 3);
    
    cvCvtColor(img, imgC, CV_GRAY2BGR);
    
    for (int i = 0; i < numberOfFeatures; i++) {
        CvScalar color = cvScalar(255, 0, 0);
        cvCircle(imgC, cvPointFrom32f(points[i]), 2, color, -1);
    }

    cvNamedWindow(name, 1);
    cvShowImage(name, imgC);

    cvReleaseImage(&imgC);
}

void CImageRegistration::findPairs(IplImage * img1, IplImage * img2, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat, bool useAffine, bool useRegions, CvPoint2D32f * initialPoints, int nInitialP) {
    if (useAffine) {
        CvPoint2D32f * meshed, * corners1, * corners2;
        int nMeshed, nCorners1, nCorners2;
        this->findInitialPoints(img1, corners1, nCorners1, meshed, nMeshed);
        this->findInitialPoints(img2, corners2, nCorners2);

        CvPoint2D32f * flow1, * flow2;
        int oFlowN;
        this->findOFlowPairs(img1, img2, meshed, nMeshed, flow1, flow2, oFlowN);

        CvPoint2D32f * affine1 = new CvPoint2D32f[MAX_FEATURES];
        CvPoint2D32f * affine2 = new CvPoint2D32f[MAX_FEATURES];
        int affineN = MAX_FEATURES;

        this->findDistanceBasedPairs(flow1, flow2, oFlowN, corners1, nCorners1, corners2, nCorners2, affine1, affine2, affineN);

        points1 = new CvPoint2D32f[oFlowN + affineN];
        points2 = new CvPoint2D32f[oFlowN + affineN];

        for (int i = 0; i < oFlowN; i++) {
            points1[i] = flow1[i];
            points2[i] = flow2[i];
        }
        nFeat = oFlowN;
        for (int i = 0; i < affineN - 1; i++) {
            bool used = false;
            for (int j = 0; j < oFlowN; j++) {
                if (((int)points1[j].x == (int)affine1[i].x) && ((int)points1[j].y == (int)affine1[i].y)) {
                    used = true;
                    break;
                }
            }
            if (! used) {
                points1[nFeat] = affine1[i];
                points2[nFeat] = affine2[i];
                nFeat++;                
            }
        }
        delete meshed;
        delete corners1;
        delete corners2;
        delete flow1;
        delete flow2;
        delete affine1;
        delete affine2;
    } else {
        CvPoint2D32f * meshed = NULL, * corners1 = NULL;
        int nMeshed, nCorners1;
        if (nInitialP == 0) {
            cout << "nInitialP == 0" << endl;
            this->findInitialPoints(img1, corners1, nCorners1, meshed, nMeshed);            
        } else {
            cout << "nInitialP != 0" << endl;
            cout << nInitialP << endl;
            meshed = new CvPoint2D32f[nInitialP];
            for (int i = 0; i < nInitialP; i++) {
                cout << i << ": " << initialPoints[i].x << ", " << initialPoints[i].y << endl;
                meshed[i] = initialPoints[i];
            }
            meshed = initialPoints;
            nMeshed = nInitialP;            
        }        

        CvPoint2D32f * flow1 = NULL, * flow2 = NULL;
        int oFlowN;
        this->findOFlowPairs(img1, img2, meshed, nMeshed, flow1, flow2, oFlowN);
        cout << "Salio, " << oFlowN << endl;

        points1 = new CvPoint2D32f[oFlowN];
        points2 = new CvPoint2D32f[oFlowN];        
        for (int i = 0; i < oFlowN; i++) {
            points1[i] = flow1[i];
            points2[i] = flow2[i];
        }
        
        nFeat = oFlowN;

        //showPairs("Post", img1, img2, points1, points2, nFeat);        

        if (meshed != NULL) delete meshed;
        if (corners1 != NULL) delete corners1;
        if (flow1 != NULL) delete flow1;
        if (flow2 != NULL) delete flow2;
        
    }

    if (! useRegions) {
        cleanFeat(points1, points2, nFeat);
        //showPairsRelationship("PairsChart", img1, img2, points1, points2, nFeat);
        //cleanWithPCACutreishon(img1, img2, points1, points2, nFeat);
    } else {
        showPairs("Pre", img1, img2, points1, points2, nFeat);
        t_moment * moments1, * moments2;
        int nMoments1, nMoments2;
        mesrTest(img1, "mser1", moments1, nMoments1);
        mesrTest(img2, "mser2", moments2, nMoments2);
        vector<t_moment *> regionPairs;
        matchMserByMoments(img1, img2, moments1, moments2, nMoments1, nMoments2, "Match", regionPairs);
        CvPoint2D32f * regionPoints1, * regionPoints2;
        int nRegionPoints;
        cleanMatches(img1, img2, regionPairs, "CleanRegions", regionPoints1, regionPoints2, nRegionPoints);

        CvPoint2D32f * oldPoints1 = points1;
        CvPoint2D32f * oldPoints2 = points2;
        points1 = new CvPoint2D32f[nFeat + nRegionPoints];
        points2 = new CvPoint2D32f[nFeat + nRegionPoints];
        for (int i = 0; i < nFeat; i++) {
            points1[i] = oldPoints1[i];
            points2[i] = oldPoints2[i];
        }
        for (int i = 0; i < nRegionPoints; i++) {
            points1[i + nFeat] = regionPoints1[i];
            points2[i + nFeat] = regionPoints2[i];
        }
        nFeat += nRegionPoints;

        //cleanUsingSplines(points1, points2, nFeat);

        CvMat * p1 = cvCreateMat(1, nFeat, CV_32FC2);
        CvMat * p2 = cvCreateMat(1, nFeat, CV_32FC2);

        for (int i = 0; i < nFeat; i++) {
            cvSet2D(p1, 0, i, cvScalar(points1[i].x, points1[i].y));
            cvSet2D(p2, 0, i, cvScalar(points2[i].x, points2[i].y));
        }

        CvMat * F = cvCreateMat(3, 3, CV_32FC1);
        CvMat *status = cvCreateMat(1, p1->cols, CV_8UC1);
        int fm_count = cvFindFundamentalMat(p1, p2, F, CV_FM_RANSAC, 1., 0.99, status);
        removeOutliers(&p1, &p2, status);
        //showPairs2("post", img1, img2, p1, p2);

        nFeat = p1->cols;
        points1 = new CvPoint2D32f[nFeat];
        points2 = new CvPoint2D32f[nFeat];

        CvScalar pA, pB;
        for (int i = 0; i < nFeat; i++) {
            pA = cvGet2D(p1, 0, i);
            pB = cvGet2D(p2, 0, i);
            points1[i] = cvPoint2D32f(pA.val[0], pA.val[1]);
            points2[i] = cvPoint2D32f(pB.val[0], pB.val[1]);
        }

        cleanFeat(points1, points2, nFeat);

        //cleanUsingSplines(points1, points2, nFeat);

        showPairsRelationship("PairsChart", img1, img2, points1, points2, nFeat);
        cleanWithPCACutreishon(img1, img2, points1, points2, nFeat);

        //cleanFeat(points1, points2, nFeat);
        //showPairs("Post", img1, img2, points1, points2, nFeat);
    }
}

void CImageRegistration::cleanUsingSplines(CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat) {
    IplImage * mesh = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * mesh1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * mesh2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvZero(mesh);
    
    for (int i = 0; i < (size.width / 10 + 1); i++) {
        cvLine(mesh, cvPoint(i * 10, 0), cvPoint(i * 10, size.height - 1), cvScalar(255, 0, 0), 1);
    }
    for (int j = 0; j < (size.height / 10 + 1); j++) {
        cvLine(mesh, cvPoint(0, j * 10), cvPoint(size.width - 1, j * 10), cvScalar(255, 0, 0), 1);
    }

    cvCvtColor(mesh, mesh1, CV_GRAY2BGR);
    TPS(mesh, points2, points1, nFeat);
    cvCvtColor(mesh, mesh2, CV_GRAY2BGR);

    for (int i = 0; i < nFeat; i++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        cvCircle(mesh1, cvPointFrom32f(points1[i]), 2, color, -1);
        cvCircle(mesh2, cvPointFrom32f(points2[i]), 2, color, -1);
    }

    cvNamedWindow("Mesh1", 1);
    cvShowImage("Mesh1", mesh1);


    cvNamedWindow("Mesh2", 1);
    cvShowImage("Mesh2", mesh2);

    cvReleaseImage(&mesh);
    cvReleaseImage(&mesh1);
    cvReleaseImage(&mesh2);    

    /*CvPoint2D32f * oldPoints1 = points1;
    CvPoint2D32f * oldPoints2 = points2;
    points1 = new CvPoint2D32f[nFeat];
    points2 = new CvPoint2D32f[nFeat];
    int oldNFeat = nFeat;
    nFeat = 0;

    for (int i = 0; i < oldNFeat; i++) {
        double minDist1 = DBL_MAX;
        double minDist2 = DBL_MAX;

        int minIndex1 = -1;
        int minIndex2 = -1;
        
        for (int j = 0; j < oldNFeat; j++) {
            if (i == j) continue;

            double dist1 = hypot(oldPoints1[i].x - oldPoints1[j].x, oldPoints1[i].y - oldPoints1[j].y);
            double dist2 = hypot(oldPoints2[i].x - oldPoints2[j].x, oldPoints2[i].y - oldPoints2[j].y);
            if (dist1 < minDist1) {
                minDist1 = dist1;
                minIndex1 = j;
            }
            if (dist2 < minDist2) {
                minDist2 = dist2;
                minIndex2 = j;
            }
        }

        if (minIndex1 == minIndex2) {
            points1[nFeat] = oldPoints1[i];
            points2[nFeat] = oldPoints2[i];
            nFeat++;
        }
    }*/
}

void CImageRegistration::cleanUsingDelaunay(IplImage * img1, IplImage * img2, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int numberOfFeatures) {
    clock_t myTime = clock();
    // Calculamos los triangulos

        // Inicializamos las estructuras
    CvMat * refs1 = cvCreateMat(size.height, size.width, CV_32SC1);
    CvMat * refs2 = cvCreateMat(size.height, size.width, CV_32SC1);
    cvSet(refs1, cvScalar(-1));
    cvSet(refs2, cvScalar(-1));

    int numTriangles[numberOfFeatures];

    CvRect rect1 = cvRect(-1, -1, size.width + 2, size.height + 2);
    CvRect rect2 = cvRect(-1, -1, size.width + 2, size.height + 2);

    CvMemStorage * storage1 = cvCreateMemStorage(0);
    CvMemStorage * storage2 = cvCreateMemStorage(0);

    CvSubdiv2D * subdiv1 = cvCreateSubdivDelaunay2D(rect1, storage1);
    CvSubdiv2D * subdiv2 = cvCreateSubdivDelaunay2D(rect2, storage2);

    for (int i = 0; i < numberOfFeatures; i++) {
        if ((points1[i].x < 0) || (points1[i].y < 0) || (points2[i].x < 0) || (points2[i].y < 0) ||
            (points1[i].x >= size.width) || (points1[i].y >= size.height) ||
            (points2[i].x >= size.width) || (points2[i].y >= size.height)) continue;

        cvSubdivDelaunay2DInsert(subdiv1, points1[i]);
        cvSubdivDelaunay2DInsert(subdiv2, points2[i]);

        CvPoint p1 = cvPointFrom32f(points1[i]);
        CvPoint p2 = cvPointFrom32f(points2[i]);
        cvSetReal2D(refs1, p1.y, p1.x, i);
        cvSetReal2D(refs2, p2.y, p2.x, i);

        numTriangles[i] = 0;
    }
    cvCalcSubdivVoronoi2D(subdiv1);
    cvCalcSubdivVoronoi2D(subdiv2);

    vector <t_triangle> triangles1;
    vector <t_triangle> triangles2;

    vector<t_triangle> trianglesRef1[size.height][size.width];
    vector<t_triangle> trianglesRef2[size.height][size.width];

    // Recorremos los triángulos para el conjunto 1 de puntos
    CvSeqReader  reader;
    int total = subdiv1->edges->total;
    int elem_size = subdiv1->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv1->edges), &reader, 0 );

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            t_triangle tri;
            CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_NEXT_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles1.size(); j++) {
                    if (areSameTriangles(tri, triangles1.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles1.push_back(tri);
                    trianglesRef1[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef1[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef1[tri.p3.y][tri.p3.x].push_back(tri);

                    (numTriangles[(int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x)])++;
                }
            }

            edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles1.size(); j++) {
                    if (areSameTriangles(tri, triangles1.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles1.push_back(tri);
                    trianglesRef1[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef1[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef1[tri.p3.y][tri.p3.x].push_back(tri);

                    (numTriangles[(int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x)])++;
                }
            }

        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    // Recorremos los triángulos para el conjunto 2 de puntos
    total = subdiv2->edges->total;
    elem_size = subdiv2->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv2->edges), &reader, 0 );

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            t_triangle tri;
            CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_NEXT_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs2, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs2, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs2, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles2.size(); j++) {
                    if (areSameTriangles(tri, triangles2.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles2.push_back(tri);
                    trianglesRef2[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef2[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef2[tri.p3.y][tri.p3.x].push_back(tri);
                }
            }

            edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs2, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs2, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs2, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles2.size(); j++) {
                    if (areSameTriangles(tri, triangles2.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles2.push_back(tri);
                    trianglesRef2[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef2[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef2[tri.p3.y][tri.p3.x].push_back(tri);
                }
            }

        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    /*vector <triangle> removedTriangles;

    /*for (int i = 0; i < triangles1.size(); i++) {
        triangle tri1 = triangles1.at(i);
        int index = (int)cvGetReal2D(refs1, tri1.p1.y, tri1.p1.x);
        CvPoint p2 = cvPointFrom32f(points2[index]);
        vector <triangle> triangles2Check = trianglesRef2[p2.y][p2.x];
        bool exists = false;
        for (int j = 0; j < triangles2Check.size(); j++) {
            if (areSameTriangles(tri1, triangles2.at(j)) == true) {
                exists = true;
                break;
            }
        }
        if (exists == false) {
            removedTriangles.push_back(tri1);

            (numTriangles[(int)cvGetReal2D(refs1, tri1.p1.y, tri1.p1.x)])--;
            (numTriangles[(int)cvGetReal2D(refs1, tri1.p2.y, tri1.p2.x)])--;
            (numTriangles[(int)cvGetReal2D(refs1, tri1.p3.y, tri1.p3.x)])--;
        }
    }*/
    vector <int> accepted;

    for (int i = 0; i < triangles1.size(); i++) {
        t_triangle tri1 = triangles1.at(i);
        for (int j = 0; j < triangles2.size(); j++) {
            if ((areSameTrianglesIndexed(tri1, triangles2.at(j)) == true) &&
                (areTrianglesEquipositioned(tri1, triangles2.at(j)) == true)) {
                //hasCorresp = true;
                bool add1 = true, add2 = true, add3 = true;
                /*for (int k = 0; k < accepted.size(); i++) {
                    if (accepted.at(k) == tri1.index1) add1 = false;
                    if (accepted.at(k) == tri1.index2) add2 = false;
                    if (accepted.at(k) == tri1.index3) add3 = false;

                    if ((add1 || add2 || add3) == false)
                        break;
                }*/
                //cout << "Accepted " << endl;
                if (add1) accepted.push_back(tri1.index1);
                if (add2) accepted.push_back(tri1.index2);
                if (add3) accepted.push_back(tri1.index3);
                break;
            }
        }
    }

    CvPoint2D32f * tmp1 = points1;
    CvPoint2D32f * tmp2 = points2;
    int oldFeatures = numberOfFeatures;

    points1 = new CvPoint2D32f[accepted.size()];
    points2 = new CvPoint2D32f[accepted.size()];

    for (int i = 0; i < accepted.size(); i++) {
        points1[i] = tmp1[accepted.at(i)];
        points2[i] = tmp2[accepted.at(i)];
    }
    numberOfFeatures = accepted.size();

    /*for (int i = 0; i < numberOfFeatures; i++) {
        /*CvPoint p1 = cvPointFrom32f(points1[i]);
        CvPoint p2 = cvPointFrom32f(points2[i]);

        vector <triangle> listTriangles1 = trianglesRef1[p1.y][p1.x];
        vector <triangle> listTriangles2 = trianglesRef2[p2.y][p2.x];

        cout << "Triangulos1: ";
        for (int i = 0; i < listTriangles1.size(); i++) {
            triangle triCoord = listTriangles1.at(i);
            triangle tri;
            tri.p1 = cvPoint((int)cvGetReal2D)
            cout << "[ " << tri.p1.x << ", " << tri.p1.y << " ;; ";
            cout << tri.p2.x << ", " << tri.p2.y << " ;; ";
            cout << tri.p3.x << ", " << tri.p3.y << " ] ";
        }
        cout << endl << "Triangulos2";
        for (int i = 0; i < listTriangles2.size(); i++) {
            triangle tri = listTriangles2.at(i);
            cout << "[ " << tri.p1.x << ", " << tri.p1.y << " ;; ";
            cout << tri.p2.x << ", " << tri.p2.y << " ;; ";
            cout << tri.p3.x << ", " << tri.p3.y << " ] ";
        }
        cout << endl;

        */

        /*bool accepted = false;
        for (int j = 0; j < listTriangles1.size(); j++) {
            for (int k = 0; k < listTriangles2.size(); k++) {
                if (areSameTriangles(listTriangles1.at(j), listTriangles2.at(k)) == true) {
                    accepted = true;
                    break;
                }
            }
            if (accepted == true)
                break;
        }
        if (accepted == true) {
            accepted1.push_back(p1);
            accepted2.push_back(p2);
        } else {
            removedPoints.push_back(p1);
        }* /

    }*/

    // Mostramos los resultados, para comprobar que todo fue bien
    IplImage * delaunay1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * delaunay2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * removed = cvCreateImage(size, IPL_DEPTH_8U, 3);

    cvCvtColor(img1, delaunay1, CV_GRAY2BGR);
    cvCvtColor(img2, delaunay2, CV_GRAY2BGR);
    cvCvtColor(img2, removed, CV_GRAY2BGR);

    //cvZero(delaunay1);
    //cvZero(delaunay2);

    for (int i = 0; i < triangles1.size(); i++) {
        t_triangle tri = triangles1.at(i);

        CvPoint poly[] = { tri.p1, tri.p2, tri.p3 };
        cvFillConvexPoly(delaunay1, poly, 3, CV_RGB(rand()&255,rand()&255,rand()&255));
        cvCircle(removed, cvPointFrom32f(points1[tri.index1]), 2, cvScalar(255, 0, 0));
        cvCircle(removed, cvPointFrom32f(points1[tri.index2]), 2, cvScalar(255, 0, 0));
        cvCircle(removed, cvPointFrom32f(points1[tri.index3]), 2, cvScalar(255, 0, 0));
        //cvFillConvexPoly(delaunay1, poly, 3, CV_RGB(255, 255, 255));
        //cvFillConvexPoly(removed, poly, 3, CV_RGB(255, 255, 255));
    }

    for (int i = 0; i < triangles2.size(); i++) {
        t_triangle tri = triangles2.at(i);

        CvPoint poly[] = { tri.p1, tri.p2, tri.p3 };
        cvFillConvexPoly(delaunay2, poly, 3, CV_RGB(rand()&255,rand()&255,rand()&255));
        //cvCircle(delaunay2, tri.p1, 2, cvScalar(255, 0, 0));
        //cvFillConvexPoly(delaunay2, poly, 3, CV_RGB(255, 255, 255));
    }

    for (int i = 0; i < oldFeatures; i++) {
        cvCircle(removed, cvPointFrom32f(tmp1[i]), 2, cvScalar(0, 0, 255));
    }
    for (int i = 0; i < numberOfFeatures; i++) {
        cvCircle(removed, cvPointFrom32f(points1[i]), 2, cvScalar(255, 0, 0));
    }

    /*for (int i = 0; i < removedTriangles.size(); i++) {
        triangle tri = removedTriangles.at(i);

        CvPoint poly[] = { tri.p1, tri.p2, tri.p3 };
        cvFillConvexPoly(removed, poly, 3, CV_RGB(255, 0, 0));
    }*/

    cvNamedWindow("Delaunay1");
    cvShowImage("Delaunay1", delaunay1);
    cvNamedWindow("Delaunay2");
    cvShowImage("Delaunay2", delaunay2);
    cvNamedWindow("Removed");
    cvShowImage("Removed", removed);

    cvReleaseImage(&delaunay1);
    cvReleaseImage(&delaunay2);
    cvReleaseImage(&removed);

    cvReleaseMat(&refs1);
    cvReleaseMat(&refs2);

    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo cleanUsingDelaunay = " << time << endl;

}

bool CImageRegistration::areSameTriangles(t_triangle tri1, t_triangle tri2) {
    if ((tri1.p1.x == tri2.p1.x) && (tri1.p1.y == tri2.p1.y) &&
        (tri1.p2.x == tri2.p2.x) && (tri1.p2.y == tri2.p2.y) &&
        (tri1.p3.x == tri2.p3.x) && (tri1.p3.y == tri2.p3.y))
        return true;
    if ((tri1.p1.x == tri2.p2.x) && (tri1.p1.y == tri2.p2.y) &&
        (tri1.p2.x == tri2.p3.x) && (tri1.p2.y == tri2.p3.y) &&
        (tri1.p3.x == tri2.p1.x) && (tri1.p3.y == tri2.p1.y))
        return true;
    if ((tri1.p1.x == tri2.p3.x) && (tri1.p1.y == tri2.p3.y) &&
        (tri1.p2.x == tri2.p1.x) && (tri1.p2.y == tri2.p1.y) &&
        (tri1.p3.x == tri2.p2.x) && (tri1.p3.y == tri2.p2.y))
        return true;
    if ((tri1.p1.x == tri2.p3.x) && (tri1.p1.y == tri2.p3.y) &&
        (tri1.p2.x == tri2.p2.x) && (tri1.p2.y == tri2.p2.y) &&
        (tri1.p3.x == tri2.p1.x) && (tri1.p3.y == tri2.p1.y))
        return true;
    if ((tri1.p1.x == tri2.p2.x) && (tri1.p1.y == tri2.p2.y) &&
        (tri1.p2.x == tri2.p1.x) && (tri1.p2.y == tri2.p1.y) &&
        (tri1.p3.x == tri2.p3.x) && (tri1.p3.y == tri2.p3.y))
        return true;
    if ((tri1.p1.x == tri2.p1.x) && (tri1.p1.y == tri2.p1.y) &&
        (tri1.p2.x == tri2.p3.x) && (tri1.p2.y == tri2.p3.y) &&
        (tri1.p3.x == tri2.p2.x) && (tri1.p3.y == tri2.p2.y))
        return true;

    return false;
}

bool CImageRegistration::areSameTrianglesIndexed(t_triangle tri1, t_triangle tri2) {
    if ((tri1.index1 == tri2.index1) && (tri1.index2 == tri2.index2) && (tri1.index3 == tri2.index3))
        return true;

    return false;
}

bool CImageRegistration::areTrianglesEquipositioned(t_triangle tri1, t_triangle tri2) {

    if ((tri1.p1.x < tri1.p2.x) && (tri2.p1.x >= tri2.p2.x)) return false;
    if ((tri1.p1.x < tri1.p3.x) && (tri2.p1.x >= tri2.p3.x)) return false;

    if ((tri1.p1.x > tri1.p2.x) && (tri2.p1.x <= tri2.p2.x)) return false;
    if ((tri1.p1.x > tri1.p3.x) && (tri2.p1.x <= tri2.p3.x)) return false;

    if ((tri1.p1.y < tri1.p2.y) && (tri2.p1.y >= tri2.p2.y)) return false;
    if ((tri1.p1.y < tri1.p3.y) && (tri2.p1.y >= tri2.p3.y)) return false;

    if ((tri1.p1.y > tri1.p2.y) && (tri2.p1.y <= tri2.p2.y)) return false;
    if ((tri1.p1.y > tri1.p3.y) && (tri2.p1.y <= tri2.p3.y)) return false;

    if ((tri1.p2.x < tri1.p3.x) && (tri2.p2.x >= tri2.p3.x)) return false;
    if ((tri1.p2.x > tri1.p3.x) && (tri2.p2.x <= tri2.p3.x)) return false;

    if ((tri1.p2.y < tri1.p3.y) && (tri2.p2.y >= tri2.p3.y)) return false;
    if ((tri1.p2.y > tri1.p3.y) && (tri2.p2.y <= tri2.p3.y)) return false;

    return true;
}

void CImageRegistration::getPairsFromPreviousImg(IplImage * imgDB, IplImage * imgRT, CvPoint2D32f * &pointsDB, CvPoint2D32f * &pointsRT, int &nFeat) {
    CvPoint2D32f * pointsDB1, * pointsDB2, * pointsRT1, * pointsRT2;
    int nFeatDB, nFeatRT;
    findPairs(oldImgDB, imgDB, pointsDB1, pointsDB2, nFeatDB, false, USE_REGIONS, oldPointsDB, oldNumberOfFeatures);
    findPairs(oldImgRT, imgRT, pointsRT1, pointsRT2, nFeatRT, false, USE_REGIONS, oldPointsRT, oldNumberOfFeatures);
    
    showPairs("oldDB", oldImgDB, imgDB, pointsDB1, pointsDB2, nFeatDB);
    showPairs("oldRT", oldImgRT, imgRT, pointsRT1, pointsRT2, nFeatRT);
    showPairs("old", oldImgDB, oldImgRT, oldPointsDB, oldPointsRT, oldNumberOfFeatures);
}

void CImageRegistration::findPairsWithCorrelation(IplImage * imgDB, IplImage * imgRT, CvPoint2D32f * &pointsDB, CvPoint2D32f * &pointsRT, int &nFeat) {
    int wSize = 9;
    double thresh = 0.7;

    CvPoint2D32f * cornersDB = NULL;
    CvPoint2D32f * cornersRT = NULL;
    int nCornersDB, nCornersRT;

    findInitialPoints(imgDB, cornersDB, nCornersDB);
    findInitialPoints(imgRT, cornersRT, nCornersRT);

    CvMat * corrMat = cvCreateMat(nCornersDB, nCornersRT, CV_32FC1);

    CvRect rectDB = cvRect(0, 0, wSize, wSize);
    CvRect rectRT = cvRect(0, 0, wSize, wSize);
    int r = (wSize - 1) / 2;

    IplImage * w1 = cvCreateImage(cvSize(wSize, wSize), IPL_DEPTH_8U, 1);
    IplImage * w2 = cvCreateImage(cvSize(wSize, wSize), IPL_DEPTH_8U, 1);

    CvScalar mean, sdv;
    double mean1, mean2;
    double sdv1, sdv2;

    for (int i = 0; i < nCornersDB; i++) {
        if ((cornersDB[i].x < wSize) || (cornersDB[i].y < wSize) ||
            (cornersDB[i].x > imgDB->width - wSize - 1) || (cornersDB[i].y > imgDB->height - wSize - 1))
            continue;

        rectDB.x = cornersDB[i].x - r;
        rectDB.y = cornersDB[i].y - r;

        cvSetImageROI(imgDB, rectDB);
        cvCopyImage(imgDB, w1);

        cvAvgSdv(w1, &mean, &sdv);
        //double mean1 = cvAvg(w1).val[0];
        mean1 = mean.val[0];
        sdv1 = sdv.val[0];

        for (int j = 0; j < nCornersRT; j++) {
            if ((cornersRT[i].x < wSize) || (cornersRT[i].y < wSize) ||
            (cornersRT[i].x > imgRT->width - wSize - 1) || (cornersRT[i].y > imgRT->height - wSize - 1))
            continue;

            rectRT.x = cornersRT[i].x - r;
            rectRT.y = cornersRT[i].y - r;

            cvSetImageROI(imgRT, rectRT);
            cvCopyImage(imgRT, w2);

            cvAvgSdv(w2, &mean, &sdv);
            //double mean2 = cvAvg(w2).val[0];
            mean2 = mean.val[0];
            sdv2 = sdv.val[0];
            

            double corr = 0;
            for (int a = 0; a < wSize; a++) {
                for (int b = 0; b < wSize; b++) {
                    corr += (cvGetReal2D(w1, a, b) - mean1) * (cvGetReal2D(w2, a, b) - mean2);
                }
            }
            corr /= (wSize * wSize - 1) * sdv1 * sdv2;

            cvSetReal2D(corrMat, i, j, corr);
        }
    }

    cvReleaseImage(&w1);
    cvReleaseImage(&w2);
    cvResetImageROI(imgDB);
    cvResetImageROI(imgRT);

    pointsDB = new CvPoint2D32f[nCornersDB * nCornersRT];
    pointsRT = new CvPoint2D32f[nCornersDB * nCornersRT];

    nFeat = 0;
    for (int i = 0; i < nCornersDB; i++) {
        bool follow = false;
        for (int j = 0; j < nCornersRT; j++) {            
            if (cvGetReal2D(corrMat, i, j) > thresh) {
                pointsDB[nFeat] = cornersDB[i];
                pointsRT[nFeat] = cornersRT[j];
                nFeat++;
                follow = true;
                break;
            }            
        }
        if (follow) continue;
    }//*/    
}