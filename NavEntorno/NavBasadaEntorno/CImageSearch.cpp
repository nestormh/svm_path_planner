/* 
 * File:   CImageSearch.cpp
 * Author: neztol
 * 
 * Created on 24 de mayo de 2010, 11:50
 */

#include "CImageSearch.h"

CImageSearch::CImageSearch(string dbName, string dbST, string dbRT, string pathBase, bool useIMU, CvRect rect) {
    this->dbST = dbST;
    this->dbRT = dbRT;
    this->pathBase = pathBase;
    this->rect = rect;
    this->size = cvSize(rect.width, rect.height);

    if (sqlite3_open(dbName.c_str(), &db) != SQLITE_OK){
        cerr << "Error al abrir la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    sqlite3_stmt *statement;

    // Buscamos el índice y la extensión de la ruta
    const char *sql = "SELECT * FROM route where (name == ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }

    sqlite3_bind_text(statement, 1, dbST.c_str(), -1, NULL);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        indexST = sqlite3_column_int(statement, 0);
        unsigned char * charExt = (unsigned char *)sqlite3_column_text(statement, 3);
        extST = string((char *)charExt);
    }
    if (sqlite3_reset(statement) != SQLITE_OK) {
        cerr << "Error al resetear el statement" << endl;
    }

    sqlite3_bind_text(statement, 1, dbRT.c_str(), -1, NULL);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        indexRT = sqlite3_column_int(statement, 0);
        unsigned char * charExt = (unsigned char *)sqlite3_column_text(statement, 3);
        extRT = string((char *)charExt);
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    // Recorremos la ruta estática y calculamos las distancias
    sql = "SELECT localX, localY, angleGPS, angleIMU FROM points WHERE (route == ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_double(statement, 1, indexST);
    
    vector<CvPoint2D32f> points;
    vector<double> angles;

    while (sqlite3_step(statement) == SQLITE_ROW) {
        double localX = sqlite3_column_double(statement, 0);
        double localY = sqlite3_column_int(statement, 1);
        //double angleGPS = sqlite3_column_int(statement, 2);
        double angleIMU = sqlite3_column_int(statement, 3);

        double angle = 0;
        if (useIMU) {                        
            angle = angleIMU;
        } else {
            double angleGPS = 0;
            if (points.size() > 3) {
                CvPoint2D32f p = points.at(points.size() - 3);
                angleGPS = atan2(localY - p.y, localX - p.x);
            }
            angle = angleGPS * 180 / CV_PI;
        }

        points.push_back(cvPoint2D32f(localX, localY));
        angles.push_back(angle);        
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    mapDistances = cvCreateMat(points.size(), points.size(), CV_64FC1);
    mapAngles = cvCreateMat(points.size(), points.size(), CV_64FC1);
    currentNearestPoints = cvCreateMat(points.size(), points.size(), CV_64FC1);

    img1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    img2 = cvCreateImage(size, IPL_DEPTH_8U, 1);


    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < points.size(); j++) {
            double dist = sqrt(pow(points.at(i).x - points.at(j).x, 2.0) + pow(points.at(i).y - points.at(j).y, 2.0));
            cvSetReal2D(mapDistances, i, j, dist);

            double distAng = abs(angles.at(i) - angles.at(j));
            distAng = min(distAng, 360 - distAng);

            cvSetReal2D(mapAngles, i, j, distAng);            
        }
    }

    indexST = 0;
    indexRT = 0;

    IplImage * imgRT;
    getRTImage(imgRT);
    IplImage * imgST;

    getInitialImage(imgRT, imgST);

    cvShowImage("imgRT", imgRT);
    cvShowImage("imgST", imgST);

    cvWaitKey(0);

    cvReleaseImage(&imgRT);
    cvReleaseImage(&imgST);
}

CImageSearch::~CImageSearch() {
    if (sqlite3_close(db) != SQLITE_OK) {
        cerr << "Error al cerrar la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    cvReleaseMat(&mapDistances);
    cvReleaseMat(&mapAngles);
}

void CImageSearch::getRTImage(IplImage * &imgRT) {
    char imageName[1024];

    sprintf(imageName, "%s/%s/Camera2/Image%d%s", pathBase.c_str(), dbRT.c_str(), indexRT, extRT.c_str());

    IplImage * tmpImg = cvLoadImage(imageName, 0);
    cvSetImageROI(tmpImg, rect);
    imgRT = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
    cvCopyImage(tmpImg, imgRT);

    cvReleaseImage(&tmpImg);
}

void CImageSearch::getSTImage(IplImage * &imgST, int index) {
    char imageName[1024];

    if (index == -1)
        index = indexST;

    sprintf(imageName, "%s/%s/Camera2/Image%d%s", pathBase.c_str(), dbST.c_str(), index, extST.c_str());

    IplImage * tmpImg = cvLoadImage(imageName, 0);
    cvSetImageROI(tmpImg, rect);
    imgST = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
    cvCopyImage(tmpImg, imgST);

    cvReleaseImage(&tmpImg);
}

inline void CImageSearch::testFast(IplImage * img, vector<CvPoint2D32f> &points) {
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

inline void CImageSearch::findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures) {
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

inline void CImageSearch::oFlow(vector <CvPoint2D32f> &points1, vector <t_Pair> &pairs, IplImage * &img1, IplImage * &img2) {
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

inline void CImageSearch::cleanRANSAC(int method, vector<t_Pair> &pairs) {
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


inline void CImageSearch::checkCoveredArea(IplImage * imgB, IplImage * imgA, int &coveredArea) {
    if ((cvCountNonZero(imgA) == 0) || (cvCountNonZero(imgB) == 0)) {
        coveredArea = 0;
        return;
    }

    cvCopyImage(imgA, img1);
    cvCopyImage(imgB, img2);

    testFast(img1, points1);
    if (points1.size() < MIN_NFEAT) {
        coveredArea = 0;

        return;
    }

    oFlow(points1, pairs, img1, img2);

    if (pairs.size() < MIN_NFEAT) {
        coveredArea = 0;

        return;
    }

    /*cleanRANSAC(CV_FM_RANSAC, pairs);

    if (pairs.size() < MIN_NFEAT) {
        coveredArea = 0;
        return;
    }

//    setMaskFromPoints(mask1, 2);

//    coveredArea = cvCountNonZero(mask1);*/
    coveredArea = pairs.size();
}

void CImageSearch::getInitialImage(IplImage * imgRT, IplImage * &imgST) {
    int maxCovered = 0;
    imgST = cvCreateImage(size, IPL_DEPTH_8U, 1);
    for (int i = 0; i < currentNearestPoints->width; i++) {
        IplImage * tmpImgST;
        getSTImage(tmpImgST, i);
        int covered;
        checkCoveredArea(imgRT, tmpImgST, covered);
        if (covered > maxCovered) {
            maxCovered = covered;
            cvCopyImage(tmpImgST, imgST);
            indexST = i;
        }
        cvReleaseImage(&tmpImgST);
    }
}

void CImageSearch::getNearestImage(IplImage * imgRT, IplImage * &imgST) {
    for (int i = 0; i < currentNearestPoints->width; i++) {
        
    }
}