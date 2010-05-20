/* 
 * File:   CRoadDetection.cpp
 * Author: neztol
 * 
 * Created on 13 de mayo de 2010, 13:03
 */

#include "CRoadDetection.h"
#include "fast/cvfast.h"

#define SUB_DISTANCE 2

CRoadDetection::CRoadDetection(CvSize size) {
    ruta = new CRutaDB2("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERBase2", "Rutas/pruebaITERConObs2", "/home/neztol/doctorado/Datos/DB");
    img1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    img2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    diff = cvCreateImage(size, IPL_DEPTH_8U, 1);
    maskResult = cvCreateImage(size, IPL_DEPTH_8U, 1);

    cvSet(mask, cvScalar(255));

    this->size = size;

    //cvNamedWindow("Img1", 1);
    //cvNamedWindow("Img2", 1);
    //cvNamedWindow("Diff", 1);
    //cvNamedWindow("Diff2", 1);
    cvNamedWindow("Points", 1);

    aco = new CAntColony(size);
}

CRoadDetection::CRoadDetection(const CRoadDetection& orig) {
    delete aco;
}

CRoadDetection::~CRoadDetection() {
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&mask);
    cvReleaseImage(&diff);
    cvReleaseImage(&maskResult);
}

void CRoadDetection::detect(int index, IplImage * result) {
    IplImage * img;
    ruta->getImageAt(img, 2, index);
    cvCopyImage(img, img1);
    int index2 = index - SUB_DISTANCE;
    if (index2 < 0)
        index2 = index + SUB_DISTANCE;
    ruta->getImageAt(img, 2, index2);
    cvCopyImage(img, img2);

    //cvShowImage("Img1", img1);
    //cvShowImage("Img2", img2);

    //cvAbsDiff(img1, img2, diff);
    calcPCA(img1, img2, diff, mask);
    //cvShowImage("Diff", diff);
    obstacleDetectionQuartile(diff, mask);
    cvZero(diff);
    cvCopy(img1, diff, maskResult);
    //cvShowImage("Diff2", diff);

    cvCopyImage(maskResult, result);

    cvReleaseImage(&img);
}

void CRoadDetection::obstacleDetectionQuartile(IplImage * pcaResult, IplImage * mask) {

    IplImage * obstaclesMask = cvCreateImage(size, IPL_DEPTH_8U, 1);

    double k = 2;

    // Ordenamos los datos
    vector<double> data;
    int levels[255];
    for (int i = 0; i < 255; i++) {
        levels[i] = 0;
    }

    for (int i = 0; i < pcaResult->height; i++) {
        for (int j = 0; j < pcaResult->width; j++) {
            if (cvGetReal2D(mask, i, j) != 255) continue;

            double val = cvGetReal2D(pcaResult, i, j);

            levels[(int) val]++;
        }
    }
    for (int i = 0; i < 255; i++) {
        for (int j = 0; j < levels[i]; j++) {
            data.push_back(i);
        }
    }

    double median = 0;
    double q1 = 0;
    double q2 = 0;
    if (data.size() % 2 == 0) {
        int index = (data.size() / 2) - 1;
        median = (data.at(index) + data.at(index + 1)) / 2;
        index = (data.size() / 4) - 1;
        q1 = (data.at(index) + data.at(index + 1)) / 2;
        index = (data.size() * 3 / 4) - 1;
        q2 = (data.at(index) + data.at(index + 1)) / 2;
    } else {
        int index = data.size() / 2;
        median = data.at(index);
        index = data.size() / 4;
        q1 = data.at(index);
        index = data.size() * 3 / 4;
        q2 = data.at(index);
    }

    double maxThresh = q2 + k * (q2 - q1);

    cvZero(obstaclesMask);
    cvCopy(pcaResult, obstaclesMask, mask);
    cvThreshold(obstaclesMask, maskResult, maxThresh, 255, CV_THRESH_BINARY_INV);
    //cvRectangle(obstaclesMask, cvPoint(0, 0), cvPoint(size.width - 1, 100), cvScalar(0), CV_FILLED);

    /*cvNamedWindow("ObstacleQ", 1);
    cvShowImage("ObstacleQ", obstaclesMask);

    cvErode(obstaclesMask, obstaclesMask, 0, 2);

    //cvNamedWindow("ObstacleQEroded", 1);
    //cvShowImage("ObstacleQEroded", obstaclesMask);

    cvDilate(obstaclesMask, obstaclesMask, 0, 2);

    cvNamedWindow("ObstacleQDilated", 1);
    cvShowImage("ObstacleQDilated", obstaclesMask);//*/

    //detectObstacles(obstaclesMask);

    cvReleaseImage(&obstaclesMask);
}

void CRoadDetection::detectObstacles(IplImage * mask) {
    // Ahora buscamos contornos
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contour = 0;

    cvFindContours(mask, storage, &contour, sizeof (CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    cvZero(maskResult);

    CvSeq* maxContour = 0;
    float maxArea = DBL_MIN;
    for (; contour != 0; contour = contour->h_next) {
        double area = fabs(cvContourArea(contour));
        if (area > maxArea) {
            maxArea = area;
            maxContour = contour;
        }
    }

    cvDrawContours(maskResult, maxContour, cvScalar(255), cvScalar(255), 0, CV_FILLED, 8);

    cvNamedWindow("ResultadoPCA", 1);
    cvShowImage("ResultadoPCA", maskResult);

    //CvSeq* poly = cvApproxPoly(contour, sizeof(CvContour), storage,
    //	                    CV_POLY_APPROX_DP, cvContourPerimeter(contour)*0.05, 0);


    //cvDrawContours(gris2, contour, cvScalar(255), cvScalar(0), -1, -1, 8);

    /*cvDilate(maskResult, maskResult);
    cvErode(maskResult, maskResult);
    IplImage * tmpImg = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvZero(tmpImg);*/

    maxContour = cvApproxPoly( maxContour, maxContour->header_size, storage,CV_POLY_APPROX_DP, cvContourPerimeter(maxContour)*0.01);
    CvSeqReader reader;
    int N = maxContour->total;
    CvPoint p;
    CvPoint pMax;

    for (; maxContour != NULL; maxContour = maxContour->h_next) {

        cvStartReadSeq(maxContour, &reader);

        for (int i = 0; i < N; ++i) {

            CV_READ_SEQ_ELEM(p, reader);
            if (p.y < pMax.y) {
                pMax = p;
            }
            printf("%d %d \n", p.x, p.y);
        }
    }
    cvDrawContours(maskResult, maxContour, cvScalar(255), cvScalar(255), 0, CV_FILLED, 8);
    cvCircle(maskResult, pMax, 5, cvScalar(255), 1);
    /*cvAbsDiff(tmpImg, maskResult, maskResult);
    cvReleaseImage(&tmpImg);*/

    cvNamedWindow("poly", 1);
    cvShowImage("poly", maskResult);

    cvReleaseMemStorage(&storage);
    delete contour;
}

void CRoadDetection::calcPCA(IplImage * img1, IplImage * img2, IplImage * diff, IplImage * mask) {
    int length = cvCountNonZero(mask);
    CvMat * data = cvCreateMat(2, length, CV_64FC1);
    CvMat * data1 = cvCreateMat(1, length, CV_64FC1);
    CvMat * data2 = cvCreateMat(1, length, CV_64FC1);
    CvMat * corr = cvCreateMat(2, 2, CV_64FC1);
    CvMat * avg = cvCreateMat(1, 2, CV_64FC1);
    CvMat * eigenValues = cvCreateMat(1, 2, CV_64FC1);
    CvMat * eigenVectors = cvCreateMat(2, 2, CV_64FC1);
    CvMat * pcaData = cvCreateMat(2, length, CV_64FC1);
    CvMat * dataX = cvCreateMat(1, length, CV_64FC1);
    CvMat * dataY = cvCreateMat(1, length, CV_64FC1);
    CvMat * distPCA = cvCreateMat(img1->height, img1->width, CV_64FC1);
    CvScalar xMean, yMean, xSdv, ySdv;

    vector<CvPoint> origPos;

    data1 = cvGetRow(data, data1, 0);
    data2 = cvGetRow(data, data2, 1);

    int pos = 0;
    for (int i = 0; i < img2->width; i++) {
        for (int j = 0; j < img2->height; j++) {
            if (cvGetReal2D(mask, j, i) != 0) {
                cvmSet(data, 0, pos, cvGetReal2D(img1, j, i));
                cvmSet(data, 1, pos, cvGetReal2D(img2, j, i));
                origPos.push_back(cvPoint(i, j));

                pos++;
            }
        }
    }

    double m1 = cvMean(data1);
    double m2 = cvMean(data2);

    cvSubS(data1, cvScalar(m1), data1);
    cvSubS(data2, cvScalar(m2), data2);

    cvMulTransposed(data, corr, 0);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cvmSet(corr, i, j, cvmGet(corr, i, j) / (1 - img2->width * img2->height));
        }
    }

    cvCalcPCA(corr, avg, eigenValues, eigenVectors, CV_PCA_DATA_AS_ROW);

    double a = cvmGet(corr, 0, 0);
    double b = cvmGet(corr, 0, 1);
    double c = cvmGet(corr, 1, 0);
    double d = cvmGet(corr, 1, 1);

    double m = (-a) - d;
    double n = a * d - b * c;

    double lambda1 = (-m + sqrt(pow(m, 2.0) - 4 * n)) / 2.0;
    double lambda2 = (-m - sqrt(pow(m, 2.0) - 4 * n)) / 2.0;

    double e2 = sqrt(pow(c, 2.0) / (pow(-(d - lambda1), 2.0) + pow(c, 2.0)));
    double e1 = -(d - lambda1) * e2 / c;

    cvmSet(eigenVectors, 0, 0, e1);
    cvmSet(eigenVectors, 1, 0, e2);

    e2 = sqrt(pow(c, 2.0) / (pow(-(d - lambda2), 2.0) + pow(c, 2.0)));
    e1 = e1 = -(d - lambda2) * e2 / c;

    cvmSet(eigenVectors, 0, 1, e1);
    cvmSet(eigenVectors, 1, 1, e2);

    cvMatMul(eigenVectors, data, pcaData);

    // Gets ACP vectors X and Y
    dataX = cvGetRow(pcaData, dataX, 1);
    dataY = cvGetRow(pcaData, dataY, 0);

    // Calculates mean and stdev
    cvAvgSdv(dataX, &xMean, &xSdv);
    cvAvgSdv(dataY, &yMean, &ySdv);

    // Draws ACP graphic data
    /*for (int i = 0, pos = 0; i < img1->width; i++) {
        for (int j = 0; j < img2->height; j++, pos++) {
            cvSetReal2D(plinear, j, i, abs(cvGetReal1D(dataY, pos) - yMean.val[0]));
        }
    }*/
    for (int i = 0; i < origPos.size(); i++) {
        CvPoint p = origPos.at(i);

        cvSetReal2D(diff, p.y, p.x, abs(cvGetReal1D(dataY, i) - yMean.val[0]));
    }

    cvReleaseMat(&data);
    cvReleaseMat(&data1);
    cvReleaseMat(&data2);
    cvReleaseMat(&corr);
    cvReleaseMat(&avg);
    cvReleaseMat(&eigenValues);
    cvReleaseMat(&eigenVectors);
    cvReleaseMat(&pcaData);
    cvReleaseMat(&dataX);
    cvReleaseMat(&dataY);
    cvReleaseMat(&distPCA);
}

inline void CRoadDetection::testFast(IplImage * img, vector<CvPoint2D32f> &points) {
    int inFASTThreshhold = 5; //80
    int inNpixels = 9;
    int inNonMaxSuppression = 0;

    CvPoint* corners;
    int numCorners;

    cvCornerFast(img, inFASTThreshhold, inNpixels, inNonMaxSuppression, &numCorners, & corners);

    points.clear();
    for (int i = 0; i < numCorners; i++) {
        points.push_back(cvPointTo32f(corners[i]));
    }

    delete corners;
}

inline void CRoadDetection::testShiTomasi(IplImage * img, vector<CvPoint2D32f> &points) {
    int numCorners = 10000;
    IplImage * eigen = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * tmp = cvCreateImage(size, IPL_DEPTH_8U, 1);

    CvPoint2D32f * corners = new CvPoint2D32f[10000];
    int defFeatMinDist = 1;

    // Corners are obtained
    cvGoodFeaturesToTrack(img, eigen, tmp, corners, &numCorners,
        DEFAULT_FEATURES_QUALITY, defFeatMinDist, NULL, 3, 0, 0.04);

    points.clear();
    for (int i = 0; i < numCorners; i++) {
        points.push_back(corners[i]);
    }

    delete corners;
}

void CRoadDetection::drawTriangles(vector<CvPoint2D32f> points, IplImage * img) {
    CvRect rect = cvRect(0, 0, size.width, size.height);
    CvMemStorage * storage = cvCreateMemStorage(0);

    CvSubdiv2D * subdiv = cvCreateSubdivDelaunay2D(rect, storage);
    for (vector<CvPoint2D32f>::iterator it = points.begin(); it != points.end(); it++) {
        if ((it->x < 0) || (it->y < 0) ||
                (it->x > size.width - 1) || (it->y > size.height - 1)) {
            continue;
        }
        try {
            cvSubdivDelaunay2DInsert(subdiv, *it);
        } catch (cv::Exception e) {
        }
    }

    CvSeqReader reader;
    int i, total = subdiv->edges->total;
    int elem_size = subdiv->edges->elem_size;

    cvStartReadSeq((CvSeq*) (subdiv->edges), &reader, 0);

    IplImage * imgDelaunay = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * maskDelaunay = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvCvtColor(img, imgDelaunay, CV_GRAY2BGR);
    cvSet(imgDelaunay, cvScalarAll(255));

    cvNamedWindow("Delaunay", 1);

    for (i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D*) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {


            CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_RIGHT);
            CvSubdiv2DEdge edge2 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge1, CV_PREV_AROUND_RIGHT);

            CvPoint * p = new CvPoint[3];
            p[0] = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            p[1] = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge1)->pt);
            p[2] = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge2)->pt);
            int npts = 3;

            double l1 = (pow(p[0].x - p[1].x, 2.0) + pow(p[0].y - p[1].y, 2.0));
            double l2 = (pow(p[1].x - p[2].x, 2.0) + pow(p[1].y - p[2].y, 2.0));
            double l3 = (pow(p[2].x - p[0].x, 2.0) + pow(p[2].y - p[0].y, 2.0));            
            double maxL = max(l1, max(l2, l3));

            //cout << abs(cvTriangleArea(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt, cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge1)->pt, cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge2)->pt)) << endl;
            if (abs(cvTriangleArea(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt, cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge1)->pt, cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge2)->pt)) < 500) {
                //CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
                //cvFillPoly(imgDelaunay, &p, &npts, 1, color);
                //if (maxL <= 2 * minL) {
                if (maxL < 20*20) {
                    cvFillPoly(imgDelaunay, &p, &npts, 1, cvScalarAll(0), 8);
                }
            }//*/

            //cvLine(imgDelaunay, cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt), cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt), cvScalar(0, 0, 255));
            //cvLine(imgDelaunay, cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge1)->pt), cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge1)->pt), cvScalar(0, 0, 255));
            //cvLine(imgDelaunay, cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge2)->pt), cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge2)->pt), cvScalar(0, 0, 255));


            /*int maxDist = 5000;
            if (((pow(p[0].x - p[1].x, 2.0) + pow(p[0].y - p[1].y, 2.0)) > maxDist * maxDist) ||
                ((pow(p[1].x - p[2].x, 2.0) + pow(p[1].y - p[2].y, 2.0)) > maxDist * maxDist) ||
                ((pow(p[2].x - p[0].x, 2.0) + pow(p[2].y - p[0].y, 2.0)) > maxDist * maxDist)) {

                cout << sqrt(pow(p[0].x - p[1].x, 2.0) + pow(p[0].y - p[1].y, 2.0)) << ", ";
                cout << sqrt(pow(p[1].x - p[2].x, 2.0) + pow(p[1].y - p[2].y, 2.0)) << ", ";
                cout << sqrt(pow(p[2].x - p[0].x, 2.0) + pow(p[2].y - p[0].y, 2.0)) << endl;

                cvFillPoly(imgDelaunay, &p, &npts, 1, cvScalarAll(255));
            }//*/
            delete p;

        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    cvRectangle(imgDelaunay, cvPoint(0, 0), cvPoint(5, size.height - 1), cvScalarAll(0), CV_FILLED);
    cvRectangle(imgDelaunay, cvPoint(size.width - 1, 0), cvPoint(size.width - 6, size.height - 1), cvScalarAll(0), CV_FILLED);
    cvRectangle(imgDelaunay, cvPoint(0, 0), cvPoint(size.width - 1, 100), cvScalarAll(0), CV_FILLED);
    cvCvtColor(imgDelaunay, maskDelaunay, CV_BGR2GRAY);
    detectObstacles(maskDelaunay);
    cvShowImage("Delaunay", imgDelaunay);

    cvReleaseImage(&imgDelaunay);
    cvReleaseImage(&maskDelaunay);
    cvReleaseMemStorage(&storage);
}

void CRoadDetection::cleanNeighbors(vector<CvPoint2D32f> &points) {

    vector<CvPoint2D32f> tmpPoints(points);
    points.clear();
    for (vector<CvPoint2D32f>::iterator it1 = tmpPoints.begin(); it1 != tmpPoints.end(); it1++) {
        int nPoints = 0;
        for (vector<CvPoint2D32f>::iterator it2 = tmpPoints.begin(); it2 != tmpPoints.end(); it2++) {
            if ((pow(it1->x - it2->x, 2.0) + pow(it1->y - it2->y, 2.0)) < 10 * 10) {
                nPoints++;
            }
        }
        if (nPoints > 5) {
            cout << nPoints << endl;
            points.push_back(*it1);
        }
    }
}

void CRoadDetection::detectRoadWithFAST(int index, IplImage * result) {
    IplImage * img;
    ruta->getImageAt(img, 2, index);
    cvCopyImage(img, img1);
    int index2 = index - SUB_DISTANCE;
    if (index2 < 0)
        index2 = index + SUB_DISTANCE;
    ruta->getImageAt(img, 2, index2);
    cvCopyImage(img, img2);

    cvShowImage("Img1", img1);
    cvShowImage("Img2", img2);

    cvErode(img1, img1);

    vector<CvPoint2D32f> points;
    testFast(img1, points);
    //testShiTomasi(img1, points);
    cleanNeighbors(points);

    IplImage * drawPoints = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvCvtColor(img1, drawPoints, CV_GRAY2BGR);
    for (vector<CvPoint2D32f>::iterator it = points.begin(); it != points.end(); it++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        cvCircle(drawPoints, cvPointFrom32f(*it), 2, color, -1);
    }
    cvShowImage("Points", drawPoints);

    drawTriangles(points, img1);

    cvAbsDiff(img1, maskResult, img1);
    cvNamedWindow("Road", 1);
    cvShowImage("Road", img1);

    cvCopyImage(maskResult, result);

    cvReleaseImage(&drawPoints);
    cvReleaseImage(&img);
}

void CRoadDetection::detectRoadWithFATPoints(int index, IplImage * result) {
    IplImage * img;
    ruta->getImageAt(img, 2, index);
    cvCopyImage(img, img1);
    int index2 = index - SUB_DISTANCE;
    if (index2 < 0)
        index2 = index + SUB_DISTANCE;
    ruta->getImageAt(img, 2, index2);
    cvCopyImage(img, img2);

    cvShowImage("Img1", img1);
    cvShowImage("Img2", img2);

    vector<CvPoint2D32f> points;
    testFast(img1, points);
    //testShiTomasi(img1, points);
    //cleanNeighbors(points);

    IplImage * fatPoints = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvSet(fatPoints, cvScalar(255));

    for (vector<CvPoint2D32f>::iterator it = points.begin(); it != points.end(); it++) {
        cvCircle(fatPoints, cvPointFrom32f(*it), 6, cvScalar(0), -1);
    }
    cvRectangle(fatPoints, cvPoint(0, 0), cvPoint(size.width - 1, 100), cvScalarAll(0), CV_FILLED);
    cvShowImage("fatPoints", fatPoints);

    detectObstacles(fatPoints);

    cvCopyImage(maskResult, result);

    cvAdd(img, maskResult, img);
    cvShowImage("MaskFat", img);

    cvReleaseImage(&fatPoints);
    cvReleaseImage(&img);
}

void CRoadDetection::detectOcclusions(int index, IplImage * result) {
    IplImage * img;
    ruta->getImageAt(img, 2, index);
    cvCopyImage(img, img1);
    int index2 = index - SUB_DISTANCE;
    if (index2 < 0)
        index2 = index + SUB_DISTANCE;
    ruta->getImageAt(img, 2, index2);
    cvCopyImage(img, img2);

    //cvShowImage("Img1", img1);
    //cvShowImage("Img2", img2);

    //cvAbsDiff(img1, img2, diff);
    calcPCA(img1, img2, diff, mask);
    //cvShowImage("Diff", diff);
    
    obstacleDetectionQuartile(diff, mask);
    //cvShowImage("Diff2", maskResult);

    cvCopyImage(maskResult, result);

    cvReleaseImage(&img);
}

void CRoadDetection::detectACO(int index, IplImage * result) {
    IplImage * img;
    IplImage * imgC = cvCreateImage(size, IPL_DEPTH_8U, 3);
    ruta->getImageAt(img, 2, index);
    cvCvtColor(img, imgC, CV_GRAY2BGR);

    CvPoint * poly = aco->iterate(imgC);
    int npts = 4;    
    cvZero(result);
    cvFillPoly(result, &poly, &npts, 1, cvScalar(255));
    delete poly;

    cvAbsDiff(img, result, img);
    cvNamedWindow("Road", 1);
    cvShowImage("Road", img);

    cvReleaseImage(&img);
    cvReleaseImage(&imgC);
}

void CRoadDetection::detectACO(IplImage * img, IplImage * result) {
    IplImage * imgC = cvCreateImage(size, IPL_DEPTH_8U, 3);

    cvCvtColor(img, imgC, CV_GRAY2BGR);

    CvPoint * poly = aco->iterate(imgC);
    int npts = 4;
    cvZero(result);
    cvFillPoly(result, &poly, &npts, 1, cvScalar(255));
    delete poly;

    cvAbsDiff(img, result, img);
    cvNamedWindow("Road", 1);
    cvShowImage("Road", img);

    cvReleaseImage(&imgC);
}

void CRoadDetection::detectFixed(IplImage * result) {

    CvPoint * poly = new CvPoint[4];
    int npts = 4;
    poly[0] = cvPoint(0, size.height - 10);
    poly[1] = cvPoint((size.width/2) - 20, 100);
    poly[2] = cvPoint((size.width/2) + 30, 100);
    poly[3] = cvPoint(size.width - 1, size.height - 10);
    cvZero(result);
    cvFillPoly(result, &poly, &npts, 1, cvScalar(255));
    delete poly;

    cvShowImage("Fixed", result);
}
