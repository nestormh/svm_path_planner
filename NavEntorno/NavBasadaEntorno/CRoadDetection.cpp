/* 
 * File:   CRoadDetection.cpp
 * Author: neztol
 * 
 * Created on 13 de mayo de 2010, 13:03
 */

#include "CRoadDetection.h"

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

    cvNamedWindow("Img1", 1);
    cvNamedWindow("Img2", 1);
    cvNamedWindow("Diff", 1);
    cvNamedWindow("Diff2", 1);
}

CRoadDetection::CRoadDetection(const CRoadDetection& orig) {
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

    cvShowImage("Img1", img1);
    cvShowImage("Img2", img2);

    //cvAbsDiff(img1, img2, diff);
    calcPCA(img1, img2, diff, mask);
    cvShowImage("Diff", diff);
    obstacleDetectionQuartile(diff, mask);
    cvZero(diff);
    cvCopy(img1, diff, maskResult);    
    cvShowImage("Diff2", diff);

    cvCopyImage(result, maskResult);

    cvReleaseImage(&img);
}

/*void CRealMatches::startTestRoadDetection() {
    CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERConObs2", "Rutas/pruebaITERBase2", "/home/neztol/doctorado/Datos/DB");
    IplImage * imgDB;
    IplImage * imgRT;
    IplImage * lastImg = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvNamedWindow("road", 1);

    ruta.getNextImage(imgRT, imgDB);
    cvCopyImage(imgRT, lastImg);
    //cvNamedWindow("ImgDB", 1);
    //cvNamedWindow("ImgRT", 1);
    int index = 4;
    ruta.setCurrentPoint(index);
    while (true) {
        ruta.getNextImage(imgRT, imgDB);
        index += 4;
        ruta.setCurrentPoint(index);

        cvCopyImage(imgRT, img1);
        cvCopyImage(imgDB, img2);

        clock_t myTime = clock();

        cvSet(mask1, cvScalar(255));
        //calcPCA(img1, lastImg, mask1);
        cvAbsDiff(imgRT, lastImg, plinear);
        cvShowImage("road", plinear);
        obstacleDetectionQuartile(plinear, mask1);

        cvCopyImage(img1, lastImg);

        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;
        cout << "Index = " << (index - 1) << endl;

        int key = cvWaitKey(0);
        if (key == 27)
            exit(0);
        if (key == 32)
            cvWaitKey(0);

        cvReleaseImage(&imgDB);
        cvReleaseImage(&imgRT);
    }
}
//*/

void CRoadDetection::obstacleDetectionQuartile(IplImage * pcaResult, IplImage * mask) {

    IplImage * obstaclesMask = cvCreateImage(size, IPL_DEPTH_8U, 1);

    double k = 0.15;

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

            levels[(int)val]++;
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
    cvThreshold(obstaclesMask, obstaclesMask, maxThresh, 255, CV_THRESH_BINARY_INV);
    cvRectangle(obstaclesMask, cvPoint(0, 0), cvPoint(size.width - 1, 100), cvScalar(0), CV_FILLED);

    cvNamedWindow("ObstacleQ", 1);
    cvShowImage("ObstacleQ", obstaclesMask);

    cvErode(obstaclesMask, obstaclesMask, 0, 2);

    //cvNamedWindow("ObstacleQEroded", 1);
    //cvShowImage("ObstacleQEroded", obstaclesMask);

    cvDilate(obstaclesMask, obstaclesMask, 0, 2);

    cvNamedWindow("ObstacleQDilated", 1);
    cvShowImage("ObstacleQDilated", obstaclesMask);//*/

    detectObstacles(obstaclesMask);

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