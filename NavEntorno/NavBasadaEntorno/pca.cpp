#include "CRealMatches.h"

void CRealMatches::calcPCA(IplImage * img1, IplImage * img2, IplImage * mask) {
    int length = cvCountNonZero(mask);
    CvMat * data = cvCreateMat(2, length, CV_64FC1);
    CvMat * data1 = cvCreateMatHeader(1, length, CV_64FC1);
    CvMat * data2 = cvCreateMatHeader(1, length, CV_64FC1);
    CvMat * corr = cvCreateMat(2, 2, CV_64FC1);
    CvMat * avg = cvCreateMat(1, 2, CV_64FC1);
    CvMat * eigenValues = cvCreateMat(1, 2, CV_64FC1);
    CvMat * eigenVectors = cvCreateMat(2, 2, CV_64FC1);
    CvMat * pcaData = cvCreateMat(2, length, CV_64FC1);
    CvMat * dataX = cvCreateMatHeader(1, length, CV_64FC1);
    CvMat * dataY = cvCreateMatHeader(1, length, CV_64FC1);
    CvMat * distPCA = cvCreateMat(img1->height, img1->width, CV_64FC1);
    CvScalar xMean, yMean, xSdv, ySdv;

    vector<CvPoint> origPos;

    cvGetRow(data, data1, 0);
    cvGetRow(data, data2, 1);    

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
    cvGetRow(pcaData, dataX, 1);
    cvGetRow(pcaData, dataY, 0);

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
        
        cvSetReal2D(plinear, p.y, p.x, abs(cvGetReal1D(dataY, i) - yMean.val[0]));
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

    origPos.clear();
}