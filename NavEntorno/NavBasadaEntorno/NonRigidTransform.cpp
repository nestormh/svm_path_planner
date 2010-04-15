#include "ImageRegistration.h"

void CImageRegistration::TPS(IplImage * &img, CvPoint2D32f * points1, CvPoint2D32f * points2, int nFeat) {

    double * fCoefs, * gCoefs;
    clock_t myTime = clock();
    getCoefsAM(points1, points2, nFeat, fCoefs, gCoefs);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en getCoefsAM = " << time << endl;

    double u, v;
    myTime = clock();
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            calculateAM(cvPoint2D32f(j, i), points1, nFeat, fCoefs, gCoefs, u, v);            
            cvSetReal2D(remapX, i, j, u);
            cvSetReal2D(remapY, i, j, v);
        }
    }
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en calculateAM = " << time << endl;

    cvRemap(img, tps, remapX, remapY, CV_INTER_AREA+CV_WARP_FILL_OUTLIERS, cvScalar(0));
    cvCopyImage(tps, img);
}

inline void CImageRegistration::getCoefsAM(CvPoint2D32f * p1, CvPoint2D32f * p2, int nFeat, double * &coefs1, double * &coefs2) {

    CvMat * A = cvCreateMat(nFeat + 3, nFeat + 3, CV_64FC1);
    CvMat * Bu = cvCreateMat(nFeat + 3, 1, CV_64FC1);
    coefs1 = new double[nFeat + 3];
    CvMat Xu = cvMat(nFeat + 3, 1, CV_64FC1, coefs1);
    CvMat * Bv = cvCreateMat(nFeat + 3, 1, CV_64FC1);
    coefs2 = new double[nFeat + 3];
    CvMat Xv = cvMat(nFeat + 3, 1, CV_64FC1, coefs2);

    cvZero(A);
    cvZero(Bu);
    cvZero(Bv);
    for (int i = 0; i < nFeat; i++) {
        cvmSet(A, i, 0, 1);
        cvmSet(A, i, 1, p1[i].x);
        cvmSet(A, i, 2, p1[i].y);
        for (int j = 0; j < nFeat; j++) {
            if (i == j) continue;
            double r2 = pow(p1[i].x - p1[j].x, 2.0) + pow(p1[i].y - p1[j].y, 2.0);
            if (r2 != 0) {
                cvmSet(A, i, j + 3, r2 * log(r2));
            }
        }
        cvmSet(A, nFeat, i + 3, 1);
        cvmSet(A, nFeat + 1, i + 3, p1[i].x);
        cvmSet(A, nFeat + 2, i + 3, p1[i].y);

        cvmSet(Bu, i, 0, p2[i].x);
        cvmSet(Bv, i, 0, p2[i].y);
    }

    cvSolve(A, Bu, &Xu, CV_SVD);
    cvSolve(A, Bv, &Xv, CV_SVD);
}

inline void CImageRegistration::calculateAM(CvPoint2D32f point, CvPoint2D32f * p1, int nFeat, double * coefs1, double * coefs2, double &u, double &v) {
    u = coefs1[0] + coefs1[1] * point.x + coefs1[2] * point.y;
    v = coefs2[0] + coefs2[1] * point.x + coefs2[2] * point.y;

    for (int j = 0; j < nFeat; j++) {
        double r2 = (point.x - p1[j].x) * (point.x - p1[j].x) + (point.y - p1[j].y) * (point.y - p1[j].y);
        double ln = log(r2);

        u += r2 * ln * coefs1[j + 3];
        v += r2 * ln * coefs2[j + 3];
    }
}
