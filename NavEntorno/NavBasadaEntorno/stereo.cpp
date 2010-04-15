#include "CRealMatches.h"

void CRealMatches::test3D() {
    int nPairs = pairs.size();
    CvMat * p1 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * p2 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat* H1 = cvCreateMat(3, 3, CV_64FC1);
    CvMat* H2 = cvCreateMat(3, 3, CV_64FC1);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat * epilines = cvCreateMat(3, nPairs, CV_32FC1);
    CvMat *statusM = cvCreateMat(1, nPairs, CV_8UC1);

    for (int i = 0; i < nPairs; i++) {
        cvSet2D(p1, 0, i, cvScalar(pairs.at(i).p1.x, pairs.at(i).p1.y));
        cvSet2D(p2, 0, i, cvScalar(pairs.at(i).p2.x, pairs.at(i).p2.y));
    }

    cvFindFundamentalMat(p1, p2, F, CV_FM_RANSAC, 3., 0.99, statusM);

    cvComputeCorrespondEpilines(p1, 1, F, epilines);

    cvStereoRectifyUncalibrated(p1, p2, F, size, H1, H2, 5);

    // NOTA: R_rect = M*H*M
    // M = camera matrix
    // Falta por saber coeffs y M
    
}
