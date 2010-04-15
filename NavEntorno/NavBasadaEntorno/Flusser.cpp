#include "ViewMorphing.h"

#define TITA_THRESH 0.1
#define AM_SUBDIVISIONS 2 //  AM_SUBDIVISIONS x AM_SUBDIVISIONS

void CViewMorphing::flusserTransform(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask) {    
    //oFlowMeshedAndDetectedFeatureTracker(img2, img1);
    AffineAndEuclidean(img1, img2, img1C, img2C, featureMask);
    //cleanUsingDelaunay(img1C, img2C);

    remapX = cvCreateMat(size.height, size.width, CV_32FC1);
    remapY = cvCreateMat(size.height, size.width, CV_32FC1);

    double * fCoefs, * gCoefs;
    double * fEstim, * gEstim;
    //getCoefsAM(points1, points2, numberOfFeatures, fCoefs, gCoefs);
    //calculateAM(points1, numberOfFeatures, fCoefs, gCoefs, fEstim, gEstim);

    flusser = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvZero(flusser);
    clock_t myTime = clock();
    trans(points1, points2, numberOfFeatures, fEstim, gEstim, cvPoint2D32f(0, 0), cvPoint2D32f(size.width - 1, size.height - 1), 0);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en Flusser = " << time << endl;
    /*for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            cvSetReal2D(remapX, i, j, j);
            cvSetReal2D(remapY, i, j, i);
        }
    }*/

    cvNamedWindow("Rectangle", 1);
    cvShowImage("Rectangle", flusser);
    cvRemap(img1C, flusser, remapX, remapY);
    cvNamedWindow("Flusser", 1);
    cvShowImage("Flusser", flusser);
    cvAbsDiff(img2C, flusser, flusser);
    cvNamedWindow("DiffFlusser", 1);
    cvShowImage("DiffFlusser", flusser);
    //cvReleaseImage(&flusser);
    warpedImg = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvRemap(img1C, warpedImg, remapX, remapY);
    mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvSet(mask, cvScalar(255));
    extraeObstaculos(img2C);

    
}

void CViewMorphing::trans(CvPoint2D32f * p1, CvPoint2D32f * p2, int nFeat, double * fEstim, double * gEstim, CvPoint2D32f ul, CvPoint2D32f lr, int level) {

    //cvRectangle(flusser, cvPointFrom32f(ul),cvPointFrom32f(lr), cvScalar(0, 255, 0));
    //nFeat = 3;
    double * pCoefs, * qCoefs;
    double * pEstim, * qEstim;
    clock_t myTime = clock();
    getCoefsAM(p1, p2, nFeat, pCoefs, qCoefs);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en obtener coeficientes = " << time << endl;
    //calculateAM(p1, nFeat, pCoefs, qCoefs, pEstim, qEstim);

    /*double tita = 0;
    for (int i = 0; i < nFeat; i++) {
        tita += pow(fEstim[i] - pEstim[i], 2.0) + pow(gEstim[i] - qEstim[i], 2.0);        
    }
    tita /= 2.0;    */

    //cout << nFeat << endl;
    //if (tita <= TITA_THRESH) {
    //if ((tita <= TITA_THRESH) && (level != 0)) {
    //if (nFeat < 8) {
        //cout << "Entro" << endl;
        //cout << "LR " << lr.x << ", " << lr.y << endl;
        //cout << "UL " << ul.x << ", " << ul.y << endl;
    myTime = clock();
        double u, v;
        for (int i = ul.y; i <= lr.y; i++) {
            //cout << "i " << i << endl;
            for (int j = ul.x; j <= lr.x; j++) {
                //cout << "j " << j << endl;
                //cout << "u = " << u << endl;
                calculateAM(cvPoint2D32f(j, i), p1, nFeat, pCoefs, qCoefs, u, v);
                cvSetReal2D(remapX, i, j, u);
                cvSetReal2D(remapY, i, j, v);
            }
        }
        time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo invertido en mapeo = " << time << endl;    /*} else {
        int incX = (int)(((lr.x - ul.x) + 1) / AM_SUBDIVISIONS);
        int incY = (int)(((lr.y - ul.y) + 1) / AM_SUBDIVISIONS);

        bool restX = (((int)(lr.x - ul.x) + 1) % AM_SUBDIVISIONS == 0)? false:true;
        bool restY = (((int)(lr.y - ul.y) + 1) % AM_SUBDIVISIONS == 0)? false:true;

        for (int i = 0; i < AM_SUBDIVISIONS; i++) {
            for (int j = 0; j < AM_SUBDIVISIONS; j++) {
                CvPoint2D32f myUL = cvPoint2D32f(ul.x + incX * i, ul.y + incY * j);
                CvPoint2D32f myLR = cvPoint2D32f(ul.x + incX * (i + 1) - 1, ul.y + incY * (j + 1) - 1);

                if ((i == AM_SUBDIVISIONS - 1) && (restX))
                    myLR.x += 1;
                if ((j == AM_SUBDIVISIONS - 1) && (restY))
                    myLR.y += 1;

                int newNFeat = 0;
                double * newfEstim = new double[nFeat];
                double * newgEstim = new double[nFeat];
                CvPoint2D32f * newP1 = new CvPoint2D32f[nFeat];
                CvPoint2D32f * newP2 = new CvPoint2D32f[nFeat];

                for (int k = 0; k < nFeat; k++) {
                    if ((p1[k].x >= myUL.x) && (p1[k].x <= myLR.x) &&
                            (p1[k].y >= myUL.y) && (p1[k].y <= myLR.y)) {
                        
                        newP1[newNFeat] = p1[k];
                        newP2[newNFeat] = p2[k];
                        newfEstim[newNFeat] = fEstim[k];
                        newgEstim[newNFeat] = gEstim[k];
                        newNFeat++;
                    }
                }

                //cout << "Caracteristicas encontradas: " << newNFeat << " de " << nFeat << endl;
                //cout << "Subdivisión: [" << myUL.x << ", " << myUL.y << "]->[" << myLR.x << ", " << myLR.y << "]->[" << endl;
                //if (newNFeat > 4) {
                //if (level == 0)
                    trans(newP1, newP2, newNFeat, newfEstim, newgEstim, myUL, myLR, level + 1);
                //}
            }
        }
    }*/
    
    //exit(0);
}

void CViewMorphing::getCoefsAM(CvPoint2D32f * p1, CvPoint2D32f * p2, int nFeat, double * &coefs1, double * &coefs2) {

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

void CViewMorphing::calculateAM(CvPoint2D32f * p1, int nFeat, double * coefs1, double * coefs2, double * &u, double * &v) {
    u = new double[nFeat];
    v = new double[nFeat];
    
    for (int i = 0; i < nFeat; i++) {
        u[i] = coefs1[0] + coefs1[1] * p1[i].x + coefs1[2] * p1[i].y;
        v[i] = coefs2[0] + coefs2[1] * p1[i].x + coefs2[2] * p1[i].y;

        for (int j = 0; j < nFeat; j++) {
            if (i == j) continue;

            double r2 = pow(p1[i].x - p1[j].x, 2.0) + pow(p1[i].y - p1[j].y, 2.0);

            u[i] += r2 * log(r2) * coefs1[j + 3];
            v[i] += r2 * log(r2) * coefs2[j + 3];
        }
    }
}

void CViewMorphing::calculateAM(CvPoint2D32f point, CvPoint2D32f * p1, int nFeat, double * coefs1, double * coefs2, double &u, double &v) {
    u = coefs1[0] + coefs1[1] * point.x + coefs1[2] * point.y;
    v = coefs2[0] + coefs2[1] * point.x + coefs2[2] * point.y;

    for (int j = 0; j < nFeat; j++) {
        double r2 = (point.x - p1[j].x) * (point.x - p1[j].x) + (point.y - p1[j].y) * (point.y - p1[j].y);
        double ln = log(r2);

        u += r2 * ln * coefs1[j + 3];
        v += r2 * ln * coefs2[j + 3];
    }
}