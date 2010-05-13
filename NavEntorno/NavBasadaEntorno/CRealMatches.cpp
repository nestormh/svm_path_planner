/* 
 * File:   CRealMatches.cpp
 * Author: neztol
 * 
 * Created on 22 de febrero de 2010, 16:19
 */

#include "CRealMatches.h"
#include "fast/cvfast.h"
#include "CRutaDB2.h"
#include "CRoadDetection.h"

#define MIN_DIST 15

#define SIZE1 cvSize(800, 600)
#define SIZE2 cvSize(640, 480)
#define SIZE3 cvSize(320, 240)
#define SIZE4 cvSize(160, 120)
#define SIZE5 cvSize(315, 240)


CRealMatches::CRealMatches(bool usePrevious) {
    currentPoint1 = cvPoint2D32f(-1, -1);
    currentIndex1 = -1;

    size = SIZE5;

    img1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    img2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    mask1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    mask2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    plinear = cvCreateImage(size, IPL_DEPTH_8U, 1);

    M1 = cvCreateMat(3, 3, CV_64F);
    M2 = cvCreateMat(3, 3, CV_64F);
    D1 = cvCreateMat(1, 5, CV_64F);
    D2 = cvCreateMat(1, 5, CV_64F);

    this->usePrevious = usePrevious;
    if (usePrevious) {
        img1Prev = cvCreateImage(size, IPL_DEPTH_8U, 1);
        img2Prev = cvCreateImage(size, IPL_DEPTH_8U, 1);
    }
    aco = new CAntColony(size);
    acoImg = cvCreateImage(size, IPL_DEPTH_8U, 3);
}

CRealMatches::CRealMatches(const CRealMatches& orig) {
}

CRealMatches::~CRealMatches() {
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&mask1);
    cvReleaseImage(&mask2);
    cvReleaseImage(&plinear);

    cvReleaseMat(&M1);
    cvReleaseMat(&M2);
    cvReleaseMat(&D1);
    cvReleaseMat(&D2);

    if (interestMask != NULL) {
        cvReleaseImage(&interestMask);
    }

    if (usePrevious) {
        cvReleaseImage(&img1Prev);
        cvReleaseImage(&img2Prev);
    }

    points1.clear();
    points2.clear();
    pairs.clear();

    delete aco;
    cvReleaseImage(&acoImg);
}

void onMouseTest1(int event, int x, int y, int flags, void * param) {
    CRealMatches * rm = (CRealMatches *) param;
    rm->onMouse1(event, x, y, flags, NULL);
}

void onMouseTest2(int event, int x, int y, int flags, void * param) {
    CRealMatches * rm = (CRealMatches *) param;
    rm->onMouse2(event, x, y, flags, NULL);
}

void CRealMatches::onMouse1(int event, int x, int y, int flags, void * param) {
    switch (event) {
        case CV_EVENT_LBUTTONDOWN:
        {
            /*double minDist = MIN_DIST;
            for (int i = 0; i < points1.size(); i++) {
                if ((points1.at(i).x == -1) && (points1.at(i).y == -1)) continue;

                double dist = sqrt(pow(points1.at(i).x - x, 2.0) + pow(points1.at(i).y - y, 2.0));
                if (dist < minDist) {
                    dist = minDist;
                    currentPoint1 = cvPoint2D32f(points1.at(i).x, points1.at(i).y);
                    currentIndex1 = i;
                }
            }*/

            paint();
            break;
        }
        case CV_EVENT_RBUTTONDOWN:
        {
            currentPoint1 = cvPoint2D32f(-1, -1);
            currentIndex1 = -1;

            double minDist = MIN_DIST;
            vector<t_Pair>::iterator pair = pairs.end();
            for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
                double dist = sqrt(pow(it->p1.x - x, 2.0) + pow(it->p1.y - y, 2.0));
                if (dist < minDist) {
                    dist = minDist;
                    pair = it;
                }
            }
            if (pair != pairs.end()) {
                points1.push_back(pair->p1);
                points2.push_back(pair->p2);
                pairs.erase(pair);
            }

            paint();
            break;
        }
    }
}

void CRealMatches::onMouse2(int event, int x, int y, int flags, void * param) {

    switch (event) {
        case CV_EVENT_LBUTTONDOWN:
        {
            /*if (currentIndex1 == -1)
                return;            

            CvPoint2D32f currentPoint2 = cvPoint2D32f(-1, -1);
            int currentIndex2 = -1;
            double minDist = MIN_DIST;
            for (int i = 0; i < points2.size(); i++) {
                if ((points2.at(i).x == -1) && (points2.at(i).y == -1)) continue;

                double dist = sqrt(pow(points2.at(i).x - x, 2.0) + pow(points2.at(i).y - y, 2.0));
                if (dist < minDist) {
                    dist = minDist;
                    currentPoint2 = cvPoint2D32f(points2.at(i).x, points2.at(i).y);
                    currentIndex2 = i;
                }
            }

            if (currentIndex2 == -1)
                return;

            t_Pair pair;
            pair.p1 = cvPoint2D32f(currentPoint1.x, currentPoint1.y);
            pair.p2 = cvPoint2D32f(currentPoint2.x, currentPoint2.y);

            points1[currentIndex1] = cvPoint2D32f(-1, -1);
            points2[currentIndex2] = cvPoint2D32f(-1, -1);
            pairs.push_back(pair);
            currentIndex1 = -1;
            currentPoint1 = cvPoint2D32f(-1, -1);*/

            paint();

            break;
        }
        case CV_EVENT_RBUTTONDOWN:
        {
            currentPoint1 = cvPoint2D32f(-1, -1);
            currentIndex1 = -1;

            double minDist = MIN_DIST;
            vector<t_Pair>::iterator pair = pairs.end();
            for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
                double dist = sqrt(pow(it->p2.x - x, 2.0) + pow(it->p2.y - y, 2.0));
                if (dist < minDist) {
                    dist = minDist;
                    pair = it;
                }
            }
            if (pair != pairs.end()) {
                points1.push_back(pair->p1);
                points2.push_back(pair->p2);
                pairs.erase(pair);
            }

            paint();

            break;
        }
    }

}

inline void CRealMatches::getPoints(IplImage * img, vector<CvPoint2D32f> &points) {
    int nCorners = 500;
    CvPoint2D32f * tmpPoints = new CvPoint2D32f[nCorners];

    IplImage * eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
    IplImage * tmp = cvCreateImage(size, IPL_DEPTH_32F, 1);

    // Corners are obtained
    cvGoodFeaturesToTrack(img, eigen, tmp, tmpPoints, &nCorners,
            0.01, MIN_DIST, NULL, 6, 0, 0.04);

    // Add corners to the final points detected    
    for (int i = 0; i < nCorners; i++) {
        points.push_back(tmpPoints[i]);
    }

    delete tmpPoints;
    cvReleaseImage(&eigen);
    cvReleaseImage(&tmp);
}

inline void CRealMatches::getOflow(IplImage * img1, IplImage * img2, vector<CvPoint2D32f> points, vector<t_Pair> &pairs) {

    int nCorners = points.size();
    CvPoint2D32f * pointsOrig = new CvPoint2D32f[nCorners];
    CvPoint2D32f * pointsDest = new CvPoint2D32f[nCorners];

    int i = 0;
    for (vector<CvPoint2D32f>::iterator it = points.begin(); it != points.end(); it++, i++) {
        pointsOrig[i] = *it;
    }

    int pSize = 5;
    CvSize pyramidSize = cvSize(pSize, pSize);
    int depth = 4;

    CvSize pyramidImageSize = cvSize(size.width + 8, size.height / 3);
    IplImage * pyramidImage1 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
    IplImage * pyramidImage2 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
    char * status0 = new char[nCorners];

    cvCalcOpticalFlowPyrLK(img1, img2, pyramidImage1, pyramidImage2,
            pointsOrig, pointsDest, nCorners,
            pyramidSize, depth, status0, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);
    //pyramidSize, depth, status0, 0, cvTermCriteria(CV_TERMCRIT_EPS, 5, 1), 0);

    for (int i = 0; i < nCorners; i++) {
        if (status0[i] != 0) {
            t_Pair pair;
            pair.p1 = pointsOrig[i];
            pair.p2 = pointsDest[i];
            pairs.push_back(pair);
        }
    }

    delete pointsOrig;
    delete pointsDest;
    delete status0;

    cvReleaseImage(&pyramidImage1);
    cvReleaseImage(&pyramidImage2);
}

inline void CRealMatches::fusePairs(vector<t_Pair> pairs1, vector<t_Pair> pairs2, bool crossed) {
    if (crossed == true) {
        for (vector<t_Pair>::iterator it1 = pairs1.begin(); it1 != pairs1.end(); it1++) {
            for (vector<t_Pair>::iterator it2 = pairs2.begin(); it2 != pairs2.end(); it2++) {
                if (((int) it1->p1.x == (int) it2->p2.x) &&
                        ((int) it1->p2.x == (int) it2->p1.x) &&
                        ((int) it1->p1.y == (int) it2->p2.y) &&
                        ((int) it1->p2.y == (int) it2->p1.y)) {

                    pairs.push_back(*it1);
                    break;
                }
            }
        }
    } else {
        for (vector<t_Pair>::iterator it = pairs1.begin(); it != pairs1.end(); it++) {
            pairs.push_back(*it);
        }
        for (vector<t_Pair>::iterator it = pairs2.begin(); it != pairs2.end(); it++) {
            pairs.push_back(*it);
        }
    }
}

inline void CRealMatches::removeOutliers(CvMat **points1, CvMat **points2, CvMat *status) {
    CvMat *points1_ = *points1;
    CvMat *points2_ = *points2;
    int count = 0;
    for (int i = 0; i < status->cols; i++)
        if (cvGetReal2D(status, 0, i))
            count++;
    if (!count) { // no inliers
        *points1 = NULL;
        *points2 = NULL;
    } else {
        *points1 = cvCreateMat(1, count, CV_32FC2);
        *points2 = cvCreateMat(1, count, CV_32FC2);
        int j = 0;
        for (int i = 0; i < status->cols; i++) {
            if (cvGetReal2D(status, 0, i)) {
                (*points1)->data.fl[j * 2] = points1_->data.fl[i * 2];
                //p1->x
                (*points1)->data.fl[j * 2 + 1] = points1_->data.fl[i * 2 + 1];
                //p1->y
                (*points2)->data.fl[j * 2] = points2_->data.fl[i * 2];
                //p2->x
                (*points2)->data.fl[j * 2 + 1] = points2_->data.fl[i * 2 + 1];
                //p2->y
                j++;
            }
        }
    }
    cvReleaseMat(&points1_);
    cvReleaseMat(&points2_);
}

inline void CRealMatches::cleanRANSAC(int method, vector<t_Pair> &pairs) {
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

inline void CRealMatches::cleanPairsByDistance(vector<t_Pair> input, vector<t_Pair> &pairs) {
    pairs.clear();
    for (vector<t_Pair>::iterator it = input.begin(); it != input.end(); it++) {
        bool add = true;
        for (vector<t_Pair>::iterator it2 = pairs.begin(); it2 != pairs.end(); it2++) {
            double a1 = it->p1.x - it2->p1.x;
            double b1 = it->p1.y - it2->p1.y;
            double a2 = it->p2.x - it2->p2.x;
            double b2 = it->p2.y - it2->p2.y;
            double dist1Sqr = a1 * a1 + b1 * b1;
            double dist2Sqr = a2 * a2 + b2 * b2;

            if ((dist1Sqr < MIN_DIST_SQR) || (dist2Sqr < MIN_DIST_SQR)) {
                add = false;
                break;
            }
        }
        if (add) {
            pairs.push_back(*it);
        }
    }

    /*for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        for (vector<t_Pair>::iterator it2 = it + 1; it2 != pairs.end(); it2++) {
            double a1 = it->p1.x - it2->p1.x;
            double b1 = it->p1.y - it2->p1.y;
            double a2 = it->p2.x - it2->p2.x;
            double b2 = it->p2.y - it2->p2.y;
            double dist1Sqr = a1 * a1 + b1 * b1;
            double dist2Sqr = a2 * a2 + b2 * b2;

            if ((dist1Sqr < MIN_DIST_SQR) || (dist2Sqr < MIN_DIST_SQR)) {
                pairs.erase(it2);
            }
        }
    }*/
}

inline void CRealMatches::paint(char * img1Name, char * img2Name, char * plinearName, char * diffName) {    
    IplImage * tmpImg1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * tmpImg2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * resta = cvCreateImage(size, IPL_DEPTH_8U, 1);

    cvCvtColor(img1, tmpImg1, CV_GRAY2BGR);
    cvCvtColor(img2, tmpImg2, CV_GRAY2BGR);

    /*for (vector<CvPoint2D32f>::iterator it = points1.begin(); it != points1.end(); it++) {
        cvCircle(tmpImg1, cvPointFrom32f(*it), 2, cvScalar(255, 0, 0), 1);
    }
    for (vector<CvPoint2D32f>::iterator it = points2.begin(); it != points2.end(); it++) {
        cvCircle(tmpImg2, cvPointFrom32f(*it), 2, cvScalar(255, 0, 0), 1);
    }//*/

    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
    //for (vector<t_Pair>::iterator it = tmpPairs.begin(); it != tmpPairs.end(); it++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        cvCircle(tmpImg1, cvPointFrom32f(it->p1), 2, color, -1);
        cvCircle(tmpImg2, cvPointFrom32f(it->p2), 2, color, -1);
    }//*/

    //cvCircle(tmpImg1, cvPointFrom32f(currentPoint1), 4, cvScalar(0, 255, 0), 1);//*/

    cvShowImage(img1Name, tmpImg1);
    cvShowImage(img2Name, tmpImg2);

    cvZero(resta);
    cvCopy(plinear, resta, mask1);
    cvCopy(resta, plinear);

    cvShowImage(plinearName, plinear);

    cvZero(resta);
    cvCopy(img1, resta, mask1);
    cvAbsDiff(resta, plinear, resta);

    cvShowImage(diffName, resta);
    cvShowImage("Mask", mask1);//*/

    /*if (usePrevious) {
        IplImage * prev1a = cvCreateImage(size, IPL_DEPTH_8U, 3);
        IplImage * prev1b = cvCreateImage(size, IPL_DEPTH_8U, 3);
        IplImage * prev2a = cvCreateImage(size, IPL_DEPTH_8U, 3);
        IplImage * prev2b = cvCreateImage(size, IPL_DEPTH_8U, 3);
        IplImage * prev1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
        IplImage * prev2 = cvCreateImage(size, IPL_DEPTH_8U, 3);

        cvCvtColor(img1Prev, prev1a, CV_GRAY2BGR);
        cvCvtColor(img2Prev, prev2a, CV_GRAY2BGR);
        cvCvtColor(img1Prev, prev1, CV_GRAY2BGR);
        cvCvtColor(img2Prev, prev2, CV_GRAY2BGR);
        cvCvtColor(img1, prev1b, CV_GRAY2BGR);
        cvCvtColor(img2, prev2b, CV_GRAY2BGR);

        for (vector<t_Pair>::iterator it = pairs1.begin(); it != pairs1.end(); it++) {        
            CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
            cvCircle(prev1a, cvPointFrom32f(it->p1), 2, color, -1);
            cvCircle(prev1b, cvPointFrom32f(it->p2), 2, color, -1);
        }
        for (vector<t_Pair>::iterator it = pairs2.begin(); it != pairs2.end(); it++) {        
            CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
            cvCircle(prev2a, cvPointFrom32f(it->p1), 2, color, -1);
            cvCircle(prev2b, cvPointFrom32f(it->p2), 2, color, -1);
        }
        for (vector<t_Pair>::iterator it = pairsPrev.begin(); it != pairsPrev.end(); it++) {
            CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
            cvCircle(prev1, cvPointFrom32f(it->p1), 2, color, -1);
            cvCircle(prev2, cvPointFrom32f(it->p2), 2, color, -1);
        }

        cvShowImage("Prev1a", prev1a);
        cvShowImage("Prev1b", prev1b);
        cvShowImage("Prev2a", prev2a);
        cvShowImage("Prev2b", prev2b);
        cvShowImage("Prev1", prev1);
        cvShowImage("Prev2", prev2);

        cvReleaseImage(&prev1a);
        cvReleaseImage(&prev1b);
        cvReleaseImage(&prev2a);
        cvReleaseImage(&prev2b);
        cvReleaseImage(&prev1);
        cvReleaseImage(&prev2);
    }//*/
    
    cvReleaseImage(&tmpImg1);
    cvReleaseImage(&tmpImg2);
    cvReleaseImage(&resta);
}

inline void CRealMatches::testSurf(IplImage * img1, IplImage * img2) {
    clock_t myTime = clock();

    CvSeq *kp1 = NULL, *kp2 = NULL;
    CvSeq *desc1 = NULL, *desc2 = NULL;
    CvMemStorage *storage = cvCreateMemStorage(0);
    cvExtractSURF(img1, NULL, &kp1, &desc1, storage, cvSURFParams(300, 0));
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en surf1 = " << time << endl;
    myTime = clock();
    cvExtractSURF(img2, NULL, &kp2, &desc2, storage, cvSURFParams(300, 0));
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en surf2 = " << time << endl;


    CvMat * points1, * points2;
    myTime = clock();
    bruteMatch(img1, img2, &points1, &points2, kp1, desc1, kp2, desc2);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en match = " << time << endl;

    myTime = clock();
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat *status = cvCreateMat(1, points1->cols, CV_8UC1);
    int fm_count = cvFindFundamentalMat(points1, points2, F, CV_FM_RANSAC, 1., 0.99, status);
    removeOutliers(&points1, &points2, status);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en RANSAC = " << time << endl;

    CvScalar p;

    for (int i = 0; i < points1->cols; i++) {
        t_Pair pair;

        p = cvGet2D(points1, 0, i);
        pair.p1 = cvPoint2D32f(p.val[0], p.val[1]);
        p = cvGet2D(points2, 0, i);
        pair.p2 = cvPoint2D32f(p.val[0], p.val[1]);

        pairs.push_back(pair);
    }

    //showPairs2("surf", img1, img2, points1, points2);
}

// http://mirror2image.wordpress.com/2009/01/25/fast-with-surf-descriptor/

inline void CRealMatches::testFast(IplImage * img, vector<CvPoint2D32f> &points) {
    int inFASTThreshhold = 10; //30; //80
    int inNpixels = 9;
    int inNonMaxSuppression = 1;

    CvPoint* corners;
    int numCorners;

    cvCornerFast(img, inFASTThreshhold, inNpixels, inNonMaxSuppression, &numCorners, & corners);

    points.clear();
    for (int i = 0; i < numCorners; i++) {
        if (cvGetReal2D(mask1, corners[i].y, corners[i].x) == 255)
            points.push_back(cvPointTo32f(corners[i]));
    }

    delete corners;
}

inline void CRealMatches::updatePrevious() {
    points1.clear();
    points2.clear();
    pairs1.clear();
    pairs2.clear();
    for (vector<t_Pair>::iterator it = pairsPrev.begin(); it != pairsPrev.end(); it++) {
        points1.push_back(it->p1);
        points2.push_back(it->p2);
    }    
    oFlow(points1, pairs1, img1Prev, img1);
    oFlow(points2, pairs2, img2Prev, img2);

    tmpPairs.clear();
    for (vector<t_Pair>::iterator it = pairs1.begin(); it != pairs1.end(); it++) {
        t_Pair pair;
        pair.p1 = it->p2;
        for (vector<t_Pair>::iterator it2 = pairsPrev.begin(); it2 != pairsPrev.end(); it2++) {
            if ((it2->p1.x == it->p1.x) && (it2->p1.y == it->p1.y)) {
                for (vector<t_Pair>::iterator it3 = pairs2.begin(); it3 != pairs2.end(); it3++) {
                    if ((it3->p1.x == it2->p2.x) && (it3->p1.y == it2->p2.y)) {
                        pair.p2 = it3->p2;
                        break;
                    }
                }
                break;
            }
        }
        tmpPairs.push_back(pair);
    }
    if (tmpPairs.size() < MIN_NFEAT) {
        cerr << "No se han encontrado puntos suficientes para hacer el emparejamiento" << endl;
        return;
    }
    cleanRANSAC(CV_FM_RANSAC, tmpPairs);
    for (vector<t_Pair>::iterator it = tmpPairs.begin(); it != tmpPairs.end(); it++) {
        pairs.push_back(*it);
    }
}

inline void CRealMatches::findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures) {
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

inline void CRealMatches::oFlow(vector <CvPoint2D32f> &points1, vector <t_Pair> &pairs, IplImage * &img1, IplImage * &img2) {
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

inline void CRealMatches::setMaskFromPoints(IplImage * &mask, int index) {
    CvPoint* pts = (CvPoint*) malloc(pairs.size() * sizeof (pts[0]));
    int* hull = (int*) malloc(pairs.size() * sizeof (hull[0]));
    CvMat point_mat = cvMat(1, pairs.size(), CV_32SC2, pts);
    CvMat hull_mat = cvMat(1, pairs.size(), CV_32SC1, hull);

    CvPoint pt;
    if (index == 0) {
        for (int i = 0; i < pairs.size(); i++) {
            pts[i] = cvPointFrom32f(pairs.at(i).p2);
        }
    } else {
        for (int i = 0; i < pairs.size(); i++) {
            pts[i] = cvPointFrom32f(pairs.at(i).p1);
        }
    }

    cvConvexHull2(&point_mat, &hull_mat, CV_CLOCKWISE, 0);
    int hullcount = hull_mat.cols;

    pt = pts[hull[hullcount - 1]];

    CvPoint * poly = new CvPoint[hullcount];
    for (int i = 0; i < hullcount; i++) {
        poly[i] = pt;
        pt = pts[hull[i]];
    }

    cvZero(mask);
    cvFillConvexPoly(mask, poly, hullcount, cvScalar(255));

    cvErode(mask, mask);

    delete pts;
    delete hull;
    delete poly;
}

inline void CRealMatches::remap(CImageRegistration ir) {
    CvPoint2D32f * points1 = new CvPoint2D32f[pairs.size()];
    CvPoint2D32f * points2 = new CvPoint2D32f[pairs.size()];

    /*IplImage * img1 = cvCreateImage(cvSize(160, 120), IPL_DEPTH_8U, 1);
    IplImage * img2 = cvCreateImage(cvSize(160, 120), IPL_DEPTH_8U, 1);
    cvResize(this->img1, img1);
    cvResize(this->img2, img2);*/

    int i = 0;
    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++, i++) {
        points1[i] = it->p1;
        points2[i] = it->p2;
        //points1[i] = cvPoint2D32f(it->p1.x / 2.0, it->p1.y / 2.0);
        //points2[i] = cvPoint2D32f(it->p2.x / 2.0, it->p2.y / 2.0);
    }

    IplImage * imgTPS = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvCopyImage(img2, imgTPS);

    ir.TPS(imgTPS, points1, points2, pairs.size());

    cvNamedWindow("TPS", 1);
    cvShowImage("TPS", imgTPS);

    cvAbsDiff(img1, imgTPS, imgTPS);
    cvNamedWindow("Diff", 1);
    cvShowImage("Diff", imgTPS);

    cvReleaseImage(&imgTPS);
    delete points1;
    delete points2;
}

inline void CRealMatches::mainTest() {
    //getPoints(img1, points1);
    //getPoints(img2, points2);
    clock_t myTime = clock();
    cvCvtColor(img2, acoImg, CV_GRAY2BGR);
    //CvPoint * road = aco->iterate(acoImg);
    CvPoint * road = new CvPoint[4];
    road[0] = cvPoint(0, size.height - 70);
    road[1] = cvPoint(184, 102);
    road[2] = cvPoint(207, 102);
    road[3] = cvPoint(size.width - 1, size.height - 70);
    int npts = 4;
    cvZero(interestMask);
    cvFillPoly(interestMask, &road, &npts, 1, cvScalar(255));
    road[1].x = 0;
    road[2].x = size.width - 1;
    cvZero(mask1);
    cvFillPoly(mask1, &road, &npts, 1, cvScalar(255));
    delete road;    
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en ACO = " << time << endl;

    myTime = clock();
    testFast(img1, points1);
    //testFast(img2, points2);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en fast = " << time << endl;

    myTime = clock();
    //getOflow(img1, img2, points1, pairs1);
    //getOflow(img2, img1, points2, pairs2);
    oFlow(points1, pairs, img1, img2);
    //testSurf(img1, img2);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en oFlow = " << time << endl;

    if (pairs.size() < MIN_NFEAT) {
        cerr << "No se han encontrado puntos suficientes para hacer el emparejamiento" << endl;
        return;
    }

    /*myTime = clock();
    fusePairs(pairs1, pairs2, false);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en fusion = " << time << endl;//*/

    myTime = clock();    
    cleanRANSAC(CV_FM_RANSAC, pairs);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en ransac = " << time << endl;
    /*myTime = clock();
    cleanRANSAC(CV_FM_LMEDS);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en lmeds = " << time << endl;//*/
    //cleanRANSAC(CV_FM_RANSAC);

    /*myTime = clock();
    remap(ir);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en TPS = " << time << endl;//*/

    if (usePrevious) {
        updatePrevious();
    }
    //cleanPairsByDistance(pairs, pairs);

    if (pairs.size() < MIN_NFEAT) {
        cerr << "No hay puntos suficientes tras la limpieza para hacer el emparejamiento" << endl;
        return;
    }

    myTime = clock();
    setMaskFromPoints(mask1, 1);
    //cvErode(mask1, mask1);
    //setMaskFromPoints(mask2, 0);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en mask = " << time << endl; //*/

    //drawTriangles("Delaunay1", "Delaunay1", true);
    //drawTriangles("Delaunay2", "Delaunay2", false);
    
    //cleanByTriangles();
    //drawTriangles("Delaunay3", "Delaunay3", true);
    //drawTriangles("Delaunay4", "Delaunay4", false);

    myTime = clock();
    //cvSmooth(img1, img1, CV_GAUSSIAN, 3);
    //cvSmooth(img2, img2, CV_GAUSSIAN, 3);    
    pieceWiseLinear();
    /*CViewMorphing wm(cvGetSize(img1));
    IplImage * img1C = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    IplImage * img2C = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    cvCvtColor(img1, img1C, CV_GRAY2BGR);
    cvCvtColor(img2, img2C, CV_GRAY2BGR);
    wm.points1 = new CvPoint2D32f[pairs.size()];
    wm.points2 = new CvPoint2D32f[pairs.size()];
    wm.numberOfFeatures = pairs.size();
    for (int i = 0; i < pairs.size(); i++) {
        wm.points1[i] = pairs.at(i).p1;
        wm.points2[i] = pairs.at(i).p2;
    }    
    wm.pieceWiseLinear(img2, img1, img1C, img2C, mask2);
    cvReleaseImage(&img1C);
    cvReleaseImage(&img2C);
    cvCvtColor(wm.warpedImg, plinear, CV_BGR2GRAY);
    cvThreshold(plinear, mask1, 0, 255, CV_THRESH_BINARY);//*/
    //cvAnd(mask1, mask2, mask1);
    cvNamedWindow("plinear2", 1);
    cvShowImage("plinear2", plinear);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en pieceWise = " << time << endl; //*/
    //wm(pairs, img1, img2);    

    myTime = clock();
    if (interestMask != NULL) {
        cvAnd(mask1, interestMask, mask1);
    }
    if (cvCountNonZero(mask1) == 0) {
        cerr << "El área de trabajo es demasiado pequeña" << endl;
        return;
    }
    calcPCA(img1, plinear, mask1);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en calcPCA = " << time << endl; //*/
    
    myTime = clock();
    //obstacleDetectionChauvenet(plinear, mask1);
    obstacleDetectionQuartile(plinear, mask1);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en obstacleDetection = " << time << endl; //*/

    //test3D();
    //test3D_2();

    /*CViewMorphing vm(cvGetSize(img1));
    vm.points1 = new CvPoint2D32f[pairs.size()];
    vm.points2 = new CvPoint2D32f[pairs.size()];
    for (int i = 0; i < pairs.size(); i++) {
        vm.points1[i] = pairs.at(i).p1;
        vm.points2[i] = pairs.at(i).p2;
    }
    IplImage * color1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    IplImage * color2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    cvCvtColor(img1, color1, CV_GRAY2BGR);
    cvCvtColor(img2, color2, CV_GRAY2BGR);
    vm.numberOfFeatures = pairs.size();
    vm.pieceWiseLinear(img1, img2, color1, color2, mask);*/

    myTime = clock();
    setMaskFromPoints(mask2, 0);
    paint();
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en paint = " << time << endl;//*/

    /*pairs.clear();
    testSurf(img1, img2);
    paint("surf1", "surf2");*/

    //cleanRANSAC(CV_FM_LMEDS);
    //testFast(img1, points1);
    //testFast(img2, points2);

    //paint();

    if (usePrevious) {
        cvCopyImage(img1, img1Prev);
        cvCopyImage(img2, img2Prev);
        pairsPrev.clear();
        for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
            pairsPrev.push_back(*it);
        }
    }
}

void CRealMatches::startTest(string path, string filename, string testName) {

    //calibrateCameras();

    string filepath = path;
    filepath += filename;
    string imgBasePath = path;

    ifstream ifs(filepath.c_str(), ifstream::in);

    char line[1024];
    ifs.getline(line, 1024);
    imgBasePath += line;
    imgBasePath += ".JPG";
    cout << "Imagen = " << imgBasePath << endl;
    ifs.getline(line, 1024);

    cvNamedWindow("Img1", 1);
    cvNamedWindow("Img2", 1);
    cvNamedWindow("Mask", 1);

    if (usePrevious) {
        cvNamedWindow("Prev1a", 1);
        cvNamedWindow("Prev1b", 1);
        cvNamedWindow("Prev2a", 1);
        cvNamedWindow("Prev2b", 1);
        cvMoveWindow("Img1", 0, 0);
        cvMoveWindow("Prev1a", 0, -1);
        cvMoveWindow("Prev1b", 0, -1);
        cvMoveWindow("Img2", size.width + 10, 0);
        cvMoveWindow("Prev2a", size.width + 10, 0);
        cvMoveWindow("Prev2b", size.width + 10, 0);
    }

    cvSetMouseCallback("Img1", onMouseTest1, this);
    cvSetMouseCallback("Img2", onMouseTest2, this);
    
    while (ifs.good()) {        
        int dist, ang;
        ifs >> dist;
        ifs >> ang;
        ifs.ignore(2);
        ifs.getline(line, 1024);
        //if ((abs(dist) > 1) || (abs(ang) > 225)) continue;
        if (abs(dist) > 1) continue;
        if (abs(ang) > 10) continue;
        if ((dist < 0) && (ang < 0)) continue;
        if ((dist > 0) && (ang > 0)) continue;

        IplImage * img1L = cvLoadImage(imgBasePath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        cvResize(img1L, img1);
        cvReleaseImage(&img1L);

        cout << "Tomando " << dist << ", " << ang << endl;
        string imgPath = path + line + ".JPG";
        IplImage * img2L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        cvResize(img2L, img2);
        cvReleaseImage(&img2L);

        clock_t myTime = clock();
        mainTest();
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;

        if (cvWaitKey(0) == 27)
            exit(0);

    }    
}

void CRealMatches::startTest2() {

    string path = "/home/neztol/doctorado/Datos/imagenesSueltas/";
    string filepath = path;
    filepath += "datosTmp.txt";

    ifstream ifs(filepath.c_str(), ifstream::in);

    char line[1024];

    cvNamedWindow("Img1", 1);
    cvNamedWindow("Img2", 1);

    //cvSetMouseCallback("Img1", onMouseTest1, this);
    //cvSetMouseCallback("Img2", onMouseTest2, this);

    while (ifs.good()) {
        ifs.getline(line, 1024);
        string imgPath = path + line;
        cout << imgPath << ", ";        
        IplImage * img1L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        cvResize(img1L, img1);
        //cvSmooth(img1, img1, CV_GAUSSIAN, 3);
        cvReleaseImage(&img1L);

        ifs.getline(line, 1024);
        imgPath = path + line;
        cout << imgPath << endl;
        IplImage * img2L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        cvResize(img2L, img2);
        //cvSmooth(img2, img2, CV_GAUSSIAN, 3);
        cvReleaseImage(&img2L);

        clock_t myTime = clock();
        mainTest();
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;

        if (cvWaitKey(0) == 27)
            exit(0);

    }    
}

void CRealMatches::startTest3() {
    cvNamedWindow("Img1", 1);
    cvNamedWindow("Img2", 1);
    cvNamedWindow("Mask", 1);
    cvNamedWindow("PLinear", 1);
    cvNamedWindow("Resta", 1);

    cvMoveWindow("Img1", 0, 0);
    cvMoveWindow("Img2", size.width + 10, 0);
    cvMoveWindow("Mask", (size.width + 10) * 2, 0);
    cvMoveWindow("PLinear", (size.width + 10) * 2, size.height + 20);
    cvMoveWindow("Resta", (size.width + 10) * 2, (size.height + 20) * 2);

    /*if (usePrevious) {
        cvNamedWindow("Prev1a", 1);
        cvNamedWindow("Prev1b", 1);
        cvNamedWindow("Prev2a", 1);
        cvNamedWindow("Prev2b", 1);
        cvNamedWindow("Prev1", 1);
        cvNamedWindow("Prev2", 1);
        
        cvMoveWindow("Prev1a", 0, (size.height + 20) * 2);
        cvMoveWindow("Prev1b", 0, size.height + 20);        
        cvMoveWindow("Prev2a", size.width + 10, (size.height + 20) * 2);
        cvMoveWindow("Prev2b", size.width + 10, size.height + 20);
        cvMoveWindow("Prev1", (size.width + 10) * 2, 0);
        cvMoveWindow("Prev2", (size.width + 10) * 3, 0);
    }//*/


    string path = "/home/neztol/doctorado/Datos/estereo/";

    vector<string> pruebas;
    //pruebas.push_back("jesusYeray1_");
    //pruebas.push_back("jesusYeray2_");
    pruebas.push_back("jesusYeray3_");
    pruebas.push_back("jesusYeray4_");
    pruebas.push_back("jesusYeray5_");

    string leftPath;
    string rightPath;
    fstream fin;
    char buffer[10];

    for (vector<string>::iterator it = pruebas.begin(); it != pruebas.end(); it++) {
        int index = 1;
        while (true) {
            leftPath = path;
            leftPath += *it;
            leftPath += "left_";
            sprintf(buffer,"%d", index);
            leftPath += buffer;
            leftPath += ".bmp";            
            
            fin.open(leftPath.c_str(), ios::in);
            if(! fin.is_open() ) {
                break;
            }
            fin.close();

            rightPath = path;
            rightPath += *it;
            rightPath += "right_";
            sprintf(buffer,"%d", index);
            rightPath += buffer;
            rightPath += ".bmp";

            IplImage * img1L = cvLoadImage(leftPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cvResize(img1L, img1);
            //cvSmooth(img1, img1, CV_GAUSSIAN, 3);
            cvReleaseImage(&img1L);
            IplImage * img2L = cvLoadImage(rightPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cvResize(img2L, img2);
            //cvSmooth(img2, img2, CV_GAUSSIAN, 3);
            cvReleaseImage(&img2L);

            clock_t myTime = clock();
            mainTest();
            time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
            cout << "Tiempo TOTAL = " << time << endl;

            if (index != 1) {
                int key = cvWaitKey(10);
                if (key == 27)
                    exit(0);
                if (key == 98)
                    break;
            }

            index++;
        }        
    }       
}

void CRealMatches::startTest4() {
    cvNamedWindow("Img1", 1);
    cvNamedWindow("Img2", 1);
    cvNamedWindow("Mask", 1);
    cvNamedWindow("PLinear", 1);
    cvNamedWindow("Resta", 1);

    cvMoveWindow("Img1", 0, 0);
    cvMoveWindow("Img2", size.width + 10, 0);
    cvMoveWindow("Mask", (size.width + 10) * 2, 0);
    cvMoveWindow("PLinear", (size.width + 10) * 2, size.height + 20);
    cvMoveWindow("Resta", (size.width + 10) * 2, (size.height + 20) * 2);

    /*if (usePrevious) {
        cvNamedWindow("Prev1a", 1);
        cvNamedWindow("Prev1b", 1);
        cvNamedWindow("Prev2a", 1);
        cvNamedWindow("Prev2b", 1);
        cvNamedWindow("Prev1", 1);
        cvNamedWindow("Prev2", 1);

        cvMoveWindow("Prev1a", 0, (size.height + 20) * 2);
        cvMoveWindow("Prev1b", 0, size.height + 20);
        cvMoveWindow("Prev2a", size.width + 10, (size.height + 20) * 2);
        cvMoveWindow("Prev2b", size.width + 10, size.height + 20);
        cvMoveWindow("Prev1", (size.width + 10) * 2, 0);
        cvMoveWindow("Prev2", (size.width + 10) * 3, 0);
    }//*/


    string path = "/home/neztol/doctorado/Datos/MRPT_Data/";

    vector<string> pruebas;
    pruebas.push_back("malaga2009_campus_RT/");

    string leftPath;
    string rightPath;    

    for (vector<string>::iterator it = pruebas.begin(); it != pruebas.end(); it++) {
        leftPath = path;
        leftPath += *it;
        leftPath += "left.txt";
        rightPath = path;
        rightPath += *it;
        rightPath += "right.txt";

        ifstream ifs1(leftPath.c_str(), ifstream::in);
        ifstream ifs2(rightPath.c_str(), ifstream::in);

        char line1[1024];
        char line2[1024];

        while (ifs1.good() && ifs2.good()) {
            ifs1.getline(line1, 1024);
            ifs2.getline(line2, 1024);

            string imgPath = path + (*it) + "Images/" + line1;
            cout << imgPath << ", ";
            IplImage * img1L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cvResize(img1L, img1);
            //cvSmooth(img1, img1, CV_GAUSSIAN, 3);
            cvReleaseImage(&img1L);
            
            imgPath = path + (*it) + "Images/" + line2;
            cout << imgPath << endl;
            IplImage * img2L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cvResize(img2L, img2);
            //cvSmooth(img2, img2, CV_GAUSSIAN, 3);
            cvReleaseImage(&img2L);            

            clock_t myTime = clock();
            mainTest();
            time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
            cout << "Tiempo TOTAL = " << time << endl;

            int key = cvWaitKey(10);
            if (key == 27)
                exit(0);
            if (key == 32)
                cvWaitKey(0);

        }
    }
}

void CRealMatches::startTest5() {
    CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERBase2", "Rutas/pruebaITERConObs2", "/home/neztol/doctorado/Datos/DB");
    IplImage * imgDB;
    IplImage * imgRT;
    int indexRT;
    int indexDB;

    interestMask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    /*cvZero(interestMask);
    //cvFillPoly(CvArr* img, CvPoint** pts, int* npts, int contours, CvScalar color, int lineType=8, int shift=0)
    CvPoint * interest = new CvPoint[4];
    interest[0] = cvPoint(0, size.height - 1);
    //interest[1] = cvPoint(122, 104);
    //interest[2] = cvPoint(207, 104);
    interest[1] = cvPoint(0, 104);
    interest[2] = cvPoint(size.width - 1, 104);
    interest[3] = cvPoint(size.width - 1, size.height - 1);
    int npts = 4;
    cvFillPoly(interestMask, &interest, &npts, 1, cvScalar(255));
    cvNamedWindow("Interest", 1);
    cvShowImage("Interest", interestMask);//*/

    //cvNamedWindow("ImgDB", 1);
    //cvNamedWindow("ImgRT", 1);
    int index = 392;
    ruta.setCurrentPoint(index);
    while (true) {
        ruta.getNextImage(imgRT, imgDB);
        index++;

        if (cvCountNonZero(imgDB) == 0) {
            cvReleaseImage(&imgDB);
            cvReleaseImage(&imgRT);

            continue;
        }

        cvCopyImage(imgRT, img1);
        cvCopyImage(imgDB, img2);
        //cvShowImage("ImgDB", imgDB);
        //cvShowImage("ImgRT", imgRT);
        clock_t myTime = clock();
        mainTest();
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;
        cout << "Index = " << (index - 1) << endl;

        if (index == 630) exit(0);

        int key = cvWaitKey(0);
        if (key == 27)
            exit(0);
        if (key == 32)
            cvWaitKey(0);

        cvReleaseImage(&imgDB);
        cvReleaseImage(&imgRT);
    }
}

void CRealMatches::startTestACO() {
    CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERConObs2", "Rutas/pruebaITERBase2", "/home/neztol/doctorado/Datos/DB");
    IplImage * imgDB;
    IplImage * imgRT;
    IplImage * img = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * checkResults = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvNamedWindow("checkResults", 1);

    //cvNamedWindow("ImgDB", 1);
    //cvNamedWindow("ImgRT", 1);
    int index = 0;
    ruta.setCurrentPoint(index);
    CAntColony aco(size);
    while (true) {
        ruta.getNextImage(imgRT, imgDB);
        index++;

        clock_t myTime = clock();

        //cvCopyImage(imgRT, img1);
        //cvCopyImage(imgDB, img2);
        cvCvtColor(imgRT, img, CV_GRAY2BGR);
        CvPoint * poly = aco.iterate(img);

        cvZero(checkResults);
        cvLine(checkResults, poly[0], poly[1], cvScalar(0, 255, 0), 3);
        cvLine(checkResults, poly[2], poly[3], cvScalar(0, 255, 0), 3);
        cvShowImage("checkResults", checkResults);

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
    cvReleaseImage(&img);
}

void CRealMatches::startTestRoadDetection() {
    CRoadDetection rd(size);
    IplImage * maskRoad = cvCreateImage(size, IPL_DEPTH_8U, 1);
    int index = 0;
    while (true) {        
        clock_t myTime = clock();

        rd.detect(index, maskRoad);
        index++;

        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;
        cout << "Index = " << (index - 1) << endl;

        int key = cvWaitKey(20);
        if (key == 27)
            exit(0);
        if (key == 32)
            cvWaitKey(0);
    }
}

void CRealMatches::mainTest(IplImage * img1, IplImage * img2) {
    this->img1 = img1;
    this->img2 = img2;

    mainTest();
}