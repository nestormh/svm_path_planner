/* 
 * File:   CRealMatches.cpp
 * Author: neztol
 * 
 * Created on 22 de febrero de 2010, 16:19
 */

#include "CRealMatches.h"
#include "fast/cvfast.h"
#include "CRoadDetection.h"

#define MIN_DIST 15


CRealMatches::CRealMatches(bool usePrevious, CvSize sizeIn) {
    init(usePrevious, sizeIn);
}

void CRealMatches::init(bool usePrevious, CvSize sizeIn) {
    currentPoint1 = cvPoint2D32f(-1, -1);
    currentIndex1 = -1;

    size = sizeIn;

    img1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    img2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    smallImg1 = cvCreateImage(cvSize(size.width / 2, size.height / 2), IPL_DEPTH_8U, 1);
    smallImg2 = cvCreateImage(cvSize(size.width / 2, size.height / 2), IPL_DEPTH_8U, 1);
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

    lastObst = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvZero(lastObst);

    pointsMask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    // Points mask is created
    CvPoint * pMask = new CvPoint[4];
    pMask[0] = cvPoint(0, size.height - 1);
    pMask[1] = cvPoint(0, 102);
    pMask[2] = cvPoint(size.width - 1, 102);
    pMask[3] = cvPoint(size.width - 1, size.height - 1);
    int npts = 4;
    cvZero(pointsMask);
    cvFillPoly(pointsMask, &pMask, &npts, 1, cvScalar(255));
    delete pMask;

    roadMask = NULL;
}



CRealMatches::CRealMatches(const CRealMatches& orig) {
}

CRealMatches::~CRealMatches() {
    destroy();
}

void CRealMatches::destroy() {
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&mask1);
    cvReleaseImage(&mask2);
    cvReleaseImage(&plinear);

    cvReleaseImage(&smallImg1);
    cvReleaseImage(&smallImg2);

    cvReleaseMat(&M1);
    cvReleaseMat(&M2);
    cvReleaseMat(&D1);
    cvReleaseMat(&D2);

    if (roadMask != NULL) {
        cvReleaseImage(&roadMask);
    }

    if (usePrevious) {
        cvReleaseImage(&img1Prev);
        cvReleaseImage(&img2Prev);
    }

    points1.clear();
    points2.clear();
    pairs.clear();

    cvReleaseImage(&lastObst);
    cvReleaseImage(&pointsMask);
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

    cvFindFundamentalMat(p1, p2, F, method, 3., 0.70, statusM);
    //cvFindFundamentalMat(p1, p2, F, method, 3., 0.99, statusM);

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
    cvExtractSURF(img1, NULL, &kp1, &desc1, storage, cvSURFParams(200, 0));
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en surf1 = " << time << endl;
    myTime = clock();
    cvExtractSURF(img2, NULL, &kp2, &desc2, storage, cvSURFParams(200, 0));
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
    cout << "Tiempo invertido en RANSAC = " << time << endl;//* /

    CvScalar p;

    for (int i = 0; i < points1->cols; i++) {
        t_Pair pair;

        p = cvGet2D(points1, 0, i);
        pair.p1 = cvPoint2D32f(p.val[0], p.val[1]);
        p = cvGet2D(points2, 0, i);
        pair.p2 = cvPoint2D32f(p.val[0], p.val[1]);

        pairs.push_back(pair);
    }//*/
    //this->bruteMatch2(kp1, desc1, kp2, desc2);

    //showPairs2("surf", img1, img2, points1, points2);

    IplImage * surf1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * surf2 = cvCreateImage(size, IPL_DEPTH_8U, 3);

    cvCvtColor(img1, surf1, CV_GRAY2BGR);
    cvCvtColor(img2, surf2, CV_GRAY2BGR);

    /*for (int i = 0; i < points1.size(); i++) {
        cvCircle(surf1, cvPoint(points1[i].x, points1[i].y), 2, cvScalar(0, 0, 255), -1);
    }

    for (int i = 0; i < points2.size(); i++) {
        cvCircle(surf2, cvPoint(points2[i].x, points2[i].y), 2, cvScalar(0, 0, 255), -1);
    }//*/

    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
    //for (vector<t_Pair>::iterator it = tmpPairs.begin(); it != tmpPairs.end(); it++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        cvCircle(surf1, cvPointFrom32f(it->p2), 2, color, -1);
        cvCircle(surf2, cvPointFrom32f(it->p1), 2, color, -1);
    }//*/

    cvShowImage("surf1", surf1);
    cvShowImage("surf2", surf2);

    cvReleaseImage(&surf1);
    cvReleaseImage(&surf2);//*/
}

double CRealMatches::distSquare(double *v1, double *v2, int n) {
	double dsq = 0.;
	while (n--) {
		dsq += (*v1 - *v2) * (*v1 - *v2);
		v1++;
		v2++;
	}
	return dsq;
}

// Find closest interest point in a list, given one interest point
int CRealMatches::findMatch(const ISurfPoint& ip1, const vector< ISurfPoint >& ipts, int vlen) {
	double mind = 1e100, second = 1e100;
	int match = -1;

	for (unsigned i = 0; i < ipts.size(); i++) {

		// Take advantage of Laplacian to speed up matching
		if (ipts[i].laplace != ip1.laplace)
			continue;

		double d = distSquare(ipts[i].ivec, ip1.ivec, vlen);

		if (d < mind) {
			second = mind;
			mind = d;
			match = i;
		} else if (d < second) {
			second = d;
		}

	}

	if (mind < 0.8 * second)
		return match;

	return -1;
}

// Find all possible matches between two images
vector< int > CRealMatches::findMatches(const vector< ISurfPoint >& ipts1, const vector< ISurfPoint >& ipts2, int vlen) {
	vector< int > matches(ipts1.size());
	int c = 0;
	for (unsigned i = 0; i < ipts1.size(); i++) {
		int match = findMatch(ipts1[i], ipts2, vlen);
		matches[i] = match;
		if (match != -1) {
                    c++;
		}
	}

        return matches;
}

void CRealMatches::bruteMatch2(CvSeq *kp1, CvSeq *desc1, CvSeq *kp2, CvSeq * desc2) {
    vector<ISurfPoint> points1;
    vector<ISurfPoint> points2;
    int vlen = desc1->total;
    for (int i = 0; i < kp1->total; i++) {
        ISurfPoint ip;
        CvSURFPoint * surfPt = (CvSURFPoint *)cvGetSeqElem(kp1, i);
        ip.x = surfPt->pt.x;
        ip.y = surfPt->pt.y;
        ip.scale = surfPt->size;
        ip.ori = surfPt->dir;
        ip.strength = surfPt->hessian;
        ip.laplace = surfPt->laplacian;

        ip.ivec = (double *) cvGetSeqElem(desc1, i);

        points1.push_back(ip);
    }
    for (int i = 0; i < kp2->total; i++) {
        ISurfPoint ip;
        CvSURFPoint * surfPt = (CvSURFPoint *)cvGetSeqElem(kp2, i);
        ip.x = surfPt->pt.x;
        ip.y = surfPt->pt.y;
        ip.scale = surfPt->size;
        ip.ori = surfPt->dir;
        ip.strength = surfPt->hessian;
        ip.laplace = surfPt->laplacian;

        ip.ivec = (double *) cvGetSeqElem(desc2, i);

        points2.push_back(ip);
    }
    vector<int> matches = findMatches(points1, points2, vlen);    
    
    pairs.clear();    
    for (int i = 0; i < matches.size(); i++) {        
        if (matches.at(i) != -1) {            
            bool isUsed = false;
            for (int j = 0; j < i; j++) {
                if (matches.at(j) == matches.at(i)) {
                    isUsed = true;
                    break;
                }
            }
            if (!isUsed) {                
                t_Pair pair;
                pair.p1 = cvPoint2D32f(points1.at(i).x, points1.at(i).y);
                pair.p2 = cvPoint2D32f(points2.at(matches.at(i)).x, points2.at(matches.at(i)).y);
                pairs.push_back(pair);
            }
        }        
    }              
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
        //if (cvGetReal2D(pointsMask, corners[i].y, corners[i].x) == 255)
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

    cvResize(img1, smallImg1);
    cvResize(img2, smallImg2);

    clock_t myTime = clock();
    //testFast(smallImg1, points1);
    testFast(img1, points1);
    //testFast(img2, points2);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en fast = " << time << endl;

    if (points1.size() < MIN_NFEAT) {
        cerr << "No se han encontrado puntos iniciales suficientes para hacer el emparejamiento" << endl;
        return;
    }
    
    myTime = clock();
    vector <t_Pair> pairs2;
    //getOflow(img1, img2, points1, pairs1);
    //getOflow(img2, img1, points2, pairs2);
    //oFlow(points1, pairs, smallImg1, smallImg2);
    //oFlow(points1, pairs2, img1, img2);
    //cleanRANSAC(CV_FM_RANSAC, pairs2);
    //oFlow(points1, pairs, img1, img2);
    /*vector<t_Pair> tmpPairs;
    for (int i = 0; i < pairs.size(); i++) {
        tmpPairs.push_back(pairs.at(i));
    }//*/
    //testSurf(img1, img2);
    vector<t_SURF_Pair> pairsSurf;
    t_Timings timings;
    surfGpu.testSurf(img1, img2, pairsSurf, timings);
    pairs.clear();
    for (int i = 0; i < pairsSurf.size(); i++) {
        t_Pair pair;
        pair.p1 = cvPointTo32f(pairsSurf.at(i).kp1.pt);
        pair.p2 = cvPointTo32f(pairsSurf.at(i).kp2.pt);

        pairs.push_back(pair);
    }
    /*for (int i = 0; i < pairs2.size(); i++) {
        pairs.push_back(pairs2.at(i));
    }

    /*for (int i = 0; i < tmpPairs.size(); i++) {
        pairs.push_back(tmpPairs.at(i));
    }//*/

    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en oFlow = " << time << endl;

    if (pairs.size() < MIN_NFEAT) {
        cerr << "No se han encontrado puntos suficientes para hacer el emparejamiento" << endl;
        return;
    }

    /*for (int i = 0; i < pairs.size(); i++) {
        pairs.at(i).p1.x *= 2;
        pairs.at(i).p1.y *= 2;
        pairs.at(i).p2.x *= 2;
        pairs.at(i).p2.y *= 2;
    }*/

    /*myTime = clock();
    fusePairs(pairs1, pairs2, false);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en fusion = " << time << endl;//*/


    myTime = clock();
    //cleanByTriangles();
    //cleanRANSAC(CV_FM_RANSAC, pairs);
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

    //drawTriangles("tri1", "tri2", false);
    //drawTriangles("tri1b", "tri2b", true);

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
    cvNamedWindow("plinear", 1);
    cvShowImage("plinear", plinear);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en pieceWise = " << time << endl; //*/
    //wm(pairs, img1, img2);    

    myTime = clock();
    if (roadMask != NULL) {
        //cvAnd(mask1, interestMask, mask1);
    }
    if (cvCountNonZero(mask1) < 4) {
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
    //setMaskFromPoints(mask2, 0);
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

inline void CRealMatches::checkCoveredArea(IplImage * imgB, IplImage * imgA, int &coveredArea) {
    if ((cvCountNonZero(imgA) == 0) || (cvCountNonZero(imgB) == 0)) {
        coveredArea = 0;
        return;
    }

    //cvCopyImage(imgA, img1);
    //cvCopyImage(imgB, img2);

    cvResize(imgA, smallImg1);
    cvResize(imgB, smallImg2);

    testFast(smallImg1, points1);
    if (points1.size() < MIN_NFEAT) {
        coveredArea = 0;

        return;
    }

    oFlow(points1, pairs, smallImg1, smallImg2);

    if (pairs.size() < MIN_NFEAT) {
        coveredArea = 0;
        
        return;
    }

    cleanRANSAC(CV_FM_RANSAC, pairs);

    if (pairs.size() < MIN_NFEAT) {
        coveredArea = 0;
        return;
    }

//    setMaskFromPoints(mask1, 2);
    
//    coveredArea = cvCountNonZero(mask1);
    coveredArea = pairs.size();
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

    roadMask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    CRoadDetection rd(size);
    rd.detectFixed(roadMask);

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
        if (abs(ang) > 5) continue;
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

            //if (index != 1) {
                int key = cvWaitKey(0);
                if (key == 27)
                    exit(0);
                if (key == 98)
                    break;
            //}

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

            string imgPath = path + (*it) + "Images_rect/" + line1;
            cout << imgPath << ", ";
            IplImage * img1L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cvResize(img1L, img1);
            //cvSmooth(img1, img1, CV_GAUSSIAN, 3);
            cvReleaseImage(&img1L);
            
            imgPath = path + (*it) + "Images_rect/" + line2;
            cout << imgPath << endl;
            IplImage * img2L = cvLoadImage(imgPath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cvResize(img2L, img2);
            //cvSmooth(img2, img2, CV_GAUSSIAN, 3);
            cvReleaseImage(&img2L);            

            clock_t myTime = clock();
            mainTest();
            time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
            cout << "Tiempo TOTAL = " << time << endl;

            int key = cvWaitKey(20);
            if (key == 27)
                exit(0);
            if (key == 32)
                cvWaitKey(0);

        }
    }
}

void CRealMatches::startTest5() {
    CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERBase2", "Rutas/pruebaITERConObs2", "/home/neztol/doctorado/Datos/DB", cvRect(5, 5, size.width - 5, size.height - 5), size);
    CRoadDetection rd(size);    

    IplImage * imgDB;
    IplImage * imgRT;
    int indexRT;
    int indexDB;
    roadMask = cvCreateImage(size, IPL_DEPTH_8U, 1);       

    rd.detectFixed(roadMask);
    //cvSet(roadMask, cvScalar(255));

    //cvNamedWindow("ImgDB", 1);
    //cvNamedWindow("ImgRT", 1);
    int index = 0;
    ruta.setCurrentPoint(index);
    while (true) {
        ruta.getNextImage(imgRT, imgDB);
        //rd.detect(ruta.getSTPoint(), roadMask);
        //rd.detectOcclusions(ruta.getSTPoint(), roadMask);
        //cvSet(roadMask, cvScalar(255));
        //cvShowImage("roadMask", interestMask);
        //rd.detectRoadWithFATPoints(ruta.getSTPoint(), roadMask);
        /*cvCvtColor(imgDB, imgColor, CV_GRAY2BGR);
        CvPoint * poly = aco.iterate(imgColor);
        int npts = 4;        
        cvZero(roadMask);
        cvFillPoly(roadMask, &poly, &npts, 1, cvScalar(255));
        cvShowImage("roadMask", roadMask);//*/
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

void CRealMatches::startTestRoadDetection() {
    CRoadDetection rd(size);
    IplImage * maskRoad = cvCreateImage(size, IPL_DEPTH_8U, 1);
    int index = 1050;
    while (true) {        
        clock_t myTime = clock();

        //rd.detect(index, maskRoad);
        //rd.detectRoadWithFAST(index, maskRoad);
        //rd.detectRoadWithFATPoints(index, maskRoad);
        //rd.detectOcclusions(index, maskRoad);
        rd.detectACO(index, maskRoad);
        cvShowImage("aco", maskRoad);
        index++;

        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;
        cout << "Index = " << (index - 1) << endl;

        int key = cvWaitKey(0);
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

void CRealMatches::startTest6() {
    //CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERBase2", "Rutas/pruebaITERConObs2", "/home/neztol/doctorado/Datos/DB");
    CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno5.sqlite", "Rutas/iter20julBase6", "Rutas/iter20julConObs2", "/home/neztol/doctorado/Datos/DB", cvRect(6,1, 633, 475), size);
    CRoadDetection rd(size);

    IplImage * imgDB1;
    IplImage * imgDB2;
    IplImage * imgDB3;
    IplImage * imgRT;
    IplImage * imgDB;
    int indexRT;
    int indexDB;
    roadMask = cvCreateImage(size, IPL_DEPTH_8U, 1);

    rd.detectFixed(roadMask);
    //cvSet(roadMask, cvScalar(255));

    //cvNamedWindow("ImgDB", 1);
    //cvNamedWindow("ImgRT", 1);
    int index = 629;
    ruta.setCurrentPoint(index);
    while (true) {
        clock_t myTime = clock();

        ruta.getNextImage(imgRT, imgDB1, imgDB2, imgDB3);

        int area1, area2, area3;
        checkCoveredArea(imgRT, imgDB1, area1);
        checkCoveredArea(imgRT, imgDB2, area2);
        checkCoveredArea(imgRT, imgDB3, area3);        

        cout << "Area1 = " << area1 << endl;
        cout << "Area2 = " << area2 << endl;
        cout << "Area3 = " << area3 << endl;
        int maxArea = max(area1, max(area2, area3));
        if (maxArea == area1) imgDB = imgDB1;
        if (maxArea == area2) imgDB = imgDB2;
        if (maxArea == area3) imgDB = imgDB3;

        index++;

        if (maxArea == 0) {
            cout << "Aqui" << endl;
            IplImage * fullColor = cvCreateImage(size, IPL_DEPTH_8U, 3);
            cvSet(fullColor, cvScalar(rand() & 255, rand() & 255, rand() & 255));
            cvShowImage("ResultadoFinal", fullColor);

            cvWaitKey(20);
            
            cvReleaseImage(&imgDB1);
            cvReleaseImage(&imgDB2);
            cvReleaseImage(&imgDB3);
            cvReleaseImage(&imgRT);
            cvReleaseImage(&fullColor);

            continue;
        }

        //cvCopyImage(imgRT, img1);
        //cvCopyImage(imgDB, img2);
        cvResize(imgRT, img1);
        cvResize(imgDB, img2);

        //cvShowImage("ImgDB", imgDB);
        //cvShowImage("ImgRT", imgRT);
        //clock_t myTime = clock();
        mainTest();
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;
        cout << "Index = " << (index - 1) << endl;

        if (index == 1866) exit(0);

        int key = cvWaitKey(0);        
        if (key == 27)
            exit(0);
        if (key == 1048608)
            cvWaitKey(0);

        cvReleaseImage(&imgDB1);
        cvReleaseImage(&imgDB2);
        cvReleaseImage(&imgDB3);
        cvReleaseImage(&imgRT);
    }
}

void CRealMatches::startTest7() {
    int testIdx =0;
    char * PATH_BASE_IMG;    
    
    switch (testIdx) {
        case 0:                 
            PATH_BASE_IMG = "/home/neztol/doctorado/Datos/EstadisticasITER/tripode1/";
            break;
        case 1:                        
            PATH_BASE_IMG = "/home/neztol/doctorado/Datos/EstadisticasITER/tripode2/";
            break;
        case 2:
            PATH_BASE_IMG = "/home/neztol/doctorado/Datos/EstadisticasITER/tripode3/";
            break;
    }

    roadMask = cvCreateImage(size, IPL_DEPTH_8U, 1);

    CRoadDetection rd(size);
    //rd.detectFixed(roadMask);

    string maskPath = string(PATH_BASE_IMG);
    maskPath += "mask.jpg";
    IplImage * tmpMask = cvLoadImage(maskPath.c_str(), 0);
    cvResize(tmpMask, roadMask);
    cvReleaseImage(&tmpMask);


    string path(PATH_BASE_IMG);
    path += "datos_RT.txt";    

    ifstream ifs(path.c_str() , ifstream::in );

    char line[1024];

    int nRT = 0;
    ifs >> nRT;
    ifs.ignore();
    char ** rtPath = new char*[nRT];
    for (int i = 0; i < nRT; i++) {
        rtPath[i] = new char[1024];
        ifs.getline(rtPath[i], 1024);
    }    
    ifs.getline(line, 1024);

    string DBpath;
    string RTpath;
    while (ifs.good()) {
        DBpath = string(PATH_BASE_IMG);        

        int dist;
        int angle;
        ifs >> dist;
        ifs >> angle;
        ifs.ignore(2);

        ifs.getline(line, 1024);
        DBpath += line;
        DBpath += ".JPG";

        if (abs(dist) > 1) continue;
        if ((dist == -1) && (angle < 0)) continue;
        if ((dist == 1) && (angle > 0)) continue;
        if (abs(angle) > 10) continue;

        for (int i = 0; i < nRT; i++) {
        //int i = 5;
            RTpath = string(PATH_BASE_IMG);
            RTpath += rtPath[i];
            RTpath += ".JPG";

            cout << RTpath << endl;
            cout << DBpath << endl;

            IplImage * imgA = cvLoadImage(RTpath.c_str(), 0);
            IplImage * imgB = cvLoadImage(DBpath.c_str(), 0);

            cvResize(imgA, img1);
            cvResize(imgB, img2);

            cvSet(lastObst, cvScalar(255));

            //cvShowImage("Img1", img1);
            //cvShowImage("Img2", img2);            

            clock_t myTime = clock();
            mainTest();
            time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
            cout << "Tiempo TOTAL = " << time << endl;

            int key = cvWaitKey(0);
            if (key == 27)
                exit(0);
            if (key == 32)
                cvWaitKey(0);

            cvReleaseImage(&imgA);
            cvReleaseImage(&imgB);
        }
    }

    ifs.close();

}

void CRealMatches::startTestCMU(string testName, bool cabecera) {
    string leftStr;
    string rightStr;

    string PATH_BASE = "/home/neztol/doctorado/Datos/CMU Image Database Stereo/";

    string dataFileName = PATH_BASE + string("datos.txt");
    string outputFileName = string("/home/neztol/doctorado/articulos/CUDA/") + testName + string(".txt");

    ifstream ifs(dataFileName.c_str() , ifstream::in );
    ofstream ofs(outputFileName.c_str(), ios::app );
    if (cabecera) {
        ofs << "testName\texample\twidth\theight\tnPoints1\tnPoints2\tnPairs\tnPairsClean\ttSurf1\ttSurf2\ttCalcMeanSdv";
        ofs << "\ttCalcCorrelation\ttCalcBestCorr\ttCalcMatches\ttMalloc1\ttMalloc2\ttMalloc\ttMemCpy\ttFreeMem";
        ofs << "\ttPrevRANSAC\ttRANSAC\ttTotal\tthreadsPerBlock\tblocksPerGrid\tdimBlock1\tdimBlock2";
        ofs << "\tdimGrid1\tdimGrid2\n\n";
    }

    string line;

    while (ifs >> line) {
        leftStr = PATH_BASE + line + string("/left.png");
        rightStr = PATH_BASE + line + string("/right.png");

        cout << leftStr << endl;
        cout << rightStr << endl;

        IplImage * imgA = cvLoadImage(leftStr.c_str(), 0);
        IplImage * imgB = cvLoadImage(rightStr.c_str(), 0);

        cvResize(imgA, img1);
        cvResize(imgB, img2);

        cvReleaseImage(&imgA);
        cvReleaseImage(&imgB);

        //cvShowImage("img1", img1);
        //cvShowImage("img2", img2);

        //mainTest();       // Descomentar para comprobar que el método está funcionando correctamente
        vector<t_SURF_Pair> pairsSurf;
        t_Timings timings;
        surfGpu.testSurf(img1, img2, pairsSurf, timings);

        ofs << testName << "\t" << line << "\t" << size.width << "\t" << size.height << "\t" << timings.nPoints1
                << "\t" << timings.nPoints2 << "\t" << timings.nPairs << "\t" << timings.nPairsClean << "\t" << timings.tSurf1 
                << "\t" << timings.tSurf2 << "\t" << timings.tCalcMeanSdv << "\t" << timings.tCalcCorrelation << "\t"
                << timings.tCalcBestCorr << "\t" << timings.tCalcMatches << "\t" << timings.tMalloc1 << "\t" << timings.tMalloc2
                << "\t" << timings.tMalloc << "\t" << timings.tMemCpy << "\t" << timings.tFreeMem << "\t" << timings.tPrevRANSAC << "\t"
                << timings.tRANSAC << "\t" << timings.tTotal << "\t" << timings.threadsPerBlock << "\t"
                << timings.blocksPerGrid << "\t" << timings.dimBlock.x << "\t" << timings.dimBlock.y
                << "\t" << timings.dimGrid.x << "\t" << timings.dimGrid.y << endl;
        //cvWaitKey(0);
        //break;
    }

    ifs.close();
    ofs.close();
}

#define TOTAL_MATCHES 100
void CRealMatches::getNearest(IplImage * imgRT, IplImage * &imgDB, int index1, int index2, CRutaDB2 &ruta) {
    cout << index1 << ", " << index2 << endl;
    if (((int)((index2 - index1) / TOTAL_MATCHES)) <= 1) {
        int maxCoincidence = INT_MIN;
        int index = -1;
        imgDB = cvCreateImage(size, IPL_DEPTH_8U, 1);

        for (int i = index1; i <= index2; i++) {

            IplImage * tmpDB;
            ruta.getImageAt(tmpDB, TYPE_ST, i);
            int coincidences;
            checkCoveredArea(imgRT, tmpDB, coincidences);
            if (coincidences > maxCoincidence) {
                maxCoincidence = coincidences;
                cvResize(tmpDB, imgDB);
                index = i;
            }

            cvReleaseImage(&tmpDB);
        }
    } else {
        int dif = (int)((index2 - index1) / TOTAL_MATCHES);

        int maxCoincidence = INT_MIN;
        int index = -1;        

        for (int i = 0; i < TOTAL_MATCHES; i++) {            
            IplImage * tmpDB;
            ruta.getImageAt(tmpDB, TYPE_ST, i * dif + index1);            
            
            int coincidences;
            checkCoveredArea(imgRT, tmpDB, coincidences);
            if (coincidences > maxCoincidence) {
                maxCoincidence = coincidences;                
                index = i;
            }

            cvReleaseImage(&tmpDB);
        }

        int newIndex1 = index * dif + index1;
        int newIndex2 = (index + 1) * dif + index1 - 1;       

        getNearest(imgRT, imgDB, newIndex1, newIndex2, ruta);
    }
}

void CRealMatches::startTestNearestImage() {
    //CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERBase2", "Rutas/pruebaITERConObs2", "/home/neztol/doctorado/Datos/DB");
    CRutaDB2 ruta("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/urbRadazulDiciembre08Base", "Rutas/urbRadazulDiciembre08obs", "/home/neztol/doctorado/Datos/DB", cvRect(5, 0, size.width - 5, size.height), size);

    for (int i = 900; i < ruta.getMaxRTPoint(); i += 20) {
        IplImage * imgRT;
        ruta.getImageAt(imgRT, TYPE_RT, i);
        /*clock_t myTime = clock();
        IplImage * imgDB;
        getNearest(imgRT, imgDB, 0, ruta.getMaxSTPoint(), ruta);
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo = " << time << endl;//*/
        int maxCoincidence = INT_MIN;
        int index = -1;
        IplImage * imgDB = cvCreateImage(size, IPL_DEPTH_8U, 1);
        int * nCoincidences = new int[ruta.getMaxSTPoint()];
        clock_t myTime = clock();
        for (int j = 0; j < ruta.getMaxSTPoint(); j++) {
            IplImage * tmpDB;
            ruta.getImageAt(tmpDB, TYPE_ST, j);
            int coincidences;            
            checkCoveredArea(imgRT, tmpDB, coincidences);
            nCoincidences[j] = coincidences;
            if (coincidences > maxCoincidence) {
                maxCoincidence = coincidences;
                cvResize(tmpDB, imgDB);
                index = j;
            }

            cvReleaseImage(&tmpDB);
        }
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo = " << time << endl;
        cout << i << " " << index << endl;//*/

        cout << "coinc = [";
        for (int j = 0; j < ruta.getMaxSTPoint() - 1; j++) {
            cout << j << ", " << nCoincidences[j] << ";" << endl;
        }
        cout << (ruta.getMaxSTPoint() - 1) << ", " << nCoincidences[ruta.getMaxSTPoint() - 1] << "];" << endl;
        
        //cvShowImage("ImgRT", imgRT);
        //cvShowImage("ImgDB", imgDB);

        cvCopyImage(imgRT, img1);
        cvCopyImage(imgDB, img2);
        roadMask = cvCreateImage(size, IPL_DEPTH_8U, 1);
        cvSet(roadMask, cvScalar(255));
        cvSet(pointsMask, cvScalar(255));

        //cvShowImage("ImgDB", imgDB);
        //cvShowImage("ImgRT", imgRT);
        //clock_t myTime = clock();
        mainTest();

        int key = cvWaitKey(0);
        if (key == 27)
            exit(0);
        if (key == 32)
            cvWaitKey(0);

        cvReleaseImage(&imgDB);
        cvReleaseImage(&imgRT);
    }
}
