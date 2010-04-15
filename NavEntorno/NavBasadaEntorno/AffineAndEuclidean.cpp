#include "ViewMorphing.h"

#define MAX_DIST_CLOUD 2//DBL_MAX

typedef struct {
    int index;
    double dist;
} t_Pair;

void CViewMorphing::AffineAndEuclidean(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask) {
    //oFlowFeatureTracker(img1, img2);
    clock_t myTime = clock();
    oFlowMeshedAndDetectedFeatureTracker(img1, img2);

    warpAffine(img1, img2);

    cleanDuplicated();
    cleanUsingDelaunay(img2C, img1C);
    cleanDuplicated();

    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en features = " << time << endl;


    showFeatureTracking(img2, img1);

    ////cleanUsingDistances(img2C, img1C);

    //pieceWiseLinear(img2, img1, img2C, img1C, featureMask);

    if (cvWaitKey(0) == 27) exit(0);
}

void CViewMorphing::warpAffine(IplImage * &img1, IplImage * &img2) {
        double minX = DBL_MAX, maxX = DBL_MIN, minY = DBL_MAX, maxY = DBL_MIN;
        CvPoint2D32f src[4] = { cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1) };
        CvPoint2D32f dst[4] = { cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1) };
        //CvPoint2D32f src[3] = { cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1) };
        //CvPoint2D32f dst[3] = { cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1), cvPoint2D32f(-1, -1) };


        for (int i = 0; i < numberOfFeatures; i++) {
            if ((points1[i].x < 1) || (points1[i].y < 1) || (points1[i].x > img1->width) || (points1[i].y > img1->height)) continue;
            if ((points2[i].x < 1) || (points2[i].y < 1) || (points2[i].x > img1->width) || (points2[i].y > img1->height)) continue;

            if (points1[i].x < minX) {
                minX = points1[i].x;
                src[0] = points1[i];
                dst[0] = points2[i];
            }
        }
        for (int i = 0; i < numberOfFeatures; i++) {
            if ((points1[i].x < 1) || (points1[i].y < 1) || (points1[i].x > img1->width) || (points1[i].y > img1->height)) continue;
            if ((points2[i].x < 1) || (points2[i].y < 1) || (points2[i].x > img1->width) || (points2[i].y > img1->height)) continue;

            if (points1[i].x > maxX) {
                bool isUsed = false;
                for (int j = 0; j < 1; j++) {
                    if ((src[j].x == points1[i].x) && (src[j].y == points1[i].y)) {
                        isUsed = true;
                        break;
                    }
                }
                if (isUsed == false) {
                    maxX = points1[i].x;
                    src[1] = points1[i];
                    dst[1] = points2[i];
                }
            }
        }
        for (int i = 0; i < numberOfFeatures; i++) {
            if ((points1[i].x < 1) || (points1[i].y < 1) || (points1[i].x > img1->width) || (points1[i].y > img1->height)) continue;
            if ((points2[i].x < 1) || (points2[i].y < 1) || (points2[i].x > img1->width) || (points2[i].y > img1->height)) continue;

            if (points1[i].y < minY) {
                bool isUsed = false;
                for (int j = 0; j < 2; j++) {
                    if ((src[j].x == points1[i].x) && (src[j].y == points1[i].y)) {
                        isUsed = true;
                        break;
                    }
                }
                if (isUsed == false) {
                    minY = points1[i].y;
                    src[2] = points1[i];
                    dst[2] = points2[i];
                }
            }
        }
        for (int i = 0; i < numberOfFeatures; i++) {
            if ((points1[i].x < 1) || (points1[i].y < 1) || (points1[i].x > img1->width) || (points1[i].y > img1->height)) continue;
            if ((points2[i].x < 1) || (points2[i].y < 1) || (points2[i].x > img1->width) || (points2[i].y > img1->height)) continue;

            if (points1[i].y > maxY) {
                bool isUsed = false;
                for (int j = 0; j < 3; j++) {
                    if ((src[j].x == points1[i].x) && (src[j].y == points1[i].y)) {
                        isUsed = true;
                        break;
                    }
                }
                if (isUsed == false) {
                    maxY = points1[i].y;
                    src[3] = points1[i];
                    dst[3] = points2[i];
                }
            }
        }

        CvMat * X = cvCreateMat(3, 3, CV_64FC1);
        cvGetPerspectiveTransform(src, dst, X);

        //CvMat * X = cvCreateMat(2, 3, CV_64FC1);
        //cvGetAffineTransform(src, dst, X);
        CvPoint2D32f * oldPoints1 = new CvPoint2D32f[numberOfFeatures];
        CvPoint2D32f * oldPoints2 = new CvPoint2D32f[numberOfFeatures];
        int oldNFeatures = numberOfFeatures;
        for (int i = 0; i < numberOfFeatures; i++) {
            oldPoints1[i] = points1[i];
            oldPoints2[i] = points2[i];
        }

        calculatePoints(img1, img2);        

        CvPoint2D32f * oldPoints = new CvPoint2D32f[numberOfFeatures];        
        double * pts = new double[numberOfFeatures * 2];
        for (int i = 0; i < numberOfFeatures; i++) {
            pts[i * 2] = points1[i].x;
            pts[i * 2 + 1] = points1[i].y;
            oldPoints[i] = points1[i];
        }
        CvMat mPts = cvMat(1, numberOfFeatures, CV_64FC2, pts);
	cvPerspectiveTransform(&mPts, &mPts, X);
        //cvTransform(&mPts, &mPts, X);

        for (int i = 0; i < numberOfFeatures; i++) {
            points1[i].x = pts[i * 2];
            points1[i].y = pts[i * 2 + 1];
            
        }

        IplImage * points = cvCreateImage(size, IPL_DEPTH_8U, 3);
        cvZero(points);
        for (int i = 0; i < numberOfFeatures; i++) {
            cvCircle(points, cvPointFrom32f(points1[i]), 2, cvScalar(0, 255, 0), -1);
            cvCircle(points, cvPointFrom32f(points2[i]), 2, cvScalar(0, 0, 255), -1);
        }

        cvNamedWindow("points", 1);
        cvShowImage("points", points);

        int nFeat = oldNFeatures;
        CvPoint2D32f * tmp1 = new CvPoint2D32f[numberOfFeatures + oldNFeatures];
        CvPoint2D32f * tmp2 = new CvPoint2D32f[numberOfFeatures + oldNFeatures];
        for (int i = 0; i < oldNFeatures; i++) {
            tmp1[i] = oldPoints1[i];
            tmp2[i] = oldPoints2[i];
        }
        for (int i = 0; i < numberOfFeatures; i++) {
            double minDist = MAX_DIST_CLOUD;
            int minIndex = -1;
            for (int j = 0; j < numberOfFeatures; j++) {
                double dist = sqrt(pow(points1[i].x - points2[j].x, 2.0) + pow(points1[i].y - points2[j].y, 2.0));
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = j;
                }
            }
            if (minIndex != -1) {
                tmp1[nFeat] = oldPoints[i];
                tmp2[nFeat] = points2[minIndex];
                nFeat++;
            }
        }
        points1 = tmp2;
        points2 = tmp1;
        numberOfFeatures = nFeat + oldNFeatures;
}

void CViewMorphing::calculatePoints(IplImage * img1, IplImage * &img2) {
	int numberOfFeatures1 = MAX_FEATURES, numberOfFeatures2 = MAX_FEATURES;

	points1 = new CvPoint2D32f[MAX_FEATURES];
        points2 = new CvPoint2D32f[MAX_FEATURES];

	IplImage * eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
	IplImage * tmp = cvCreateImage(size, IPL_DEPTH_32F, 1);

	// Obtenemos las caracter�sticas de la imagen (puntos caracter�sticos)
	cvGoodFeaturesToTrack(img1, eigen, tmp, points1, &numberOfFeatures1,
		DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);
	// Precisi�n subpixel
	if (COMPUTE_SUBPIXEL_CORNERS != 0) {
		cvFindCornerSubPix(img1, points1, numberOfFeatures1, cvSize(SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE),
			cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	}

	// Obtenemos las caracter�sticas de la imagen (puntos caracter�sticos)
	cvGoodFeaturesToTrack(img2, eigen, tmp, points2, &numberOfFeatures2,
		DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);
	// Precisi�n subpixel
	if (COMPUTE_SUBPIXEL_CORNERS != 0) {
		cvFindCornerSubPix(img2, points2, numberOfFeatures2, cvSize(SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE),
			cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	}

        numberOfFeatures = (int)min(numberOfFeatures1, numberOfFeatures2);
}
/*
 *public void cleanUsingDistance(ArrayList<Pair> pairs) {

        if (pairs == null) {
            return;
        } else if (pairs.size() < 4) {
            pairs.clear();
            return;
        }

        ArrayList<Pair> goodPairs = new ArrayList();

        for (Pair pair : pairs) {
            goodPairs.add(pair);
        }

        Collections.sort((List) pairs);

        // First quartile
        double Q1a = pairs.get((int) (pairs.size() / 4.0) - 1).getDist();
        double Q1b = pairs.get((int) (pairs.size() / 4.0)).getDist();
        double percent1 = (pairs.size() / 4.0) - ((int) (pairs.size() / 4.0));
        double Q1 = ((Q1b - Q1a) * percent1) + Q1a;

        // Third quartile
        double Q3a = pairs.get((int) (3.0 * pairs.size() / 4.0) - 1).getDist();
        double Q3b = pairs.get((int) (3.0 * pairs.size() / 4.0)).getDist();
        double percent3 = (3.0 * pairs.size() / 4.0) - ((int) (3.0 * pairs.size() / 4.0));
        double Q3 = ((Q3b - Q3a) * percent3) + Q3a;

        double lThresh = Q1 - 0.5 * (Q3 - Q1);
        double hThresh = Q3 + 0.5 * (Q3 - 1);

        pairs.clear();
        for (Pair pair : goodPairs) {
            if ((pair.getDist() < hThresh) && (pair.getDist() > lThresh)) {
                pairs.add(pair);
            }
        }
    }*/
void CViewMorphing::cleanUsingDistances(IplImage * img1, IplImage * &img2) {
    cout << "======================================" << endl;
    t_Pair * pairs = new t_Pair[numberOfFeatures];
    double * dist = new double[numberOfFeatures];

    for (int i = 0; i < numberOfFeatures; i++) {
        dist[i] = sqrt(pow(points1[i].x - points2[i].x, 2.0) +  pow(points1[i].y - points2[i].y, 2.0));
    }

    for (int i = 0; i < numberOfFeatures; i++) {
        t_Pair pair;
        pair.index = -1;        
        pair.dist = DBL_MAX;        
        for (int j = 0; j < numberOfFeatures; j++) {
            if (dist[j] < pair.dist) {
                pair.index = j;
                pair.dist = dist[j];
            }
        }
        dist[pair.index] = DBL_MAX;
        pairs[i] = pair;
        cout << pair.dist << endl;
    }

    // First quartile
    double Q1a = pairs[(int) (numberOfFeatures / 4.0) - 1].dist;
    double Q1b = pairs[(int) (numberOfFeatures / 4.0)].dist;
    double percent1 = (numberOfFeatures / 4.0) - ((int) (numberOfFeatures / 4.0));
    double Q1 = ((Q1b - Q1a) * percent1) + Q1a;

    // Third quartile
    double Q3a = pairs[(int) (2.0 * numberOfFeatures / 4.0) - 1].dist;
    double Q3b = pairs[(int) (2.0 * numberOfFeatures / 4.0)].dist;
    double percent3 = (2.0 * numberOfFeatures / 4.0) - ((int) (2.0 * numberOfFeatures / 4.0));
    double Q3 = ((Q3b - Q3a) * percent3) + Q3a;

    double lThresh = Q1 - 0.5 * (Q3 - Q1);
    double hThresh = Q3 + 0 * (Q3 - 1);

    cout << "L = " << lThresh << endl;
    cout << "H = " << hThresh << endl;

    IplImage * lienzo = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvZero(lienzo);

    for (int i = 0; i < numberOfFeatures; i++) {
        CvScalar color;
        if ((pairs[i].dist < hThresh) && (pairs[i].dist > lThresh)) {
            color = cvScalar(0, 255, 0);
        } else {
            color = cvScalar(0, 0, 255);
        }
        cvCircle(lienzo, cvPointFrom32f(points1[pairs[i].index]), 2, color);
    }

    cvNamedWindow("Lienzo", 1);
    cvShowImage("Lienzo", lienzo);
}