
#include <vector>

#include "ViewMorphing.h"

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"

typedef struct {
    int index2;
    double corr;
} t_match;

CViewMorphing::CViewMorphing(CvSize size) {
	this->size = size;
}

CViewMorphing::~CViewMorphing(void)
{
}

void CViewMorphing::oFlowFeatureTracker(IplImage * img1, IplImage * &img2) {
	numberOfFeatures = MAX_FEATURES;

	CvPoint2D32f * features = new CvPoint2D32f[MAX_FEATURES];

	IplImage * eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
	IplImage * tmp = cvCreateImage(size, IPL_DEPTH_32F, 1);
		
	// Obtenemos las caracter�sticas de la imagen (puntos caracter�sticos)
	cvGoodFeaturesToTrack(img1, eigen, tmp, features, &numberOfFeatures, 
		DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);

	// Precisi�n subpixel
	if (COMPUTE_SUBPIXEL_CORNERS != 0) {
		cvFindCornerSubPix(img1, features, numberOfFeatures, cvSize(SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE),
			cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	}	

	// Buscamos las correspondencias de la imagen 1 a la 2
	CvSize pyramidImageSize = cvSize(size.width + 8, size.height / 3);
	IplImage * pyramidImage1 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
	IplImage * pyramidImage2 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);

	char * status0 = new char[numberOfFeatures];
	char * status1 = new char[numberOfFeatures];

	CvPoint2D32f ** oFlowPoints1To2 = new CvPoint2D32f *[2];
	CvPoint2D32f ** oFlowPoints2To1 = new CvPoint2D32f *[2];

	oFlowPoints1To2[0] = features;
	oFlowPoints1To2[1] = new CvPoint2D32f[MAX_FEATURES + 4];
	oFlowPoints2To1[0] = new CvPoint2D32f[MAX_FEATURES + 4];
	oFlowPoints2To1[1] = new CvPoint2D32f[MAX_FEATURES + 4];

	cvCalcOpticalFlowPyrLK(img1, img2, pyramidImage1, pyramidImage2,
		oFlowPoints1To2[0], oFlowPoints1To2[1], numberOfFeatures,
		cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
		PYRAMID_DEPTH, status0, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

	// Cruzamos las correspondencias en los dos sentidos para ver si son las mismas 
	for (int i = 0; i < MAX_FEATURES; i++) {
		oFlowPoints2To1[0][i].x = oFlowPoints1To2[1][i].x;
		oFlowPoints2To1[0][i].y = oFlowPoints1To2[1][i].y;
	}

	// Buscamos las correspondencias de la imagen 2 a la 1
	cvCalcOpticalFlowPyrLK(img2, img1, pyramidImage2, pyramidImage1,
		oFlowPoints2To1[0], oFlowPoints2To1[1], numberOfFeatures,
		cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
		PYRAMID_DEPTH, status1, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

	points1 = new CvPoint2D32f[MAX_FEATURES + 4];
	points2 = new CvPoint2D32f[MAX_FEATURES + 4];

	numberOfFeatures = 0;

	// Comparamos ambas caracter�sticas y obtenemos el resultado final
	for (int i = 0; i < MAX_FEATURES; i++) {
		if (status0[i] && status1[i]) {
			float distance = sqrt( pow( (double)(oFlowPoints2To1[1][i].x -
										oFlowPoints1To2[0][i].x), 2.0) +
								   pow( (double)(oFlowPoints2To1[1][i].y -
										oFlowPoints1To2[0][i].y), 2.0));
			if (distance < MAX_PIXEL_DISTANCE) {
				points1[numberOfFeatures].x = (oFlowPoints1To2[0][i].x + oFlowPoints2To1[1][i].x) / 2.0;
				points1[numberOfFeatures].y = (oFlowPoints1To2[0][i].y + oFlowPoints2To1[1][i].y) / 2.0;
				points2[numberOfFeatures].x = (oFlowPoints1To2[1][i].x + oFlowPoints2To1[0][i].x) / 2.0;
				points2[numberOfFeatures].y = (oFlowPoints1To2[1][i].y + oFlowPoints2To1[0][i].y) / 2.0;
				//points1[numberOfFeatures] = oFlowPoints1To2[0][i];
				//points2[numberOfFeatures] = oFlowPoints1To2[1][i];
				numberOfFeatures++;
			}
		}
	}		//*/

	//warpPoints(img2);
}

void CViewMorphing::oFlowMeshedFeatureTracker(IplImage * img1, IplImage * &img2) {
        clock_t myTime = clock();
        int numberOfPixelsPerDivision = 20;
	numberOfFeatures = ((img1->width / numberOfPixelsPerDivision) - 1) * ((img1->height / numberOfPixelsPerDivision) - 1);
	CvPoint2D32f * features = new CvPoint2D32f[numberOfFeatures];

        for (int index = 0, i = numberOfPixelsPerDivision; i < img1->width; i += numberOfPixelsPerDivision) {
            for (int j = numberOfPixelsPerDivision; j < img1->height; j += numberOfPixelsPerDivision, index++) {
                features[index] = cvPoint2D32f(i, j);                
            }
        }       

	// Buscamos las correspondencias de la imagen 1 a la 2
	CvSize pyramidImageSize = cvSize(size.width + 8, size.height / 3);
	IplImage * pyramidImage1 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
	IplImage * pyramidImage2 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);

	char * status0 = new char[numberOfFeatures];
	char * status1 = new char[numberOfFeatures];

	CvPoint2D32f ** oFlowPoints1To2 = new CvPoint2D32f *[2];
	CvPoint2D32f ** oFlowPoints2To1 = new CvPoint2D32f *[2];

	oFlowPoints1To2[0] = features;
	oFlowPoints1To2[1] = new CvPoint2D32f[numberOfFeatures + 4];
	oFlowPoints2To1[0] = new CvPoint2D32f[numberOfFeatures + 4];
	oFlowPoints2To1[1] = new CvPoint2D32f[numberOfFeatures + 4];       

	cvCalcOpticalFlowPyrLK(img1, img2, pyramidImage1, pyramidImage2,
		oFlowPoints1To2[0], oFlowPoints1To2[1], numberOfFeatures,
		cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
		PYRAMID_DEPTH, status0, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);
        
	// Cruzamos las correspondencias en los dos sentidos para ver si son las mismas
	for (int i = 0; i < numberOfFeatures; i++) {
		oFlowPoints2To1[0][i].x = oFlowPoints1To2[1][i].x;
		oFlowPoints2To1[0][i].y = oFlowPoints1To2[1][i].y;
	}

	// Buscamos las correspondencias de la imagen 2 a la 1
	cvCalcOpticalFlowPyrLK(img2, img1, pyramidImage2, pyramidImage1,
		oFlowPoints2To1[0], oFlowPoints2To1[1], numberOfFeatures,
		cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
		PYRAMID_DEPTH, status1, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

	points1 = new CvPoint2D32f[numberOfFeatures + 4];
	points2 = new CvPoint2D32f[numberOfFeatures + 4];

        int maxFeat = numberOfFeatures;
	numberOfFeatures = 0;

	// Comparamos ambas caracter�sticas y obtenemos el resultado final
	for (int i = 0; i < maxFeat; i++) {
		if (status0[i] && status1[i]) {
			float distance = sqrt( pow( (double)(oFlowPoints2To1[1][i].x -
										oFlowPoints1To2[0][i].x), 2.0) +
								   pow( (double)(oFlowPoints2To1[1][i].y -
										oFlowPoints1To2[0][i].y), 2.0));
			if (distance < MAX_PIXEL_DISTANCE) {
				points1[numberOfFeatures].x = (oFlowPoints1To2[0][i].x + oFlowPoints2To1[1][i].x) / 2.0;
				points1[numberOfFeatures].y = (oFlowPoints1To2[0][i].y + oFlowPoints2To1[1][i].y) / 2.0;
				points2[numberOfFeatures].x = (oFlowPoints1To2[1][i].x + oFlowPoints2To1[0][i].x) / 2.0;
				points2[numberOfFeatures].y = (oFlowPoints1To2[1][i].y + oFlowPoints2To1[0][i].y) / 2.0;
				//points1[numberOfFeatures] = oFlowPoints1To2[0][i];
				//points2[numberOfFeatures] = oFlowPoints1To2[1][i];
				numberOfFeatures++;
			}
		}
	}		//*/

        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo = " << time << endl;

	//warpPoints(img2);
}

void CViewMorphing::oFlowMeshedAndDetectedFeatureTracker(IplImage * img1, IplImage * &img2) {
    clock_t myTime = clock();
        int numberOfPixelsPerDivision = 20; //img1->width * img1->height; //20;
	int numberOfMeshedFeatures = ((img1->width / numberOfPixelsPerDivision) - 1) * ((img1->height / numberOfPixelsPerDivision) - 1);
	CvPoint2D32f * meshedFeatures = new CvPoint2D32f[numberOfMeshedFeatures];

        for (int index = 0, i = numberOfPixelsPerDivision; i < img1->width; i += numberOfPixelsPerDivision) {
            for (int j = numberOfPixelsPerDivision; j < img1->height; j += numberOfPixelsPerDivision, index++) {
                meshedFeatures[index] = cvPoint2D32f(i, j);
            }
        }

        int numberOfDetectedFeatures = MAX_FEATURES;

	CvPoint2D32f * detectedFeatures = new CvPoint2D32f[MAX_FEATURES];

	IplImage * eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
	IplImage * tmp = cvCreateImage(size, IPL_DEPTH_32F, 1);

	// Obtenemos las caracter�sticas de la imagen (puntos caracter�sticos)
	cvGoodFeaturesToTrack(img1, eigen, tmp, detectedFeatures, &numberOfDetectedFeatures,
		DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);

	// Precisi�n subpixel
	/*if (COMPUTE_SUBPIXEL_CORNERS != 0) {
		cvFindCornerSubPix(img1, detectedFeatures, numberOfDetectedFeatures, cvSize(SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE),
			cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	}*/

        numberOfFeatures = numberOfDetectedFeatures + numberOfMeshedFeatures;
        CvPoint2D32f * features = new CvPoint2D32f[numberOfFeatures];

        for (int i = 0; i < numberOfMeshedFeatures; i++) {
            features[i] = meshedFeatures[i];
        }
        for (int i = 0; i < numberOfDetectedFeatures; i++) {
            features[i + numberOfMeshedFeatures] = detectedFeatures[i];
        }

	// Buscamos las correspondencias de la imagen 1 a la 2
	CvSize pyramidImageSize = cvSize(size.width + 8, size.height / 3);
	IplImage * pyramidImage1 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);
	IplImage * pyramidImage2 = cvCreateImage(pyramidImageSize, IPL_DEPTH_8U, 1);

	char * status0 = new char[numberOfFeatures];
	char * status1 = new char[numberOfFeatures];

	CvPoint2D32f ** oFlowPoints1To2 = new CvPoint2D32f *[2];
	CvPoint2D32f ** oFlowPoints2To1 = new CvPoint2D32f *[2];

	oFlowPoints1To2[0] = features;
	oFlowPoints1To2[1] = new CvPoint2D32f[numberOfFeatures + 4];
	oFlowPoints2To1[0] = new CvPoint2D32f[numberOfFeatures + 4];
	oFlowPoints2To1[1] = new CvPoint2D32f[numberOfFeatures + 4];

	cvCalcOpticalFlowPyrLK(img1, img2, pyramidImage1, pyramidImage2,
		oFlowPoints1To2[0], oFlowPoints1To2[1], numberOfFeatures,
		cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
		PYRAMID_DEPTH, status0, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

	// Cruzamos las correspondencias en los dos sentidos para ver si son las mismas
	for (int i = 0; i < numberOfFeatures; i++) {
		oFlowPoints2To1[0][i].x = oFlowPoints1To2[1][i].x;
		oFlowPoints2To1[0][i].y = oFlowPoints1To2[1][i].y;
	}

	// Buscamos las correspondencias de la imagen 2 a la 1
	cvCalcOpticalFlowPyrLK(img2, img1, pyramidImage2, pyramidImage1,
		oFlowPoints2To1[0], oFlowPoints2To1[1], numberOfFeatures,
		cvSize(PYRAMIDRICAL_SEARCH_WINDOW_SIZE, PYRAMIDRICAL_SEARCH_WINDOW_SIZE),
		PYRAMID_DEPTH, status1, 0, cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04), 0);

	points1 = new CvPoint2D32f[numberOfFeatures + 4];
	points2 = new CvPoint2D32f[numberOfFeatures + 4];

        int maxFeat = numberOfFeatures;
	numberOfFeatures = 0;

	// Comparamos ambas caracter�sticas y obtenemos el resultado final
	for (int i = 0; i < maxFeat; i++) {
		if (status0[i] && status1[i]) {
			float distance = sqrt( pow( (double)(oFlowPoints2To1[1][i].x -
										oFlowPoints1To2[0][i].x), 2.0) +
								   pow( (double)(oFlowPoints2To1[1][i].y -
										oFlowPoints1To2[0][i].y), 2.0));
			if (distance < MAX_PIXEL_DISTANCE) {
				points1[numberOfFeatures].x = (oFlowPoints1To2[0][i].x + oFlowPoints2To1[1][i].x) / 2.0;
				points1[numberOfFeatures].y = (oFlowPoints1To2[0][i].y + oFlowPoints2To1[1][i].y) / 2.0;
				points2[numberOfFeatures].x = (oFlowPoints1To2[1][i].x + oFlowPoints2To1[0][i].x) / 2.0;
				points2[numberOfFeatures].y = (oFlowPoints1To2[1][i].y + oFlowPoints2To1[0][i].y) / 2.0;
				//points1[numberOfFeatures] = oFlowPoints1To2[0][i];
				//points2[numberOfFeatures] = oFlowPoints1To2[1][i];
				numberOfFeatures++;
			}
		}
	}		//*/

        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo = " << time << endl;

	//warpPoints(img2);
}

void CViewMorphing::LMedSFeatureTracker(IplImage * img1, IplImage * &img2) {
	int numberOfFeatures1 = MAX_FEATURES;
        int numberOfFeatures2 = MAX_FEATURES;

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

        numberOfFeatures = min(numberOfFeatures1, numberOfFeatures2);
	//warpPoints(img2);
}

// From the article Fast Algorithm for Point Pattern Matching, S. Chang, F.Cheng and al.
void CViewMorphing::PointPatternMatchingFeatureTracker(IplImage * img1, IplImage * &img2) {
	int numberOfFeatures1 = MAX_FEATURES;
        int numberOfFeatures2 = MAX_FEATURES;

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

        int m = numberOfFeatures1;
        int n = numberOfFeatures2;
        int M[m][n];
        //int match_flag[m][n];
        /*for (int i = 0; i < m; i++) {
            for  (int a = 0; a < n; a++) {
                match_flag[i][a] = 1;
            }
        }*/

        // Step 1: Matching pairs support finding algorithm
        for (int i = 0; i < m; i++) {
            for  (int a = 0; a < n; a++) {
                for (int j = 0; j < m; j++) {
                    for (int b = 0; b < n; b++) {
                        if ((i != j) && (a != b)) {
                            /*if (match_flag[j][b] == 1) {
                                cout << "m = " << m << "; n = " << n << ": " << i << ", " << a << ", " << j << ", " << b << endl;
                            }*/
                        }
                    }
                }
                // Punto 5
            }
        }

        numberOfFeatures = min(numberOfFeatures1, numberOfFeatures2);
	//warpPoints(img2);
}


void CViewMorphing::CorrelationFeatureTracker(IplImage * img1, IplImage * &img2) {

	int numberOfFeatures1 = MAX_FEATURES;
        int numberOfFeatures2 = MAX_FEATURES;

	CvPoint2D32f * tmpPoints1 = new CvPoint2D32f[MAX_FEATURES];
	CvPoint2D32f * tmpPoints2 = new CvPoint2D32f[MAX_FEATURES];

	IplImage * eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
	IplImage * tmp = cvCreateImage(size, IPL_DEPTH_32F, 1);

	// Obtenemos las caracter�sticas de la imagen (puntos caracter�sticos)
	cvGoodFeaturesToTrack(img1, eigen, tmp, tmpPoints1, &numberOfFeatures1,
		DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);

	// Precisi�n subpixel
	if (COMPUTE_SUBPIXEL_CORNERS != 0) {
		cvFindCornerSubPix(img1, tmpPoints1, numberOfFeatures1, cvSize(SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE),
			cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	}

        // Obtenemos las caracter�sticas de la imagen (puntos caracter�sticos)
	cvGoodFeaturesToTrack(img2, eigen, tmp, tmpPoints2, &numberOfFeatures2,
		DEFAULT_FEATURES_QUALITY, DEFAULT_FEATURES_MIN_DIST, NULL, 3, USE_HARRIS, 0.04);

	// Precisi�n subpixel
	if (COMPUTE_SUBPIXEL_CORNERS != 0) {
		cvFindCornerSubPix(img2, tmpPoints2, numberOfFeatures2, cvSize(SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE),
			cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	}

        int wW = 2, wH = 2;
        t_match matches[numberOfFeatures1];
        numberOfFeatures = 0;
        for (int i = 0; i < numberOfFeatures1; i++) {
            matches[i].index2 = -1;
            for (int j = 0; j < numberOfFeatures2; j++) {
                double mean1 = 0.0, mean2 = 0.0;
                for (int a = -wH; a < wH; a++) {
                    if ((tmpPoints1[i].y + a < 0) || (tmpPoints1[i].y + a >= img1->height)) continue;
                    if ((tmpPoints2[j].y + a < 0) || (tmpPoints2[j].y + a >= img2->height)) continue;
                    for (int b = -wW; b < wW; b++) {
                        if ((tmpPoints1[i].x + b < 0) || (tmpPoints1[i].x + b >= img1->width)) continue;
                        if ((tmpPoints2[j].x + b < 0) || (tmpPoints2[j].x + b >= img2->width)) continue;
                        mean1 += cvGetReal2D(img1, tmpPoints1[i].y + a, tmpPoints1[i].x + b);
                        mean2 += cvGetReal2D(img2, tmpPoints2[j].y + a, tmpPoints2[j].x + b);
                    }
                }
                double num = 0.0;
                double s1 = 0.0;
                double s2 = 0.0;
                for (int a = -wH; a < wH; a++) {
                    if ((tmpPoints1[i].y + a < 0) || (tmpPoints1[i].y + a >= img1->height)) continue;
                    if ((tmpPoints2[j].y + a < 0) || (tmpPoints2[j].y + a >= img2->height)) continue;
                    for (int b = -wW; b < wW; b++) {
                        if ((tmpPoints1[i].x + b < 0) || (tmpPoints1[i].x + b >= img1->width)) continue;
                        if ((tmpPoints2[j].x + b < 0) || (tmpPoints2[j].x + b >= img2->width)) continue;
                        
                        num += (cvGetReal2D(img1, tmpPoints1[i].y + a, tmpPoints1[i].x + b) - mean1) * (cvGetReal2D(img2, tmpPoints2[j].y + a, tmpPoints2[j].x + b) - mean2);
                        s1 += pow((cvGetReal2D(img1, tmpPoints1[i].y + a, tmpPoints1[i].x + b) - mean1), 2.0);
                        s2 += pow((cvGetReal2D(img2, tmpPoints2[j].y + a, tmpPoints2[j].x + b) - mean2), 2.0);
                    }
                }
                int n = (2 * wH + 1) * (2 * wW + 1);
                s1 = sqrt(s1 / (double)(n - 1));
                s2 = sqrt(s2 / (double)(n - 1));

                double denom = (n - 1) * s1 * s2;

                double pearson = num / denom;                

                if (pearson > 0.99) {
                    if (matches[i].index2 == -1) {
                        matches[i].index2 = j;
                        matches[i].corr = pearson;
                        numberOfFeatures++;
                    } else if (matches[i].corr < pearson) {
                        matches[i].index2 = j;
                        matches[i].corr = pearson;
                    }
                }
            }
        }
        
        points1 = new CvPoint2D32f[numberOfFeatures];
        points2 = new CvPoint2D32f[numberOfFeatures];
        for (int index = 0, i = 0; i < numberOfFeatures; i++) {
            if (matches[i].index2 != -1) {
                points1[index] = tmpPoints1[i];
                points2[index] = tmpPoints2[matches[i].index2];
                index++;
            }
        }
	//warpPoints(img2);
}

// Function that find a surf feature in a image represented by an Image structure
vector< Ipoint > CViewMorphing::surfFeatureFinder(IplImage * img, double thresh, int &vLength) {

  Image * im = new Image(img->width, img->height);
  for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
          im->setPix(x, y, cvGetReal2D(img, y, x));
      }
  }

  // Length of the descriptor vector
  int nFeatures = 0;
  // Initial sampling step (default 2)
  int samplingStep = 2;
  // Number of analysed octaves (default 4)
  int octaves = 4;
  // Blob response treshold
  double thres = 4.0; //thresh / 10000.0;//4.0;
  // Set this flag "true" to double the image size
  bool doubleImageSize = true;
  // Initial lobe size, default 3 and 5 (with double image size)
  int initLobe = 3;
  // Upright SURF or rotation invaraiant
  bool upright = true;
  // If the extended flag is turned on, SURF 128 is used
  bool extended = true;
  // Spatial size of the descriptor window (default 4)
  int indexSize = 6;

  // Create the integral image
  Image iimage(im, doubleImageSize);

  // These are the interest points
  vector< Ipoint > ipts;
  ipts.reserve(1000);

  // Extract interest points with Fast-Hessian
  FastHessian fh(&iimage, /* pointer to integral image */
                 ipts,
                 thres, /* blob response threshold */
                 doubleImageSize, /* double image size flag */
                 initLobe * 3 /* 3 times lobe size equals the mask size */,
                 samplingStep, /* subsample the blob response map */
                 octaves /* number of octaves to be analysed */);

  fh.getInterestPoints();

  // Initialise the SURF descriptor
  Surf des(&iimage, /* pointer to integral image */
           doubleImageSize, /* double image size flag */
           upright, /* rotation invariance or upright */
           extended, /* use the extended descriptor */
           indexSize /* square size of the descriptor window (default 4x4)*/);

  // Get the length of the descriptor vector resulting from the parameters
  nFeatures = des.getVectLength();  

  // Compute the orientation and the descriptor for every interest point
  for (unsigned n=0; n<ipts.size(); n++){
    //for (Ipoint *k = ipts; k != NULL; k = k->next){
    // set the current interest point
    des.setIpoint(&ipts[n]);
    // assign reproducible orientation
    des.assignOrientation();
    // make the SURF descriptor
    des.makeDescriptor();

    vLength = des.getVectLength();
  }

  delete im;

  return ipts;
}

// Calculate square distance of two vectors
double CViewMorphing::distSquare(double *v1, double *v2, int n) {
	double dsq = 0.;
	while (n--) {
		dsq += (*v1 - *v2) * (*v1 - *v2);
		v1++;
		v2++;
	}
	return dsq;
}

// Find closest interest point in a list, given one interest point
int CViewMorphing::findSurfMatch(const Ipoint& ip1, const vector< Ipoint >& ipts, int vlen) {
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

	if (mind < 0.5 * second)
		return match;

	return -1;
}

// Find all possible matches between two images
void CViewMorphing::findSurfMatches(const vector< Ipoint >& ipts1, const vector< Ipoint >& ipts2, int vLength) {
	vector< int > matches(ipts1.size());
	int c = 0;
	for (unsigned i = 0; i < ipts1.size(); i++) {            
		int match = findSurfMatch(ipts1[i], ipts2, vLength);
		matches[i] = match;
		if (match != -1) {
			c++;
		}
	}

        numberOfFeatures = c;
        points1 = new CvPoint2D32f[numberOfFeatures];
        points2 = new CvPoint2D32f[numberOfFeatures];

        c = 0;
        for (int i = 0; i < numberOfFeatures; i++) {
            if (matches[i] != -1) {
                points1[c] = cvPoint2D32f(ipts1[i].x, ipts1[i].y);
                points2[c] = cvPoint2D32f(ipts2[matches[i]].x, ipts2[matches[i]].y);
                c++;
            }
        }
}

void CViewMorphing::SurfFeatureTracker(IplImage * img1, IplImage * &img2) {
    clock_t myTime = clock();
    clock_t stepTime = clock();
    int vLength = 0;
    vector< Ipoint > ipts1 = surfFeatureFinder(img1, 400, vLength);
    time_t time = (double(clock() - stepTime) / CLOCKS_PER_SEC * 1000);
    cout << "surfFind1 = " << time << endl;
    cout << "surfFind1 = " << ipts1.size() << " puntos" << endl;
    stepTime = clock();
    vector< Ipoint > ipts2 = surfFeatureFinder(img2, 400, vLength);
    time = (double(clock() - stepTime) / CLOCKS_PER_SEC * 1000);
    cout << "surfFind2 = " << time << endl;
    cout << "surfFind2 = " << ipts2.size() << " puntos" << endl;
    stepTime = clock();

    findSurfMatches(ipts1, ipts2, vLength);

    time = (double(clock() - stepTime) / CLOCKS_PER_SEC * 1000);
    cout << "surfMatch = " << time << endl;

    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo surf = " << time << endl;
}

/*
void CViewMorphing::cleanUsingDelaunay(IplImage * img1, IplImage * img2) {
    	CvRect rect1 = cvRect(-1, -1, size.width + 2, size.height + 2);
        CvRect rect2 = cvRect(-1, -1, size.width + 2, size.height + 2);

        CvMemStorage * storage1 = cvCreateMemStorage(0);
        CvMemStorage * storage2 = cvCreateMemStorage(0);

	CvSubdiv2D * subdiv1 = cvCreateSubdivDelaunay2D(rect1, storage1);
        CvSubdiv2D * subdiv2 = cvCreateSubdivDelaunay2D(rect2, storage2);
	for (int i = 0; i < numberOfFeatures; i++) {
		cvSubdivDelaunay2DInsert(subdiv1, points1[i]);
                cvSubdivDelaunay2DInsert(subdiv2, points2[i]);
	}
	cvCalcSubdivVoronoi2D(subdiv1);
        cvCalcSubdivVoronoi2D(subdiv2);

	/*CvSeqReader reader;
        int total = subdiv->edges->total;
        int elem_size = subdiv->edges->elem_size;

        cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );* /

	IplImage * imgDelaunay = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage * imgDelaunay2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
        cvCopyImage(img1, imgDelaunay);
        cvCopyImage(img2, imgDelaunay2);

        cvNamedWindow("Delaunay", 1);
	cvNamedWindow("Delaunay2", 1);

        for (int i = 0; i < numberOfFeatures; i++) {            
            CvSubdiv2DEdge e = 0, e0 = 0;
            CvSubdiv2DPoint * p = 0;
            CvPoint pts[3];

            cvSubdiv2DLocate(subdiv1, points1[i], &e0, &p);

            if( e0 ) {
                e = e0;                
                int index = 0;
                do {
                    CvSubdiv2DPoint * ptSub = cvSubdiv2DEdgeOrg(e);
                    //Subdivision vertex point
                    CvPoint2D32f pt32f = ptSub->pt;
                    pts[index] = cvPointFrom32f(pt32f); // to an integer point
                    index++;
                    e = cvSubdiv2DGetEdge(e,CV_NEXT_AROUND_LEFT);
                } while( e != e0 );
                cout << index << endl;
            } else {
                cout << "e0 nanai" << endl;
            }
            CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
            cout << "Puntos = ";
            for (int j = 0; j < 3; j++) {
                cout << "[" << pts[j].x << ", " << pts[j].x << "] ";
            }
            cout << endl;
            cvFillConvexPoly(imgDelaunay, pts, 3, color, -1);
        }

        /*for(int i = 0; i < total; i++ ) {
            CvQuadEdge2D* edge = (CvQuadEdge2D*) (reader.ptr);

            if (CV_IS_SET_ELEM(edge)) {

                CvPoint2D32f p1OrgF = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt;
                CvPoint2D32f p1DestF = cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt;

                CvPoint org = cvPointFrom32f(p1OrgF);
                CvPoint dest = cvPointFrom32f(p1DestF);

                cvLine(imgDelaunay2, org, dest, cvScalar(255));

                CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_RIGHT);
                CvSubdiv2DEdge edge2 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge1, CV_NEXT_AROUND_LEFT);

                cvLine(imgDelaunay, dest, cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt), cvScalar(255, 0, 255));
                cvLine(imgDelaunay, cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt), cvPointFrom32f(cvSubdiv2DEdgeDst(edge2)->pt), cvScalar(0, 255, 255));
                cvCircle(imgDelaunay, org, 2, cvScalar(255, 0, 0), -1);
                cvCircle(imgDelaunay, dest, 2, cvScalar(255, 0, 0), 1);

                CvPoint puntos[3];
                int nPuntos = 3;
                puntos[0] = org;
                puntos[1] = dest;
                puntos[2] = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

                cvFillConvexPoly(imgDelaunay, &puntos[0], nPuntos, cvScalar(0, 0, 255)); //CV_RGB(rand()&255,rand()&255,rand()&255));
            }

            CV_NEXT_SEQ_ELEM(elem_size, reader);
        }* /

	cvShowImage("Delaunay", imgDelaunay);
	cvShowImage("Delaunay2", imgDelaunay2);

	cvReleaseImage(&imgDelaunay);
	cvReleaseImage(&imgDelaunay2);
}
*/

bool CViewMorphing::areSameTriangles(triangle tri1, triangle tri2) {
    if ((tri1.p1.x == tri2.p1.x) && (tri1.p1.y == tri2.p1.y) &&
        (tri1.p2.x == tri2.p2.x) && (tri1.p2.y == tri2.p2.y) &&
        (tri1.p3.x == tri2.p3.x) && (tri1.p3.y == tri2.p3.y))
        return true;
    if ((tri1.p1.x == tri2.p2.x) && (tri1.p1.y == tri2.p2.y) &&
        (tri1.p2.x == tri2.p3.x) && (tri1.p2.y == tri2.p3.y) &&
        (tri1.p3.x == tri2.p1.x) && (tri1.p3.y == tri2.p1.y))
        return true;
    if ((tri1.p1.x == tri2.p3.x) && (tri1.p1.y == tri2.p3.y) &&
        (tri1.p2.x == tri2.p1.x) && (tri1.p2.y == tri2.p1.y) &&
        (tri1.p3.x == tri2.p2.x) && (tri1.p3.y == tri2.p2.y))
        return true;
    if ((tri1.p1.x == tri2.p3.x) && (tri1.p1.y == tri2.p3.y) &&
        (tri1.p2.x == tri2.p2.x) && (tri1.p2.y == tri2.p2.y) &&
        (tri1.p3.x == tri2.p1.x) && (tri1.p3.y == tri2.p1.y))
        return true;
    if ((tri1.p1.x == tri2.p2.x) && (tri1.p1.y == tri2.p2.y) &&
        (tri1.p2.x == tri2.p1.x) && (tri1.p2.y == tri2.p1.y) &&
        (tri1.p3.x == tri2.p3.x) && (tri1.p3.y == tri2.p3.y))
        return true;
    if ((tri1.p1.x == tri2.p1.x) && (tri1.p1.y == tri2.p1.y) &&
        (tri1.p2.x == tri2.p3.x) && (tri1.p2.y == tri2.p3.y) &&
        (tri1.p3.x == tri2.p2.x) && (tri1.p3.y == tri2.p2.y))
        return true;

    return false;
}

bool CViewMorphing::areSameTrianglesIndexed(triangle tri1, triangle tri2) {
//    cout << "Comparing: [" << tri1.index1 << ", " << tri1.index2 << ", " << tri1.index3 << "] ::: [";
//    cout << tri2.index1 << ", " << tri2.index2 << ", " << tri2.index3 << endl;

    if ((tri1.index1 == tri2.index1) && (tri1.index2 == tri2.index2) && (tri1.index3 == tri2.index3))
        return true;

    /*if ((tri1.index1 == tri2.index2) && (tri1.index2 == tri2.index3) && (tri1.index3 == tri2.index1))
        return true;

    if ((tri1.index1 == tri2.index3) && (tri1.index2 == tri2.index1) && (tri1.index3 == tri2.index2))
        return true;

    /*if ((tri1.index1 == tri2.index3) && (tri1.index2 == tri2.index2) && (tri1.index3 == tri2.index1))
        return true;

    if ((tri1.index1 == tri2.index2) && (tri1.index2 == tri2.index1) && (tri1.index3 == tri2.index3))
        return true;

    if ((tri1.index1 == tri2.index1) && (tri1.index2 == tri2.index3) && (tri1.index3 == tri2.index2))
        return true;*/

    return false;
}

bool CViewMorphing::areTrianglesEquipositioned(triangle tri1, triangle tri2) {

    if ((tri1.p1.x < tri1.p2.x) && (tri2.p1.x >= tri2.p2.x)) return false;
    if ((tri1.p1.x < tri1.p3.x) && (tri2.p1.x >= tri2.p3.x)) return false;

    if ((tri1.p1.x > tri1.p2.x) && (tri2.p1.x <= tri2.p2.x)) return false;
    if ((tri1.p1.x > tri1.p3.x) && (tri2.p1.x <= tri2.p3.x)) return false;

    if ((tri1.p1.y < tri1.p2.y) && (tri2.p1.y >= tri2.p2.y)) return false;
    if ((tri1.p1.y < tri1.p3.y) && (tri2.p1.y >= tri2.p3.y)) return false;

    if ((tri1.p1.y > tri1.p2.y) && (tri2.p1.y <= tri2.p2.y)) return false;
    if ((tri1.p1.y > tri1.p3.y) && (tri2.p1.y <= tri2.p3.y)) return false;

    if ((tri1.p2.x < tri1.p3.x) && (tri2.p2.x >= tri2.p3.x)) return false;
    if ((tri1.p2.x > tri1.p3.x) && (tri2.p2.x <= tri2.p3.x)) return false;

    if ((tri1.p2.y < tri1.p3.y) && (tri2.p2.y >= tri2.p3.y)) return false;
    if ((tri1.p2.y > tri1.p3.y) && (tri2.p2.y <= tri2.p3.y)) return false;

    return true;
}


void CViewMorphing::cleanUsingDelaunay(IplImage * img1, IplImage * img2) {
    clock_t myTime = clock();
    // Calculamos los triangulos

        // Inicializamos las estructuras
    CvMat * refs1 = cvCreateMat(size.height, size.width, CV_32SC1);
    CvMat * refs2 = cvCreateMat(size.height, size.width, CV_32SC1);
    cvSet(refs1, cvScalar(-1));
    cvSet(refs2, cvScalar(-1));

    int numTriangles[numberOfFeatures];
    
    CvRect rect1 = cvRect(-1, -1, size.width + 2, size.height + 2);
    CvRect rect2 = cvRect(-1, -1, size.width + 2, size.height + 2);

    CvMemStorage * storage1 = cvCreateMemStorage(0);
    CvMemStorage * storage2 = cvCreateMemStorage(0);

    CvSubdiv2D * subdiv1 = cvCreateSubdivDelaunay2D(rect1, storage1);
    CvSubdiv2D * subdiv2 = cvCreateSubdivDelaunay2D(rect2, storage2);

    for (int i = 0; i < numberOfFeatures; i++) {
        if ((points1[i].x < 0) || (points1[i].y < 0) || (points2[i].x < 0) || (points2[i].y < 0) ||
            (points1[i].x >= size.width) || (points1[i].y >= size.height) ||
            (points2[i].x >= size.width) || (points2[i].y >= size.height)) continue;

        cvSubdivDelaunay2DInsert(subdiv1, points1[i]);
        cvSubdivDelaunay2DInsert(subdiv2, points2[i]);

        CvPoint p1 = cvPointFrom32f(points1[i]);
        CvPoint p2 = cvPointFrom32f(points2[i]);
        cvSetReal2D(refs1, p1.y, p1.x, i);
        cvSetReal2D(refs2, p2.y, p2.x, i);

        numTriangles[i] = 0;
    }
    cvCalcSubdivVoronoi2D(subdiv1);
    cvCalcSubdivVoronoi2D(subdiv2);   

    vector <triangle> triangles1;
    vector <triangle> triangles2;

    vector<triangle> trianglesRef1[size.height][size.width];
    vector<triangle> trianglesRef2[size.height][size.width];   

    // Recorremos los triángulos para el conjunto 1 de puntos
    CvSeqReader  reader;
    int total = subdiv1->edges->total;
    int elem_size = subdiv1->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv1->edges), &reader, 0 );

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            triangle tri;
            CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_NEXT_AROUND_ORG);            

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles1.size(); j++) {                    
                    if (areSameTriangles(tri, triangles1.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles1.push_back(tri);
                    trianglesRef1[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef1[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef1[tri.p3.y][tri.p3.x].push_back(tri);

                    (numTriangles[(int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x)])++;
                }
            }

            edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles1.size(); j++) {
                    if (areSameTriangles(tri, triangles1.at(j)) == true) {                        
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles1.push_back(tri);
                    trianglesRef1[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef1[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef1[tri.p3.y][tri.p3.x].push_back(tri);

                    (numTriangles[(int)cvGetReal2D(refs1, tri.p1.y, tri.p1.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p2.y, tri.p2.x)])++;
                    (numTriangles[(int)cvGetReal2D(refs1, tri.p3.y, tri.p3.x)])++;
                }
            }

        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    // Recorremos los triángulos para el conjunto 2 de puntos
    total = subdiv2->edges->total;
    elem_size = subdiv2->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv2->edges), &reader, 0 );

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            triangle tri;
            CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_NEXT_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs2, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs2, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs2, tri.p3.y, tri.p3.x);

                bool add = true;
                for (int j = 0; j < triangles2.size(); j++) {
                    if (areSameTriangles(tri, triangles2.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles2.push_back(tri);
                    trianglesRef2[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef2[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef2[tri.p3.y][tri.p3.x].push_back(tri);
                }
            }

            edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_ORG);

            tri.p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            tri.p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);
            tri.p3 = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            if (! ((tri.p1.x < 0) || (tri.p1.y < 0) || (tri.p1.x >= size.width) || (tri.p1.y >= size.height) ||
                    (tri.p2.x < 0) || (tri.p2.y < 0) || (tri.p2.x >= size.width) || (tri.p2.y >= size.height) ||
                    (tri.p3.x < 0) || (tri.p3.y < 0) || (tri.p3.x >= size.width) || (tri.p3.y >= size.height))) {

                tri.index1 = (int)cvGetReal2D(refs2, tri.p1.y, tri.p1.x);
                tri.index2 = (int)cvGetReal2D(refs2, tri.p2.y, tri.p2.x);
                tri.index3 = (int)cvGetReal2D(refs2, tri.p3.y, tri.p3.x);                

                bool add = true;
                for (int j = 0; j < triangles2.size(); j++) {
                    if (areSameTriangles(tri, triangles2.at(j)) == true) {
                        add = false;
                        break;
                    }
                }
                if (add == true) {
                    triangles2.push_back(tri);
                    trianglesRef2[tri.p1.y][tri.p1.x].push_back(tri);
                    trianglesRef2[tri.p2.y][tri.p2.x].push_back(tri);
                    trianglesRef2[tri.p3.y][tri.p3.x].push_back(tri);
                }
            }

        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    /*vector <triangle> removedTriangles;

    /*for (int i = 0; i < triangles1.size(); i++) {
        triangle tri1 = triangles1.at(i);
        int index = (int)cvGetReal2D(refs1, tri1.p1.y, tri1.p1.x);
        CvPoint p2 = cvPointFrom32f(points2[index]);
        vector <triangle> triangles2Check = trianglesRef2[p2.y][p2.x];
        bool exists = false;
        for (int j = 0; j < triangles2Check.size(); j++) {
            if (areSameTriangles(tri1, triangles2.at(j)) == true) {
                exists = true;
                break;
            }
        }
        if (exists == false) {
            removedTriangles.push_back(tri1);

            (numTriangles[(int)cvGetReal2D(refs1, tri1.p1.y, tri1.p1.x)])--;
            (numTriangles[(int)cvGetReal2D(refs1, tri1.p2.y, tri1.p2.x)])--;
            (numTriangles[(int)cvGetReal2D(refs1, tri1.p3.y, tri1.p3.x)])--;
        }
    }*/
    vector <int> accepted;    

    for (int i = 0; i < triangles1.size(); i++) {
        triangle tri1 = triangles1.at(i);        
        for (int j = 0; j < triangles2.size(); j++) {
            if ((areSameTrianglesIndexed(tri1, triangles2.at(j)) == true) &&
                (areTrianglesEquipositioned(tri1, triangles2.at(j)) == true)) {
                //hasCorresp = true;
                bool add1 = true, add2 = true, add3 = true;
                /*for (int k = 0; k < accepted.size(); i++) {
                    if (accepted.at(k) == tri1.index1) add1 = false;
                    if (accepted.at(k) == tri1.index2) add2 = false;
                    if (accepted.at(k) == tri1.index3) add3 = false;

                    if ((add1 || add2 || add3) == false)
                        break;
                }*/
                //cout << "Accepted " << endl;
                if (add1) accepted.push_back(tri1.index1);
                if (add2) accepted.push_back(tri1.index2);
                if (add3) accepted.push_back(tri1.index3);
                break;
            }
        }
    }
    
    CvPoint2D32f * tmp1 = points1;
    CvPoint2D32f * tmp2 = points2;
    int oldFeatures = numberOfFeatures;

    points1 = new CvPoint2D32f[accepted.size()];
    points2 = new CvPoint2D32f[accepted.size()];

    for (int i = 0; i < accepted.size(); i++) {        
        points1[i] = tmp1[accepted.at(i)];
        points2[i] = tmp2[accepted.at(i)];
    }
    numberOfFeatures = accepted.size();

    /*for (int i = 0; i < numberOfFeatures; i++) {
        /*CvPoint p1 = cvPointFrom32f(points1[i]);
        CvPoint p2 = cvPointFrom32f(points2[i]);

        vector <triangle> listTriangles1 = trianglesRef1[p1.y][p1.x];
        vector <triangle> listTriangles2 = trianglesRef2[p2.y][p2.x];

        cout << "Triangulos1: ";
        for (int i = 0; i < listTriangles1.size(); i++) {
            triangle triCoord = listTriangles1.at(i);
            triangle tri;
            tri.p1 = cvPoint((int)cvGetReal2D)
            cout << "[ " << tri.p1.x << ", " << tri.p1.y << " ;; ";
            cout << tri.p2.x << ", " << tri.p2.y << " ;; ";
            cout << tri.p3.x << ", " << tri.p3.y << " ] ";
        }
        cout << endl << "Triangulos2";
        for (int i = 0; i < listTriangles2.size(); i++) {
            triangle tri = listTriangles2.at(i);
            cout << "[ " << tri.p1.x << ", " << tri.p1.y << " ;; ";
            cout << tri.p2.x << ", " << tri.p2.y << " ;; ";
            cout << tri.p3.x << ", " << tri.p3.y << " ] ";
        }
        cout << endl;

        */

        /*bool accepted = false;
        for (int j = 0; j < listTriangles1.size(); j++) {
            for (int k = 0; k < listTriangles2.size(); k++) {
                if (areSameTriangles(listTriangles1.at(j), listTriangles2.at(k)) == true) {
                    accepted = true;
                    break;
                }
            }
            if (accepted == true)
                break;
        }
        if (accepted == true) {
            accepted1.push_back(p1);
            accepted2.push_back(p2);
        } else {
            removedPoints.push_back(p1);
        }* /

    }*/

    // Mostramos los resultados, para comprobar que todo fue bien
    IplImage * delaunay1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * delaunay2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * removed = cvCreateImage(size, IPL_DEPTH_8U, 3);

    cvCopyImage(img1, delaunay1);
    cvCopyImage(img2, delaunay2);
    cvCopyImage(img2, removed);

    //cvZero(delaunay1);
    //cvZero(delaunay2);

    for (int i = 0; i < triangles1.size(); i++) {
        triangle tri = triangles1.at(i);

        CvPoint poly[] = { tri.p1, tri.p2, tri.p3 };
        cvFillConvexPoly(delaunay1, poly, 3, CV_RGB(rand()&255,rand()&255,rand()&255));
        cvCircle(removed, cvPointFrom32f(points1[tri.index1]), 2, cvScalar(255, 0, 0));
        cvCircle(removed, cvPointFrom32f(points1[tri.index2]), 2, cvScalar(255, 0, 0));
        cvCircle(removed, cvPointFrom32f(points1[tri.index3]), 2, cvScalar(255, 0, 0));
        //cvFillConvexPoly(delaunay1, poly, 3, CV_RGB(255, 255, 255));
        //cvFillConvexPoly(removed, poly, 3, CV_RGB(255, 255, 255));
    }

    for (int i = 0; i < triangles2.size(); i++) {
        triangle tri = triangles2.at(i);

        CvPoint poly[] = { tri.p1, tri.p2, tri.p3 };
        cvFillConvexPoly(delaunay2, poly, 3, CV_RGB(rand()&255,rand()&255,rand()&255));
        //cvCircle(delaunay2, tri.p1, 2, cvScalar(255, 0, 0));
        //cvFillConvexPoly(delaunay2, poly, 3, CV_RGB(255, 255, 255));
    }

    for (int i = 0; i < oldFeatures; i++) {
        cvCircle(removed, cvPointFrom32f(tmp1[i]), 2, cvScalar(0, 0, 255));
    }
    for (int i = 0; i < numberOfFeatures; i++) {
        cvCircle(removed, cvPointFrom32f(points1[i]), 2, cvScalar(255, 0, 0));
    }

    /*for (int i = 0; i < removedTriangles.size(); i++) {
        triangle tri = removedTriangles.at(i);

        CvPoint poly[] = { tri.p1, tri.p2, tri.p3 };
        cvFillConvexPoly(removed, poly, 3, CV_RGB(255, 0, 0));
    }*/

    cvNamedWindow("Delaunay1");
    cvShowImage("Delaunay1", delaunay1);
    cvNamedWindow("Delaunay2");
    cvShowImage("Delaunay2", delaunay2);
    cvNamedWindow("Removed");
    cvShowImage("Removed", removed);

    cvReleaseImage(&delaunay1);
    cvReleaseImage(&delaunay2);
    cvReleaseImage(&removed);

    cvReleaseMat(&refs1);
    cvReleaseMat(&refs2);

    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo cleanUsingDelaunay = " << time << endl;

}

void CViewMorphing::generatePointsByDelaunay(IplImage * img1, IplImage * img2) {
    clock_t myTime = clock();
    // Calculamos los triangulos

    CvRect rect1 = cvRect(-1, -1, size.width + 2, size.height + 2);
    CvRect rect2 = cvRect(-1, -1, size.width + 2, size.height + 2);

    CvMemStorage * storage1 = cvCreateMemStorage(0);
    CvMemStorage * storage2 = cvCreateMemStorage(0);

    CvSubdiv2D * subdiv1 = cvCreateSubdivDelaunay2D(rect1, storage1);
    CvSubdiv2D * subdiv2 = cvCreateSubdivDelaunay2D(rect2, storage2);

    CvMat * refs1 = cvCreateMat(size.height, size.width, CV_32SC1);
    CvMat * refs2 = cvCreateMat(size.height, size.width, CV_32SC1);
    cvSet(refs1, cvScalar(-1));
    cvSet(refs2, cvScalar(-1));

    for (int i = 0; i < numberOfFeatures; i++) {
        if ((points1[i].x < 0) || (points1[i].y < 0) || (points2[i].x < 0) || (points2[i].y < 0) ||
            (points1[i].x >= size.width) || (points1[i].y >= size.height) ||
            (points2[i].x >= size.width) || (points2[i].y >= size.height)) continue;

        cvSubdivDelaunay2DInsert(subdiv1, points1[i]);
        cvSubdivDelaunay2DInsert(subdiv2, points2[i]);

        CvPoint p1 = cvPointFrom32f(points1[i]);
        CvPoint p2 = cvPointFrom32f(points2[i]);
        cvSetReal2D(refs1, p1.y, p1.x, i);
        cvSetReal2D(refs2, p2.y, p2.x, i);
    }
    cvCalcSubdivVoronoi2D(subdiv1);
    cvCalcSubdivVoronoi2D(subdiv2);

    /*CvRect rect = cvRect(-1, -1, size.width + 2, size.height + 2);
    CvMemStorage * storage = cvCreateMemStorage(0);
    CvSubdiv2D * subdiv = cvCreateSubdivDelaunay2D(rect, storage);

    for (int i = 0; i < numberOfFeatures; i++) {
        if ((points1[i].x < 0) || (points1[i].y < 0) || (points2[i].x < 0) || (points2[i].y < 0) ||
            (points1[i].x >= size.width) || (points1[i].y >= size.height) ||
            (points2[i].x >= size.width) || (points2[i].y >= size.height)) continue;

        cvSubdivDelaunay2DInsert(subdiv, points1[i]);

    }
    cvCalcSubdivVoronoi2D(subdiv);

    // Recorremos los triángulos para el conjunto 1 de puntos
    CvSeqReader  reader;
    int total = subdiv->edges->total;
    int elem_size = subdiv->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv->edges), &reader, 0 );  

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            CvPoint p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            CvPoint p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);

            if ((p1.x < 0) || (p1.y < 0) || (p2.x < 0) || (p2.y < 0) ||
                (p1.x >= size.width) || (p1.y >= size.height) ||
                (p2.x >= size.width) || (p2.y >= size.height)) {

                
            }
        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }*/

    IplImage * imgDel1 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgDel2 = cvCreateImage(size, IPL_DEPTH_8U, 3);

    cvCopyImage(img2, imgDel1);
    cvCopyImage(img1, imgDel2);

    // Recorremos los triángulos para el conjunto 1 de puntos
    CvSeqReader  reader;
    int total = subdiv1->edges->total;
    int elem_size = subdiv1->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv1->edges), &reader, 0 );

    vector <CvPoint> pts1;
    vector <CvPoint> pts2;

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            CvPoint p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            CvPoint p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);            

            if ((p1.x < 0) || (p1.y < 0) || (p2.x < 0) || (p2.y < 0) ||
                (p1.x >= size.width) || (p1.y >= size.height) ||
                (p2.x >= size.width) || (p2.y >= size.height)) {

                CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
                cvLine(imgDel1, p1, p2, color, 1);
                
                int index = -1;
                
                if ((p1.x < 0) || (p1.y < 0) || (p1.x >= size.width) || (p1.y >= size.height)) {                    
                    if (! ((p2.x < 0) || (p2.y < 0) || (p2.x >= size.width) || (p2.y >= size.height))) {
                        index = (int)cvGetReal2D(refs1, p2.y, p2.x);
                        CvPoint p3 = calculaCorte(p2, p1);
                        CvPoint p4 = calculaCorte(cvPointFrom32f(points2[index]), p1);

                        CvPoint p3b = calculaPuntoRelativo(p1, cvPointFrom32f(points2[index]), p3);
                        CvPoint p4b = calculaPuntoRelativo(p1, p2, p4);

                        if (! ((p3.x < 0) || (p3.y < 0) || (p3b.x < 0) || (p3b.y < 0) ||
                               (p3.x >= size.width) || (p3.y >= size.height) ||
                               (p3b.x >= size.width) || (p3b.y >= size.height))) {

                            pts1.push_back(p3);
                            pts2.push_back(p3b);

                            CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
                            cvCircle(imgDel1, p3, 2, color, -1);
                            cvCircle(imgDel2, p3b, 2, color, -1);
                        }

                        if (! ((p4.x < 0) || (p4.y < 0) || (p4b.x < 0) || (p4b.y < 0) ||
                               (p4.x >= size.width) || (p4.y >= size.height) ||
                               (p4b.x >= size.width) || (p4b.y >= size.height))) {

                            pts1.push_back(p4);
                            pts2.push_back(p4b);

                            CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
                            cvCircle(imgDel2, p4, 2, color, -1);
                            cvCircle(imgDel1, p4b, 2, color, -1);
                        }

                        // Habría que calcular los puntos relativos al corte en ambos sentidos
                        // (con el corte p2-p3 hacia la imagen 2, y points2[index]-p4 hacia la imagen 1)
                        // Para ambos pares se comprueba que queden dentro de la imagen, si no no se añaden
                        // NOTA: Sólo una de las dos parejas debería poderse añadir a la vez.
                    }
                } else {                    
                    if (! ((p1.x < 0) || (p1.y < 0) || (p1.x >= size.width) || (p1.y >= size.height))) {
                        index = (int)cvGetReal2D(refs1, p1.y, p1.x);
                    }
                }

                //if (index != -1)
                //    cvCircle(imgDel2, cvPointFrom32f(points2[index]), 2, color, -1);               

                //cvLine(imgDel1, p1, p2, cvScalar(0, 255, 0), 1);
            } else {
                cvLine(imgDel1, p1, p2, cvScalar(0, 255, 0), 1);
            }
        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    total = subdiv2->edges->total;
    elem_size = subdiv2->edges->elem_size;

    cvStartReadSeq((CvSeq*)(subdiv2->edges), &reader, 0 );

    for (int i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D *) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {
            CvPoint p1 = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt);
            CvPoint p2 = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt);

            cvLine(imgDel2, p1, p2, cvScalar(0, 255, 0), 1);
        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    cvNamedWindow("imgDel1", 1);
    cvNamedWindow("imgDel2", 1);

    cvShowImage("imgDel1", imgDel1);
    cvShowImage("imgDel2", imgDel2);

    cvReleaseImage(&imgDel1);
    cvReleaseImage(&imgDel2);

    /*CvPoint2D32f * tmp1 = points1;
    CvPoint2D32f * tmp2 = points2;
    
    points1 = new CvPoint2D32f[numberOfFeatures + 3];
    points2 = new CvPoint2D32f[numberOfFeatures + 3];
    for (int i = 0; i < numberOfFeatures; i++) {
        points1[i] = tmp1[i];
        points2[i] = tmp2[i];
    }
        
    points1[numberOfFeatures] = cvPoint2D32f(-1, 965);
    points2[numberOfFeatures] = cvPoint2D32f(-1, 965);

    points1[numberOfFeatures + 1] = cvPoint2D32f(965, -1);
    points2[numberOfFeatures + 1] = cvPoint2D32f(965, -1);

    points1[numberOfFeatures + 2] = cvPoint2D32f(-967, -967);
    points2[numberOfFeatures + 2] = cvPoint2D32f(-967, -967);

    numberOfFeatures += 3;*/
}

// a---------------b
// |               |
// |      ·p1 ------------ ·p2
// |               |
// c---------------d
CvPoint CViewMorphing::calculaCorte(CvPoint p1, CvPoint p2) {
    double m = (double)(p2.y - p1.y) / (double)(p2.x - p1.x);

    if (p2.y <= p1.y) {
        // Comprobamos para la recta a-b
        double xTemp = ((m * p1.x) - p1.y) / m;        
        if ((xTemp >= 0) && (xTemp <= size.width - 1))
            return cvPoint(xTemp, 0);
    } else {
        // Comprobamos para la recta c-d
        double xTemp = ((size.height - 1) + (m * p1.x) - p1.y) / m;        
        if ((xTemp >= 0) && (xTemp <= size.width - 1))
            return cvPoint(xTemp, size.height - 1);
    }
    
    if (p2.x <= p1.x) {
        // Comprobamos para la recta a-c
        double yTemp = -(m * p1.x) + p1.y;        
        if ((yTemp > 0) && (yTemp < size.height - 1))
            return cvPoint(0, yTemp);

    } else {
        // Comprobamos para la recta b-d
        double yTemp = (m * (size.width - 1)) - (m * p1.x) + p1.y;        
        if ((yTemp > 0) && (yTemp < size.height - 1))
            return cvPoint(size.width - 1, yTemp);
    }

    return cvPoint(-1, -1);
}

CvPoint CViewMorphing::calculaPuntoRelativo(CvPoint p1, CvPoint p2, CvPoint corte) {
    double d = sqrt(pow(p1.x - corte.x, 2.0) + pow(p1.y - corte.y, 2.0));
    double h = sqrt(pow(p1.x - p2.x, 2.0) + pow(p1.y - p2.y, 2.0));

    double a = p2.y - p1.y;
    double b = p2.x - p1.x;

    double x = (d * b / h) + p1.x;
    double y = (a * d / h) + p1.y;

    return cvPoint(x, y);
}

void CViewMorphing::showFeatureTracking(IplImage * img1, IplImage * img2) {
	cvNamedWindow("Imagen1", 1);
	cvNamedWindow("Imagen2", 1);
	//cvNamedWindow("Tracking", 1);
	cvNamedWindow("Tracking2", 1);

	IplImage * imgA = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage * imgB = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage * imgC = cvCreateImage(cvSize(size.width * 2, size.height), IPL_DEPTH_8U, 3);

	cout << numberOfFeatures << endl;
	cvCvtColor(img1, imgA, CV_GRAY2BGR);
	cvCvtColor(img2, imgB, CV_GRAY2BGR);	
	for (int i = 0; i < numberOfFeatures; i++) {
		//cout << points1[i].x << ", " << points1[i].y << endl;
		cvCircle(imgA, cvPointFrom32f(points1[i]), 2, cvScalar(255, 0, 0));
		cvCircle(imgB, cvPointFrom32f(points2[i]), 2, cvScalar(255, 0, 0));
	}	

	cvShowImage("Imagen1", imgA);
	cvShowImage("Imagen2", imgB);

	/*cvWaitKey(0);

	for (int i = 0; i < numberOfFeatures; i++) {
		//if (points1[i].x > 100) continue;
		cvZero(imgC);
		cvCvtColor(img1, imgA, CV_GRAY2BGR);
		cvSetImageROI(imgC, cvRect(0, 0, size.width, size.height));
		cvAdd(imgA, imgC, imgC);
		cvCvtColor(img2, imgB, CV_GRAY2BGR);
		cvSetImageROI(imgC, cvRect(size.width, 0, size.width, size.height));
		cvAdd(imgB, imgC, imgC);
		cvResetImageROI(imgC);

		cvLine(imgC, cvPointFrom32f(points1[i]), cvPoint((int)points2[i].x + size.width, (int)points2[i].y), cvScalar(0, 255, 0));

		cvShowImage("Tracking", imgC);

		int code = cvWaitKey(0);

		if (code == 27) break;
	}

	cvZero(imgC);
	cvCvtColor(img1, imgA, CV_GRAY2BGR);
	cvSetImageROI(imgC, cvRect(0, 0, size.width, size.height));
	cvAdd(imgA, imgC, imgC);
	cvCvtColor(img2, imgB, CV_GRAY2BGR);
	cvSetImageROI(imgC, cvRect(size.width, 0, size.width, size.height));
	cvAdd(imgB, imgC, imgC);
	cvResetImageROI(imgC);

	for (int i = 0; i < numberOfFeatures; i++) {
		cvLine(imgC, cvPointFrom32f(points1[i]), cvPoint((int)points2[i].x + size.width, (int)points2[i].y), cvScalar(0, 255, 0));
	}

	cvShowImage("Tracking", imgC);*/

	cvZero(imgC);
	cvSetImageROI(imgC, cvRect(0, 0, size.width, size.height));
	cvAdd(imgA, imgC, imgC);
	cvCvtColor(img2, imgB, CV_GRAY2BGR);
	cvSetImageROI(imgC, cvRect(size.width, 0, size.width, size.height));
	cvAdd(imgB, imgC, imgC);
	cvResetImageROI(imgC);
	for (int i = 0; i < numberOfFeatures; i++) {
		CvScalar color = cvScalar(rand() % 255, rand() % 255, rand() % 255);
		cvCircle(imgC, cvPointFrom32f(points1[i]), 2, color, -1);
		cvCircle(imgC, cvPoint((int)points2[i].x + size.width, (int)points2[i].y), 2, color, -1);
	}

	cvShowImage("Tracking2", imgC);

	//cvWaitKey(0);

	cvReleaseImage(&imgA);
	cvReleaseImage(&imgB);
	cvReleaseImage(&imgC);
}

void CViewMorphing::imageMorphing(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * &morphedImage) {
	if (numberOfFeatures < 8) {
		cerr << "Method cannot be applied if there are less than 8 points" << endl;
		return;
	}	

	cvNamedWindow("Img1CAntes", 1);
	cvShowImage("Img1CAntes", img1C);
	cvNamedWindow("Img2CAntes", 1);
	cvShowImage("Img2CAntes", img2C);

	IplImage * img1CAntes = cvCloneImage(img1C);
	IplImage * img2CAntes = cvCloneImage(img2C);
	
	// Computes fundamental matrix
	CvMat * fundamentalMatrix = cvCreateMat(3, 3, CV_32FC1);
	//CvMat * mPoints1 = cvCreateMat(2, numberOfFeatures, CV_32FC1);
	//CvMat * mPoints2 = cvCreateMat(2, numberOfFeatures, CV_32FC1);
	CvMat * mStatus = cvCreateMat(1, numberOfFeatures, CV_8UC1);

	/*for (int i = 0; i < numberOfFeatures; i++) {
		cvmSet(mPoints1, 0, i, points1[i].x);
		cvmSet(mPoints1, 1, i, points1[i].y);

		cvmSet(mPoints2, 0, i, points2[i].x);
		cvmSet(mPoints2, 1, i, points2[i].y);
	}*/

	vector <CvPoint2D32f> points[2];
	vector <CvPoint2D32f> p2;

	for (int i = 0; i < numberOfFeatures; i++) {
		points[0].push_back(points1[i]);
		points[1].push_back(points2[i]);
	/*	cvmSet(mPoints1, 0, i, points1[i].x);
		cvmSet(mPoints1, 1, i, points1[i].y);

		cvmSet(mPoints2, 0, i, points2[i].x);
		cvmSet(mPoints2, 1, i, points2[i].y);*/
	}

	CvMat mPoints1 = cvMat(1, numberOfFeatures, CV_32FC2, &points[0][0]);
	CvMat mPoints2 = cvMat(1, numberOfFeatures, CV_32FC2, &points[1][0]);

	int res = cvFindFundamentalMat(&mPoints1, &mPoints2, fundamentalMatrix, CV_FM_RANSAC, 1.0, 0.99, mStatus);

	if (! res) {
		cerr << "Fundamental matrix was not found" << endl;
		return;
	}

	// Compute corresponding epilines
	CvMat * epilines1 = cvCreateMat(3, numberOfFeatures, CV_32FC1);
	CvMat * epilines2 = cvCreateMat(3, numberOfFeatures, CV_32FC1);

	cvComputeCorrespondEpilines(&mPoints1, 1, fundamentalMatrix, epilines1);
	cvComputeCorrespondEpilines(&mPoints2, 2, fundamentalMatrix, epilines2);

	// Compute number and length of epilines
	int lineCount = 0;	
	CvMatrix3 fundamentalMatrix3;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamentalMatrix3.m[i][j] = cvmGet(fundamentalMatrix, j, i);			
		}		
	}

	cvMakeScanlines(&fundamentalMatrix3, size, 0, 0, 0, 0, &lineCount);

	int * lengthEpilines1 = (int*)(calloc( lineCount * 2, sizeof(int) * 4));
	int * lengthEpilines2 = (int*)(calloc( lineCount * 2, sizeof(int) * 4));

	int * scanLines1 = (int*)(calloc( lineCount * 2, sizeof(int) * 4));
	int * scanLines2 = (int*)(calloc( lineCount * 2, sizeof(int) * 4));

	cvMakeScanlines(&fundamentalMatrix3, size, scanLines1, scanLines2, lengthEpilines1, lengthEpilines2, &lineCount);

	// Prewarp the source images
	uchar * preWarpData1 = new uchar[max(size.width, size.height) * lineCount * 3];
	uchar * preWarpData2 = new uchar[max(size.width, size.height) * lineCount * 3];

	cvPreWarpImage(lineCount, img1C, preWarpData1, lengthEpilines1, scanLines1);
	cvPreWarpImage(lineCount, img2C, preWarpData2, lengthEpilines2, scanLines2);

	// Find runs for the rectified image (series of pixels with similar intensity)
	int * numRuns1 = new int[lineCount];
	int * numRuns2 = new int[lineCount];

	int * runs1 = new int[size.width * lineCount];
	int * runs2 = new int[size.width * lineCount];	

	cvFindRuns(lineCount, preWarpData1, preWarpData2, lengthEpilines1, lengthEpilines2, runs1, runs2, numRuns1, numRuns2);

	// Find correspondences between runs
	int * runCorrelation1 = new int[max(size.width, size.height) * lineCount * 3];
	int * runCorrelation2 = new int[max(size.width, size.height) * lineCount * 3];

	cvDynamicCorrespondMulti(lineCount, runs1, numRuns1, runs2, numRuns2, runCorrelation1, runCorrelation2);

	// Morph the images
	uchar * tmpDataImageDst = new uchar[size.width * (size.height + 1) * 3];
	int * numScanLinesMorphedImage = (int*)(calloc(lineCount * 2, sizeof(int) * 4));

	cvMorphEpilinesMulti(lineCount, preWarpData1, lengthEpilines1,
						preWarpData2, lengthEpilines2, tmpDataImageDst,
						numScanLinesMorphedImage, ALPHA_WEIGHT, runs1, numRuns1, 
						runs2, numRuns2, runCorrelation1, runCorrelation2);

	// Post warp the image
	int * scanLinesMorphedImage = (int*)(calloc( lineCount * 2, sizeof(int) * 4));
	morphedImage = cvCloneImage(img2C);   

	cvPostWarpImage(lineCount, tmpDataImageDst, numScanLinesMorphedImage, morphedImage, scanLinesMorphedImage);		

	cvDeleteMoire(morphedImage);

	cvAbsDiff(img1CAntes, morphedImage, img1C);
	cvAbsDiff(img2CAntes, morphedImage, img2C);

	cvNamedWindow("Morph", 1);
	cvShowImage("Morph", morphedImage);
	cvNamedWindow("Img1C", 1);
	cvShowImage("Img1C", img1C);
	cvNamedWindow("Img2C", 1);
	cvShowImage("Img2C", img2C);

	cvWaitKey(0);
}

void CViewMorphing::morphingByStereo(IplImage * img1C, IplImage * img2C) {

	// Computes fundamental matrix
	CvMat * fundamentalMatrix = cvCreateMat(3, 3, CV_64FC1);
	//CvMat * mPoints1 = cvCreateMat(2, numberOfFeatures, CV_64FC1);
	//CvMat * mPoints2 = cvCreateMat(2, numberOfFeatures, CV_64FC1);	
	CvMat * mStatus = cvCreateMat(1, numberOfFeatures, CV_8UC1);
	CvMat * H1 = cvCreateMat(3, 3, CV_64FC1);
	CvMat * H2 = cvCreateMat(3, 3, CV_64FC1);
	CvMat * M1 = cvCreateMat(3, 3, CV_64FC1);
	CvMat * M2 = cvCreateMat(3, 3, CV_64FC1);
	CvMat * Mi = cvCreateMat(3, 3, CV_64FC1);
	CvMat * R1 = cvCreateMat(3, 3, CV_64FC1);
	CvMat * R2 = cvCreateMat(3, 3, CV_64FC1);
	CvMat * D1 = cvCreateMat(1, 5, CV_64FC1);
	CvMat * D2 = cvCreateMat(1, 5, CV_64FC1);
	CvMat * mx1 = cvCreateMat(size.height, size.width, CV_32FC1);
	CvMat * my1 = cvCreateMat(size.height, size.width, CV_32FC1);
	CvMat * mx2 = cvCreateMat(size.height, size.width, CV_32FC1);
	CvMat * my2 = cvCreateMat(size.height, size.width, CV_32FC1);

	IplImage * morphedImage1 = cvCloneImage(img1C); 
	IplImage * morphedImage2 = cvCloneImage(img2C); 

	vector <CvPoint2D32f> points[2];
	vector <CvPoint2D32f> p2;

	for (int i = 0; i < numberOfFeatures; i++) {
		points[0].push_back(points1[i]);
		points[1].push_back(points2[i]);
	/*	cvmSet(mPoints1, 0, i, points1[i].x);
		cvmSet(mPoints1, 1, i, points1[i].y);

		cvmSet(mPoints2, 0, i, points2[i].x);
		cvmSet(mPoints2, 1, i, points2[i].y);*/
	}

	CvMat mPoints1 = cvMat(1, numberOfFeatures, CV_32FC2, &points[0][0]);
	CvMat mPoints2 = cvMat(1, numberOfFeatures, CV_32FC2, &points[1][0]);

	int res = cvFindFundamentalMat(&mPoints1, &mPoints2, fundamentalMatrix, CV_FM_RANSAC, 1.0, 0.99, mStatus);

	cvStereoRectifyUncalibrated(&mPoints1, &mPoints2, fundamentalMatrix, size, H1, H2, 5);

	cvSetIdentity(M1);
	cvInvert(M1, Mi);
	cvMatMul(H1, M1, R1);
	cvMatMul(Mi, R1, R1);
	cvInvert(M2, Mi);
	cvMatMul(H2, M2, R2);
	cvMatMul(Mi, R2, R2);

	cvInitUndistortRectifyMap(M1, D1, R1, M1, mx1, my1);
	cvInitUndistortRectifyMap(M2, D2, R2, M2, mx2, my2);

	cvRemap(img1C, morphedImage1, mx1, my1);
	cvRemap(img2C, morphedImage2, mx2, my2);	

	cvNamedWindow("Imagen1", 1);
	cvShowImage("Imagen1", img1C);
	cvNamedWindow("Imagen2", 1);
	cvShowImage("Imagen2", img2C);
	cvNamedWindow("Morphed1", 1);
	cvShowImage("Morphed1", morphedImage1);
	cvNamedWindow("Morphed2", 1);
	cvShowImage("Morphed2", morphedImage2);

	cvAbsDiff(img2C, morphedImage1, morphedImage1);

	cvNamedWindow("Sub", 1);
	cvShowImage("Sub", morphedImage1);

	cvWaitKey(0);

}

vector <int> CViewMorphing::findInsideOut() {
	vector <int> indexes;

	for (int i = 0; i < triangles->rows; i++) {
		int index1 = cvGetReal2D(triangles, i, 1);
		int index2 = cvGetReal2D(triangles, i, 0);
		double sumXY = (points1[index1].x - points1[index2].x) * ((points1[index1].y + points1[index2].y));
		double sumUV = (affinePoints[index1].x - affinePoints[index2].x) * ((affinePoints[index1].y + affinePoints[index2].y));
		
		index1 = cvGetReal2D(triangles, i, 2);
		index2 = cvGetReal2D(triangles, i, 1);
		sumXY += (points1[index1].x - points1[index2].x) * ((points1[index1].y + points1[index2].y));
		sumUV += (affinePoints[index1].x - affinePoints[index2].x) * ((affinePoints[index1].y + affinePoints[index2].y));
		
		index1 = cvGetReal2D(triangles, i, 0);
		index2 = cvGetReal2D(triangles, i, 2);
		sumXY += (points1[index1].x - points1[index2].x) * ((points1[index1].y + points1[index2].y));
		sumUV += (affinePoints[index1].x - affinePoints[index2].x) * ((affinePoints[index1].y + affinePoints[index2].y));

		if (sumXY * sumUV < 0)		
			indexes.push_back(i);
	}	

	return indexes;
}

void CViewMorphing::transformaALoBruto(IplImage * img1, IplImage * img2) {
	IplImage * maskTriangulo1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * maskTriangulo2 = cvCreateImage(size, IPL_DEPTH_8U, 1);

	CvPoint2D32f puntosControl1[4];
	CvPoint2D32f puntosControl2[4];
	CvPoint pintaTriangulo1[3];
	CvPoint pintaTriangulo2[3];

	CvMat * transform = cvCreateMat(3, 3, CV_64FC1);

	IplImage * prev = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * post = cvCreateImage(size, IPL_DEPTH_8U, 1);

	cvNamedWindow("Mask1", 1);
	cvNamedWindow("Mask2", 1);
	cvNamedWindow("Prev", 1);
	cvNamedWindow("Post", 1);
	cvNamedWindow("Img2", 1);

	cvZero(post);

	for (int i = 0; i < triangles->rows; i++) {		
		puntosControl1[0] = points1[(int)cvGetReal2D(triangles, i, 0)];
		puntosControl1[1] = points1[(int)cvGetReal2D(triangles, i, 1)];
		puntosControl1[2] = points1[(int)cvGetReal2D(triangles, i, 2)];
		puntosControl1[3] = cvPoint2D32f(0, 0);

		puntosControl2[0] = points2[(int)cvGetReal2D(triangles, i, 0)];
		puntosControl2[1] = points2[(int)cvGetReal2D(triangles, i, 1)];
		puntosControl2[2] = points2[(int)cvGetReal2D(triangles, i, 2)];
		puntosControl2[3] = cvPoint2D32f(0, 0);

		for (int j = 0; j < 3; j++) {
			puntosControl1[3].x += puntosControl1[j].x;
			puntosControl1[3].y += puntosControl1[j].y;

			puntosControl2[3].x += puntosControl2[j].x;
			puntosControl2[3].y += puntosControl2[j].y;

			pintaTriangulo1[j] = cvPointFrom32f(puntosControl1[j]);
			pintaTriangulo2[j] = cvPointFrom32f(puntosControl2[j]);
		}

		puntosControl1[3].x /= 3;
		puntosControl1[3].y /= 3;

		puntosControl2[3].x /= 3;
		puntosControl2[3].y /= 3;

		// Pintamos el tri�ngulo actual		
		cvZero(maskTriangulo1);
		cvZero(maskTriangulo2);
		cvFillConvexPoly(maskTriangulo1, pintaTriangulo1, 3, cvScalar(255));
		cvFillConvexPoly(maskTriangulo2, pintaTriangulo2, 3, cvScalar(255));

		//cvGetAffineTransform(puntosControl2, puntosControl1, transform);
		cvGetPerspectiveTransform(puntosControl1, puntosControl2, transform);

		cvZero(prev);		

		cvCopy(img1, prev, maskTriangulo1);

		//cvWarpAffine(prev, prev, transform, CV_INTER_LINEAR);
		cvWarpPerspective(prev, prev, transform, CV_INTER_LINEAR);

		cvAdd(prev, post, post);

		cvShowImage("Mask1", maskTriangulo1);
		cvShowImage("Mask2", maskTriangulo2);
		cvShowImage("Prev", prev);
		cvShowImage("Post", post);					

		//cvWaitKey(0);
	}

	cvAbsDiff(img2, post, prev);
	cvShowImage("Img2", prev);	

	cvWaitKey(0);

	cvReleaseImage(&maskTriangulo1);
	cvReleaseImage(&maskTriangulo2);
}

/*% For each triangle, finding a plane to warp xy into uv is an affine 
% transformation.
%
% For an affine transformation:
%
%                   [ A D ]
% [u v] = [x y 1] * [ B E ]
%                   [ C F ]
%
% [ u1 v1 ]   [ x1 y1 1 ]   [ A D ]
% [ u2 v2 ] = [ x2 y2 1 ] * [ B E ]
% [ u3 v3 ]   [ x3 y3 1 ]   [ C F ]
%
% Rewriting the above matrix equation:
% U = X * T, where T = [A B C; D E F]'
%
% With the 3 correspondence points of each triangle, we can solve for T,
% T = X\U 
%
% see "Piecewise linear mapping functions for image registration" Ardeshir
% Goshtasby, Pattern Recognition, Vol 19, pp. 459-466, 1986.*/
void CViewMorphing::pieceWiseLinear(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask) {
	/*numberOfFeatures = 13;
	points1 = new CvPoint2D32f[numberOfFeatures];
	points2 = new CvPoint2D32f[numberOfFeatures];
	allPoints = new 
	
	CvPoint2D32f[numberOfFeatures];
	affinePoints = new CvPoint2D32f[numberOfFeatures];

	points1[0] = cvPoint2D32f(283.1923, 6.8960);
	points1[1] = cvPoint2D32f(218.6652, 19.6197);
	points1[2] = cvPoint2D32f(289.5541, 103.2322);
	points1[3] = cvPoint2D32f(223.2094, 88.6909);
	points1[4] = cvPoint2D32f(167.7707, 76.8761);
	points1[5] = cvPoint2D32f(164.1353, 45.9758);
	points1[6] = cvPoint2D32f(141.4145, 66.8789);
	points1[7] = cvPoint2D32f(220.4829, 165.9416);
	points1[8] = cvPoint2D32f(190.4915, 135.9501);
	points1[9] = cvPoint2D32f(103.2436, 77.7849);
	points1[10] = cvPoint2D32f(113.2407, 39.6140);
	points1[11] = cvPoint2D32f(50.5313, 31.4345);
	points1[12] = cvPoint2D32f(25.9929, 150.4915);

	points2[0] = cvPoint2D32f(310.4300, 13.8629);
	points2[1] = cvPoint2D32f(237.5157, 25.7114);
	points2[2] = cvPoint2D32f(319.5443, 109.5629);
	points2[3] = cvPoint2D32f(241.1614, 95.8914);
	points2[4] = cvPoint2D32f(173.7157, 83.1314);
	points2[5] = cvPoint2D32f(170.9814, 53.0543);
	points2[6] = cvPoint2D32f(141.8157, 73.1057);
	points2[7] = cvPoint2D32f(270.3271, 169.7171);
	points2[8] = cvPoint2D32f(227.4900, 140.5514);
	points2[9] = cvPoint2D32f(109.0043, 84.0429);
	points2[10] = cvPoint2D32f(109.9157, 44.8514);
	points2[11] = cvPoint2D32f(57.0529, 35.7371);
	points2[12] = cvPoint2D32f(64.3443, 155.1343);

	allPoints[0] = cvPoint2D32f(223.2094, 88.6909);
	allPoints[1] = cvPoint2D32f(218.6652, 19.6197);
	allPoints[2] = cvPoint2D32f(283.1923, 6.8960);
	allPoints[3] = cvPoint2D32f(289.5541, 103.2322);
	allPoints[4] = cvPoint2D32f(167.7707, 76.8761);
	allPoints[5] = cvPoint2D32f(141.4145, 66.8789);
	allPoints[6] = cvPoint2D32f(190.4915, 135.9501);
	allPoints[7] = cvPoint2D32f(220.4829, 165.9416);
	allPoints[8] = cvPoint2D32f(50.5313, 31.4345);
	allPoints[9] = cvPoint2D32f(25.9929, 150.4915);
	allPoints[10] = cvPoint2D32f(103.2436, 77.7849);
	allPoints[11] = cvPoint2D32f(113.2407, 39.6140);
	allPoints[12] = cvPoint2D32f(164.1353, 45.9758);

	affinePoints[0] = cvPoint2D32f(298.0669, 6.5847);
	affinePoints[1] = cvPoint2D32f(227.9957, 18.9620);
	affinePoints[2] = cvPoint2D32f(285.2192, 103.4305);
	affinePoints[3] = cvPoint2D32f(215.7642, 89.9982);
	affinePoints[4] = cvPoint2D32f(156.2220, 77.4315);
	affinePoints[5] = cvPoint2D32f(160.3803, 46.9936);
	affinePoints[6] = cvPoint2D32f(128.9474, 67.4475);
	affinePoints[7] = cvPoint2D32f(226.3251, 164.5921);
	affinePoints[8] = cvPoint2D32f(193.1904, 135.2865);
	affinePoints[9] = cvPoint2D32f(96.1692, 78.6924);
	affinePoints[10] = cvPoint2D32f(105.7257, 39.0075);
	affinePoints[11] = cvPoint2D32f(58.8603, 30.0558);
	affinePoints[12] = cvPoint2D32f(39.0581, 150.9038); // */

	// Lleg� el momento de usar Delaunay (cp2tform.m, linea: 567)
	CvRect rect = cvRect(0, 0, size.width, size.height);
	CvMemStorage * storage = cvCreateMemStorage(0);

	CvSubdiv2D * subdiv = cvCreateSubdivDelaunay2D(rect, storage);	
	for (int i = 0; i < numberOfFeatures; i++) {			
		if (points1[i].x < 0 || points1[i].y < 0) continue;
		cvSubdivDelaunay2DInsert(subdiv, points1[i]);
	}

	cvCalcSubdivVoronoi2D(subdiv);

	drawDelaunay(subdiv, img1);

	//navegaPorTriangulos(subdiv);	

	/*/ FAKE_TRIANGLES - START
	// Creamos nuestros falsos triangulos para comparar con Matlab
	cvReleaseMat(&triangles);
	triangles = cvCreateMat(19, 3, CV_8UC1);

	cvSetReal2D(triangles, 0, 0, 1);
	cvSetReal2D(triangles, 1, 0, 10);
	cvSetReal2D(triangles, 2, 0, 9);
	cvSetReal2D(triangles, 3, 0, 9);
	cvSetReal2D(triangles, 4, 0, 11);
	cvSetReal2D(triangles, 5, 0, 11);
	cvSetReal2D(triangles, 6, 0, 5);
	cvSetReal2D(triangles, 7, 0, 7);
	cvSetReal2D(triangles, 8, 0, 7);
	cvSetReal2D(triangles, 9, 0, 4);
	cvSetReal2D(triangles, 10, 0, 4);
	cvSetReal2D(triangles, 11, 0, 4);
	cvSetReal2D(triangles, 12, 0, 4);
	cvSetReal2D(triangles, 13, 0, 4);
	cvSetReal2D(triangles, 14, 0, 6);
	cvSetReal2D(triangles, 15, 0, 6);
	cvSetReal2D(triangles, 16, 0, 6);
	cvSetReal2D(triangles, 17, 0, 6);
	cvSetReal2D(triangles, 18, 0, 6);

	cvSetReal2D(triangles, 0, 1, 2);
	cvSetReal2D(triangles, 1, 1, 12);
	cvSetReal2D(triangles, 2, 1, 8);
	cvSetReal2D(triangles, 3, 1, 10);
	cvSetReal2D(triangles, 4, 1, 2);
	cvSetReal2D(triangles, 5, 1, 10);
	cvSetReal2D(triangles, 6, 1, 9);
	cvSetReal2D(triangles, 7, 1, 11);
	cvSetReal2D(triangles, 8, 1, 5);
	cvSetReal2D(triangles, 9, 1, 5);
	cvSetReal2D(triangles, 10, 1, 8);
	cvSetReal2D(triangles, 11, 1, 9);
	cvSetReal2D(triangles, 12, 1, 1);
	cvSetReal2D(triangles, 13, 1, 1);
	cvSetReal2D(triangles, 14, 1, 7);
	cvSetReal2D(triangles, 15, 1, 4);
	cvSetReal2D(triangles, 16, 1, 4);
	cvSetReal2D(triangles, 17, 1, 11);
	cvSetReal2D(triangles, 18, 1, 7);

	cvSetReal2D(triangles, 0, 2, 12);
	cvSetReal2D(triangles, 1, 2, 13);
	cvSetReal2D(triangles, 2, 2, 13);
	cvSetReal2D(triangles, 3, 2, 13);
	cvSetReal2D(triangles, 4, 2, 12);
	cvSetReal2D(triangles, 5, 2, 12);
	cvSetReal2D(triangles, 6, 2, 10);
	cvSetReal2D(triangles, 7, 2, 10);
	cvSetReal2D(triangles, 8, 2, 10);
	cvSetReal2D(triangles, 9, 2, 9);
	cvSetReal2D(triangles, 10, 2, 3);
	cvSetReal2D(triangles, 11, 2, 8);
	cvSetReal2D(triangles, 12, 2, 3);
	cvSetReal2D(triangles, 13, 2, 2);
	cvSetReal2D(triangles, 14, 2, 5);
	cvSetReal2D(triangles, 15, 2, 2);
	cvSetReal2D(triangles, 16, 2, 5);
	cvSetReal2D(triangles, 17, 2, 2);
	cvSetReal2D(triangles, 18, 2, 11);

	cvSubS(triangles, cvScalar(1), triangles);	
	//*/  // FAKE_TRIANGLES - END

	//vector<int> bad_triangles = findInsideOut();
	// TODO: De momento estoy asumiendo que todos mis tri�ngulos son buenos.
	// Cuando todo est� funcionando, hacer lo que se hace a partir de la l�nea 576
	// de cp2tform.m

	//transformaALoBruto(img1, img2);

	// TODO: Probar qu� es lo que ocurre si hay un punto mal puesto
        clock_t myTime = clock();
        pruebaPaperPiecewiseLinear(img1C, img2C, featureMask, subdiv);

        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo piecewise linear = " << time << endl;

        myTime = clock();
	//extraeObstaculos(img1C);
        time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo obstaculos = " << time << endl;

}

void CViewMorphing::extraeObstaculos(IplImage * img) {
    cout << "extrae" << endl;
	IplImage * imgTReal = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * imgWarped = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * tmpGray = cvCreateImage(size, IPL_DEPTH_8U, 1);

	cvZero(imgTReal);
	cvZero(imgWarped);

	//preprocesa(img);
	//preprocesa(warpedImg);

        cout << 1 << endl;
	cvCvtColor(img, tmpGray, CV_BGR2GRAY);
        cout << 2 << endl;
	cvCopy(tmpGray, imgTReal, mask);
        cout << 3 << endl;
	cvCvtColor(warpedImg, tmpGray, CV_BGR2GRAY);
        cout << 4 << endl;
	cvCopy(tmpGray, imgWarped, mask);

        cout << "Hasta aqui bien" << endl;
#ifdef WITH_PCA

	cvNamedWindow("Prep1", 1);
	cvShowImage("Prep1", img);
	cvNamedWindow("Prep2", 1);
	cvShowImage("Prep2", warpedImg);

	nav.getPCADifs(imgWarped, imgTReal, mask, PAINT_PCA);
	IplImage * pcaDif = nav.getDifPCA();

        cvNamedWindow("distPCA", 1);
        cvShowImage("distPCA", pcaDif);

        //statItem->mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
        //cvCopyImage(mask, statItem->mask);
        statItem->areaCubierta = cvCountNonZero(mask);
        //statItem->result = cvCreateImage(size, IPL_DEPTH_8U, 1);
        //cvCopyImage(mask, statItem->result);
        stringstream ss;
        ss << ".dist" << statItem->distance << ".angle" << statItem->angle << ".size" << statItem->size.width << "x" << statItem->size.height << ".zoom" << statItem->zoom << ".b1_" << statItem->blur1 << ".b2_" <<         ss << ".dist" << statItem->distance << ".angle" << statItem->angle << ".size" << statItem->size.width << "x" << statItem->size.height << ".zoom" << statItem->zoom << ".b1_" << statItem->blur1 << ".b2_" <<         ss << ".dist" << statItem->distance << ".angle" << statItem->angle << ".size" << statItem->size.width << "x" << statItem->size.height << ".zoom" << statItem->zoom << ".b1_" << statItem->blur1 << ".b2_" << statItem->blur2;

        cout << ss.str() << endl;
        string pathMask = statItem->path + "." + ss.str() + ".mask.JPG";
        string pathRes = statItem->path + "." + ss.str() + ".result.JPG";

        //cvSaveImage(pathMask.c_str(), mask);
        //0cvSaveImage(pathRes.c_str(), pcaDif);

	cvErode(pcaDif, pcaDif, 0, 1);        
	//cvDilate(pcaDif, pcaDif, 0, 1);

	cvThreshold(pcaDif, pcaDif, 20, 255, CV_THRESH_BINARY);        

	cvDilate(pcaDif, pcaDif, 0, 1);
        
        statItem->pixelsDiferentes = cvCountNonZero(pcaDif);

	// Ahora buscamos contornos
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	IplImage * maskResult = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage * recuadro = cvCloneImage(img);		

	cvFindContours(pcaDif, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    cvZero(maskResult);

	for( ; contour != 0; contour = contour->h_next ) {
		CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );		
		if (fabs(cvContourArea(contour)) > 50) {
			/* replace CV_FILLED with 1 to see the outlines */
			cvDrawContours(maskResult, contour, color, color, -1, CV_FILLED, 8 );

			/*CvPoint * puntos = new CvPoint[contour->total];
			for (int i = 0; i < contour->total; i++) {
				CvPoint * p = CV_GET_SEQ_ELEM(CvPoint, contour, i);
				puntos[i] = *p;
				cout << puntos[i].x << ", " << puntos[i].y << endl;
			}*/			
			CvPoint p[4];
			p[0] = cvPoint(10, 10);
			p[1] = cvPoint(20, 10);
			p[2] = cvPoint(20, 20);
			p[3] = cvPoint(10, 20);
			CvRect rect = cvBoundingRect(contour);
			cvRectangle(recuadro, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height), cvScalar(0, 255, 0));
		}
    }

	cvNamedWindow("ResultadoPCA", 1);
	cvShowImage("ResultadoPCA", maskResult);

	cvNamedWindow("ResultadoFinal", 1);
	cvShowImage("ResultadoFinal", recuadro);

	//char cadena[1024];
	//sprintf(cadena, "C:\\Documents and Settings\\neztol\\Escritorio\\Doctorado\\Deteccion\\resultados\\Figura%d.jpg", rand() % 100);
	//cvSaveImage(cadena, recuadro);
#endif
}

void CViewMorphing::navegaPorTriangulos(CvSubdiv2D * subdiv) {
	CvSeqReader  reader;
    int i, total = subdiv->edges->total;
    int elem_size = subdiv->edges->elem_size;
	
	vector <int *> tri;

    cvStartReadSeq((CvSeq*)(subdiv->edges), &reader, 0 );	

    for( i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

        if(CV_IS_SET_ELEM(edge)) {
			CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge)edge, CV_PREV_AROUND_RIGHT);

			CvPoint trianglePoints[3];

			trianglePoints[0] = cvPointFrom32f(cvSubdiv2DEdgeOrg((CvSubdiv2DEdge)edge)->pt);
			trianglePoints[1] = cvPointFrom32f(cvSubdiv2DEdgeDst((CvSubdiv2DEdge)edge)->pt);
			trianglePoints[2] = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);	

			bool innerTriangle = true;
			/*for (int i = 0; i < 3; i++) {
				if (trianglePoints[i].x <= 0 || trianglePoints[i].x > size.width ||
					trianglePoints[i].y <= 0 || trianglePoints[i].y > size.height) {									

					innerTriangle = false;
				}
			}*/
			if (innerTriangle == true) {	
				int * index = new int[3];				
				for (int i = 0; i < numberOfFeatures; i++) {
					for (int j = 0; j < 3; j++) {
						if (cvPointFrom32f(points1[i]).x == trianglePoints[j].x &&
							cvPointFrom32f(points1[i]).y == trianglePoints[j].y)						
							index[j] = i;
					}
				}

				tri.push_back(index);
			}
        }

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }		
	
	// Hacemos limpieza de tri�ngulos
	/*int countMax = 0;
	for (int i = 0; i < tri.size(); i++) {
		bool add = true;
		for (int j = 0; j < tri.size(); i++) {
		}
	}*/

	triangles = cvCreateMat(tri.size(), 3, CV_8UC1);

	for (int i = 0; i < tri.size(); i++) {	
		for (int j = 0; j < 3; j++) {
			cvSetReal2D(triangles, i, j, tri.at(i)[j]);			
		}		
	}
	
	tri.clear();	
}

void CViewMorphing::drawDelaunay(CvSubdiv2D * subdiv, IplImage * img) {

    CvSeqReader  reader;
    int i, total = subdiv->edges->total;
    int elem_size = subdiv->edges->elem_size;

    cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );

	
	IplImage * imgDelaunay = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage * imgDelaunay2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
	cvCvtColor(img, imgDelaunay, CV_GRAY2BGR);	
	cvZero(imgDelaunay2);
	cvNamedWindow("Delaunay", 1);
	cvNamedWindow("Delaunay2", 1);

    for( i = 0; i < total; i++ ) {
        CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

        if( CV_IS_SET_ELEM( edge )) {			

			CvPoint2D32f p1OrgF = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge)edge)->pt;
			CvPoint2D32f p1DestF = cvSubdiv2DEdgeDst((CvSubdiv2DEdge)edge)->pt;

			CvPoint org = cvPointFrom32f(p1OrgF);
			CvPoint dest = cvPointFrom32f(p1DestF);

			cvLine(imgDelaunay2, org, dest, cvScalar(255));

			CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge)edge, CV_PREV_AROUND_RIGHT);
			CvSubdiv2DEdge edge2 = cvSubdiv2DGetEdge((CvSubdiv2DEdge)edge1, CV_NEXT_AROUND_LEFT);

			cvLine(imgDelaunay, dest, cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt), cvScalar(255, 0, 255));
			cvLine(imgDelaunay, cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt), cvPointFrom32f(cvSubdiv2DEdgeDst(edge2)->pt), cvScalar(0, 255, 255));
			cvCircle(imgDelaunay, org, 2, cvScalar(255, 0, 0), -1);
			cvCircle(imgDelaunay, dest, 2, cvScalar(255, 0, 0), 1);

			CvPoint puntos[3];
			int nPuntos = 3;
			puntos[0] = org;
			puntos[1] = dest;
			puntos[2] = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

			cvFillConvexPoly(imgDelaunay, &puntos[0], nPuntos, cvScalar(0, 0, 255));//CV_RGB(rand()&255,rand()&255,rand()&255));
        }

        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }
	
	cvShowImage("Delaunay", imgDelaunay);
	cvShowImage("Delaunay2", imgDelaunay2);	

	cvReleaseImage(&imgDelaunay);
	cvReleaseImage(&imgDelaunay2);
}

double CViewMorphing::calculaDet(CvMat * m) {
	return /*cvmGet(m, 0, 0) **/ cvGet2D(m, 1, 1).val[0]; // * cvmGet(m, 2, 2);
	/* +
				cvmGet(m, 1, 0) * cvmGet(m, 2, 1) * cvmGet(m, 0, 2) +
				cvmGet(m, 2, 0) * cvmGet(m, 0, 1) * cvmGet(m, 1, 2) - 				
				cvmGet(m, 0, 2) * cvmGet(m, 1, 1) * cvmGet(m, 2, 0) -
				cvmGet(m, 1, 2) * cvmGet(m, 2, 1) * cvmGet(m, 0, 0) -
				cvmGet(m, 2, 2) * cvmGet(m, 0, 1) * cvmGet(m, 1, 0);*/
}

void CViewMorphing::pruebaPaperPiecewiseLinear(IplImage * img1, IplImage * img2, IplImage * featureMask, CvSubdiv2D * subdiv) {
	// Primero automatizamos las correspondencias entre los puntos
	CvMat * mCorrespPoints = cvCreateMat(size.height, size.width, CV_64FC2);
	cvZero(mCorrespPoints);

	CvMat * A = cvCreateMat(3, 3, CV_64FC1);
	CvMat * B = cvCreateMat(3, 3, CV_64FC1);
	CvMat * C = cvCreateMat(3, 3, CV_64FC1);
	CvMat * D = cvCreateMat(3, 3, CV_64FC1);

	cvSet(A, cvScalar(1));
	cvSet(B, cvScalar(1));
	cvSet(C, cvScalar(1));
	cvSet(D, cvScalar(1));

	for (int i = 0; i < numberOfFeatures; i++) {
		CvPoint p1 = cvPointFrom32f(points1[i]);
		CvPoint p2 = cvPointFrom32f(points2[i]);
		cvSet2D(mCorrespPoints, p1.y, p1.x, cvScalar(p2.x, p2.y));		
	}

	IplImage * muestraPCorresp1 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
	IplImage * muestraPCorresp2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
	warpedImg = cvCreateImage(size, IPL_DEPTH_8U, img1->nChannels);
	
	IplImage * resta = cvCreateImage(size, IPL_DEPTH_8U, img1->nChannels);
	mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
	cvZero(mask);

	cvNamedWindow("Corresp1", 1);
	cvNamedWindow("Corresp2", 1);

	
	CvMat * remapX = cvCreateMat(size.height, size.width, CV_32FC1);
	CvMat * remapY = cvCreateMat(size.height, size.width, CV_32FC1);

	for (int m = 0; m < size.width; m++) {
		for (int n = 0; n < size.height; n++) {
			CvPoint2D32f myPoint = cvPoint2D32f(m, n);

			CvSubdiv2DEdge e;
			CvSubdiv2DEdge e0;
			CvSubdiv2DPoint * p = 0;

			cvSubdiv2DLocate(subdiv, myPoint, &e0, &p);

			CvPoint2D32f point1[3];
			CvPoint2D32f point2[3];

			int index = 0;

			if (e0) {
                            e = e0;
                            bool toTransform = true;
				do {	
					CvPoint2D32f p1 = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge)e)->pt;
					CvPoint2D32f p2;

					if (p1.x < 0 || p1.y < 0 || p1.x >= size.width || p1.y >= size.height) {
                                            toTransform = false;
						break; //p2 = p1;
					} else {
						CvScalar p2Val = cvGet2D(mCorrespPoints, cvRound(p1.y), cvRound(p1.x));

						p2 = cvPoint2D32f(p2Val.val[0], p2Val.val[1]);
					}

					point1[index] = p1;
					point2[index] = p2;
					index++;

					e = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_LEFT);
				} while(e != e0);
                                if (toTransform == false)
                                    continue;
			} else {
				cout << "No se encontro e0" << endl;
				continue;
			}

			// Ahora vamos a calcular el punto central que nos deber�a quedar				
			// Calculamos A
			for (int j = 0; j < 3; j++) {
				cvmSet(A, j, 0, point1[j].y);
				cvmSet(A, j, 1, point2[j].x);
				cvmSet(B, j, 0, point1[j].x);
				cvmSet(B, j, 1, point2[j].x);
				cvmSet(C, j, 0, point1[j].x);
				cvmSet(C, j, 1, point1[j].y);
				cvmSet(D, j, 0, point1[j].x);
				cvmSet(D, j, 1, point1[j].y);
				cvmSet(D, j, 2, point2[j].x);
			}

			double a = cvDet(A);
			double b = -cvDet(B);
			double c = cvDet(C);
			double d = -cvDet(D);

			CvPoint2D32f newPos = cvPoint2D32f(0, 0);
			newPos.x = (-a * myPoint.x - b * myPoint.y - d) / c;

			// Calculamos la componente y
			for (int j = 0; j < 3; j++) {
				cvmSet(A, j, 0, point1[j].y);
				cvmSet(A, j, 1, point2[j].y);
				cvmSet(B, j, 0, point1[j].x);
				cvmSet(B, j, 1, point2[j].y);
				cvmSet(C, j, 0, point1[j].x);
				cvmSet(C, j, 1, point1[j].y);
				cvmSet(D, j, 0, point1[j].x);
				cvmSet(D, j, 1, point1[j].y);
				cvmSet(D, j, 2, point2[j].y);
			}

			a = cvDet(A);
			b = -cvDet(B);
			c = cvDet(C);
			d = -cvDet(D);

			newPos.y = (-a * myPoint.x - b * myPoint.y - d) / c;

			//if (newPos.x >= 320) newPos.x = 319;
			//if (newPos.y >= 240) newPos.y = 239;
			//if (newPos.x < 0) newPos.x = 0;
			//if (newPos.y < 0) newPos.y = 0;

			if (MASK_DELAUNAY == 1) {
				bool pintar = true;
				CvPoint poly[3];
				for (int i = 0; i < 3; i++) {
					if (point2[i].x < 0 || point2[i].y < 0 ||
						point2[i].x >= size.width || point2[i].y >= size.height) {
							pintar = false;					
							break;
					}				
					poly[i] = cvPointFrom32f(point2[i]);				
				}			
				if (pintar == true) {				
					cvFillConvexPoly(mask, poly, 3, cvScalar(255));
				}	
			}

			//cout << cvFloor(myPoint.y) << ", " << cvFloor(myPoint.x) << endl;
			//cout << cvFloor(newPos.y) << ", " << cvFloor(newPos.x) << endl;

			cvSetReal2D(remapX, cvRound(myPoint.y), cvRound(myPoint.x), newPos.x);
			cvSetReal2D(remapY, cvRound(myPoint.y), cvRound(myPoint.x), newPos.y);

			/*if (newPos.x < 0 || newPos.x >= size.width ||
				newPos.y < 0 || newPos.y >= size.height)
				//continue;
				//cvSet2D(warpedImg, cvFloor(myPoint.y), cvFloor(myPoint.x) , cvScalarAll(0));
			else {
				//cvSet2D(warpedImg, cvFloor(myPoint.y), cvFloor(myPoint.x) , cvGet2D(img2, cvFloor(newPos.y), cvFloor(newPos.x)));				
			}*/
		}
	}       

        cvSet2D(img2, 0, 0, cvScalarAll(0));
	cvRemap(img2, warpedImg, remapX, remapY, CV_WARP_FILL_OUTLIERS + CV_INTER_CUBIC);
	if (featureMask != NULL) {
            cout << "Hola" << endl;
            cvSet2D(featureMask, 0, 0, cvScalarAll(0));
		IplImage * feat = cvCreateImage(size, IPL_DEPTH_8U, 1);
		cvRemap(featureMask, feat, remapX, remapY, CV_WARP_FILL_OUTLIERS + CV_INTER_CUBIC);

		cvAnd(mask, feat, mask);
                cvErode(mask, mask);

		cvNamedWindow("featureMask", 1);
		cvShowImage("featureMask", feat);
	}       

	if (MASK_DELAUNAY == 0) {
		cvThreshold(warpedImg, mask, 1, 255, CV_THRESH_BINARY);
	}
        //cvCvtColor(warpedImg, mask, CV_BGR2GRAY);
        //cvRemap(featureMask, mask, remapX, remapY, CV_WARP_FILL_OUTLIERS + CV_INTER_CUBIC);
        //cvThreshold(mask, mask, 1, 255, CV_THRESH_BINARY);

	cvShowImage("Corresp1", warpedImg);
	cvShowImage("Corresp2", img1);

	cvNamedWindow("Mask", 1);
	cvShowImage("Mask", mask);       

	cvNamedWindow("Resta", 1);
	cvAbsDiff(warpedImg, img1, resta);
	cvShowImage("Resta", resta);

	cvReleaseMat(&mCorrespPoints);
}

void CViewMorphing::warpPoints(IplImage * &img1) {	
	CvMat * A = cvCreateMat(2 * numberOfFeatures, 8, CV_64FC1);
	CvMat * B = cvCreateMat(2 * numberOfFeatures, 1, CV_64FC1);

	double x[9];
	CvMat X = cvMat(8, 1, CV_64FC1, x);	

	CvMat * point = cvCreateMat(3, 1, CV_64FC1);	

	cvZero(A);
	for (int i = 0; i < numberOfFeatures; i++) {
		cvmSet(A, i, 0, points1[i].x);
		cvmSet(A, i, 1, points1[i].y);
		cvmSet(A, i, 2, 1);
		cvmSet(A, i, 6, -points1[i].x * points2[i].x);
		cvmSet(A, i, 7, -points1[i].y * points2[i].x);
		cvmSet(A, i + numberOfFeatures, 3, points1[i].x);
		cvmSet(A, i + numberOfFeatures, 4, points1[i].y);
		cvmSet(A, i + numberOfFeatures, 5, 1);
		cvmSet(A, i + numberOfFeatures, 6, -points1[i].x * points2[i].y);
		cvmSet(A, i + numberOfFeatures, 7, -points1[i].y * points2[i].y);

		cvmSet(B, i, 0, points2[i].x);
		cvmSet(B, i + numberOfFeatures, 0, points2[i].y);
	}

	cvSolve(A, B, &X, CV_SVD);		
	
	x[8] = 1;

	X = cvMat(3, 3, CV_64FC1, x);

	/*for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cout << cvmGet(&X, i, j) << " ";
		}
		cout << endl;
	}
	cout << endl;*/

	affine = cvCreateImage(size, IPL_DEPTH_8U, 1);
	cvWarpPerspective(img1, affine, &X, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
	//cvWarpAffine(img1, affine, &X, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);

	//X = cvMat(3, 3, CV_64FC1, x);	

	//cvSetIdentity(&X);

	double ptsOrig[] = { 0, 0, size.width - 1, 0, size.width - 1, size.height -1, 0, size.height - 1 };
	double pts[] = { 0, 0, size.width - 1, 0, size.width - 1, size.height -1, 0, size.height - 1 };

	CvMat rect = cvMat(1, 4, CV_64FC2, pts);
	cvPerspectiveTransform(&rect, &rect, &X);

	cvCircle(affine, cvPoint(pts[0], pts[1]), 2, cvScalar(255), -1);
	cvCircle(affine, cvPoint(pts[2], pts[3]), 2, cvScalar(255), -1);
	cvCircle(affine, cvPoint(pts[4], pts[5]), 2, cvScalar(255), -1);
	cvCircle(affine, cvPoint(pts[6], pts[7]), 2, cvScalar(255), -1);

	for (int i = 0; i < 8; i += 2) {
		if (pts[i] >= 0 && pts[i] < size.width &&
			pts[i + 1] >= 0 && pts[i + 1] < size.height) {
				points1[numberOfFeatures] = cvPoint2D32f(ptsOrig[i], ptsOrig[i + 1]);
				points2[numberOfFeatures] = cvPoint2D32f(pts[i], pts[i + 1]);
				numberOfFeatures++;			
		}

		// Intersecciones
		double x0 = pts[i];
		double y0 = pts[i + 1];
		double x1, y1;
		if (i == 0) {
			x1 = pts[6];
			y1 = pts[7];
		} else {
			x1 = pts[i - 2];
			y1 = pts[i - 1];
		}
		double m = (y1 - y0) / (x1 - x0);
		if (i == 0 || i == 4) {
			double myY = m * (ptsOrig[i] - x0) + y0;				

			double distAB = sqrt(pow(y1 - y0, 2.0) + pow(x1 - x0, 2.0));
			double dist = sqrt(pow(myY - y0, 2.0) + pow(ptsOrig[i] - x0, 2.0));

			double prop = dist / distAB;
			double newY = prop * (y1 - y0) + y0;			

			if (newY >= 0 && newY < size.height &&
				myY >= 0 && myY < size.height) {
				points1[numberOfFeatures] = cvPoint2D32f(ptsOrig[i], newY);
				points2[numberOfFeatures] = cvPoint2D32f(ptsOrig[i], myY);
				numberOfFeatures++;			
			}
			
			cvCircle(affine, cvPoint(ptsOrig[i], myY), 2, cvScalar(255), -1);
		} else {
			double myX = (ptsOrig[i + 1] - y0) / m + x0;			
			cvCircle(affine, cvPoint(myX, ptsOrig[i + 1]), 2, cvScalar(255), -1);

			double distAB = sqrt(pow(y1 - y0, 2.0) + pow(x1 - x0, 2.0));
			double dist = sqrt(pow(ptsOrig[i + 1] - y0, 2.0) + pow(myX - x0, 2.0));

			double prop = dist / distAB;
			double newX = prop * (x1 - x0) + x0;
						
			if (myX >= 0 && myX < size.width &&
				newX >= 0 && newX < size.width) {
				points1[numberOfFeatures] = cvPoint2D32f(newX, ptsOrig[i + 1]);
				points2[numberOfFeatures] = cvPoint2D32f(myX, ptsOrig[i + 1]);
				numberOfFeatures++;
			}
		}
	}

	// Intersecci�n 1 ( y = m * (x - x0) + y0, m = (y1 - y0) / (x1 - x0))

	cvNamedWindow("Affine", 1);
	cvShowImage("Affine", affine);
}

void CViewMorphing::preprocesa(IplImage * img) {
	IplImage * r = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * g = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * b = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage * rgb = cvCreateImage(size, IPL_DEPTH_8U, 1);

	cvSplit(img, b, g, r, 0);

	//cvAdd(r, g, rgb);
	//cvAdd(b, rgb, rgb);

	//cvDiv(r, rgb, r);
	//cvDiv(g, rgb, g);

	//cvCvtColor(img, img, CV_BGR2RGB);

	cvZero(b);

	cvMerge(b, g, r, 0, img);	
}

void CViewMorphing::cleanDuplicated() {
    CvPoint2D32f * tmp1 = new CvPoint2D32f[numberOfFeatures];
    CvPoint2D32f * tmp2 = new CvPoint2D32f[numberOfFeatures];
    int nFeat = 0;

    for (int i = 0; i < numberOfFeatures; i++) {
        bool exists = false;
        if (points1[i].x < 5 && points1[i].y < 5) {
            exists = true;
        } else {
            for (int j = 0; j < nFeat; j++) {
                if (((int)tmp1[j].x == (int)points1[i].x) && ((int)tmp1[j].y == (int)points1[i].y)) {
                    exists = true;
                    break;
                }
            }
        }
        if (exists == false) {
            tmp1[nFeat] = points1[i];
            tmp2[nFeat] = points2[i];
            nFeat++;
        }
    }

    numberOfFeatures = nFeat;
    points1 = tmp1;
    points2 = tmp2;
}

void CViewMorphing::viewMorphing(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask, t_Statistic_Item * item) {

    clock_t myTime = clock();
    	///////oFlowFeatureTracker(img2, img1);
        //oFlowMeshedFeatureTracker(img2, img1);


        //oFlowMeshedAndDetectedFeatureTracker(img2, img1);


        //LMedSFeatureTracker(img2, img1);
        //PointPatternMatchingFeatureTracker(img2, img1);
        //CorrelationFeatureTracker(img2, img1);
        //SurfFeatureTracker(img2, img1);
	//siftFeatureTracker(img2, img1);
	//showFeatureTracking(img2, img1);

        //cleanUsingDelaunay(img2C, img1C);

        //generatePointsByDelaunay(img1C, img2C);

    statItem = item;
    CImageRegistration ir(size);
    ir.findPairs(img2, img1, points1, points2, numberOfFeatures, false, false);
        showFeatureTracking(img2, img1);
        statItem->nPuntosEmparejados = numberOfFeatures;
        ir.~CImageRegistration();
	
	//imageMorphing(img1, img2, img1C, img2C, img2);
	
	//morphingByStereo(img1, img2);
	
	//pruebaWarp1Matlab(img1, img2, 6);

    

	//http://www.cs.wright.edu/~agoshtas/ardy.html
        try {
        	pieceWiseLinear(img2, img1, img2C, img1C, featureMask);
        } catch(...) {
            cout << "Excepción imprevista en línea 2034 de ViewMorphing.cpp" << endl;
        }


        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo total de los totales = " << time << endl;


	//cvWaitKey(0);
}



void inicio2(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask) {
	CViewMorphing vm(cvGetSize(img1));

	//vm.viewMorphing(img1, img2, img1C, img2C, featureMask);
        //vm.contourMatching(img1, img2);

        //vm.AffineAndEuclidean(img1, img2, img1C, img2C, featureMask);
        vm.flusserTransform(img1, img2, img1C, img2C, featureMask);
}