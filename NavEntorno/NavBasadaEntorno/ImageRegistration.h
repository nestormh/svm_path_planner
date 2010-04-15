/* 
 * File:   ImageRegistration.h
 * Author: neztol
 *
 * Created on 3 de noviembre de 2009, 11:29
 */

#ifndef _IMAGEREGISTRATION_H
#define	_IMAGEREGISTRATION_H

#include "stdafx.h"

// Corner detection constants
#define MESH_DISTANCE 20

#define MAX_FEATURES 700
#define DEFAULT_FEATURES_QUALITY 0.01
#define DEFAULT_FEATURES_MIN_DIST 15//5

#define USE_HARRIS 0

// Optical flow constants
#define PYRAMIDRICAL_SEARCH_WINDOW_SIZE 5//2 //10
#define PYRAMID_DEPTH 4

#define MAX_PIXEL_DISTANCE 1

// Distance matching constants
#define MAX_DIST_CLOUD 1//DBL_MAX

// Cleaning features
#define CLEAN_THRESH 25

#define CLEAN_DIST_SPLINES 20

#define USE_REGIONS false

typedef struct {
    CvPoint p1, p2, p3;
    int index1, index2, index3;
} t_triangle;

class CImageRegistration {
public:
    CImageRegistration(CvSize size);
    void registration(IplImage * imgBD, IplImage * imgRT);
    // Just for testing
    void registration(IplImage * imgDB1, IplImage * imgDB2, IplImage * imgRT);
    void findPairs(IplImage * img1, IplImage * img2, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat, bool useAffine, bool useRegions, CvPoint2D32f * initialPoints = NULL, int nInitialP = 0);
    void findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures);
    void getPairsOnBigImg(IplImage * imgDB, IplImage * imgRT, IplImage * imgDBC, IplImage * imgRTC);
    void TPS(IplImage * &img, CvPoint2D32f * points1, CvPoint2D32f * points2, int nFeat);
    ~CImageRegistration();

    void cleanFeat(CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat);
private:
    CvSize size;

    IplImage * imgDB;
    IplImage * imgRT;
    IplImage * oldImgDB;
    IplImage * oldImgRT;

    int numberOfFeatures;
    int oldNumberOfFeatures;
    CvPoint2D32f * pointsDB;
    CvPoint2D32f * pointsRT;
    CvPoint2D32f * oldPointsDB;
    CvPoint2D32f * oldPointsRT;

    // Aux params
    IplImage * img32fc1_a;
    IplImage * img32fc1_b;    

    // Giving a more understandable name
    IplImage * eigen;
    IplImage * tmp;

    // For corner detection
    int numberOfMeshedFeatures;
    CvPoint2D32f * meshedFeatures;

    // For optical flow
    IplImage * pyramidImage1;
    IplImage * pyramidImage2;

    // For non-rigid transform
    CvMat * remapX;
    CvMat * remapY;
    IplImage * tps;
    
    // Functions    
    void findInitialPoints(IplImage * img, CvPoint2D32f * &corners, int &nCorners, CvPoint2D32f * &meshed, int &nMeshed);
    void findInitialPoints(IplImage * img, CvPoint2D32f * &corners, int &nCorners);
    void findDistanceBasedPairs(const CvPoint2D32f * flow1, const CvPoint2D32f * flow2, int oFlowN,
                                const CvPoint2D32f * origPoints1, int nOrig1, const CvPoint2D32f * origPoints2, int nOrig2,
                                CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures);
    void getPairsFromPreviousImg(IplImage * imgDB, IplImage * imgRT, CvPoint2D32f * &pointsDB, CvPoint2D32f * &pointsRT, int &nFeat);
    void findPairsWithCorrelation(IplImage * imgDB, IplImage * imgRT, CvPoint2D32f * &pointsDB, CvPoint2D32f * &pointsRT, int &nFeat);

    

    
    inline void getCoefsAM(CvPoint2D32f * p1, CvPoint2D32f * p2, int nFeat, double * &coefs1, double * &coefs2);
    inline void calculateAM(CvPoint2D32f point, CvPoint2D32f * p1, int nFeat, double * coefs1, double * coefs2, double &u, double &v);

    void showPairs(char * name, IplImage * img1, IplImage * img2, CvPoint2D32f * points1, CvPoint2D32f * points2, int numberOfFeatures);
    void showFeat(char * name, IplImage * img, CvPoint2D32f * points, int numberOfFeatures);

    void showPairsRelationship(char * name, IplImage * img1, IplImage * img2, CvPoint2D32f * points1, CvPoint2D32f * points2, int numberOfFeatures);
    void cleanWithPCACutreishon(IplImage * img1, IplImage * img2, CvPoint2D32f * points1, CvPoint2D32f * points2, int numberOfFeatures);

    void cleanUsingSplines(CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat);
    
    void cleanUsingDelaunay(IplImage * img1, IplImage * img2, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int numberOfFeatures);
    bool areSameTriangles(t_triangle tri1, t_triangle tri2);
    bool areSameTrianglesIndexed(t_triangle tri1, t_triangle tri2);
    bool areTrianglesEquipositioned(t_triangle tri1, t_triangle tri2);
};



typedef struct {
        CvMoments moments;
        CvHuMoments hu;
        CvBox2D box;
        CvPoint * points;
        int nPoints;
        double normHu[7];
    } t_moment;

void cjtosImagenes();
void statistics();
void pruebaSurf(IplImage * img1, IplImage * img2);
double calcCorr(t_moment mmt1, t_moment mmt2, int method);
double calcCCorr(IplImage * img1, IplImage * img2, t_moment mmt1, t_moment mmt2, bool show);
int mesrTest(const IplImage * rsp, char * name, t_moment * &momentList, int &nMoments);
void matchMserByMoments(IplImage * img1, IplImage * img2, t_moment * momentList1, t_moment * momentList2, int nMoment1, int nMoment2, char * name, vector<t_moment *> &regionPairs);
void cleanMatches(IplImage * img1, IplImage * img2, vector<t_moment *> &regionPairs, char * name);
int starTest(const IplImage * rsp, char * name);
void cleanMatches(IplImage * img1, IplImage * img2, vector<t_moment *> &regionPairs, char * name, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat);

void bruteMatch(IplImage * img1, IplImage * img2, CvMat **points1, CvMat **points2, CvSeq *kp1, CvSeq *desc1, CvSeq *kp2, CvSeq * desc2);
void showPairs2(char * name, IplImage * img1, IplImage * img2, CvMat * points1, CvMat * points2);
void removeOutliers(CvMat **points1, CvMat **points2, CvMat *status);


#endif	/* _IMAGEREGISTRATION_H */

