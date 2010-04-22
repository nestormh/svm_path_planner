#pragma once

#include "stdafx.h"

#include "Surf/imload.h"
#include "Surf/surflib.h"
#include "Surf/image.h"
#include "Surf/os_mapping.h"
#include "Surf/ipoint.h"

#include "NavEntorno.h"
#include "ImageRegistration.h"

using namespace surf;

typedef struct {
    CvPoint p1, p2, p3;
    int index1, index2, index3;
} triangle;


// http://read.pudn.com/downloads46/sourcecode/windows/bitmap/154585/ViewMorphing/VM.cpp__.htm

#define MAX_FEATURES 400 //400
#define DEFAULT_FEATURES_QUALITY 0.01
#define DEFAULT_FEATURES_MIN_DIST 5

#define USE_HARRIS 1

#define COMPUTE_SUBPIXEL_CORNERS 0

#define SEARCH_WINDOW_SIZE 5

#define PYRAMIDRICAL_SEARCH_WINDOW_SIZE 2 //10
#define PYRAMID_DEPTH 4

#define MAX_PIXEL_DISTANCE 5

#define MASK_DELAUNAY 1

// Weight for the two source images
#define ALPHA_WEIGHT 0.5

#define WITH_PCA 0

#define PAINT_PCA false

typedef struct {
    double distance;
    double angle;
    string path;
    string pathBase;
    CvSize size;
    IplImage * result;
    IplImage * mask;
    int nPuntosEmparejados;
    int areaCubierta;
    int pixelsDiferentes;    
    int zoom;
    int blur1;
    int blur2;
} t_Statistic_Item;

void inicio2(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask);

class CViewMorphing {
private:
	CvSize size;	
	CvPoint2D32f * allPoints;
	CvPoint2D32f * affinePoints;
	CvMat * triangles;

	IplImage * mask;
	IplImage * warpedImg;
	IplImage * affine;
        IplImage * flusser;

        CvMat * remapX;
        CvMat * remapY;

	CNavEntorno nav;

        t_Statistic_Item * statItem;

        // Function that find a surf feature in a image represented by an Image structure
        vector< Ipoint > surfFeatureFinder(IplImage * img, double thresh, int &vLength);
        // Calculate square distance of two vectors
        double distSquare(double *v1, double *v2, int n);
        // Find closest interest point in a list, given one interest point
        int findSurfMatch(const Ipoint& ip1, const vector< Ipoint >& ipts, int vlen);
        // Find all possible matches between two images
        void findSurfMatches(const vector< Ipoint >& ipts1, const vector< Ipoint >& ipts2, int vLength);
        // Finds and matches the features by using SURF
        void SurfFeatureTracker(IplImage * img1, IplImage * &img2);

        void cleanUsingDelaunay(IplImage * img1, IplImage * img2);
        bool areSameTriangles(triangle tri1, triangle tri2);
        bool areSameTrianglesIndexed(triangle tri1, triangle tri2);
        bool areTrianglesEquipositioned(triangle tri1, triangle tri2);

        CvPoint calculaCorte(CvPoint p1, CvPoint p2);
        CvPoint calculaPuntoRelativo(CvPoint p1, CvPoint p2, CvPoint corte);
        void generatePointsByDelaunay(IplImage * img1, IplImage * img2);

	void oFlowFeatureTracker(IplImage * img1, IplImage * &img2);
        void CorrelationFeatureTracker(IplImage * img1, IplImage * &img2);
        void oFlowMeshedAndDetectedFeatureTracker(IplImage * img1, IplImage * &img2);
        void LMedSFeatureTracker(IplImage * img1, IplImage * &img2);
        void PointPatternMatchingFeatureTracker(IplImage * img1, IplImage * &img2);
        void oFlowMeshedFeatureTracker(IplImage * img1, IplImage * &img2);        
	void showFeatureTracking(IplImage * img1, IplImage * img2);
	void imageMorphing(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * &morphedImage);
	void morphingByStereo(IplImage * img1, IplImage * img2);	
	void drawDelaunay(CvSubdiv2D * subdiv, IplImage * img);
	
	void navegaPorTriangulos(CvSubdiv2D * subdiv);
	vector <int> findInsideOut();
	void transformaALoBruto(IplImage * img1, IplImage * img2);
	double calculaDet(CvMat * m);
	void pruebaPaperPiecewiseLinear(IplImage * img1, IplImage * img2, IplImage * featureMask, CvSubdiv2D * subdiv);
	void extraeObstaculos(IplImage * img);	
	void warpPoints(IplImage * &img1);
	void preprocesa(IplImage * img1);
        void cleanDuplicated();

        double pghMatchShapes(CvSeq *shape1, CvSeq *shape2);
        double treeMatchShapes(CvSeq *shape1, CvSeq *shape2);
        void drawBox2D(IplImage * img, CvBox2D box);

        void warpAffine(IplImage * &img1, IplImage * &img2);
        void calculatePoints(IplImage * img1, IplImage * &img2);
        void cleanUsingDistances(IplImage * img1, IplImage * &img2);

        void trans(CvPoint2D32f * p1, CvPoint2D32f * p2, int nFeat, double * fEstim, double * gEstim, CvPoint2D32f ul, CvPoint2D32f lr, int level);
        void getCoefsAM(CvPoint2D32f * p1, CvPoint2D32f * p2, int nFeat, double * &coefs1, double * &coefs2);
        void calculateAM(CvPoint2D32f * p1, int nFeat, double * coefs1, double * coefs2, double * &u, double * &v);
        void calculateAM(CvPoint2D32f point, CvPoint2D32f * p1, int nFeat, double * coefs1, double * coefs2, double &u, double &v);

public:
	CViewMorphing(CvSize size);
	~CViewMorphing(void);
	void viewMorphing(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask, t_Statistic_Item * item = NULL);
        void contourMatching(IplImage * img1, IplImage * &img2);
        void AffineAndEuclidean(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask);
        void flusserTransform(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask);
        void pieceWiseLinear(IplImage * img1, IplImage * img2, IplImage * img1C, IplImage * img2C, IplImage * featureMask);

        int numberOfFeatures;
	CvPoint2D32f * points1;
	CvPoint2D32f * points2;

};