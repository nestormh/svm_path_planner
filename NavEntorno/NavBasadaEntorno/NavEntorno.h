#pragma once

#include "stdafx.h"

using namespace std;

#define W_FLOW1	"FlujoOptico1"
#define W_FLOW2	"FlujoOptico2"
#define W_RESTA	"Resta"
#define W_PLOT "Plot"
#define W_PLOT_PCA "PlotPCA"
#define W_DISTPCA "DistPCA"
#define W_VEL_OFLOW "VelocidadOFlow"
#define W_PERSP "Persp"

#define winSize cvSize(5, 5)
#define step	10		// Must be image's widh-height divisor

class CNavEntorno {
private:
	// X & Y velocity vectors. Tells how much pixels have been displaced from the first image
	CvMat * velx;
	CvMat * vely;

	// Input images
	IplImage * img1;
	IplImage * img2;

	// A, B, X & warp matrix. Used for the transformation of img1
	CvMat * A;
	CvMat * B;
	CvMat * X;
	CvMat * warp;	
	
	IplImage * persp;			// Warped image	
	IplImage * perspMask;		// Mask defining persp limits
	IplImage * imgShaped;		// Image with a ROI defined by perpMask

	// ACP Matrix
	CvMat * data;	// Original data matrix (gray level 1st image x gray level 2nd image x optical flow velocity)
	CvMat * data1;	// First row of data pointer
	CvMat * data2;	// Second row of data pointer	
	CvMat * pcaData;	// Data after PCA
	CvMat * corr;	// Correlation matrix
	CvMat * avg;	// Average matrix
	CvMat * eigenValues; // Eigen Values
	CvMat * eigenVectors; // Eigen Vectors
	CvMat * dataX; // X axis of PCA transformed data
	CvMat * dataY; // Y axis of PCA transformed data
	CvMat * distPCA; // distances to the correlation line of each point after PCA
	CvMat * vel; // optical flow distances matrix

	// Statistical information on PCA results
	CvScalar xMean, yMean, xSdv, ySdv; 

	IplImage * subImages; // Image resulting from substracting img2 and persp

	inline void init(IplImage * img);			// initializes class variables	
	inline void showData(bool paint, double time, double absDist, 
		double angDist, double lateralDist);			// Shows excecution information on screen
	inline void warpImage();					// Transforms the base image	
	inline void calcPCA();								// Calculates PCA for persp and img2
	inline void getDifsOnPCA();						// Gets the differences using the points obtained in calcPCA
	inline void calcOFlowDistancesAndSub();					// Calculates distances based on optical flow
	inline void logicTransf();	
	inline void igualaBrilloContraste();
public:
	CNavEntorno(void);					// Constructor
	~CNavEntorno(void);					// Destructor

	// Fits the first image to keep similarities at the same pixels
	void matchImages(IplImage * img1, IplImage * img2, IplImage * persp, 
		double &time, double absDist, double angDist, double lateralDist, bool paint);
	void getPCADifs(IplImage * persp, IplImage * img2, IplImage * mask, bool paint);
	IplImage * getDifPCA();
};
