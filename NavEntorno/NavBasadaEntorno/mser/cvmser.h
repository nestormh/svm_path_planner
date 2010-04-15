#ifndef GUARD_cvmser_h
#define GUARD_cvmser_h

#include "cv.h"

typedef struct CvMSERParams
{
	// delta, in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta}
	int delta;
	// prune the area which bigger/smaller than max_area/min_area
	int max_area;
	int min_area;
	// prune the area have simliar size to its children
	float max_variation;
	// trace back to cut off mser with diversity < min_diversity
	float min_diversity;
	/* the next few params for MSER of color image */
	// for color image, the evolution steps
	int max_evolution;
	// the area threshold to cause re-initialize
	double area_threshold;
	// ignore too small margin
	double min_margin;
	// the aperture size for edge blur
	int edge_blur_size;
}
CvMSERParams;

CvMSERParams cvMSERParams( int delta = 5, int min_area = 60, int max_area = 14400, float max_variation = .25, float min_diversity = .2, int max_evolution = 200, double area_threshold = 1.01, double min_margin = .003, int edge_blur_size = 5 );

void cvExtractMSER( CvArr* _img, CvArr* _mask, CvSeq** contours, CvMemStorage* storage, CvMSERParams params );

#endif
