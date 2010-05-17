/*
 * Regions.h
 *
 *  Created on: 23/02/2010
 *      Author: jonatan
 */

#include <cv.h>
#include <highgui.h>
#include "Equivalence.h"

#ifndef REGIONS_H_
#define REGIONS_H_

#define EQ_SIZE 50				// Tamaño inicial de la tabla de equivalencias

class Regions;

typedef struct labeled {
	int i, j, label;

} labeled;

typedef struct properties {
	int index;
	CvPoint centroid;
	int perimeter;
	long area;					// En píxeles
	double fr;
} properties;


class Regions {


private:
//	IplImage *img;
	IplImage *regions;
	int nRegions;				// No. Regiones actuales
	int sizeProps;			// Tamaño de la zona de memoria reservada para las propiedades


public:
	properties *props;

	Regions();
	Regions(IplImage *src);
	Regions(IplImage *src, int nRegions);
	int getNRegions();
	int getSizeProps();
	IplImage *getImage();
	void Insert (int index, CvPoint centroid, int perimeter, long area, double fr);
	void Insert (properties props);
	void Neighbors (int i, int j, CvSeq *a);
	void Classical (IplImage *img);
	void CalcProps();
	void Paint ();
	void PaintSeq ();
	virtual ~Regions();
};

#endif /* REGIONS_H_ */
