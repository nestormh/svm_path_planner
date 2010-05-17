/*
 * main.cpp
 *
 *  Created on: 11/02/2010
 *      Author: jonatan
 */

#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
//#include <time.h>
//#include <math.h>
#include "Regions.h"
#include "Equivalence.h"

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
IplImage *threshold (IplImage *src, int r, int g, int b) {
	IplImage  *auxR,
			 *auxG,
			 *auxB,
			 *dst;

	dst = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	auxR = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	auxG = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	auxB = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);

	cvSplit (src, auxB, auxG, auxR, 0);

	cvCmpS(auxR, r, auxR, CV_CMP_GE);
	cvCmpS(auxG, g, auxG, CV_CMP_GE);
	cvCmpS(auxB, b, auxB, CV_CMP_LE);

	cvAnd(auxR, auxG, dst);
	cvAnd(dst, auxB, dst);

	cvReleaseImage (&auxR);
	cvReleaseImage (&auxG);
	cvReleaseImage (&auxB);

//	cvSet (dst, cvScalar(255), dst);

	return (dst);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/

Regions *onlyCones (Regions *reg, IplImage *img = NULL) {
	int i;
	int nRegions;
	Regions *cones;

	nRegions = reg->getNRegions();
	printf ("------------->Hay %d regiones\n", nRegions);

	cones = new Regions(reg->getImage(), reg->getNRegions());

	for (i = 0; i < nRegions; i++) {


		if ((reg->props[i].area > 20) && (reg->props[i].area < 5000) &&
			(reg->props[i].fr > 20) && (reg->props[i].fr < 45)) {
			printf ("Mostrando región %d. Área=%ld, Perímetro=%d, Centroide=(%d, %d)\n", i, reg->props[i].area, reg->props[i].perimeter, reg->props[i].centroid.x, reg->props[i].centroid.y);
			printf ("Factor redondez: %f\n", reg->props[i].fr);

			if (img) {
				cvCircle(img, cvPoint(reg->props[i].centroid.x, reg->props[i].centroid.y), 20, cvScalar(255,0,0), 1);
				cvShowImage("Regiones", img);
				cvWaitKey(0);
				cones->Insert(reg->props[i]);
			}
		}

	}

	return (cones);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int main () {
	IplImage *imagen,
			 *imagenBW;

	Regions *reg;
	Regions *cones;
	IplConvKernel *kernel;


	imagen = cvLoadImage("./Imagenes/cono3_150_izquierda.bmp");
//	imagen = cvLoadImage("./Imagenes/cono3_150_derecha.bmp");
//	imagen = cvLoadImage("./Imagenes/cono_140_izquierda.bmp");
//	imagen = cvLoadImage("./Imagenes/cono_130_derecha.bmp");

	if (!imagen) {
		printf ("Imagen no válida.\n");
		exit(-1);
	}


	cvNamedWindow("Imagen", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Regiones", CV_WINDOW_AUTOSIZE);

	cvShowImage("Imagen", imagen);
	cvWaitKey(1000);

	imagenBW = threshold (imagen, 250, 100, 200);
	cvShowImage("Imagen", imagenBW);
	cvWaitKey(1000);

	kernel = cvCreateStructuringElementEx(3, 3, 2, 1, CV_SHAPE_CROSS);

//	cvErode(imagenBW, imagenBW, kernel);
//	cvShowImage("Imagen", imagenBW);
//	cvWaitKey(1000);
//
//	cvDilate(imagenBW, imagenBW, kernel);
//	cvShowImage("Imagen", imagenBW);

	cvMorphologyEx(imagenBW, imagenBW, NULL, kernel, CV_MOP_OPEN);			//Open = erosión + dilatación
	cvWaitKey(1000);



	reg = (Regions*) new Regions (imagenBW);

	reg->Classical(imagenBW);

	reg->CalcProps();

	cones = onlyCones(reg, imagenBW);

	reg->Paint();
	cvWaitKey(1000);


	cones->Paint();
	cvWaitKey(1000);

	delete (reg);
	cvReleaseImage (&imagen);
	cvReleaseImage (&imagenBW);
	cvDestroyWindow("Imagen");
	printf("fin\n");
}
