/*
 * Regions.cpp
 *
 *  Created on: 23/02/2010
 *      Author: jonatan
 */

#include "Regions.h"

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Regions::Regions() {
	// TODO Auto-generated constructor stub

	regions = NULL;
	nRegions = 0;
	sizeProps = 0;
	props = NULL;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Regions::Regions(IplImage *src) {
	// TODO Auto-generated constructor stub

	this->regions = cvCreateImage(cvGetSize(src), IPL_DEPTH_16U, 1);
	cvSetZero(regions);
	this->regions->origin = 0;
	this->nRegions = 0;
	this->sizeProps = 0;
	this->props = NULL;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN: Inicializa la estructura con memoria reservada para un número de regiones determinado
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Regions::Regions(IplImage *src, int sizeProps) {

	this->regions = src;

	this->regions->origin = 0;
	this->nRegions = 0;													// No hay regiones inicializadas

	this->sizeProps = sizeProps;
	this->props = (properties *) malloc (sizeProps * sizeof(properties));
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int Regions::getNRegions(){
	return this->nRegions;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int Regions::getSizeProps(){
	return this->nRegions;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
IplImage *Regions::getImage(){
	return this->regions;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::Insert(int index, CvPoint centroid, int perimeter, long area, double fr){

	this->props[this->nRegions].index = index;
	this->props[this->nRegions].centroid = centroid;
	this->props[this->nRegions].perimeter = perimeter;
	this->props[this->nRegions].area = area;
	this->props[this->nRegions].fr= fr;

	this->nRegions ++;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::Insert(properties props){

	this->props[this->nRegions] = props;

	this->nRegions ++;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::Neighbors (int i, int j, CvSeq *a) {
	labeled elem;

	if (j >= 1) {
		elem.j = j - 1;

		if (i >= 1) {
			elem.i = i - 1;
			elem.label = ((ushort *)(regions->imageData + (i-1)*regions->widthStep))[j-1];
			if (elem.label)
				cvSeqPush(a, &elem);
		}

		elem.i = i;
		elem.label = ((ushort *)(regions->imageData + i*regions->widthStep))[j-1];
		if (elem.label)
			cvSeqPush(a, &elem);

		elem.i = i + 1;
		if (i < regions->height - 1) {
			elem.label = ((ushort *)(regions->imageData + (i+1)*regions->widthStep))[j-1];
			if (elem.label)
				cvSeqPush(a, &elem);
		}
	}

	if (i >= 1) {
		elem.i = i-1;
		elem.j = j;
		elem.label = ((ushort *)(regions->imageData + (i-1)*regions->widthStep))[j];
		if (elem.label)
			cvSeqPush(a, &elem);
	}

}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
static int cmp_fnc( const void* _a, const void* _b, void* userdata ) {
    labeled* a = (labeled*)_a;
    labeled* b = (labeled*)_b;

    int y_diff = a->label - b->label;
    return y_diff;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::Classical (IplImage *img) {
	int i, j;
	int nlines, npixels, currentLabel, label;
	CvSeq *a;
	CvMemStorage *storage;
	labeled *head;
	int *lut;
	Equivalence *eqtable;

	storage= cvCreateMemStorage(0);
	a = cvCreateSeq( 0, sizeof(CvSeq), sizeof(labeled), storage);

	eqtable = (Equivalence*) new Equivalence(EQ_SIZE);

	// Top-down pass 1
	nlines = img->height;
	npixels = img->width;

	currentLabel = 0;

	head = (labeled *) malloc (sizeof(labeled));

	for (i = 0; i < nlines; i++) {
		for (j = 0; j < npixels; j++) {
			if (((uchar *)(img->imageData + i*(img->widthStep)))[j]) {
//				printf ("1 Pixel (%d, %d) a: %hu\n", i, j, ((uchar *)(img->imageData + i*(this->img->widthStep)))[j]);
				this->Neighbors(i, j, a);
				if (a->total == 0) {				// No hay vecinos etiquetados -> región nueva
					currentLabel ++;
					label = currentLabel;
				}  else {					// Hay vecinos -> etiqueta más pequeña
					cvSeqSort(a, cmp_fnc, 0);

					cvSeqPopFront(a, head);
					label = head->label;
				}

				((ushort *)(this->regions->imageData + i*(this->regions->widthStep)))[j] = label;
//				printf ("Etiqueta:%d\n", label);

				while (a->total) {
					cvSeqPopFront(a, head);

					if (label != head->label) {
//						printf ("Insertar(%d, %d)\n", label, head->label);
						eqtable->add(label, head->label);
						eqtable->add(head->label, label);
					}
				}
				cvClearSeq(a);
			}

		}

	}
	free (head);

	// Find equivalence classes

	eqtable->resolve();
	lut = eqtable->buildLUT(currentLabel, &this->nRegions);

	// Top-down pass 2
	for (i = 0; i < nlines; i++) {
		for (j = 0; j < npixels; j++) {
			label = ((ushort *)(this->regions->imageData + i*(this->regions->widthStep)))[j];
			((ushort *)(this->regions->imageData + i*(this->regions->widthStep)))[j] = lut[label];
		}
	}

//	printf ("width:%d height:%d\n----------------------\n", this->img->width, this->img->height);

//	cvShowImage("Imagen", this->regions);
//	cvWaitKey(1000);

	delete (eqtable);
	free (lut);
	cvReleaseMemStorage (&storage);

}



/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::CalcProps() {
	int i, j, k;
	IplImage *layer, *perimeter;
	long int sumaX, sumaY, area;
//	double fr;

	this->sizeProps = nRegions;
	this->props = (properties *) malloc (this->sizeProps * sizeof(properties));
	layer = cvCreateImage(cvGetSize(this->regions), IPL_DEPTH_8U, 1);
	perimeter = cvCreateImage(cvGetSize(this->regions), IPL_DEPTH_8U, 1);

	for (k = 0; k < this->nRegions; k++) {
		cvCmpS (this->regions, k, layer, CV_CMP_EQ);

//		cvSetZero(perimeter);

		sumaX = 0;
		sumaY = 0;
		area = 0;
		for (i = 0; i < this->regions->height; i++) {
			for (j = 0; j < this->regions->width; j++) {
				if ((((ushort *)(this->regions->imageData + i * this->regions->widthStep))[j]) == k) {
					sumaX += j;					// Acumular coordenadas X
					sumaY += i;					// Acumular coordenadas Y
					area ++;					// Contar No. de píxeles
				}
			}
		}

		// Se calcula el perímetro restando a la región el resultado de erosionar 1 píxel
//		cvErode(layer, perimeter);
//		cvSub (layer, perimeter, perimeter);

		// Se calcula el perímetro restando a la región dilatada en un píxel la región original
		cvDilate(layer, perimeter);
		cvSub (perimeter, layer, perimeter);

		this->props[k].index = k;
		this->props[k].area = area;
		if ((double)this->props[k].area) {
			this->props[k].centroid.x = sumaX/area;				// centroide = media de coordenadas de la región
			this->props[k].centroid.y = sumaY/area;
			this->props[k].perimeter = cvCountNonZero (perimeter);

			this->props[k].fr = (double)this->props[k].perimeter * (double)this->props[k].perimeter / (double)this->props[k].area;
		} else {
			printf ("Región %d con área 0\n", k);
			this->props[k].fr = -1;
		}

//		if ((area > 100) && (fr > 30) && (fr < 60)) {
//			printf ("Mostrando región %d. Área=%ld, Perímetro=%d, Centroide=(%d, %d)\n", k, area, this->props[k].perimeter, this->props[k].centroid.x, this->props[k].centroid.y);
//			printf ("SumaX= %ld. SumaY=%ld\n", sumaX, sumaY);
//			printf ("Factor redondez: %f\n", fr);
//
//			cvShowImage("Regiones", perimeter);
//			cvWaitKey(0);
//			cvShowImage("Regiones", layer);
//			cvWaitKey(0);
//		}
	}

	cvReleaseImage (&layer);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN: Pinta todas las regiones de colores diferentes
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::Paint() {
	int i;
	int step;
	IplImage *bwLayer, *colorLayer;
	int r, g, b;

	step = 0xFFFFFF / this->nRegions;

	colorLayer = cvCreateImage(cvGetSize(this->regions), IPL_DEPTH_8U, 3);
	bwLayer = cvCreateImage(cvGetSize(this->regions), IPL_DEPTH_8U, 1);


	for (i = 0; i < this->nRegions; i++) {
		cvCmpS (this->regions, this->props[i].index, bwLayer, CV_CMP_EQ);
//		cvSetZero(colorLayer);
		r = (((i + 1) * step) >> 16) & 0x0000FF;
		g = (((i + 1) * step) >> 8) & 0x0000FF;
		b = ((i + 1) * step) & 0x0000FF;
		cvSet(colorLayer, CV_RGB(r, g, b), bwLayer);
//		cvShowImage("Regiones", colorLayer);
//		printf ("Mostrando región %d: %d (%x) [%d, %d, %d] [%x, %x, %x]\n", i, i*step, i*step, r, g, b, r, g, b);
//		cvWaitKey(100);
	}

	cvShowImage("Regiones", colorLayer);
//	cvSaveImage ("superpuestos.bmp", colorLayer);

	cvReleaseImage (&colorLayer);
}



/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void Regions::PaintSeq() {
	int i;
	IplImage *layer;

	layer = cvCreateImage(cvGetSize(this->regions), IPL_DEPTH_8U, 1);

	for (i = 0; i < this->nRegions; i++) {
		cvCmpS (this->regions, i, layer, CV_CMP_EQ);
		cvShowImage("Regiones", layer);
//		printf ("Mostrando región %d\n", i);
//		cvWaitKey(100);
	}

	cvReleaseImage (&layer);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
Regions::~Regions() {
	// TODO Auto-generated destructor stub

	cvReleaseImage (&(this->regions));
	free (props);
}
