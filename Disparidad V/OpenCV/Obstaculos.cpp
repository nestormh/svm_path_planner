/*
 * Obstaculos.cpp
 *
 *  Created on: 09/11/2009
 *      Author: jonatan
 */

#include "Obstaculos.h"


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
Obstaculos::Obstaculos() {
	// TODO Auto-generated constructor stub
printf ("Constructor\n");

	n = 0;

	this->storage = cvCreateMemStorage(0);
	this->list = cvCreateSeq (0, sizeof(CvSeq), sizeof(obstaculo), storage);

}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::Insert(int delta, int u, int v, int width, int height) {
	obstaculo *nuevo;

	nuevo = new(obstaculo);

	nuevo->delta = delta;
	nuevo->u = u;
	nuevo->v = v;
	nuevo->width = width;
	nuevo->height = height;

	cvSeqPush(this->list, nuevo);

	this->n++;

}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::Print(){
	int i;
	obstaculo *aux;

	for (i = 0; i < n; i++){
		aux = (obstaculo *) cvGetSeqElem(this->list, i);
		printf ("[delta=%d, u=%d, v=%d, ancho=%d, alto=%d] -> Distancia:%f\n", aux-> delta, aux->u, aux->v, aux->width, aux->height, (float)(0.545*425/aux->delta));
	}

}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::Draw(IplImage* src){
	IplImage* color_dst;
	CvFont font;
	double hScale,
		   vScale;
	int lineWidth;
	CvScalar rectColor;
	char auxText[255];
	obstaculo *aux;
	int i;


	/* Configurar la fuente para el texto en im�genes */
	hScale = 0.5;
	vScale = 0.5;
	lineWidth = 0;
	cvInitFont (&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, lineWidth);

	// Para mostrar el rect�ngulo
	color_dst = cvCloneImage (src);
	color_dst->origin = src->origin;


	for (i = 0; i < this->n; i++){
		aux = (obstaculo *) cvGetSeqElem(this->list, i);

		if (aux->delta > 46)
			rectColor = CV_RGB(255,0,0);
		else if (aux->delta > 23)
			rectColor = CV_RGB(255,255,0);
		else
			rectColor = CV_RGB(0,255,0);

		cvRectangle(color_dst, cvPoint(aux->u, aux->v), cvPoint(aux->u + aux->width, aux->v + aux->height), rectColor);

		sprintf(auxText, "%.2f m",(float)(0.545*425/aux->delta));
		cvPutText (color_dst, auxText, cvPoint(aux->u, aux->v - 1), &font, rectColor);

	}
	cvShowImage( "Obstaculos", color_dst );

	cvReleaseImage(&color_dst);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::getObstacle(int index, obstaculo *returned){

	if (index > this->n){
		returned = (obstaculo *) cvGetSeqElem(this->list, index);
	} else
		returned = NULL;

}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::unlink(){
	this->n = 0;
	this->list = NULL;
	this->storage = NULL;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
Obstaculos::~Obstaculos() {
	// TODO Auto-generated destructor stub

	printf ("Destructor\n");

	if (this->list != NULL) {
		cvClearSeq (this->list);
	}

	if (this->storage != NULL) {
		cvReleaseMemStorage (&storage);
	}
}

