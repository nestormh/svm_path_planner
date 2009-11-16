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

	n = 0;
	this->frame = 0;
	this->storage = NULL;
	this->list = NULL;

}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
Obstaculos::Obstaculos(int frame) {
	// TODO Auto-generated constructor stub

	n = 0;
	this->frame = frame;
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

	printf("Frame %d (%d Obstáculos)\n", this->frame, this->n);
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
	PARÁMETROS: Desvincula los punteros a las listas de obstáculos.
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::Unlink(){
	this->n = 0;
	this->list = NULL;
	this->storage = NULL;
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: Save
	   FUNCIÓN: Escribe en el fichero que se le indica un objeto de la clase Obstaculos con el formato:
				   <No de frame> <No. de obstáculos>
				   <delta> <u> <v> <width> <height>
				   ...
				   <delta> <u> <v> <width> <height>
	PARÁMETROS: FILE *filename -> Fichero en el que se escribirá.
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::Save(FILE *filename) {
	int i;
	obstaculo *aux;

	fprintf (filename, "%d %d\n", this->frame, this->n);
	for (i = 0; i < n; i++){
		aux = (obstaculo *) cvGetSeqElem(this->list, i);
		fprintf (filename, "%d %d %d %d %d\n", aux->delta, aux->u, aux->v, aux->width, aux->height);
	}
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: Compare
	   FUNCIÓN: Compara dos elementos que se pasarán como punteros void para devolver un entero que indique
			    en qué orden deben ir.
	PARÁMETROS: const void *a -> Primer elemento a comparar
				const void *b -> Segundo elemento a comparar
				void *userdata -> Puntero a un entero, con typecast a void*
					0 -> Ordenar por delta
					1 -> Ordenar por u
					2 -> Ordenar por v
					3 -> Ordenar por ancho
					4 -> Ordenar por alto
	  DEVUELVE: -1 -> El primer elemento es menor
				 0 -> Son iguales
				 1 -> El segundo elemento es menor
-----------------------------------------------------------------------------------------------------------------*/
int Obstaculos::Compare (const void * a, const void * b, void *userdata) {
	obstaculo *o1, *o2;
	int campo, comparacion;

	o1 = (obstaculo *)a;
	o2 = (obstaculo *)b;

	campo = *(int *) userdata;

	switch (campo) {
	case 1:
		comparacion = o1->u - o2->u;
		break;
	case 2:
		comparacion = o1->v - o2->v;
		break;
	case 3:
		comparacion = o1->width - o2->width;
		break;
	case 4:
		comparacion = o1->height - o2->height;
		break;
	case 0:
	default:
		comparacion = o1->delta - o2->delta;
		break;
	}
	return (comparacion);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: Sort
	   FUNCIÓN: Ordena la secuencia de menor a mayor ordenando por el campo que se le indique.
	PARÁMETROS: order -> Especifica el campo por el cual se ordenará la secuencia
					0 -> Ordenar por delta
					1 -> Ordenar por u
					2 -> Ordenar por v
					3 -> Ordenar por ancho
					4 -> Ordenar por alto
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void Obstaculos::Sort(int order) {

	cvSeqSort (this->list, Compare, (void*)&order);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
Obstaculos::~Obstaculos() {
	// TODO Auto-generated destructor stub

	if (this->list != NULL) {
		cvClearSeq (this->list);
	}

	if (this->storage != NULL) {
		cvReleaseMemStorage (&storage);
	}
}

