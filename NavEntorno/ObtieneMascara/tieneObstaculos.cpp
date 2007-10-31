#include "StdAfx.h"
#include "tieneObstaculos.h"

tieneObstaculos::tieneObstaculos() {
	bSize = 73;
	params = 6;
	puntos = 8;
	bufSize = 5;
	orAnd = 3;
	umbral = 128;
	puntos1.p1 = cvPoint(0, 0);
	puntos1.p2 = cvPoint(0, 0);
	puntos1.p3 = cvPoint(0, 0);
	puntos1.p4 = cvPoint(0, 0);
	puntos1.p5 = cvPoint(0, 0);	

	hormigas = new Hormigas("Hormigas");
}

tieneObstaculos::tieneObstaculos(int bSize, int params, int puntos, int bufSize) {	
	this->bSize = bSize;
	this->params = params;
	this->puntos = puntos;
	this->bufSize = bufSize;
	umbral = 128;
	puntos1.p1 = cvPoint(0, 0);
	puntos1.p2 = cvPoint(0, 0);
	puntos1.p3 = cvPoint(0, 0);
	puntos1.p4 = cvPoint(0, 0);
	puntos1.p5 = cvPoint(0, 0);

	hormigas = new Hormigas("Hormigas");
}

tieneObstaculos::~tieneObstaculos() {
	delete hormigas;
}

int tieneObstaculos::updateBSize(int val) {	
	if (val < 3)
		val = 3;
	if (val % 2 == 0)
		val++;
	bSize = val;
	return val;
}

int tieneObstaculos::updateParams(int val) {
	if (val < 3)
		val = 3;
	if (val % 2 == 0)
		val++;
	params = val;
	return val;
}

int tieneObstaculos::updateBuffer(int val) {
	if (val < 1)
		val = 1;
	buf.setMaxBuffer(val);
	return val;
}

int tieneObstaculos::updatePuntos(int val) {
	if (val < 1)
		val = 1;
	puntos = val;
	return val;
}

int tieneObstaculos::updateOrAnd(int val) {		
	return (orAnd = val);
}

int tieneObstaculos::updateUmbral(int val) {		
	return (umbral = val);
}

void tieneObstaculos::getContornos(IplImage * img, IplImage * mascara) {
	IplImage * horm = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);	
	
	cvCopyImage(img, horm);	

	secuenceStruct left, right;

	CvMemStorage * storage = cvCreateMemStorage (1000);
	CvSeq *comp;
	cvPyrSegmentation(img, horm, storage, &comp, 4, 255, 20);	
	cvReleaseMemStorage(&storage);

	cvShowImage("Pyramid", horm);

	int total = 0;
	CvPoint * puntos = hormigas->obtieneMascara(horm, &left, &right, &total);

	if (puntos == NULL)
		cout << "NULL" << endl;

	for (int i = 1; i < total; i++) {		
		//cout << puntos[i].x << ", " << puntos[i].y << endl;
		cvLine(mascara, puntos[i - 1], puntos[i], cvScalarAll(255), 10);
	}

	//cvWaitKey(0);

	//cvLine(mascara, mask.a, mask.b, cvScalarAll(255), 20);
	//cvLine(mascara, mask.a, mask.d, cvScalarAll(255), 20);
	//cvLine(mascara, mask.c, mask.b, cvScalarAll(255), 20);	

	cvShowImage("Umbral", mascara);

	cvReleaseImage(&horm);		
}

void tieneObstaculos::creaMascaraDesdePoligono(IplImage * imagen, IplImage * mascara) {	
	IplImage * circulo = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 1);
	cvZero(circulo);
	
	cvEllipse(circulo, cvPoint(circulo->width / 2, circulo->height / 2 - 10), cvSize(circulo->width / 2 - 10, circulo->height / 2 - 10), 0, 180, 360, cvScalar(255), -1); 
	cvRectangle(circulo, cvPoint(10, 0), cvPoint(imagen->width - 10, imagen->height / 2), cvScalar(255), CV_FILLED);
	//cvRectangle(circulo, cvPoint(0, 0), cvPoint(imagen->width - 1, 20), cvScalar(0), CV_FILLED);	
		
	cvAnd(mascara, circulo, mascara);	

	cvReleaseImage(&circulo);
}

void tieneObstaculos::anadeObjetos(IplImage * imagen, IplImage * mascara) {
	IplImage * gris = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 1);
	IplImage * poligono = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 1);

	cvCopyImage(mascara, poligono);
	cvFloodFill(poligono, cvPoint(poligono->width / 2, poligono->height - 1), cvScalar(255));
	
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contornos = 0;	
	CvSeq* result;

	cvCvtColor(imagen, gris, CV_BGR2GRAY);

	cvAdaptiveThreshold(gris, gris, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, bSize, params);
	
	cvFindContours(gris, storage, &contornos, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);   					
	cvZero(gris);
	
	for( ; contornos != 0; contornos = contornos->h_next ) {								
		if (fabs(cvContourArea(contornos)) > 300) {					
			cvDrawContours(gris, contornos, cvScalar(255), cvScalar(255), -1, -1, 8);
		}
	}		

	cvDilate(gris, gris);
	cvAnd(gris, poligono, mascara);

	cvShowImage("Umbral", mascara);
	
	cvReleaseMemStorage(&storage);		

	cvReleaseImage(&gris);
	cvReleaseImage(&poligono);
}

IplImage * tieneObstaculos::getMask(IplImage * imagen) {	
	IplImage * mascara = cvCreateImage(cvGetSize(imagen), IPL_DEPTH_8U, 1);	
	cvZero(mascara);	
	
	getContornos(imagen, mascara);		

	//anadeObjetos(imagen, mascara);

	//creaMascaraDesdePoligono(imagen, mascara);

	if (cvCountNonZero(mascara) < 10) {
		cvSet(mascara, cvScalar(255));
	}

	cvShowImage("Imagen", imagen);	

	cvSubS(imagen, cvScalar(100, 100, 100), imagen);
	cvAddS(imagen, cvScalar(100, 100, 100), imagen, mascara);
	cvShowImage("Debug", imagen);
		
	cvReleaseImage(&imagen);
	
	return mascara;
}