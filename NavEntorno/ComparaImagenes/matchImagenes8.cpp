// matchImagenes6.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "comparaImagenes.h"
#include "CRuta.h"

int inicio = 275 , fin = 1390, vel = 0;
int pos = 0, next = 3;

comparaImagenes comp;

void initControl() {
	cvNamedWindow("Control", 0);
	cvMoveWindow("Control", 700, 10);

	cvCreateTrackbar("Inicio", "Control", &inicio, 726, NULL);
	cvCreateTrackbar("Fin", "Control", &fin, 726, NULL);
	cvCreateTrackbar("Pos", "Control", &pos, 726, NULL);
	cvCreateTrackbar("Vel", "Control", &vel, 1000, NULL);
	cvCreateTrackbar("Next", "Control", &next, 100, NULL);		
}

void initGUI() {	
	initControl();

	cvNamedWindow("Imagen1", 1);
	cvMoveWindow("Imagen1", 10, 10);
	cvNamedWindow("Imagen2", 1);	
	cvMoveWindow("Imagen2", 10, 280);
	
	cvNamedWindow("Persp", 1);
	cvMoveWindow("Persp", 360, 10);	

	cvNamedWindow("Resta", 1);
	cvMoveWindow("Resta", 360, 280);	

	cvNamedWindow("Mascara", 1);
	cvMoveWindow("Mascara", 10, 550);	
}

int inicio0() {	
	CRuta ruta1("C:\\Proyecto\\Datos", "iter");
	CRuta ruta2("C:\\Proyecto\\Datos", "iter");

	IplImage * resta = NULL;

	initGUI();
		
	tCoord coord;
	
	for (pos = inicio - 1; pos < fin - next; pos++) {
		cvSetTrackbarPos("Pos", "Control", pos);
	
		coord = ruta2.getPosicion(pos + next);				

		IplImage * img1 = ruta1.getImagenCercana(coord);
		IplImage * mask = ruta1.getMask();
		if (resta == NULL)
			resta = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);

		// Aquí debería tomarse la imagen desde la cámara		
		coord = ruta2.getPosicion(pos);		
		IplImage * img2 = ruta2.getImagenCercana(coord);

		comp.compara(img1, img2, mask, resta);		

		cvWaitKey(vel);		

		cvReleaseImage(&img1);
		cvReleaseImage(&img2);
		cvReleaseImage(&mask);
		
		if (pos > fin - next - 2)
			pos = inicio - 1;	
	}

	cvWaitKey(0);

	cvReleaseImage(&resta);

	return 0;
}

int inicio1() {	
	CRuta ruta1("C:\\Proyecto\\Datos", "320e");
	CRuta ruta2("C:\\Proyecto\\Datos", "320f");

	IplImage * resta = NULL;

	initGUI();
		
	tCoord coord;
	
	for (pos = inicio - 1; pos < fin - next; pos++) {
		cvSetTrackbarPos("Pos", "Control", pos);
	
		coord = ruta2.getPosicion(pos);		

		IplImage * img1 = ruta1.getImagenCercana(coord);
		IplImage * mask = ruta1.getMask();

		if (resta == NULL)
			resta = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);


		// Aquí debería tomarse la imagen desde la cámara		
		IplImage * img2 = ruta2.getImagenAt(pos);

		comp.compara(img1, img2, mask, resta);		

		cvWaitKey(vel);		

		cvReleaseImage(&img1);
		cvReleaseImage(&img2);
		cvReleaseImage(&mask);
		
		if (pos > fin - next - 2)
			pos = inicio - 1;	
	}

	cvWaitKey(0);

	cvReleaseImage(&resta);

	return 0;
}

int _tmain(int argc, _TCHAR* argv[]) {
	return inicio0();
}


