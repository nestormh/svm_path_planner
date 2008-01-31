// matchImagenes6.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "comparaImagenes.h"
#include "CRuta.h"

int inicio = 850, fin = 1617, vel = 200;
int pos = 0, next = 3;

comparaImagenes comp;

void initControl() {
	cvNamedWindow("Control", 0);
	cvMoveWindow("Control", 700, 10);

	cvCreateTrackbar("Inicio", "Control", &inicio, 726, NULL);
	cvCreateTrackbar("Fin", "Control", &fin, 2000, NULL);
	cvCreateTrackbar("Pos", "Control", &pos, 2000, NULL);
	cvCreateTrackbar("Vel", "Control", &vel, 1000, NULL);
	cvCreateTrackbar("Next", "Control", &next, 100, NULL);		
}

void initGUI() {	
	initControl();

	/*cvNamedWindow("Imagen1", 1);
	cvMoveWindow("Imagen1", 10, 10);
	cvNamedWindow("Imagen2", 1);	
	cvMoveWindow("Imagen2", 10, 280);
	
	cvNamedWindow("Persp", 1);
	cvMoveWindow("Persp", 360, 10);*/

	cvNamedWindow("Resta", 1);
	cvMoveWindow("Resta", 360, 320);	

	/*cvNamedWindow("Mascara", 1);
	cvMoveWindow("Mascara", 10, 550);*/

	cvNamedWindow("Debug", 1);
	cvMoveWindow("Debug", 10, 10);

	cvNamedWindow("Debug2", 1);
	cvMoveWindow("Debug2", 360, 10);
}

void creaMascara(IplImage * mascara) {
	cvZero(mascara);
	CvPoint poly[] = { cvPoint(122,41), cvPoint(141, 40), cvPoint(360, 240), cvPoint(-196, 240) };
	cvFillConvexPoly(mascara, poly, 4, cvScalar(255));
}

int inicio0() {	
	//CRuta ruta1("C:\\Proyecto\\Datos", "iter");
	//CRuta ruta2("C:\\Proyecto\\Datos", "iter");
	CRuta ruta1("C:\\Proyecto\\Datos", "rutaBase");
	CRuta ruta2("C:\\Proyecto\\Datos", "rutaBase");

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
	/*CRuta ruta1("C:\\Proyecto\\Datos", "320e");
	CRuta ruta2("C:\\Proyecto\\Datos", "320f");*/
	CRuta ruta1("C:\\Proyecto\\Datos", "aerop14EneSinObs");
	CRuta ruta2("C:\\Proyecto\\Datos", "aerop14EneConObs");

	IplImage * resta = NULL;

	initGUI();
		
	tCoord coord;
	
	for (pos = inicio - 1; pos < fin - next; pos++) {
		cvSetTrackbarPos("Pos", "Control", pos);
	
		coord = ruta2.getPosicion(pos);		

		IplImage * img1 = ruta1.getImagenCercana(coord);
		IplImage * mask = ruta1.getMask();
		//IplImage * mask = NULL;
		creaMascara(mask);

		if (resta == NULL)
			resta = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);


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

int captura2camaras() {
	CvCapture* capture1 = cvCaptureFromCAM(0);
    CvCapture* capture2 = cvCaptureFromCAM(0);

    IplImage* img1 = 0;
    IplImage* img2 = 0;

	// create a window
    cvNamedWindow("Left", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Left", 100, 100);

    cvNamedWindow("Right", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Right", 200, 100);

	while(true) {
		if(!cvGrabFrame(capture1)){              // capture a frame
			printf("Could not grab a frame from camera 1\n\7");
			exit(0);
		}
		if(!cvGrabFrame(capture2)){              // capture a frame
			printf("Could not grab a frame from camera 2\n\7");
			exit(0);
		}

		img1=cvRetrieveFrame(capture1);           // retrieve the captured frame
		img2=cvRetrieveFrame(capture2);

		// show the image
		cvShowImage("Left", img1 );
		cvShowImage("Right", img2 );

		// wait for a key
		cvWaitKey(20);
	}

	cvReleaseImage(&img1);
	cvReleaseImage(&img2);

	//Release the image
	cvReleaseCapture(&capture1);
	cvReleaseCapture(&capture2);

	
	return 0;

}

int _tmain(int argc, _TCHAR* argv[]) {
	return inicio1();
}


