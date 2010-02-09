/**************************************************************************
 * Proyecto de reconocimiento de carreteras mediante visión artificial.   *
 * Autor: Rafael Arnay del Arco.                                          *
 * Fecha de última modificación: 22 Enero 2009                            *
 **************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <shmCInterface.h>
#include "aco.h"






int bordeSup = 30;
int horizonSlider = 90;
int edgeSlider = 12;//75
int kBar = 80, kpBar = 70, kdBar = 30, izqBar = 160, dchaBar  = 160;
int searchAreas = 10;
int consigna = 0;
float anguloCamara  = 170;
int errorAnt;





void on_trackbar(int h)
{
	shmWriteRoadSegmentationParams(horizonSlider,edgeSlider,searchAreas);
   if (horizonSlider >= 1)
		bordeSup = horizonSlider;
}
void on_trackbar2(int h)
{  
	shmWriteRoadSegmentationParams(horizonSlider,edgeSlider,searchAreas);
}
void on_trackbar3(int h)
{  
	shmWriteRoadSegmentationParams(horizonSlider,edgeSlider,searchAreas);
}


/*************************************************************************
 * Procedimiento con las siguientes etapas:
 * 1. Obtiene una captura de vídeo.
 * 		1.1 De un fichero
 * 		1.2 De una capturadora de vídeo conectada al ordenador
 * 2. Preprocesado de la imágen
 * 		2.1 Canny
 * 		2.2 Operaciones morfológicas
 * 3. Obtención de los caminos mínimos a través de OCH
 * 4. Actualización del punto de atracción.
 * 5. Actualización del patrón de la carretera.
 *************************************************************************/ 
void detectarCarretera(void) {

IplImage* img = 0;


IplImage* ed = 0;
IplImage* gray = 0;
IplImage* dst = 0;
IplImage* mask = 0;
IplImage* shadows = 0;



//IplImage* h = 0;
//IplImage* l = 0;
//IplImage* s = 0;
//IplImage* hls = 0;

IplImage* bordes = 0;
IplImage* shadowMask = 0;
IplImage* segmented = 0;
IplImage* traces = 0;
IplImage* hough = 0;
IplImage* tracesaux = 0;
IplImage* tempImage = 0;
IplImage* tempCapture = 0;





int keyCode;

//variables de la colonia

colonyStruct *colony;
secuenceStruct shortestLeftPath;
secuenceStruct shortestRightPath;

colony = (colonyStruct*)malloc(sizeof(colonyStruct));

int attractionX = 160, attractionXAnt = -1,attractionY = 270;
int refLeftY = -1,refRightY = -1;
int aRef = -1,bRef = -1,cRef = -1,dRef = -1;
int selectedImage = 1;
int corte = 0;




CvCapture* capture;											//Estructura donde almacenar el stream de vídeo de entrada
CvMemStorage* storage = cvCreateMemStorage(0);				//storage y contours: necesarios para la función findcontours
CvSeq* contour = 0;

cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);				//Creación y colocación de las ventanas
cvNamedWindow("sliderWin", CV_WINDOW_AUTOSIZE);
cvMoveWindow("mainWin", 100, 100);

//system ("v4l2-ctl --dev /dev/video2 -i 2");
//capture = cvCaptureFromAVI("videofinal5.avi");				//Captura archivo de vídeo de la ruta especificada
capture = cvCaptureFromCAM(-1);							//Captura a través de una capturadora de vídeo
if(!cvGrabFrame(capture)){              
		printf("El fichero no se encuentra o no tiene el formato adecuado\n");
		return;
}

img=cvRetrieveFrame(capture);           					//nos quedamos con el primer frame de la captura
img->origin = 1;											//herencia del Windows, la referencia en la imagen es la esquina
															//inferior izquierda.

//Creación de las imágenes

gray = cvCreateImage(cvGetSize(img),8,1); gray->origin = 1;
dst = cvCreateImage(cvGetSize(img),8,1); dst->origin =1;
segmented = cvCreateImage(cvGetSize(img),8,1); segmented->origin = 1;
bordes = cvCreateImage(cvGetSize(img),8,1); bordes->origin = 1;
shadowMask = cvCreateImage(cvGetSize(img),8,1); shadowMask->origin = 1;
mask = cvCreateImage(cvGetSize(img),8,3); mask->origin = 1;
shadows = cvCreateImage(cvGetSize(img),8,1); shadows->origin = 1;


// Necesarias si se va a trabajar con la luminancia

//hls = cvCreateImage(cvGetSize(img),8,3); hls->origin = 1;
//h = cvCreateImage(cvGetSize(img),8,1); h->origin = 1;
//s = cvCreateImage(cvGetSize(img),8,1); s->origin = 1;
//l = cvCreateImage(cvGetSize(img),8,1); l->origin = 1;

traces = cvCreateImage(cvGetSize(img),8,3); traces->origin = 1;
tempCapture = cvCreateImage(cvGetSize(img),8,3); tempCapture->origin = 1;

tracesaux = cvCreateImage(cvGetSize(img),8,3); tracesaux->origin = 1;
tempImage = cvCreateImage(cvGetSize(img),8,1);tempImage->origin = 1;
ed = cvCreateImage(cvGetSize(img),8,1); ed->origin = 1;


// Creación de las barras de desplazamiento

cvCreateTrackbar("Horiz.", "sliderWin", &horizonSlider, 200, on_trackbar);
cvCreateTrackbar("Bordes", "sliderWin", &edgeSlider, 255, on_trackbar2);
cvCreateTrackbar("Pos.Ini.", "sliderWin", &searchAreas, 170, on_trackbar3);
cvShowImage("sliderWin",traces);


cvSet(shadowMask,cvScalar(1));
consigna = 0;
while(1) {
	
	
	shmReadRoadSegmentationParams(&horizonSlider,&edgeSlider,&searchAreas); // Leemos de memoria compartida los parámetros del preprocesado
	
	cvConvertImage(img,img,1); 								//flip vertical para mostrar la imágen correctamente.
	//cvCvtColor(img,hls,CV_BGR2HLS);						//en caso de que queramos trabajar con la luminancia
	//cvSplit(hls,h,l,s,0);
	
		
	
	cvCvtColor(img,gray,CV_RGB2GRAY);						//pasamos la imagen de entrada a escala de grises

	//cvSmooth(gray,gray,2,5);								//emborronamos para filtrar las altas frecuencias 
	
	cvCanny(gray,segmented,edgeSlider,edgeSlider+70); 		//aplicamos canny para detección de bordes
	
	cvDilate(segmented,segmented,0,3);						//operación morfológica para dilatar los bordes y 
															//crear dos zonas diferenciadas en la imágen: los márgenes
															//en blanco y la carretera en negro.
	
	cvCopy(segmented,dst);									//se necesita una imagen temporal para calcular los contornos
	
	cvFindContours( dst, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );
	cvZero(bordes);
	for( ; contour != 0; contour = contour->h_next ) {
		if (contour->total > 20) {
			cvDrawContours( bordes, contour, cvScalar(255), cvScalar(255), -1, 1, 8 );
		}
	}
	
	attractionY = bordes->height-bordeSup+20;				//altura del punto de atracción
	
	//inicialización de la colonia de la izquierda
	initColony(bordes,traces,colony,bordes->width,bordes->height-bordeSup,attractionX,attractionY,searchAreas,&refLeftY);
	
	//Cálculo del camino mínimo por la izquierda
	acoMetaheuristic(bordes,traces,colony,&shortestLeftPath,3);
	
	cvCopy(traces,tracesaux);								//se salva el camino más corto de la izquierda
	
	//inicialización de la colonia de la derecha
	initColony(bordes,traces,colony,bordes->width,bordes->height-bordeSup,attractionX,attractionY,img->width-searchAreas,&refRightY);
	
	//cálculo del camino mínimo por la derecha
	acoMetaheuristic(bordes,traces,colony,&shortestRightPath,3);

	cvAdd(tracesaux,traces,tracesaux);						//juntamos los dos caminos-solución en una imágen
	
	cvAdd(img,tracesaux,tracesaux);							//solapamos los caminos-solución con la imágen de entrada de fondo
	
	cvSetZero(mask);										//inicialización del patrón de la carretera

	//actualización del punto de atracción
	setPointofAttraction(mask,shadowMask,&shortestLeftPath,&shortestRightPath,&attractionX,&aRef,&bRef,&cRef,&dRef,&corte,consigna);
	shmWriteRoadInformation(aRef,bRef,corte);

	cvAdd(img,mask,img);									//solapamos el patrón con la imágen de entrada de fondo
	
	if (attractionXAnt == -1) attractionXAnt = attractionX;	

	/***************Depuración***********************	
	 * Dependiendo del número pulsado se mostrarán 
	 * distintas imágenes correspondientes a diferentes
	 * etapas del procesado. Si se pulsa otra tecla,
	 * se termina la ejecución.
	 * **********************************************/
	switch(selectedImage) {
	case 1:
		cvShowImage("mainWin",img);
		break;
	case 2: 
		cvShowImage("mainWin",tracesaux);
		break;
	case 3: 
		cvShowImage("mainWin",bordes);	
		break;
	case 4: 
		cvShowImage("mainWin",segmented);
		break;
	}	
	keyCode = cvWaitKey(2);
	if (keyCode == '1') 
		selectedImage = 1;
	else if(keyCode == '2')
		selectedImage = 2;
	else if(keyCode == '3')
		selectedImage = 3;
	else if(keyCode == '4')
		selectedImage = 4;
	else if (keyCode == 'q')
		break;
		
	/**********Fin de depuración***************/
	
	if(!cvGrabFrame(capture)){              				// Se captura el siguiente frame 
		printf("Could not grab a frame\n\7");
		return;
	}
	cvCopy(cvRetrieveFrame(capture),img);	
	cvSetTrackbarPos( "Horiz.", "sliderWin", bordeSup);
	cvSetTrackbarPos( "Bordes", "sliderWin", edgeSlider);	
}



cvReleaseCapture(&capture);									//liberamos la memoria de la captura de vídeo

cvReleaseImage(&tempImage);									//liberamos la memoria de las imágenes
cvReleaseImage(&bordes);
cvReleaseImage(&shadowMask);
cvReleaseImage(&segmented);
cvReleaseImage(&hough);
cvReleaseImage(&mask);
cvReleaseImage(&shadows);

//cvReleaseImage(&h);
//cvReleaseImage(&l);
//cvReleaseImage(&s);
//cvReleaseImage(&hls);

cvReleaseImage(&traces);
cvReleaseImage(&tracesaux);
cvReleaseImage(&tempCapture);
cvReleaseImage(&gray);
cvReleaseImage(&dst);
cvReleaseImage(&ed);

cvDestroyWindow("mainWin");									//Liberamos la memoria de las ventanas
cvDestroyWindow("sliderWin");



}

/************************************/

int main(int argc, char *argv[]){
	shmSafeGet();
	
	detectarCarretera();
	
	shmSafeErase();
	
	/*
	int shmid;
	double* v;
	

	
	shmid = shmSafeGet();
	v = (double*)shmSafeMap(shmid);
	shmSafeDeconnect(v);	
	shmSafeErase(shmid);
	*/
	
return 0;
}

