/**************************************************************************
 * Proyecto de reconocimiento de carreteras mediante visión artificial.   *
 * Autor: Rafael Arnay del Arco.                                          *
 **************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include "aco.h"
#include "Serial.h"




int bordeSup = 30;
int horizonSlider = 40;
int edgeSlider = 75;
int kBar = 80, kpBar = 70, kdBar = 30, izqBar = 160, dchaBar  = 160;
int searchAreas = 10;
int consigna = 0;
float anguloCamara  = 170;
int errorAnt;
CSerial serial;
CSerial serial2;





/***********************************************************************
Obtenemos los valores de referencia de la carretera.
************************************************************************/
/*void media(IplImage* img,IplImage* mask,CvScalar* mean, CvScalar* sdv){

	IplImage* mask2 = cvCreateImage(cvGetSize(img),8,1); mask2->origin = 1;
		
	cvSetImageCOI(mask,2);
	cvCopy(mask,mask2);
	cvSetImageCOI(mask,0);
	cvThreshold(mask2,mask2,0,1,CV_THRESH_BINARY);
	
	cvAvgSdv(img,mean,sdv,mask2);
	//printf("Media (%f %f %f)",mean.val[0],mean.val[1],mean.val[2]);
	//printf(" Sdv (%f %f %f)\n",sdv.val[0],sdv.val[1],sdv.val[2]);

	cvReleaseImage(&mask2);
}
*/
/****************************************************************************
Calculamos un mapa de distancia a los valores de referencia en las posiciones
de los píxeles detectados como bordes
*****************************************************************************/
/*
void distance(IplImage* img, IplImage* bordes,CvScalar mean, CvScalar sdv) {

	CvPoint edgePos;
	CvScalar imgVal;
	CvScalar imgDist;
	
	

	
	
	
	for (int i = 0; i < img->width;i++){
		for (int j = 0; j < img->height;j++){
	
			imgVal = cvGet2D(bordes,j,i);
			if (imgVal.val[0]==255) {
				
				imgVal = cvGet2D(img,j,i);
				imgDist.val[1] = 255 - abs(mean.val[0] - imgVal.val[0]);
				imgDist.val[2] = 255 - abs(mean.val[2] - imgVal.val[2]);
				imgDist.val[0] = (int)floor(float(imgDist.val[1]+imgDist.val[2]) / 2.0);

				cvSet2D(bordes,j,i,imgDist);

			}
		}
	}
}


*/
/***********************************************************************
  Determina cuánto debe girar el coche y envía el comando al volante 
  @Parametros:
  x =	orientación actual medida como coordenada x en la última 
		fila de procesado de la imagen.
  xAnt = orientación anterior.
  a = corte del borde izq con la fila 0 de la imagen.
  b = corte del borde dcho con la fila 0 de la imagen.
*************************************************************************/
void setAnguloGiro(int x, int* xAnt, IplImage* img, int a,int b) {


	int variacion = abs(x - 160);
	int radius = 20;
	float kp = (float)kpBar/100.0,kd = (float)kdBar/100.0;
	int c,error;
	int rango;
	float izq, dcha, k = (float)kBar/20.0;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1,1,0,2);
	
	
	rango = 160-izqBar;
	if (a > rango){
		izq = a - rango;
		cvPutText( img, "<!", cvPoint(10,70),&font,cvScalar(255,255,0));
	}
	else
		izq = 0;
	rango = 160+dchaBar;
	if (b < rango) {
		dcha = rango - b;
		cvPutText( img, "!>", cvPoint(278,70),&font,cvScalar(255,255,0));
	}
	else
		dcha = 0;
	consigna = (int)floor(160.0 +float((dcha-izq)*k));
	error = x-consigna;

	if (errorAnt == -1)
		errorAnt = error;
	
	float comando = (error) * kp + (error - errorAnt)*kd;
	errorAnt = error;
	c = (int)floor(float((160 + comando) * 5000) / 160.0);
	if (c < 1000) c  =1000;
	else if (c > 9000) c  = 9000;
	CvPoint center = cvPoint(img->width-radius-10,radius+20);

	cvCircle( img, center, radius, cvScalar(0,0,255), 2);
	float angle = -(((float)c-1000.0)/8000.0)*160+170;
    cvLine( img, center, cvPoint( cvRound( center.x + radius*cos((angle)*CV_PI/180)),
    cvRound( center.y + radius*sin((angle)*CV_PI/180))), cvScalar(0,0,255),2 );
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5,0.5,0,2);
	
	cvPutText( img, "DIR", cvPoint(278,5),&font,cvScalar(0,0,255));
	serial.setVolante(c);
	
}

/***************************************************
Determina el ángulo de la cámara a partir del corte
de las dos rectas del patrón. La idea es mantener el
corte siempre en la zona alta de la imagen.
***************************************************/

void setAnguloCamara(int corte, IplImage* dst) {

	int radius = 20;
	if ((corte <200) && (anguloCamara < 200))
		anguloCamara += 0.2;
	else if ((corte > 240) && (anguloCamara > 80))
		anguloCamara -= 0.2;
		CvPoint center = cvPoint(radius+10,radius+20);
	
	//Imprime el indicador del ángulo de la cámara.
	cvCircle( dst, center, radius, cvScalar(255,0,0), 2);
    cvLine( dst, center, cvPoint( cvRound( center.x - radius*cos((anguloCamara)*CV_PI/180)),
    cvRound( center.y + radius*sin((anguloCamara)*CV_PI/180))), cvScalar(255,0,0),2 );

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5,0.5,0,2);
	cvPutText( dst, "CAM", cvPoint(14,5),&font,cvScalar(255,0,0));

	serial2.setCamara((unsigned char)floor (anguloCamara));
}
/******************************************/
void calcLuminosity(IplImage* l) {
	
	CvScalar mean;
	mean = cvAvg(l);
	printf ("%f\n",mean.val[0]);
}
/******************************************/
void on_trackbar(int h)
{
   if (horizonSlider >= 1)
		bordeSup = horizonSlider;
}
void on_trackbar2(int h)
{  
}
void on_trackbar3(int h)
{  
}
/*********************************
Imprime un menú por pantalla
**********************************/
int menu (const char* path) {
	int op;
	
	printf ("\n****************************\n");
	printf ("* 1 Capturar de una camara *\n");
	printf ("* 2 Capturar de un archivo *\n");
	printf ("* 3 Salir                  *\n");
	printf ("****************************\n");
	scanf("%d", &op);

	if (op == 2) {
		printf ("Ruta del archivo:\n");
		scanf("%s", path);
	}
	return op;

}
/**************************************************
Llama al algoritmo de OCH y a los procedimientos
de control de la cámara y la dirección.
Recibe un valor que indica si se ha de leer de fichero
o de una cámara.En caso de que la fuente sea un fichero,
se le pasa la ruta del mismo.
***************************************************/
void detectarCarretera(int source, const char* path) {

IplImage* img = 0;


IplImage* ed = 0;
IplImage* gray = 0;
IplImage* dst = 0;
IplImage* mask = 0;
IplImage* shadows = 0;



IplImage* h = 0;
IplImage* l = 0;
IplImage* s = 0;
IplImage* hls = 0;

IplImage* bordes = 0;
IplImage* shadowMask = 0;
IplImage* segmented = 0;
IplImage* traces = 0;
IplImage* hough = 0;
IplImage* tracesaux = 0;
IplImage* tempImage = 0;

IplConvKernel* seDisco = cvCreateStructuringElementEx(5,5,1,1,CV_SHAPE_ELLIPSE);
IplConvKernel* seDisco2= cvCreateStructuringElementEx(20,20,1,1,CV_SHAPE_ELLIPSE);



int i;
int keyCode;
bool shoot = false;

//variables de la colonia

colonyStruct colony;
secuenceStruct shortestLeftPath;
secuenceStruct shortestRightPath;
int attractionX = 160, attractionXAnt = -1,attractionY = 270;
int refLeftY = -1,refRightY = -1;
int aRef = -1,bRef = -1,cRef = -1,dRef = -1;
int startingAreasW = 10;
int selectedImage = 1;
int corte = 0;




CvCapture* capture;

CvVideoWriter* vw = cvCreateVideoWriter( "salida.avi", -1, 30, cvSize(320,240));

CvMemStorage* storage = cvCreateMemStorage(0);
CvSeq* contour = 0;

//creamos la ventana
cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
cvNamedWindow("sliderWin", CV_WINDOW_AUTOSIZE);
cvMoveWindow("mainWin", 100, 100);


	switch(source) {
	case 2:
		capture = cvCaptureFromAVI(path);
		edgeSlider = 75;
		break;
	case 1:
		capture = cvCaptureFromCAM(-1);
		edgeSlider = 120;
		break;
	}
if(!cvGrabFrame(capture)){              // capture a frame 
		printf("El fichero no se encuentra o no tiene el formato adecuado\n");
		return;
}
else {
img=cvRetrieveFrame(capture);           // retrieve the captured frame



//reservamos imagenes

gray = cvCreateImage(cvGetSize(img),8,1); gray->origin = 1;
dst = cvCreateImage(cvGetSize(img),8,1); dst->origin = 1;



segmented = cvCreateImage(cvGetSize(img),8,1); segmented->origin = 1;
bordes = cvCreateImage(cvGetSize(img),8,1); bordes->origin = 1;
shadowMask = cvCreateImage(cvGetSize(img),8,1); shadowMask->origin = 1;
mask = cvCreateImage(cvGetSize(img),8,3); mask->origin = 1;
shadows = cvCreateImage(cvGetSize(img),8,1); shadows->origin = 1;




hls = cvCreateImage(cvGetSize(img),8,3); hls->origin = 1;
h = cvCreateImage(cvGetSize(img),8,1); h->origin = 1;
s = cvCreateImage(cvGetSize(img),8,1); s->origin = 1;
l = cvCreateImage(cvGetSize(img),8,1); l->origin = 1;

traces = cvCreateImage(cvGetSize(img),8,3); traces->origin = 1;
tracesaux = cvCreateImage(cvGetSize(img),8,3); tracesaux->origin = 1;
tempImage = cvCreateImage(cvGetSize(img),8,1); tempImage->origin = 1;
ed = cvCreateImage(cvGetSize(img),8,1); ed->origin = 1;


cvCreateTrackbar("Horiz.", "sliderWin", &horizonSlider, 200, on_trackbar);
cvCreateTrackbar("Bordes", "sliderWin", &edgeSlider, 255, on_trackbar2);
cvCreateTrackbar("Pos.Ini.", "sliderWin", &searchAreas, 170, on_trackbar3);
cvCreateTrackbar("K", "sliderWin", &kBar, 100, on_trackbar3);
cvCreateTrackbar("Kp", "sliderWin", &kpBar, 100, on_trackbar3);
cvCreateTrackbar("Kd", "sliderWin", &kdBar, 100, on_trackbar3);
cvCreateTrackbar("Dist.D.", "sliderWin", &dchaBar, 300, on_trackbar3);
cvCreateTrackbar("Dist.I.", "sliderWin", &izqBar, 300, on_trackbar3);
cvShowImage("sliderWin",traces);

if (serial.Open(3, 9600)) {
 printf("Port opened successfully\n");
 
}

else
 printf("Failed to open port!\n");
/*
if (serial2.Open(8, 9600)) {
 printf("Port opened successfully\n");
 serial2.setCamara(anguloCamara);
}
else
 printf("Failed to open port!\n");
*/

cvSet(shadowMask,cvScalar(1));
consigna = 0;
while(1) {
	
	
	cvCvtColor(img,hls,CV_BGR2HLS);
	cvSplit(hls,h,l,s,0);
	calcLuminosity(l);
	
	if (shoot)
		cvSaveImage("img.jpg", img);
	cvCvtColor(img,gray,CV_RGB2GRAY);

	if (shoot)
		cvSaveImage("gray.jpg", gray);
	
	
	cvCanny(gray,segmented,edgeSlider,edgeSlider+70);
	
	if (shoot)
		cvSaveImage("canny.jpg", segmented);
	

	cvDilate(segmented,segmented,0,3);
	
	if (shoot)
		cvSaveImage("dilated.jpg", segmented);
	
	cvCopy(segmented,dst);
	
	cvFindContours( dst, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );
	cvZero(bordes);
	for( ; contour != 0; contour = contour->h_next )
        {
			if (contour->total > 20) {
				
				
				cvDrawContours( bordes, contour, cvScalar(255), cvScalar(255), -1, 1, 8 );
			}
        }
	//distance(hls,bordes,mean,sdv);

if (shoot)
		cvSaveImage("entrada.jpg", bordes);
	

	
	
	
	attractionY = bordes->height-bordeSup+20;
		
	initColony(bordes,traces,&colony,bordes->width,bordes->height-bordeSup,attractionX,attractionY,searchAreas,&refLeftY);
	acoMetaheuristic(bordes,traces,&colony,&shortestLeftPath,3);
	cvCopy(traces,tracesaux);
	initColony(bordes,traces,&colony,bordes->width,bordes->height-bordeSup,attractionX,attractionY,img->width-searchAreas,&refRightY);
	acoMetaheuristic(bordes,traces,&colony,&shortestRightPath,3);

	cvAdd(tracesaux,traces,tracesaux);
	if (shoot)
		cvSaveImage("traces.jpg", tracesaux);
	
	cvAdd(img,tracesaux,tracesaux);
if (shoot)
		cvSaveImage("tracesImg.jpg", tracesaux);
	
	
	cvSetZero(mask);

	setPointofAttraction(mask,shadowMask,&shortestLeftPath,&shortestRightPath,&attractionX,&aRef,&bRef,&cRef,&dRef,&corte,consigna);

	//media(hls,mask,&mean,&sdv);
	cvAdd(img,mask,img);
	if (shoot)
		cvSaveImage("patron.jpg", img);
	
	if (attractionXAnt == -1) attractionXAnt = attractionX;

	setAnguloGiro(attractionX,&attractionXAnt,img,cRef,dRef);
	//setAnguloCamara(corte, img);
	
	
	
	cvWriteFrame(vw,img);

	
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
case 5: 
		cvShowImage("mainWin",h);
		break;
case 6: 
		cvShowImage("mainWin",l);
		break;
case 7: 
		cvShowImage("mainWin",s);
		break;

	}
	
	shoot = false;
	keyCode = cvWaitKey(1);
	if (keyCode == '1') 
		selectedImage = 1;
	else if(keyCode == '2')
		selectedImage = 2;
	else if(keyCode == '3')
		selectedImage = 3;
	else if(keyCode == '4')
		selectedImage = 4;
	else if(keyCode == '5')
		selectedImage = 5;
	else if(keyCode == '6')
		selectedImage = 6;
	else if(keyCode == '7')
		selectedImage = 7;
	else if(keyCode == 'p')
		shoot = true;
	else if (keyCode != -1)
		break;
	
	if(!cvGrabFrame(capture)){              // capture a frame 
		printf("Could not grab a frame\n\7");
		return;
	}
	
	img=cvRetrieveFrame(capture);           // retrieve the captured frame
	
	cvSetTrackbarPos( "Horiz.", "mainWin", bordeSup);
	cvSetTrackbarPos( "Bordes", "mainWin", edgeSlider);
	
}


//release video
cvReleaseCapture(&capture);
cvReleaseVideoWriter(&vw);
//liberamos imagenes

cvReleaseImage(&tempImage);
cvReleaseImage(&bordes);
cvReleaseImage(&shadowMask);
cvReleaseImage(&segmented);
cvReleaseImage(&hough);
cvReleaseImage(&mask);
cvReleaseImage(&shadows);

cvReleaseImage(&h);
cvReleaseImage(&l);
cvReleaseImage(&s);
cvReleaseImage(&hls);

cvReleaseImage(&traces);
cvReleaseImage(&tracesaux);
cvReleaseImage(&gray);
cvReleaseImage(&dst);
cvReleaseImage(&ed);

cvDestroyWindow("mainWin");

//liberamos elementos estructurales
cvReleaseStructuringElement(&seDisco);
cvReleaseStructuringElement(&seDisco2);

}
}
/************************************/


int main(int argc, char *argv[]){
	
	int op;
	char path[200] = "";
	
	while(1) {
		op = menu(path);
		if (op == 3) break;
		detectarCarretera(op,path);
	}

return 0;
}

