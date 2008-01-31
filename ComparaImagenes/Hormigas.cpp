#include "StdAfx.h"
#include "Hormigas.h"

Hormigas::Hormigas(char * ventana_name) {

	ventana = new char[strlen(ventana_name)];
	strcpy(ventana, ventana_name);

	hormigas = false;

	bordeSup = 20;
	edgeSlider = 61;
	searchAreas = 21;
	consigna = 0;	

	selectedImage = 1;
}
Hormigas::~Hormigas() {
	releaseHormigas();
}

void Hormigas::setHorizon(int val) {
	bordeSup = val;
	cvSetTrackbarPos("Horizon", "Control", bordeSup);
}

void Hormigas::setEdge(int val) {
	edgeSlider = val;
	cvSetTrackbarPos("Edges", "Control", edgeSlider);
}

void Hormigas::setAreas(int val) {
	searchAreas = val;
	cvSetTrackbarPos("Searching Areas", "Control", searchAreas);
}

void Hormigas::setSelected(int val) {
	selectedImage = val;
	cvSetTrackbarPos("Selected", "Control", selectedImage);
}

void Hormigas::initHormigas(CvSize size) {
	seDisco = cvCreateStructuringElementEx(5,5,1,1,CV_SHAPE_ELLIPSE);
	seDisco2= cvCreateStructuringElementEx(20,20,1,1,CV_SHAPE_ELLIPSE);

	//variables de la colonia
	attractionX = 160;
	attractionXAnt = -1;
	attractionY = 270;
	refLeftY = -1;
	refRightY = -1;
	aRef = -1;
	bRef = -1;
	cRef = -1;
	dRef = -1;
	startingAreasW = 10;
	selectedImage = 1;
	corte = 0;		

	//reservamos imagenes
	img = cvCreateImage(size,8,3); img->origin = 1;
	gray = cvCreateImage(size,8,1); gray->origin = 1;
	dst = cvCreateImage(size,8,1); dst->origin = 1;
	
	segmented = cvCreateImage(size,8,1); segmented->origin = 1;
	bordes = cvCreateImage(size,8,1); bordes->origin = 1;
	shadowMask = cvCreateImage(size,8,1); shadowMask->origin = 1;
	mask = cvCreateImage(size,8,3); mask->origin = 1;
	shadows = cvCreateImage(size,8,1); shadows->origin = 1;
	
	r = cvCreateImage(size,8,1); r->origin = 1;
	g = cvCreateImage(size,8,1); g->origin = 1;
	b = cvCreateImage(size,8,1); b->origin = 1;

	traces = cvCreateImage(size,8,3); traces->origin = 1;
	tracesaux = cvCreateImage(size,8,3); tracesaux->origin = 1;
	tempImage = cvCreateImage(size,8,1); tempImage->origin = 1;
	ed = cvCreateImage(size,8,1); ed->origin = 1;
	
	carretera = cvCreateImage(size, 8, 1); carretera->origin = 1;

	storage = cvCreateMemStorage(0);
	contour = 0;


	cvSet(shadowMask,cvScalar(1));
	consigna = 0;

	hormigas = true;
}

tMascara Hormigas::obtieneMascara(IplImage * imgIn) {
	
	if (! hormigas) {
		initHormigas(cvGetSize(imgIn));
	}		

	
	cvFlip(imgIn, img, 0);

	cvCvtColor(img,gray,CV_RGB2GRAY);
	cvCanny(gray,segmented,edgeSlider,edgeSlider+70);
		
	cvDilate(segmented,segmented,0,3);
	
	cvCopy(segmented,dst);	
	cvFindContours( dst, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );
	
	cvZero(bordes);
	for( ; contour != 0; contour = contour->h_next ){
		if (contour->total > 60) {				
			cvDrawContours( bordes, contour, cvScalar(255), cvScalar(255), -1, 1, 8 );
		}
    }
		
	cvDilate(bordes,bordes);
	
	attractionY = bordes->height-bordeSup+20;	
		
	initColony(bordes,traces,&colony,bordes->width,bordes->height-bordeSup,attractionX,attractionY,searchAreas,&refLeftY);
	acoMetaheuristic(bordes,traces,&colony,&shortestLeftPath,3);
	cvCopy(traces,tracesaux);
	initColony(bordes,traces,&colony,bordes->width,bordes->height-bordeSup,attractionX,attractionY,img->width-searchAreas,&refRightY);
	acoMetaheuristic(bordes,traces,&colony,&shortestRightPath,3);
	cvAdd(tracesaux,traces,tracesaux);
	cvAdd(img,tracesaux,tracesaux);
	cvSetZero(mask);

	CvPoint * puntos = NULL;

	setPointofAttraction(mask,shadowMask,&shortestLeftPath,&shortestRightPath,&attractionX,&aRef,&bRef,&cRef,&dRef,&corte,consigna, &puntos);		

	cvAdd(img, mask, mask);	

	if (attractionXAnt == -1) attractionXAnt = attractionX;

	cvCopyImage(bordes, carretera);
		
	switch(selectedImage) {
	case 1:
		cvShowImage(ventana, mask);
		break;
	case 2: 
		cvShowImage(ventana,tracesaux);
		break;
	case 3: 
		cvShowImage(ventana,bordes);
		break;
	case 4: 
		cvShowImage(ventana,segmented);
		break;
	}
	
	cvSetTrackbarPos( "Horizon", "Control", bordeSup);
	cvSetTrackbarPos( "Edges", "Control", edgeSlider);	

	tMascara retorno;
	retorno.a = cvPoint(puntos[0].x - 10, img->height - puntos[0].y);	
	retorno.b = cvPoint(puntos[1].x + 10, img->height - puntos[1].y);
	retorno.c = cvPoint(puntos[2].x, img->height - 1);
	retorno.d = cvPoint(puntos[3].x, img->height - 1);
//*/
	/*retorno.a = cvPoint(aRef, bordeSup);
	retorno.b = cvPoint(bRef, bordeSup);
	retorno.c = cvPoint(cRef, img->height - 1);
	retorno.d = cvPoint(dRef, img->height - 1);
//*/
	return retorno;
}
	
void Hormigas::releaseHormigas() {
	if (hormigas) {
		//liberamos imagenes	
		cvReleaseImage(&img);
		cvReleaseImage(&tempImage);
		cvReleaseImage(&bordes);
		cvReleaseImage(&shadowMask);
		cvReleaseImage(&segmented);		
		cvReleaseImage(&mask);
		cvReleaseImage(&shadows);

		cvReleaseImage(&r);
		cvReleaseImage(&g);
		cvReleaseImage(&b);

		cvReleaseImage(&traces);
		cvReleaseImage(&tracesaux);
		cvReleaseImage(&gray);
		cvReleaseImage(&dst);
		cvReleaseImage(&ed);

		cvReleaseImage(&carretera);

		cvDestroyWindow(ventana);

		//liberamos elementos estructurales
		cvReleaseStructuringElement(&seDisco);
		cvReleaseStructuringElement(&seDisco2);
	}
}

int Hormigas::getHorizonte() {
	return bordeSup;
}