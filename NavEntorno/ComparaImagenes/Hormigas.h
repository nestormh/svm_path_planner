#pragma once

#include "aco.h"

struct tMascara {
	CvPoint a, b, c, d; 
};

class Hormigas
{
private:	
	int bordeSup;	
	int edgeSlider;
	int searchAreas;
	int consigna;		

	bool hormigas;

	IplImage* img;

	IplImage* ed;
	IplImage* gray;
	IplImage* dst;
	IplImage* mask;
	IplImage* shadows;

	IplImage* r;
	IplImage* g;
	IplImage* b;
	IplImage* h;
	IplImage* l;
	IplImage* s;
	IplImage* hls;
	
	IplImage* bordes;
	IplImage* shadowMask;
	IplImage* segmented;
	IplImage* traces;	
	IplImage* tracesaux;
	IplImage* tempImage;

	IplImage * carretera;

	IplConvKernel* seDisco;
	IplConvKernel* seDisco2;

	CvMemStorage* storage;
	CvSeq* contour;

	char * ventana;

	// Variables de la colonia
	colonyStruct colony;
	secuenceStruct shortestLeftPath;
	secuenceStruct shortestRightPath;
	int attractionX, attractionXAnt, attractionY;
	int refLeftY, refRightY;
	int aRef, bRef, cRef, dRef;
	int startingAreasW;
	int selectedImage;
	int corte;		

	void initHormigas(CvSize size);
	void releaseHormigas();
public:	
	Hormigas(char * ventana_name);
	~Hormigas();

	tMascara obtieneMascara(IplImage * img);

	// Setters	
	void setHorizon(int val);
	void setEdge(int val);
	void setAreas(int val);
	void setSelected(int val);

	// Getters
	int getHorizonte();
};

