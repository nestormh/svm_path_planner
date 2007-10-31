#include "StdAfx.h"
#include "comparaImagenes.h"
#include<time.h>

comparaImagenes::comparaImagenes() {		
	//criterio = cvTermCriteria(CV_TERMCRIT_EPS, 0, 0.001);
	criterio = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);		       
}

comparaImagenes::~comparaImagenes() {	
	if (gris1 != NULL) cvReleaseImage(&gris1);	
	if (gris2 != NULL) cvReleaseImage(&gris2);
	if (persp != NULL) cvReleaseImage(&persp);
	if (matrix != NULL) cvReleaseMat(&matrix);
	if (matrix != NULL) cvReleaseMat(&matrixAnt);
	if (eigen != NULL) cvReleaseImage(&eigen);			
	if (temp != NULL) cvReleaseImage(&temp);	
	if (prev_pyramid != NULL) cvReleaseImage(&prev_pyramid);	
	if (pyramid != NULL) cvReleaseImage(&pyramid);
}

void comparaImagenes::init(CvSize size) {
	gris1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
	gris2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
	persp = cvCreateImage(size, IPL_DEPTH_8U, 3);
	eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
	temp = cvCreateImage(size, IPL_DEPTH_32F, 1);
	prev_pyramid = cvCreateImage(size, IPL_DEPTH_8U, 1);
	pyramid = cvCreateImage(size, IPL_DEPTH_8U, 1);
	matrix = cvCreateMat(3, 3, CV_32FC1);
	matrixAnt = cvCreateMat(3, 3, CV_32FC1);
}

/*void comparaImagenes::cleanOpticalFlow(bool * estado, float * error) {
	
	double * distancias = new double[cuenta];

	// Calculamos las distancias entre los puntos y quitamos aquellos con dist. > 20
	int contador = 0;
	for (int i = 0; i < cuenta; i++) {
		distancias[i] = sqrt(pow(esquinas1[i].x - esquinas2[i].x, 2.0f) + pow(esquinas1[i].y - esquinas2[i].y, 2.0f));

		if (distancias[i] > 40) {
			estado[i] = false;
		}
		if (estado[i] == true)
			contador++;				
	}

	CvPoint2D32f * esquinas3 = new CvPoint2D32f[contador];
	CvPoint2D32f * esquinas4 = new CvPoint2D32f[contador];
	
	for (int i = 0, j = 0; i < cuenta; i++) {
		if (estado[i] == true) {
			esquinas3[j] = esquinas1[i];
			esquinas4[j] = esquinas2[i];
			j++;
		}
	}

	delete distancias;
	delete esquinas1;
	delete esquinas2;

	esquinas1 = esquinas3;
	esquinas2 = esquinas4;
	
	cuenta = contador;
}*/

void comparaImagenes::cleanOpticalFlow(bool * estado, float * error) {
	
	double * distancias = new double[cuenta];
	double * angulos = new double[cuenta];

	// Calculamos las distancias y los ángulos entre los puntos de origen y destino
	for (int i = 0; i < cuenta; i++) {
		distancias[i] = sqrt(pow(esquinas1[i].x - esquinas2[i].x, 2.0f) + pow(esquinas1[i].y - esquinas2[i].y, 2.0f));
		
		double a = esquinas2[i].y - esquinas1[i].y;
		double d = sqrt(pow(esquinas2[i].x - esquinas1[i].x, 2.0f) + pow(esquinas2[i].y - esquinas1[i].y, 2.0f));

		if (d != 0) {
			angulos[i] = acos(a / d);
			if (a < 0)
				angulos[i] = 2 * CV_PI - angulos[i];
		} else {
			angulos[i] = angulos[i - 1];
		}		
	}

	CvMat dist = cvMat(cuenta, 1, CV_64FC1, distancias);
	CvMat ang = cvMat(cuenta, 1, CV_64FC1, angulos);

	CvScalar mediaD, desvD, mediaA, desvA;
	cvAvgSdv(&dist, &mediaD, &desvD);
	cvAvgSdv(&ang, &mediaA, &desvA);	

	// Continuar eliminando los que se salen de la media
	int contador = 0;
	for (int i = 0; i < cuenta; i++) {
		if (distancias[i] - mediaD.val[0] > 2 * desvD.val[0]) {
			estado[i] = false;
		}		
		if (angulos[i] - mediaA.val[0] > desvA.val[0]) {
			estado[i] = false;			
		}
		if (distancias[i] > 30) {
			estado[i] = false;
		}
		if (estado[i] == true)
			contador++;		
	}	

	CvPoint2D32f * esquinas3 = new CvPoint2D32f[contador];
	CvPoint2D32f * esquinas4 = new CvPoint2D32f[contador];
	
	for (int i = 0, j = 0; i < cuenta; i++) {
		if (estado[i] == true) {
			esquinas3[j] = esquinas1[i];
			esquinas4[j] = esquinas2[i];
			j++;
		}
	}

	delete distancias;
	delete angulos;
	delete esquinas1;
	delete esquinas2;

	esquinas1 = esquinas3;
	esquinas2 = esquinas4;
	
	cuenta = contador;
}

void comparaImagenes::opticalFlow(IplImage * img1, IplImage * img2, IplImage * mask) {
	// Inicializacion
  	cuenta = 500;
	esquinas1 = new CvPoint2D32f[cuenta];
	esquinas2 = new CvPoint2D32f[cuenta];	
	
	cvCvtColor(img1, gris1, CV_BGR2GRAY);
	cvCvtColor(img2, gris2, CV_BGR2GRAY);

	// Obtiene las esquinas iniciales
	cvGoodFeaturesToTrack(gris1, eigen, temp, esquinas1, &cuenta, 0.01, 10, mask, 3, 0, 0.04 );
	cvFindCornerSubPix(gris1, esquinas1, cuenta, cvSize(10, 10), cvSize(-1,-1), criterio);

	// Si no se encontraron las esquinas, se finaliza la ejecución de la iteración
	if (cuenta == 0)
		return;
			
	// Se crean las estructuras para obtener el flujo optico
	char * status = new char[cuenta];
	bool * estado = new bool[cuenta];
	float * error = new float[cuenta];
	
	// Flujo optico
	cvCalcOpticalFlowPyrLK(gris1, gris2, prev_pyramid, pyramid,
                esquinas1, esquinas2, cuenta, cvSize(10,10), 3, status, error, criterio, 0);

	// Se actualiza el vector de validos
	for (int i = 0; i < cuenta; i++) {
		if (status[i] == 1)
			estado[i] = true;
		else
			estado[i] = false;
	}

	delete status;

	// Se limpian los vectores
	cleanOpticalFlow(estado, error);

	delete estado;
	delete error;
}

void comparaImagenes::calculaCoeficientes() {	

	int size = cuenta * 2;

	double * a = new double[size * 8];
	double * b = new double[size];
	double * x = new double[9];

	CvMat A, B, X;
	A = cvMat(size, 8, CV_64FC1, a);
	B = cvMat(size, 1, CV_64FC1, b);
	X = cvMat(8, 1, CV_64FC1, x);	

	for (int i = 0; i < cuenta; i++) {				
		a[i * 8] = a[(i + cuenta) * 8 + 3] = esquinas1[i].x;
        a[i * 8 + 1] = a[(i + cuenta) * 8 + 4] = esquinas1[i].y;
        a[i * 8 + 2] = a[(i + cuenta) * 8 + 5] = 1;
        a[i * 8 + 3] = a[i * 8 + 4] = a[i * 8 + 5] =
        a[(i + cuenta) * 8] = a[(i + cuenta) * 8 + 1] = a[(i + cuenta) * 8 + 2] = 0;
        a[i * 8 + 6] = -esquinas1[i].x*esquinas2[i].x;
        a[i * 8 + 7] = -esquinas1[i].y*esquinas2[i].x;
        a[(i + cuenta) * 8 + 6] = -esquinas1[i].x*esquinas2[i].y;
        a[(i + cuenta) * 8 + 7] = -esquinas1[i].y*esquinas2[i].y;
        b[i] = esquinas2[i].x;
        b[i+cuenta] = esquinas2[i].y;
	}
		
	cvSolve(&A, &B, &X, CV_SVD);		
	
	x[8] = 1;

	X = cvMat(3, 3, CV_64FC1, x);
	cvConvert(&X, matrix);
}

void comparaImagenes::aplicaPerspectiva(IplImage * img1, IplImage * img2) {
	if (cuenta < 10) {
		cvCopy(matrixAnt, matrix);
	} else {
		calculaCoeficientes();
	}
	cvWarpPerspective(img1, persp, matrix);

	cvCvtColor(persp, gris1, CV_BGR2GRAY);	
	cvErode(gris1, gris1, 0, 5);

	CvMemStorage * store = cvCreateMemStorage (1000);
	CvSeq* contornos = 0;
	cvFindContours(gris1, store, &contornos, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);   					

	if ((contornos == 0) || (contornos->h_next != 0)) {
		cout << "Asignando la matriz anterior" << endl;
		cvCopy(matrixAnt, matrix);
		cvWarpPerspective(img1, persp, matrix);
	} else {
		cvCopy(matrix, matrixAnt);
	}

	cvReleaseMemStorage(&store);	
}

void comparaImagenes::preProcesado(IplImage * imagen) {
	// Hacemos un suavizado gaussiano
	cvSmooth(imagen, imagen, CV_GAUSSIAN);
	
	// Hacemos una segmentación piramidal
	CvMemStorage * storPrep = cvCreateMemStorage (0);
	CvSeq *comp;
	cvPyrSegmentation(imagen, imagen, storPrep, &comp, 4, 255, 0);
	cvReleaseMemStorage(&storPrep);
}

void comparaImagenes::restaImagenes(IplImage * img, IplImage * resta) {
	//***************************************
	cvShowImage("Persp", persp);
	//***************************************

	cvCvtColor(persp, gris1, CV_BGR2GRAY);
	cvThreshold(gris1, gris1, 0, 255, CV_THRESH_BINARY);
	cvSub(persp, img, persp);
	cvZero(resta);	
	cvCopy(persp, resta, gris1);	
}

void comparaImagenes::liberaMem() {
	delete esquinas1;
	delete esquinas2;
}

void comparaImagenes::compara(IplImage * img1, IplImage * img2, IplImage * mask, IplImage * resta) {	
	if (gris1 == NULL)
		init(cvGetSize(img1));

	clock_t tiempo = clock();	

	//*************************************
	// Fase 1: preprocesado
	//*************************************
	preProcesado(img1);
	preProcesado(img2);

	//*************************************************************
	// Fase 2: Hacemos el flujo optico, limpiando los resultados
	//*************************************************************
	opticalFlow(img1, img2, mask);	

	//***************************************************************
	// Fase 3: Obtenemos la matriz de transformación y la aplicamos
	//***************************************************************
	aplicaPerspectiva(img1, img2);

	//*******************************
	// Fase 4: Calculamos la máscara
	//*******************************	
	restaImagenes(img2, resta);

	// ...


	//***************************************************************
	// Mostramos (también se muestra persp en restaImagenes):
	IplImage * conMask = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);		

	cvSubS(img1, cvScalar(100, 100, 100), conMask);
	cvAddS(conMask, cvScalar(100, 100, 100), conMask, mask);	

	for (int i = 0; i < cuenta; i++) {		
		cvCircle(img1, cvPointFrom32f(esquinas1[i]), 3, cvScalar(0, 0, 255), 1);
		cvCircle(img2, cvPointFrom32f(esquinas1[i]), 3, cvScalar(0, 0, 255), 1);
		cvCircle(img2, cvPointFrom32f(esquinas2[i]), 3, cvScalar(255, 0, 0), 1);
		cvLine(img2, cvPointFrom32f(esquinas1[i]), cvPointFrom32f(esquinas2[i]), cvScalar(0, 255, 0)); 
	}		

	cvShowImage("Imagen1", img1);
	cvShowImage("Imagen2", img2);
	cvShowImage("Mascara", conMask);	
	cvShowImage("Resta", resta);

	cvReleaseImage(&conMask);	
	//***************************************************************

	//*********************************************************************
	// Fase ???: Liberamos memoria, dejando listo para siguiente iteración
	//*********************************************************************
	liberaMem();

	cout << "Tiempo: " << double(clock() - tiempo) / CLOCKS_PER_SEC << "segundos" << endl;
}