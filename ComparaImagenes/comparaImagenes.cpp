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
	if (perspMask != NULL) cvReleaseImage(&perspMask);
	if (matrix != NULL) cvReleaseMat(&matrix);
	if (matrixAnt != NULL) cvReleaseMat(&matrixAnt);
	if (eigen != NULL) cvReleaseImage(&eigen);			
	if (temp != NULL) cvReleaseImage(&temp);	
	if (prev_pyramid != NULL) cvReleaseImage(&prev_pyramid);	
	if (pyramid != NULL) cvReleaseImage(&pyramid);
	if (mascaraCarretera != NULL) cvReleaseImage(&mascaraCarretera);
	if (hormigas != NULL) delete hormigas;
}

void comparaImagenes::init(CvSize size) {
	gris1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
	gris2 = cvCreateImage(size, IPL_DEPTH_8U, 1);
	persp = cvCreateImage(size, IPL_DEPTH_8U, 3);
	perspMask = cvCreateImage(size, IPL_DEPTH_8U, 1);
	eigen = cvCreateImage(size, IPL_DEPTH_32F, 1);
	temp = cvCreateImage(size, IPL_DEPTH_32F, 1);
	prev_pyramid = cvCreateImage(size, IPL_DEPTH_8U, 1);
	pyramid = cvCreateImage(size, IPL_DEPTH_8U, 1);
	matrix = cvCreateMat(3, 3, CV_32FC1);
	matrixAnt = cvCreateMat(3, 3, CV_32FC1);
	mascaraCarretera = cvCreateImage(size, IPL_DEPTH_8U, 1);
	hormigas = new Hormigas("Hormigas");
	ficheros = 0;	
}

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
	// Si el número de puntos es pequeño, usamos la matriz anterior
	if (cuenta < 10) {
		cvCopy(matrixAnt, matrix);
		cvWarpPerspective(img1, persp, matrix);
		return;
	} else {
		// Si hay puntos suficientes, podemos calcular la nueva matriz
		calculaCoeficientes();
	}
	cvWarpPerspective(img1, persp, matrix);

	// Si el número de 0s es grande, usamos la matriz anterior
	cvCvtColor(persp, perspMask, CV_BGR2GRAY);
	cvThreshold(perspMask, perspMask, 0, 255, CV_THRESH_BINARY);

	if (cvCountNonZero(perspMask) < (3 * img1->width * img1->height / 4)) {
		cvCopy(matrixAnt, matrix);
		cvWarpPerspective(img1, persp, matrix);
		return;
	}

	// Si la imagen se divide en 2 partes, usamos la matriz anterior
	cvErode(perspMask, gris1, 0, 5);

	CvMemStorage * store = cvCreateMemStorage (1000);
	CvSeq* contornos = 0;
	cvFindContours(gris1, store, &contornos, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);   					

	if ((contornos == 0) || (contornos->h_next != 0)) {
		cout << "Asignando la matriz anterior" << endl;
		cvCopy(matrixAnt, matrix);
		cvWarpPerspective(img1, persp, matrix);
	}

	cvReleaseMemStorage(&store);

	// Si hemos llegado hasta aquí, ya podemos copiar esta como última matriz
	cvCopy(matrix, matrixAnt);
}

/*void comparaImagenes::getCarretera(IplImage * img1) {
	pFuga = hormigas->obtieneMascara(img1);

	// Aplicamos la transformación a los puntos
	CvMat * punto1 = cvCreateMat(3, 1, CV_32FC1); 
	CvMat * punto2 = cvCreateMat(3, 1, CV_32FC1);
	CvMat * punto3 = cvCreateMat(3, 1, CV_32FC1);
	CvMat * punto4 = cvCreateMat(3, 1, CV_32FC1);
	punto1->data.fl[0] = (float)pFuga.a.x;
	punto1->data.fl[1] = (float)pFuga.a.y;
	punto1->data.fl[2] = 1;
	punto2->data.fl[0] = (float)pFuga.b.x;
	punto2->data.fl[1] = (float)pFuga.b.y;
	punto2->data.fl[2] = 1;
	punto3->data.fl[0] = (float)pFuga.c.x;
	punto3->data.fl[1] = (float)pFuga.c.y;
	punto3->data.fl[2] = 1;
	punto4->data.fl[0] = (float)pFuga.d.x;
	punto4->data.fl[1] = (float)pFuga.d.y;
	punto4->data.fl[2] = 1;
	cvMatMul(matrix, punto1, punto1);
	cvMatMul(matrix, punto2, punto2);
	cvMatMul(matrix, punto3, punto3);
	cvMatMul(matrix, punto4, punto4);
	
	pFuga.a = cvPoint(punto1->data.fl[0], punto1->data.fl[1]);
	pFuga.b = cvPoint(punto2->data.fl[0], punto2->data.fl[1]);
	pFuga.c = cvPoint(punto3->data.fl[0], punto3->data.fl[1]);
	pFuga.d = cvPoint(punto4->data.fl[0], punto4->data.fl[1]);

	pFuga.c.x += img1->height - pFuga.c.y;
	if (pFuga.c.x < img1->width) pFuga.c.x = img1->width;
	pFuga.c.y += img1->height - pFuga.c.y;
	pFuga.d.x -= img1->height - pFuga.d.y;
	if (pFuga.d.x > 0) pFuga.d.x = 0;
	pFuga.d.y += img1->height - pFuga.d.y;

	cout << "(" << pFuga.a.x << ", " << pFuga.a.y << ")" << endl;
	cout << "(" << pFuga.b.x << ", " << pFuga.b.y << ")" << endl;
	cout << "(" << pFuga.c.x << ", " << pFuga.c.y << ")" << endl;
	cout << "(" << pFuga.d.x << ", " << pFuga.d.y << ")" << endl;

	CvPoint poly[] = { pFuga.a, pFuga.b, pFuga.c, pFuga.d };
	//cvCvtColor(img1, mascaraCarretera, CV_BGR2GRAY);
	cvCopyImage(gris2, mascaraCarretera);
	//cvFlip(gris2, mascaraCarretera, 1);
	cvLine(mascaraCarretera, pFuga.a, pFuga.b, cvScalar(255));
	cvLine(mascaraCarretera, pFuga.b, pFuga.c, cvScalar(255));
	cvLine(mascaraCarretera, pFuga.c, pFuga.d, cvScalar(255));
	cvLine(mascaraCarretera, pFuga.d, pFuga.a, cvScalar(255));
	cvFillConvexPoly(mascaraCarretera, poly, 4, cvScalar(128));

	cvNamedWindow("Debug", 1);
	cvShowImage("Debug", mascaraCarretera);
}*/

void comparaImagenes::iguala3D(IplImage * img1, IplImage * img2) {
	IplImage * r1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage * g1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage * b1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);

	IplImage * r2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage * g2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage * b2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);

	cvSplit(img1, b1, g1, r1, NULL);
	cvSplit(img2, b2, g2, r2, NULL);

	iguala1D(r1, r2);
	iguala1D(g1, g2);
	iguala1D(b1, b2);
	
	cvMerge(b2, g2, r2, NULL, img2);

	cvReleaseImage(&r1);
	cvReleaseImage(&g1);
	cvReleaseImage(&b1);
	cvReleaseImage(&r2);
	cvReleaseImage(&g2);
	cvReleaseImage(&b2);
}

void comparaImagenes::iguala1D(IplImage * img1, IplImage * img2) {
	CvScalar media1, desv1;
	CvScalar media2, desv2;

	cvAvgSdv(img1, &media1, &desv1);
	cvAvgSdv(img2, &media2, &desv2);

	double brightness = ((media1.val[0] - media2.val[0]) * 100 / 128);
	double contrast = ((desv1.val[0] - desv2.val[0]) * 100 / 128);

	aplicaBrilloContraste(img2, img2, brightness, contrast);
}

void comparaImagenes::aplicaBrilloContraste(IplImage * img1, IplImage * img2, double brightness, double contrast) {
	uchar lut[256];
	CvMat* lut_mat = cvCreateMatHeader(1, 256, CV_8UC1);
    cvSetData(lut_mat, lut, 0);

	/*
     * The algorithm is by Werner D. Streidt
     * (http://visca.com/ffactory/archives/5-99/msg00021.html)
     */
    if( contrast > 0 )
    {
        double delta = 127.*contrast/100;
        double a = 255./(255. - delta*2);
        double b = a*(brightness - delta);
        for(int i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }
    else
    {
        double delta = -128.*contrast/100;
        double a = (256.-delta*2)/255.;
        double b = a*brightness + delta;
        for(int i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }

    cvLUT(img1, img2, lut_mat);
}

void comparaImagenes::preProcesado(IplImage * imagen) {
	// Hacemos un suavizado gaussiano
	//cvSmooth(imagen, imagen, CV_GAUSSIAN);
	
	// Hacemos una segmentación piramidal
	/*CvMemStorage * storPrep = cvCreateMemStorage (0);
	CvSeq *comp;
	cvPyrSegmentation(imagen, imagen, storPrep, &comp, 4, 255, 0);
	cvReleaseMemStorage(&storPrep);*/
}

void comparaImagenes::restaImagenes(IplImage * img, IplImage * resta) {
	//***************************************
	cvShowImage("Persp", persp);
	//***************************************

	// Resta en B/N:

	// Convertimos el color
	cvCvtColor(img, gris1, CV_BGR2GRAY);
	cvCvtColor(persp, gris2, CV_BGR2GRAY);

	// Hacemos la resta
	cvAbsDiff(gris1, gris2, gris1);	

	// Creamos la máscara
	cvThreshold(gris2, gris2, 0, 255, CV_THRESH_BINARY);

	// Aplicamos la máscara
	cvZero(resta);	
	cvCopy(gris1, resta, gris2);
	
	// Resta en color
	/*cvCvtColor(persp, gris1, CV_BGR2GRAY);
	cvThreshold(gris1, gris1, 0, 255, CV_THRESH_BINARY);	

	cvAbsDiff(img, persp, persp);

	cvZero(resta);	
	cvCopy(persp, resta, gris1);*/
}

/*void comparaImagenes::filtraImagen(IplImage * resta, IplImage * mask, IplImage * img2) {	
	IplImage * res = cvCreateImage(cvGetSize(resta), IPL_DEPTH_8U, 3);	
	cvZero(res);

	// Variables para la segmentación y la búsqueda de contornos
	CvMemStorage * storage = cvCreateMemStorage (0);
	CvSeq *contour;
	
	cvZero(gris2);
	cvCvtColor(resta, gris1, CV_BGR2GRAY);
	cvCopy(gris1, gris2, mask);	

	cvSobel(gris1, gris1, 2, 2, 5);
	cvDilate(gris1, gris1, 0, 1);
	cvThreshold(gris1, gris1, 70, 255, CV_THRESH_BINARY_INV);
	//cvDilate(gris1, gris1, 0, 1);
	cvErode(gris1, gris1, 0, 1);
	cvCvtColor(resta, gris2, CV_BGR2GRAY);
	cvAnd(mask, gris1, mask);
	cvZero(gris1);
	cvCopy(gris2, gris1, mask);
	cvShowImage("Debug", gris1);
	cvShowImage("Debug2", mask);

	// Y una vez tenemos la máscara, intentamos hacerla algo más uniforme
	// para detectar los obstáculos. Pero eso otro día

	//cvSaveImage("C:\\Archivos de programa\\OpenCV\\samples\\c\\ObstaculoAer.jpg", gris2);

	//cvSmooth(gris2, gris2, CV_MEDIAN);

	// Hacemos la segmentación (piramidal)
	cvPyrSegmentation(gris2, gris2, storage, &contour, 4, 57, 30);
	cvReleaseMemStorage(&storage);	

	/*cvDilate(gris2, gris1, 0, 2);
	cvThreshold(gris1, gris1, 10, 255, CV_THRESH_BINARY_INV);
	cvShowImage("Debug", gris1);*/

	/*// Umbralizamos para quitarnos del todo la carretera
	cvThreshold(gris2, gris2, 20, 0, CV_THRESH_TOZERO);	
	
	// Búsqueda de contornos
	storage = cvCreateMemStorage();
    contour = 0;	
	cvFindContours(gris2, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    	

	cvZero(gris2);
	if (contour != NULL && contour->total >= 1) {
		cout << "Total = " << contour->total << endl;
	}

	for( ; contour != 0; contour = contour->h_next) {
		//contour->storage->top->next
		if (abs(cvContourArea(contour)) > 100) {
			//cvDrawContours(res, contour, cvScalar(0, 0, 255), cvScalar(255, 0, 0), -1, -1, 8);
			
			/*CvSeq* poly = cvApproxPoly(contour, sizeof(CvContour), storage,
                    CV_POLY_APPROX_DP, cvContourPerimeter(contour)*0.05, 0);*/

			/*cvDrawContours(gris2, contour, cvScalar(255), cvScalar(0), -1, -1, 8);
		}
	}
	
	//cvErode(gris2, gris2);
	//cvDilate(gris2, gris2);
	cvCopy(img2, res, gris2);

	//cvShowImage("Debug", gris2);	
	//cvShowImage("Debug2", res);

	cvReleaseMemStorage(&storage);

	cvReleaseImage(&res);
}*/

void comparaImagenes::apaisaImagen(IplImage * img1, IplImage * img2) {
	IplImage * res = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
	cvCvtColor(img1, res, CV_GRAY2BGR);

	// NOTA: Estos cálculos pueden ser hechos offline
	// Hallamos los punto b y d
	CvPoint d = cvPoint(ORIGEN, img1->height - 1);
	CvPoint aux = cvPoint(img1->width / 2 + ORIGEN, HORIZONTE);

	double m = (double)(d.y - aux.y) / (double)(d.x - aux.x);
	double x = (-aux.y / m) + aux.x;

	CvPoint b = cvPoint(x, 0);
	b = aux;

	// Hallamos los puntos a y c
	CvPoint c = cvPoint(img1->width + ORIGEN - 1, img1->height-1);
	aux = cvPoint(img1->width / 2 + ORIGEN, HORIZONTE);

	m = (double)(c.y - aux.y) / (double)(c.x - aux.x);
	x = (-aux.y / m) + aux.x;

	CvPoint a = cvPoint(x, 0);
	a = aux;

	// Ahora calculamos los puntos b' y d'
	x = ((b.x - d.x) / 2.0) + d.x;
	cout << x << endl;
	CvPoint b1 = cvPoint(x, HORIZONTE);
	CvPoint d1 = cvPoint(x, img1->height - 1);

	// Ahora calculamos los puntos a' y c'
	x = ((c.x - a.x) / 2.0) + a.x;
	cout << x << endl;
	CvPoint a1 = cvPoint(x, HORIZONTE);
	CvPoint c1 = cvPoint(x, img1->height - 1);

	cvLine(res, cvPoint(0, HORIZONTE), cvPoint(res->width, HORIZONTE), cvScalar(255, 0, 0));
	cvLine(res, d1, b1, cvScalar(0, 255, 0));
	cvLine(res, a1, c1, cvScalar(0, 255, 0));
	cvLine(res, d, b, cvScalar(255, 255, 0));
	cvLine(res, a, c, cvScalar(255, 255, 0));
	cvCircle(res, a, 2, cvScalar(0, 255, 255), -1);

	// Calculamos la matriz de transformación
	CvPoint2D32f origen[] = { cvPointTo32f(a), cvPointTo32f(b), cvPointTo32f(c), cvPointTo32f(d) };	
	CvPoint2D32f destino[] = { cvPointTo32f(a1), cvPointTo32f(b1), cvPointTo32f(c1), cvPointTo32f(d1) };	
	
	double q[9];
	double q2[16];
	CvMat apaisaMat = cvMat(3, 3, CV_64F, q);
	CvMat apaisaMat2 = cvMat(4, 4, CV_64F, q2);	
	cvWarpPerspectiveQMatrix(origen, destino, &apaisaMat);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {						
			if ((i < 3) && (j < 3)) {
				cout << "(" << i << ", " << j << ")" << endl;
				cvmSet(&apaisaMat2, i, j, cvmGet(&apaisaMat, i, j));
			} else {
				cout << "[" << i << ", " << j << "]" << endl;
				cvmSet(&apaisaMat2, i, j, 0);
			}
		}
	}
	cvmSet(&apaisaMat2, 3, 3, 1);

	/*// Calculamos la matriz de transformación
	int cuenta = 4;
	int size = cuenta * 2;

	double * p = new double[size * 8];
	double * q = new double[size];
	double * r = new double[9];

	CvMat P, Q, R, *apaisaMat;
	P = cvMat(size, 8, CV_64FC1, p);
	Q = cvMat(size, 1, CV_64FC1, q);
	R = cvMat(8, 1, CV_64FC1, r);
	apaisaMat = cvCreateMat(3, 3, CV_32FC1);

	CvPoint punto1, punto2;
	for (int i = 0; i < cuenta; i++) {	
		switch(i){
			case 0: {
				punto1 = a;
				punto2 = a1;
				break;
			}
			case 1: {
				punto1 = b;
				punto2 = b1;
				break;
			}
			case 2: {
				punto1 = c;
				punto2 = c1;
				break;
			}
			case 3: {
				punto1 = d;
				punto2 = d1;
				break;
			}
		}
		p[i * 8] = p[(i + cuenta) * 8 + 3] = punto1.x;
        p[i * 8 + 1] = p[(i + cuenta) * 8 + 4] = punto1.y;
        p[i * 8 + 2] = p[(i + cuenta) * 8 + 5] = 1;
        p[i * 8 + 3] = p[i * 8 + 4] = p[i * 8 + 5] =
        p[(i + cuenta) * 8] = p[(i + cuenta) * 8 + 1] = p[(i + cuenta) * 8 + 2] = 0;
        p[i * 8 + 6] = -punto1.x * punto2.x;
        p[i * 8 + 7] = -punto1.y * punto2.x;
        p[(i + cuenta) * 8 + 6] = -punto1.x * punto2.y;
        p[(i + cuenta) * 8 + 7] = -punto1.y * punto2.y;
        q[i] = punto2.x;
        q[i+cuenta] = punto2.y;
	}
		
	cvSolve(&P, &Q, &R, CV_SVD);		
	
	r[8] = 1;

	R = cvMat(3, 3, CV_64FC1, r);
	cvConvert(&R, apaisaMat);*/

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << cvmGet(&apaisaMat2, i, j) << " ";
		}
		cout << endl;
	}

	// FIN de NOTA

	// Aplicamos la transformación
	cvWarpPerspective(res, res, &apaisaMat);

	cvReleaseImage(&res);
}

void comparaImagenes::difObstaculo(IplImage * img, IplImage * persp, IplImage * mask, CvScalar * media, CvScalar * desv) {
	IplImage * test1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage * test2 = cvCreateImage(cvGetSize(persp), IPL_DEPTH_8U, 3);
	cvZero(test1);
	cvZero(test2);

	cvCopy(img, test1, mask);
	cvCopy(persp, test2, mask);

	/*cvNamedWindow("Debug3", 1);
	cvShowImage("Debug3", test1);
	cvNamedWindow("Debug4", 1);
	cvShowImage("Debug4", test2);*/

	CvScalar media1, media2, desv1, desv2;
	cvAvgSdv(test1, &media1, &desv1);
	cvAvgSdv(test2, &media2, &desv2);

 	*media = cvScalar(abs(media1.val[0] - media2.val[0]), abs(media1.val[1] - media2.val[1]), abs(media1.val[2] - media2.val[2]));
	*desv = cvScalar(abs(desv1.val[0] - desv2.val[0]), abs(desv1.val[1] - desv2.val[1]), abs(desv1.val[2] - desv2.val[2]));

	cout << media1.val[0] << " - " << media2.val[0] << " = " << media->val[0] << endl;
	cout << media1.val[1] << " - " << media2.val[1] << " = " << media->val[1] << endl;
	cout << media1.val[2] << " - " << media2.val[2] << " = " << media->val[2] << endl;
	cout << desv1.val[0] << " - " << desv2.val[0] << " = " << desv->val[0] << endl;
	cout << desv1.val[1] << " - " << desv2.val[1] << " = " << desv->val[1] << endl;
	cout << desv1.val[2] << " - " << desv2.val[2] << " = " << desv->val[2] << endl;

	cvReleaseImage(&test1);
	cvReleaseImage(&test2);
}

void comparaImagenes::filtraImagen(IplImage * resta, IplImage * mask, IplImage * img2) {
	// Variables para la búsqueda de contornos
	CvMemStorage * storage = cvCreateMemStorage (0);
	CvSeq *contour;
	
	// Buscamos la máscara de lo que es carretera
	aplicaBrilloContraste(resta, gris1, 60, 60);	
	cvDilate(gris1, gris1, 0, 2);

	cvThreshold(gris1, gris1, 20, 255, CV_THRESH_BINARY_INV);
	
	cvZero(gris2);
	cvCopy(gris1, gris2, perspMask);
	cvZero(gris1);
	cvCopy(gris2, gris1, mask);		

	// Buscamos contornos
	storage = cvCreateMemStorage();
    contour = 0;	
	cvFindContours(gris1, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);    	

	if (contour == NULL || contour->total == 0) {
		cout << "contour == NULL" << endl;
		return;
	}

	CvPoint der, izq, top;
	double m1 = -1000, m2 = 1000;

	for( ; contour != 0; contour = contour->h_next) {				
		if (abs(cvContourArea(contour)) > 20000) {

			CvSeq* poly = cvApproxPoly(contour, sizeof(CvContour), storage,
                    CV_POLY_APPROX_DP, 15, 0);

			top = *(CvPoint*)cvGetSeqElem(poly, 0);
			int tope = -1;
			for (int i = 1; i < poly->total; i++) {
				CvPoint actual = *(CvPoint*)cvGetSeqElem(poly, i);
				if (actual.y < top.y) {
					top = actual;
					tope = i;
				}				
			}
	
			izq = der = top;
			m1 = -1000;
			m2 = 1000;			
			for (int i = 0; i < poly->total; i++) {
				if (i == tope) continue;
				CvPoint actual = *(CvPoint*)cvGetSeqElem(poly, i);
				double m = (double)(actual.y - top.y) / (double)(actual.x - top.x);				
				if ((m < -0.5) && (m > m1)) {
					m1 = m;
					izq = actual;
				}
				if ((m > 0.5) && (m < m2)) {
					m2 = m;
					der = actual;
				}				
			}			
		}
	}	

	// Si no los encontramos, usamos la máscara anterior
	if (m1 != -1000) {	
		// Creamos la máscara
		double x = (resta->height - 1 - top.y) / m1 + top.x;
		izq = cvPoint(x, resta->height - 1);

		x = (resta->height - 1 - top.y) / m2 + top.x;
		der = cvPoint(x, resta->height - 1);

		CvPoint puntos[] = { izq, top, der };
		cvZero(mascaraCarretera);
		cvFillConvexPoly(mascaraCarretera, puntos, 3, cvScalar(255));		
	}

	// Aplicamos la máscara recién creada
	cvZero(gris1);
	cvCopy(resta, gris1, mascaraCarretera);

	cvSmooth(gris1, gris1, CV_MEDIAN, 3, 7);
	
	aplicaBrilloContraste(gris1, gris1, 100, 100);	
	
	// Buscamos contornos: si el total encontrado es muy alto,
	// erosionamos y volvemos a buscar
	cvFindContours(gris1, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	IplImage * res = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);

	cvZero(res);


	cvZero(gris2);
	for( ; contour != 0; contour = contour->h_next) {				
		if (abs(cvContourArea(contour)) > 200) {										
			cvZero(gris1);
			cvDrawContours(gris1, contour, cvScalarAll(255), cvScalarAll(0), -1, -1);
			CvScalar media, desv;
			difObstaculo(img2, persp, gris1, &media, &desv);
			// Desechamos obstáculos falsos
			if (media.val[0] < 1 && media.val[1] < 1 && media.val[2] < 1 && 
				desv.val[0] < 4 && desv.val[1] < 4 && desv.val[2] < 4) {
				cvDrawContours(res, contour, cvScalar(0, 0, 255), cvScalarAll(0), -1, 3);
			} else {
				cvDrawContours(res, contour, cvScalar(0, 255, 0), cvScalarAll(0), -1, 3);
				cvDrawContours(gris2, contour, cvScalarAll(255), cvScalarAll(0), -1, -1);				
			}
		}
	}
	//cvDilate(gris2, gris2, 0, 2);
	
	cvCopy(img2, res, gris2);

	cvShowImage("Debug", img2);
	cvShowImage("Debug2", res);

	cvReleaseImage(&res);
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
	// Fase 1: Preprocesado
	//*************************************
	iguala3D(img1, img2);
	preProcesado(img1);
	preProcesado(img2);	

	//*************************************************************
	// Fase 2: Hacemos el flujo optico, limpiando los resultados
	//*************************************************************
	opticalFlow(img1, img2, mask);	

	//*******************************************************************
	// Fase 3: Obtenemos la máscara de la carretera con el algoritmo ACO
	//*******************************************************************
	//getCarretera(img1);

	//***************************************************************
	// Fase 4: Obtenemos la matriz de transformación y la aplicamos
	//***************************************************************
	aplicaPerspectiva(img1, img2);

	//***************************************************************
	// Fase 5: Calculamos la máscara y restamos las imágenes
	//***************************************************************
	restaImagenes(img2, resta);

	//*************************************************************************
	// Fase 6: Filtramos la imagen restada, eliminando información innecesaria
	//*************************************************************************
	filtraImagen(resta, mask, img2);

	/*char * fichero = new char[1024];
	sprintf(fichero, "C:\\Proyecto\\Datos\\Restas1\\Imagen%d.jpg", ficheros);	
	cvSaveImage(fichero, resta);

	ficheros++;*/

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