#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "..\..\CapturaImagen\CapturaImagen\CapturaVLC.h"

#define MAXD 70				// Disparidad máxima
#define MIND 15				// Mínimo valor de disparidad a tener en cuenta para detectar obstáculos

typedef struct {			// Tipo de datos para indicar los parámetros de ajuste de los diferentes algoritmos
	int filtro,
		sobel,
		umbral,
		porcentaje,
		umbralObstaculos;
} parameter;


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void printImage(IplImage *image){
	int pixelSize;				// Tipo de dato del pixel en bytes
	char *data;
	double dato;
	int signo;


	pixelSize = (image->depth & 0x0000FFFF) / 8;
	signo = (image->depth & 0xF0000000);
	
	for (int i=image->height - 1; i >= 0; i--){
		printf("Fila: %d\n", i);
		for (int j=0; j < image->width; j++){
			data = (image->imageData + (i * image->widthStep + j * pixelSize));
			switch (image->depth) {
				case IPL_DEPTH_8U:{
						printf ("%hhu ", (unsigned char)*data);
						break;
				}
				case IPL_DEPTH_8S:{
						printf ("%hhd ", *((char*)data));
						break;
				}
				case IPL_DEPTH_16U:{
						printf ("%hu ", (unsigned short int)*data);
						break;
				}
				case IPL_DEPTH_16S:{
					printf ("%hd ", *((short int*)data));	
					break;
				}
				case IPL_DEPTH_32F:{
					printf ("%f ", *((float *)data));
					break;
				}
				case IPL_DEPTH_64F:{
					printf ("%f ", *((double *)data));
					break;
				}
			}
		}
		printf("\n\n");
	}
//	printf ("Signo: %d  pixelSize: %d\n", signo, pixelSize);
	getchar();
}



/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void preprocesadoB (IplImage *left, IplImage *right, int th){
	IplImage *mask,				// Máscara
			 *temp,				// Imagen ternarizada temporal
			 *aux;				
	CvSeq *contour = 0;
	CvMemStorage *storage = cvCreateMemStorage(0);

	cvCanny(left, left, 600, 500);
	cvCanny(right, right, 600, 500);
	cvSobel(left,left,1,0);
	cvSobel(right,right,1,0);
	//cvDilate(left, left, 0, 1);
	//cvDilate(right, right, 0, 1);
/*
	cvFindContours (left, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	cvZero(left);
	for (; contour != 0; contour = contour->h_next){
		if (contour->total > 20){
			cvDrawContours (left, contour, cvScalar(255), cvScalar(255), -1, 1, 8);
		}
	}
	cvFindContours (right, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	cvZero(right);
	for (; contour != 0; contour = contour->h_next){
		if (contour->total > 20){
			cvDrawContours (right, contour, cvScalar(255), cvScalar(255), -1, 1, 8);
		}
	}
	*/
	cvNamedWindow("Izquierda", CV_WINDOW_AUTOSIZE);// Filtrado de Sobel de bordes verticales
//	cvShowImage("Izquierda", left);
	cvNamedWindow("Derecha", CV_WINDOW_AUTOSIZE);// Filtrado de Sobel de bordes verticales
//	cvShowImage("Derecha", right);
	
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void preprocesado (IplImage *left, IplImage *right, int filterSize, int sobelSize, int th){
	IplImage *mask,				// Máscara
			 *temp,				// Imagen ternarizada temporal
			 *auxSobel,
			 *auxThreshold,
			 *aux;				
	IplImage *b;
	CvScalar mean;
	double threshold;

	cvSmooth(left, left, CV_BLUR, filterSize, filterSize);			// Filtrar para eliminar bordes superfluos
	cvSmooth(right, right, CV_BLUR, filterSize, filterSize);

	mask = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	temp = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	aux = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	auxSobel = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	auxThreshold = cvCreateImage(cvGetSize(left), IPL_DEPTH_32F, 1);
	b = cvCreateImage(cvGetSize(right), IPL_DEPTH_32F, 1);

	/* Ternarización de imagen izquierda*/
	cvNamedWindow("Preprocesado Izquierda", CV_WINDOW_AUTOSIZE);
			
	cvSobel(left, auxSobel, 1, 0, sobelSize);			// Filtrado de Sobel de bordes verticales
	
	cvSetZero(b);										// Cálculo automatizado del umbral
	cvConvertScale(auxSobel, auxThreshold, 1, 0);		// Pasar a punto flotante
	cvSquareAcc(auxThreshold, b);						// Elevar al cuadrado
	mean = cvAvg(b);									// Hallar la media
	threshold = sqrt(4 * (double)mean.val[0]);
	
	cvSet (temp, cvScalar(127));						// Inicializar imagen ternarizada

	cvSetZero (aux);
	cvCmpS(auxSobel,  - (threshold / th), mask, CV_CMP_LT);			// Construir máscara para valores por debajo del umbral
	cvCopy(aux, temp, mask);							// Aplicar máscara

	cvSet (aux, cvScalar(255));
	cvCmpS(auxSobel,  threshold / th, mask, CV_CMP_GT);			// Construir máscara para valores por encima del umbral
	cvCopy(aux, temp, mask);							// Aplicar máscara

	cvCopy(temp, left);
cvShowImage("Preprocesado Izquierda", left);

	/* Ternarización de imagen derecha*/
	cvSobel(right, auxSobel, 1, 0, 3);						// Filtrado de Sobel de bordes verticales
	
	cvSetZero(b);											// Cálculo automatizado del umbral
	cvConvertScale(auxSobel, auxThreshold, 1, 0);			// Pasar a punto flotante
	cvSquareAcc(auxThreshold, b);
	mean = cvAvg(b);
	threshold = sqrt(4 * (double)mean.val[0]);

	cvSet (temp, cvScalar(128));									// Inicializar imagen ternarizada

	cvSetZero (aux);
	cvCmpS(auxSobel, - (threshold / th), mask, CV_CMP_LT);			// Construir la máscara para valores por debajo del umbral
	cvCopy(aux, temp, mask);										// Aplicar máscara

	cvSet (aux, cvScalar(255));
	cvCmpS(auxSobel,  + (threshold / th), mask, CV_CMP_GT);			// Construir máscara para valores por encima del umbral
	cvCopy(aux, temp, mask);										// Aplicar máscara

	cvCopy(temp, right);

	cvReleaseImage (&auxSobel);								// Liberar memoria
	cvReleaseImage (&auxThreshold);
	cvReleaseImage (&b);
	cvReleaseImage (&mask);
	cvReleaseImage (&aux);
	cvReleaseImage (&temp);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void correlacion (IplImage *left, IplImage *right, int d, IplImage *mapa){
	int i;
	IplImage *corr,			// Auxiliar para la correlación
			 *auxU,			// Auxiliar sin signo
			 *auxS, 		// Auxiliar con signo
			 *auxL,			// Imagen izquierda con signo
 			 *auxR,			// Imagen derecha con signo
			 *min,
			 *mask;			
	CvMat *kernel;			// Kernel de convolución

	corr = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	auxU = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	auxS = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	min = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	mask = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);

	auxL = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	auxR = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	auxL->origin = 1;
	auxR->origin = 1;

	cvScale(left, auxL, 1, -127);
	cvScale(right, auxR, 1, -127);

	kernel = cvCreateMat(1, 9, CV_8UC1);						// Inicializar el kernel de convolución
	cvSet(kernel, cvScalar(1));
	
	for (i=d-1; i > 0 ; i--){
		cvResetImageROI(auxL);									
		cvAbs(auxL, auxS);							// La parte no solapada se deja con el original en valor absoluto

		cvSetImageROI(auxL,cvRect(i, 0, auxL->width - i, auxL->height));
		cvSetImageROI(auxR,cvRect(0, 0, (auxR->width - i), auxR->height));
	
		cvSetImageROI(auxS,cvRect(i, 0, auxS->width - i, auxS->height));

		cvAbsDiff(auxL, auxR, auxS);						// Diferencia en valor absoluto entre imágenes

		cvResetImageROI(auxS);		

		cvFilter2D(auxS, corr, kernel, cvPoint(-1, -1));	// Convolución (en los bordes rellena para cubrir el kernel)

/*Construir mapa de disparidad */	
		if (i != d-1) {										
			cvMin(corr, min, min);							// Actualizar la "imagen" de mínimos
			cvCmp(corr, min, mask, CV_CMP_EQ);				// Buscar los pixeles de la capa actual que represntan minimos
			cvSet (auxU, cvScalar(i));						// Construir imagen 
			cvCopy(auxU, mapa, mask);						// Poner al valor de la "capa" los pixeles con valor minimo 	
		} else {
			cvCopy(corr, min);								// Inicializar la imagen de mínimos
			cvSet (mapa, cvScalar(0));						// Inicializar el mapa de disparidad
		}
	}


	cvResetImageROI(left);
	cvResetImageROI(right);

	cvReleaseImage (&corr);									// Liberar memoria
	cvReleaseImage (&auxR);											
	cvReleaseImage (&auxL);											
	cvReleaseImage (&auxS);											
	cvReleaseImage (&auxU);
	cvReleaseImage (&mask);
	cvReleaseImage (&min);

	cvReleaseMat (&kernel);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void crearImagen (IplImage *mapa, IplImage *imagen){
	int	i, j, count;
	IplImage *mask;

	mask = cvCreateImage(cvGetSize(mapa), 8, 1);

	for (i = 0; i < mapa->height; i ++) {
		for (j = 0; j < MAXD; j++){
			cvSetImageROI(mapa, cvRect(0, i, mapa->width, 1));			// Recorrer fila a fila de la imagen
			cvSetImageROI(mask, cvRect(0, i, mapa->width, 1));
			cvCmpS(mapa, j, mask, CV_CMP_EQ);							// Ver cuantos pixeles tienen el valor de disparidad actual
			count = cvCountNonZero(mask);								// Acumularlos
			cvSet2D(imagen, i, j, cvScalar(count));						// Rellenar con la cuenta el pixel de la imagen de disparidad
		}
	}
	cvResetImageROI(mapa);	
	
	cvReleaseImage (&mask);						// Liberar memoria
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void crearImagenH (IplImage *mapa, IplImage *imagen){
	int	i, j, count;
	IplImage *mask;

	mask = cvCreateImage(cvGetSize(mapa), 8, 1);

	for (i = 0; i < mapa->width; i ++) {
		for (j = 0; j < MAXD; j++){
			cvSetImageROI(mapa, cvRect(i, 0, 1, mapa->height));			// Recorrer columna a columna de la imagen
			cvSetImageROI(mask, cvRect(i, 0, 1, mapa->height));
			cvCmpS(mapa, j, mask, CV_CMP_EQ);							// Ver cuantos pixeles tienen el valor de disparidad actual
			count = cvCountNonZero(mask);								// Acumularlos
			cvSet2D(imagen, j, i, cvScalar(count));						// Rellenar con la cuenta el pixel de la imagen de disparidad
		}
	}
	cvResetImageROI(mapa);	
	
	cvReleaseImage (&mask);						// Liberar memoria
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void lineas(IplImage *src){
    
    CvMemStorage *storage;
    CvSeq *lines;   // NO SE ESTÁ LIBERANDO, HAY QUE VER SI SE VA A DEVOLVER O QUÉ
    int i;
	IplConvKernel *se;
	float* line;
	float rho, theta;
	CvPoint pt1, pt2;
	double a, b, x0, y0;
		
	storage = cvCreateMemStorage(0);
	lines = 0;
	se = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_ELLIPSE);

//	cvDilate(src, src, se);

	cvSetImageROI(src, cvRect(10, 0, src->width - 30, src->height));
	lines = cvHoughLines2( src, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 100, 0, 0 );

    for( i = 0; i < MIN(lines->total,100); i++ ){
		line = (float*)cvGetSeqElem(lines,i);
        rho = line[0];
        theta = line[1];
        
        a = cos(theta), b = sin(theta);
        x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cvLine( src, pt1, pt2, cvScalar(255), 3, 8 );
     }


	cvResetImageROI(src);

	cvReleaseMemStorage (&storage);
	cvReleaseStructuringElement(&se);
 //  	cvShowImage("Hough", src);
	src->origin = 1;
  
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void obstaculos (IplImage *img, int th, int factor){
	int i,
		nMax,
		counts[MAXD],
		maximos[MAXD],
		greatest;
	IplImage *salida;
		
	salida = cvCreateImage(cvGetSize(img), 8, 1);
	salida->origin = 1;

	cvThreshold(img, img, th, 255, CV_THRESH_BINARY);			// Umbralizar
	
	cvSetZero(salida);
	
	greatest = 0;
	for (i = 0, nMax = 0; i < MAXD; i++){
		cvSetImageROI(img, cvRect(i, 0, 1, img->height));					// Seleccionar columna
		counts[i] = cvCountNonZero(img);									// Contar puntos
		
		if ((i > MIND) & (i < MAXD - 1)){									// Sólo buscar máximos entre MIND y MAXD -> Descartar objetos lejanos
			if ((counts[i]>= counts[i-1]) & (counts[i] > counts[i + 1]) & (i - 1 != maximos[nMax - 1])){
				maximos[nMax] = i;
				nMax++;
				if (counts[i] > greatest)
					greatest = counts[i];
			}
		}
		
		cvCircle(salida, cvPoint(i, counts[i]), 1, cvScalar(255));
		if (i > 0)
			cvLine(salida, cvPoint(i, counts[i-1]), cvPoint(i, counts[i]), cvScalar(255));
	}
	
	printf ("greatest = %d\n", greatest);
	for (i = 0; i < nMax; i++) {
		if (counts[maximos[i]] > greatest * factor / 100)	
			printf ("Posible obstaculo a %f m (disparidad %d)\n", (float)(0.545*425/maximos[i]), maximos[i]);
	}
	
	cvShowImage("Obstaculos", salida);

	cvResetImageROI(img);
	cvReleaseImage(&salida);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void checkFilter (int id){
	int aux;

	aux = cvGetTrackbarPos("Filtro", "Controles");
	
	if (aux%2 == 0){
		cvSetTrackbarPos("Filtro", "Controles", aux + 1);
	}
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void checkSobel (int id){
	int aux;

	aux = cvGetTrackbarPos("Sobel", "Controles");
	
	if (aux%2 == 0){
		cvSetTrackbarPos("Sobel", "Controles", aux + 1);
	}
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: rotate
	   FUNCIÓN:
	PARÁMETROS: IplImage *src --> Imagen a rotar
				IplImage *dst --> Imagen de destino (debe tener las medidas apropiadas)
				double degree --> Ángulo a rotar
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void rotate(const IplImage *src, IplImage *dst, double degree){
	double angle = degree * CV_PI / 180.;			// angle in radian
	double a = sin(angle), b = cos(angle);			// sine and cosine of angle
	int w_src = src->width,							// dimensions of src, dst and actual needed size
		h_src = src->height;
	int w_dst = dst->width, 
		h_dst = dst->height;
	int w_rot = 0, 
		h_rot = 0;									// actual needed size
	double scale_w = 0.,							// scale factor for rotation
		scale_h = 0., 
		scale = 0.;	
	double map[6];									// map matrix for WarpAffine, stored in array
	CvMat map_matrix = cvMat(2, 3, CV_64FC1, map);
	CvPoint2D32f pt = cvPoint2D32f(w_src / 2, h_src / 2);// Rotation center needed for cv2DRotationMatrix
  
	// Make w_rot and h_rot according to phase
	w_rot = (int)(h_src * fabs(a) + w_src * fabs(b));
	h_rot = (int)(w_src * fabs(a) + h_src * fabs(b));
	scale_w = (double)w_dst / (double)w_rot;
	scale_h = (double)h_dst / (double)h_rot;
	scale = MIN(scale_w, scale_h);
	cv2DRotationMatrix(pt, -degree, scale, &map_matrix);
	
	// Adjust rotation center to dst's center
	map[2] += (w_dst - w_src) / 2;
	map[5] += (h_dst - h_src) / 2;
	cvWarpAffine( src, dst, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: disparity
	   FUNCIÓN:
	PARÁMETROS: IplImage *left    -> Imagen izquierda. Se asume una imagen RGB de 3 planos.
				IplImage *right   -> Imagen derecha. Se asume una imagen RGB de 3 planos.
				parameter adjusts -> Ajustes para los algoritmos que componen la disparidad.
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void disparity (IplImage *left, IplImage* right, parameter adjusts){
	IplImage *izquierda,		// Imagen izquierda en escala de grises
			 *derecha,			// Imagen derecha en escala de grises
			 *mapaDisparidad,	// Mapa de disparidad
			 *imagenDisparidadH,
			 *imagenDisparidad;

	mapaDisparidad = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	imagenDisparidad = cvCreateImage(cvSize(MAXD,cvGetSize(left).height), IPL_DEPTH_8U, 1);

imagenDisparidadH = cvCreateImage(cvSize(cvGetSize(left).width, MAXD), IPL_DEPTH_8U, 1);

	izquierda = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	derecha = cvCreateImage(cvGetSize(right), IPL_DEPTH_8U, 1);

	mapaDisparidad->origin = 1;
	imagenDisparidad->origin = 1;
	
	cvCvtColor(left, izquierda, CV_RGB2GRAY);					// Pasar a escala de grises
	izquierda->origin = 1;

	cvCvtColor(right, derecha, CV_RGB2GRAY);					// Pasar a escala de grises
	derecha->origin = 1;

clock_t start = clock();
	preprocesado (izquierda, derecha, adjusts.filtro, adjusts.sobel, adjusts.umbral);
	correlacion (izquierda, derecha, MAXD, mapaDisparidad);
	crearImagen (mapaDisparidad, imagenDisparidad);

crearImagenH(mapaDisparidad, imagenDisparidadH);
cvThreshold(imagenDisparidadH, imagenDisparidadH, 10, 255, CV_THRESH_BINARY);			// Umbralizar

	obstaculos(imagenDisparidad, adjusts.umbralObstaculos, adjusts.porcentaje * 10);
clock_t stop = clock();

	printf("%.10lf\n", (double)(stop - start)/CLOCKS_PER_SEC);

		cvShowImage ("Mapa disparidad", mapaDisparidad);	
		cvShowImage ("Imagen disparidad", imagenDisparidad);
		cvShowImage ("Imagen disparidad H", imagenDisparidadH);
	
//		lineas(imagenDisparidad);

	cvReleaseImage(&mapaDisparidad);
	cvReleaseImage(&imagenDisparidad);
	cvReleaseImage(&izquierda);
	cvReleaseImage(&derecha);

	cvWaitKey(1);

}



/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int main (int argc, char* argv[]){
	IplImage *tempImage,		// Temporal para la conversión RGB a escala de grises
			 *izquierda,		// Imagen izquierda
			 *derecha;			// Imagen derecha

	LPWSTR *lista;
	int totalDisp = 0;
	CCapturaVLC captura;
	parameter ajustes;

	lista = captura.listaDispositivos(&totalDisp);
	
	printf("TotalDisp = %d\n", totalDisp);

	for (int i = 0; i < totalDisp; i++) {
		printf("%d: %S\n", i + 1, lista[i]);
	}

	izquierda = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
	derecha = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);

	cvNamedWindow("Izquierda", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Derecha", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Obstaculos", CV_WINDOW_AUTOSIZE);
//	cvNamedWindow("Hough", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Mapa disparidad", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Imagen disparidad", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Imagen disparidad H", CV_WINDOW_AUTOSIZE);

	/* Crear la ventana de controles */
	cvNamedWindow("Controles", CV_WINDOW_AUTOSIZE);
	ajustes.filtro = 3;
	ajustes.sobel = 3;
	ajustes.umbral = 4;
	ajustes.umbralObstaculos = 10;
	ajustes.porcentaje = 5;
	cvCreateTrackbar ("Filtro", "Controles", &ajustes.filtro, 21, checkFilter);
	cvCreateTrackbar ("Sobel", "Controles", &ajustes.sobel, 7, checkSobel);
	cvCreateTrackbar ("Umbral Ter", "Controles", &ajustes.umbral, 10, NULL);
	cvCreateTrackbar ("Umbral Obs", "Controles", &ajustes.umbralObstaculos, 15, NULL);
	cvCreateTrackbar ("Porcentaje", "Controles", &ajustes.porcentaje, 10, NULL);

	while(1) {
		
		izquierda = cvLoadImage("izquierda8.jpg");
		izquierda->origin = 1;

		derecha = cvLoadImage("derecha8.jpg");
		derecha->origin = 1;

		disparity (izquierda, derecha, ajustes);

		cvWaitKey(1);
	}
	return (0);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCIÓN:
	PARÁMETROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------
int main (int argc, char* argv[]){
	IplImage *tempImage,		// Temporal para la conversión RGB a escala de grises
			 *izquierda,		// Imagen izquierda
			 *derecha,			// Imagen derecha
			 *mapaDisparidad,	// Mapa de disparidad
			 *imagenDisparidad;

	LPWSTR *lista;
	int totalDisp = 0;
	int filtro,
		sobel,
		umbral,
		porcentaje,
		umbralObstaculos;
	CCapturaVLC captura;


	lista = captura.listaDispositivos(&totalDisp);
	
	printf("TotalDisp = %d\n", totalDisp);

	for (int i = 0; i < totalDisp; i++) {
		printf("%d: %S\n", i + 1, lista[i]);
	}

	izquierda = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 1);
	derecha = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 1);
	mapaDisparidad = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 1);
	imagenDisparidad = cvCreateImage(cvSize(MAXD,240), IPL_DEPTH_8U, 1);

	cvNamedWindow("Izquierda", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Derecha", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Obstaculos", CV_WINDOW_AUTOSIZE);
//	cvNamedWindow("Hough", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Mapa disparidad", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Imagen disparidad", CV_WINDOW_AUTOSIZE);

	/* Crear la ventana de controles 
	cvNamedWindow("Controles", CV_WINDOW_AUTOSIZE);
	filtro = 3;
	sobel = 3;
	umbral = 4;
	umbralObstaculos = 10;
	porcentaje = 5;
	cvCreateTrackbar ("Filtro", "Controles", &filtro, 21, checkFilter);
	cvCreateTrackbar ("Sobel", "Controles", &sobel, 7, checkSobel);
	cvCreateTrackbar ("Umbral Ter", "Controles", &umbral, 10, NULL);
	cvCreateTrackbar ("Umbral Obs", "Controles", &umbralObstaculos, 15, NULL);
	cvCreateTrackbar ("Porcentaje", "Controles", &porcentaje, 10, NULL);

	mapaDisparidad->origin = 1;
	imagenDisparidad->origin = 1;

	while(1) {
		
		izquierda = cvLoadImage("izquierda8.jpg", 0);
		//tempImage = captura.captura(lista[0]);					// Capturar imagen izquierda
		//cvCvtColor(tempImage, izquierda, CV_RGB2GRAY);
		//cvReleaseImage(&tempImage);
		izquierda->origin = 1;

		derecha = cvLoadImage("derecha8.jpg", 0);
		//tempImage = captura.captura(lista[1]);					// Capturar imagen derecha
		//cvCvtColor(tempImage, derecha, CV_RGB2GRAY);
		//cvReleaseImage(&tempImage);
		derecha->origin = 1;

/*
cvSet(izquierda, cvScalar(0));
cvRectangle(izquierda, cvPoint(50, 0), cvPoint(70, 20), cvScalar(254),CV_FILLED);
cvSet(derecha, cvScalar(0));
cvRectangle(derecha, cvPoint(20, 0), cvPoint(40, 20), cvScalar(254), CV_FILLED);

//		cvShowImage ("Izquierda", izquierda);	
//		cvShowImage ("Derecha", derecha);	

	clock_t start = clock();
		preprocesado (izquierda, derecha, filtro, sobel, umbral);
		correlacion (izquierda, derecha, MAXD, mapaDisparidad);

//tempImage = cvLoadImage("mapa.bmp");					// Cargar mapadisparidad calculado con matlab --> A partir de aqui funciona
//cvCvtColor(tempImage, mapaDisparidad, CV_RGB2GRAY);
		crearImagen (mapaDisparidad, imagenDisparidad);
		obstaculos(imagenDisparidad, umbralObstaculos, porcentaje * 10);
	clock_t stop = clock();

		printf("%.10lf\n", (double)(stop - start)/CLOCKS_PER_SEC);

//		cvShowImage ("Mapa disparidad", mapaDisparidad);	
		cvShowImage ("Imagen disparidad", imagenDisparidad);
	
//		lineas(imagenDisparidad);
	
//		cvReleaseImage(&izquierda);
//		cvReleaseImage(&derecha);

		cvWaitKey(1);
	}
	return (0);
}*/