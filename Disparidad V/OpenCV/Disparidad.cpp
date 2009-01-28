#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "Lineas.h"
//#include "..\..\CapturaImagen\CapturaImagen\CapturaVLC.h"

#define MAXD 70				// Disparidad m�xima
#define MIND 15				// M�nimo valor de disparidad a tener en cuenta para detectar obst�culos
#define MARGIN 2			// Tamaño del margen de búsqueda en torno al valor de disparidad
#define DISCARD 15			// Se descartan las disparidades de 0 a DISCARD por corresponderse con objetos lejanos
#define RECT 5				// Margen que se considera aceptable para que una línea sea recta (diferencia de coordenadas)
#define WINDOW 11			// Ancho de la ventana para considerar dos líneas paralelas como la misma

typedef struct {			// Tipo de datos para indicar los par�metros de ajuste de los diferentes algoritmos
	int filtro,
		sobel,
		umbral,
		porcentaje,
		umbralObstaculos;
} parameter;

int source = 0;


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void printImage(IplImage *image){
	int pixelSize;				// Tipo de dato del pixel en bytes
	char *data;
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
	   FUNCI�N: Adapta el brillo y contraste de la imagen2 a los valores de la imagen1. Primero calcula ambas 
				magnitudes para las dos im�genes y calcula la diferencia entre ellas. Esta diferencia se usa para 
				modificar el brillo y contraste de imagen2.
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void iguala1D(IplImage * img1, IplImage * img2) {
	uchar lut[256];
	CvMat* lut_mat = cvCreateMatHeader(1, 256, CV_8UC1);
    cvSetData(lut_mat, lut, 0);

	CvScalar media1, desv1;
	CvScalar media2, desv2;

	// Calcular brillo y contraste de ambas im�genes
	cvAvgSdv(img1, &media1, &desv1);
	cvAvgSdv(img2, &media2, &desv2);

	double brightness = ((media1.val[0] - media2.val[0]) * 100 / 128);
	double contrast = ((desv1.val[0] - desv2.val[0]) * 100 / 128);

    /*
     * The algorithm is by Werner D. Streidt
     * (http://visca.com/ffactory/archives/5-99/msg00021.html)
     */
    if( contrast > 0 ) {
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
    } else {
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

    cvLUT(img2, img2, lut_mat);

	cvReleaseMat (&lut_mat);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void preprocesado (IplImage *left, IplImage *right, int filterSize){

	cvSmooth(left, left, CV_BLUR, filterSize, filterSize);			// Filtrar para eliminar bordes superfluos
	cvSmooth(right, right, CV_BLUR, filterSize, filterSize);

	// Cuantizar la imagen descartando los bits menos significativos
	//cvAndS(left, cvScalar(224), left);
	//cvAndS(right, cvScalar(224), right);

	//cvDilate(left, left, NULL, 1);
	//cvDilate(right, right, NULL, 1);


	//cvNamedWindow("Preprocesado Izquierda", CV_WINDOW_AUTOSIZE);
	//cvShowImage("Preprocesado Izquierda", left);
	//cvNamedWindow("Preprocesado Derecha", CV_WINDOW_AUTOSIZE);
	//cvShowImage("Preprocesado Derecha", right);
	
	
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N: Ternariza una imagen realizando previamente un filtrado para eliminar el n�mero de bordes.
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void ternarizar (IplImage *left, IplImage *right, int filterSize, int sobelSize, int th){
	IplImage *mask,				// M�scara
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

	/* Ternarizaci�n de imagen izquierda*/
	cvNamedWindow("Preprocesado", CV_WINDOW_AUTOSIZE);
			
	cvSobel(left, auxSobel, 1, 1, sobelSize);			// Filtrado de Sobel de bordes verticales
	
	/* C�lculo automatizado del umbral */
	cvSetZero(b);										
	cvConvertScale(auxSobel, auxThreshold, 1, 0);		// Pasar a punto flotante
	cvSquareAcc(auxThreshold, b);						// Elevar al cuadrado
	mean = cvAvg(b);									// Hallar la media
	threshold = sqrt(4 * (double)mean.val[0]);
	
	cvSet (temp, cvScalar(127));						// Inicializar imagen ternarizada

	cvSetZero (aux);
	cvCmpS(auxSobel,  - (threshold / th), mask, CV_CMP_LT);			// Construir m�scara para valores por debajo del umbral
	cvCopy(aux, temp, mask);							// Aplicar m�scara

	cvSet (aux, cvScalar(255));
	cvCmpS(auxSobel,  threshold / th, mask, CV_CMP_GT);			// Construir m�scara para valores por encima del umbral
	cvCopy(aux, temp, mask);							// Aplicar m�scara

	cvCopy(temp, left);
cvShowImage("Preprocesado", left);


	/* Ternarizaci�n de imagen derecha*/
	cvSobel(right, auxSobel, 1, 1, 3);						// Filtrado de Sobel de bordes verticales
	
	/* C�lculo automatizado del umbral */
	cvSetZero(b);											
	cvConvertScale(auxSobel, auxThreshold, 1, 0);			// Pasar a punto flotante
	cvSquareAcc(auxThreshold, b);
	mean = cvAvg(b);
	threshold = sqrt(4 * (double)mean.val[0]);

	cvSet (temp, cvScalar(128));									// Inicializar imagen ternarizada

	cvSetZero (aux);
	cvCmpS(auxSobel, - (threshold / th), mask, CV_CMP_LT);			// Construir la m�scara para valores por debajo del umbral
	cvCopy(aux, temp, mask);										// Aplicar m�scara

	cvSet (aux, cvScalar(255));
	cvCmpS(auxSobel,  + (threshold / th), mask, CV_CMP_GT);			// Construir m�scara para valores por encima del umbral
	cvCopy(aux, temp, mask);										// Aplicar m�scara

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
	   FUNCI�N: Ternariza una imagen realizando previamente un filtrado para eliminar el n�mero de bordes.
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void ternarizacion (IplImage *img, int filterSize, int sobelSize, int th){
	IplImage *mask,				// M�scara
			 *temp,				// Imagen ternarizada temporal
			 *auxSobel,
			 *auxThreshold,
			 *aux;				
	IplImage *b;
	CvScalar mean;
	double threshold;

	cvSmooth(img, img, CV_BLUR, filterSize, filterSize);			// Filtrar para eliminar bordes superfluos
	
	mask = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	temp = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	aux = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	auxSobel = cvCreateImage(cvGetSize(img), IPL_DEPTH_16S, 1);
	auxThreshold = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
	b = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);

	cvNamedWindow("Preprocesado", CV_WINDOW_AUTOSIZE);
			
	cvSobel(img, auxSobel, 1, 0, sobelSize);			// Filtrado de Sobel de bordes verticales
	
	/* C�lculo automatizado del umbral */
	cvSetZero(b);										
	cvConvertScale(auxSobel, auxThreshold, 1, 0);		// Pasar a punto flotante
	cvSquareAcc(auxThreshold, b);						// Elevar al cuadrado
	mean = cvAvg(b);									// Hallar la media
	threshold = sqrt(4 * (double)mean.val[0]);
	
	cvSet (temp, cvScalar(127));						// Inicializar imagen ternarizada

	cvSetZero (aux);
	cvCmpS(auxSobel,  - (threshold / th), mask, CV_CMP_LT);			// Construir m�scara para valores por debajo del umbral
	cvCopy(aux, temp, mask);							// Aplicar m�scara

	cvSet (aux, cvScalar(255));
	cvCmpS(auxSobel,  threshold / th, mask, CV_CMP_GT);			// Construir m�scara para valores por encima del umbral
	cvCopy(aux, temp, mask);							// Aplicar m�scara

	cvCopy(temp, img);
cvShowImage("Preprocesado", img);

	cvReleaseImage (&auxSobel);								// Liberar memoria
	cvReleaseImage (&auxThreshold);
	cvReleaseImage (&b);
	cvReleaseImage (&mask);
	cvReleaseImage (&aux);
	cvReleaseImage (&temp);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void correlacion (IplImage *left, IplImage *right, int d, IplImage *mapa){
	int i;
	IplImage *corr,			// Auxiliar para la correlaci�n
			 *auxU,			// Auxiliar sin signo
			 *auxS, 		// Auxiliar con signo
			 *auxL,			// Imagen izquierda con signo
 			 *auxR,			// Imagen derecha con signo
			 *min,
			 *mask;			
	CvMat *kernel;			// Kernel de convoluci�n

	corr = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	auxU = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	auxS = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	min = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	mask = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);

	auxL = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
	auxR = cvCreateImage(cvGetSize(left), IPL_DEPTH_16S, 1);
		
	auxL->origin = left->origin;
	auxR->origin = right->origin;
	
	

	cvScale(left, auxL, 1, -127);
	cvScale(right, auxR, 1, -127);

	kernel = cvCreateMat(1, 9, CV_8UC1);						// Inicializar el kernel de convoluci�n
	cvSet(kernel, cvScalar(1));
	
	for (i=d-1; i > 0 ; i--){
		cvResetImageROI(auxL);									
		cvAbs(auxL, auxS);							// La parte no solapada se deja con el original en valor absoluto

		cvSetImageROI(auxL,cvRect(i, 0, auxL->width - i, auxL->height));
		cvSetImageROI(auxR,cvRect(0, 0, (auxR->width - i), auxR->height));
	
		cvSetImageROI(auxS,cvRect(i, 0, auxS->width - i, auxS->height));

		cvAbsDiff(auxL, auxR, auxS);						// Diferencia en valor absoluto entre im�genes

		cvResetImageROI(auxS);		

		cvFilter2D(auxS, corr, kernel, cvPoint(-1, -1));	// Convoluci�n (en los bordes rellena para cubrir el kernel)

/*Construir mapa de disparidad */	
		if (i != d-1) {										
			cvMin(corr, min, min);							// Actualizar la "imagen" de m�nimos
			cvCmp(corr, min, mask, CV_CMP_EQ);				// Buscar los pixeles de la capa actual que represntan minimos
			cvSet (auxU, cvScalar(i));						// Construir imagen 
			cvCopy(auxU, mapa, mask);						// Poner al valor de la "capa" los pixeles con valor minimo 	
		} else {
			cvCopy(corr, min);								// Inicializar la imagen de m�nimos
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
	   FUNCI�N:
	PAR�METROS:
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
	   FUNCI�N:
	PAR�METROS:
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
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void crearImagenH2 (IplImage *mapa, IplImage *imagen, int center){
	int	i, j, count;
	IplImage *mask;

	mask = cvCreateImage(cvGetSize(mapa), 8, 1);

	for (i = 0; i < mapa->width; i ++) {
		for (j = center - MARGIN; j < center + MARGIN; j++){
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
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
/*void lineas(IplImage *src){         
	IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    int i;

    cvCvtColor( src, color_dst, CV_GRAY2BGR );
#if 0
        lines = cvHoughLines2( src, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 50, 0, 0 );

        for( i = 0; i < lines->total; i++ )
        {
            float* line = (float*)cvGetSeqElem(lines,i);
            float rho = line[0];
            float theta = line[1];
            CvPoint pt1, pt2;
            double a = cos(theta), b = sin(theta);
            if( fabs(a) < 0.001 )
            {
                pt1.x = pt2.x = cvRound(rho);
                pt1.y = 0;
                pt2.y = color_dst->height;
            }
            else if( fabs(b) < 0.001 )
            {
                pt1.y = pt2.y = cvRound(rho);
                pt1.x = 0;
                pt2.x = color_dst->width;
            }
            else
            {
                pt1.x = 0;
                pt1.y = cvRound(rho/b);
                pt2.x = cvRound(rho/a);
                pt2.y = 0;
            }
            cvLine( color_dst, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
        }
#else
        lines = cvHoughLines2( src, storage, CV_HOUGH_PROBABILISTIC, 2, 10*CV_PI/180, 40, 30, 20 );
        
		// Habr�a que ir construyendo una estructura en la que devolver s�lo aquellas l�neas que resulten interesantes 
		// (aprovechar para separarlas en verticales, horizontales y oblicuas)
		for( i = 0; i < lines->total; i++ ) {
            CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
            if (line[0].y <= line[1].y)						// L�neas de pendiente positiva
				cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );
			else if (abs(line[0].x - line[1].x) < 5)		// L�neas semi-verticales
				cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );
			else if (abs(line[0].y - line[1].y) < 5)		// L�neas semi-horizontales
				cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );
        }
#endif
        cvShowImage( "Obstaculos", color_dst );

	cvReleaseImage (&color_dst);		
	cvReleaseMemStorage(&storage);

}*/

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N: Sort 2d points in right-to-left top-to-bottom order 
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/

static int cmp_func( const void* _a, const void* _b, void* userdata ) {
    CvPoint* a = (CvPoint*)_a;
    CvPoint* b = (CvPoint*)_b;
    int y_diff = a->y - b->y;
    int x_diff = b->x - a->x;
    return x_diff ? x_diff : y_diff;
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void lineasV(IplImage *src, CvSeq *vertical, CvSeq *diagonal){         
	IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
	CvPoint *line, 
		    *aux;
    int i, j,
		nLines;

	int ymax, ymin;

    cvCvtColor( src, color_dst, CV_GRAY2BGR );

	cvSetImageROI(src, cvRect(DISCARD, 0, src->width - DISCARD, src->height));	// Descartar la zona correspondiente a disparidades peque�as

	line = (CvPoint*) malloc (2 * sizeof (CvPoint));
	lines = cvHoughLines2( src, storage, CV_HOUGH_PROBABILISTIC, 2, 10*CV_PI/180, 20, 20, 30 ); //2 10*CV_PI/180 40 30 20
        	
	nLines = lines->total;
	for (i = 0; i < nLines; i++) {
	
		cvSeqPopFront(lines, line);						// Sacar el primer elemento de la lista
    
		line[0].x = line[0].x + DISCARD;						// Corregir para que quede en coordenadas de la imagen original
		line[1].x = line[1].x + DISCARD;
		
		j = 0;
  		if ((line[0].x == line[1].x)					// L�neas verticales
			|| (abs((line[0].y - line[1].y) / (line[0].x - line[1].x)) > RECT)){				// L�neas semi-verticales		
			do {										// Buscar el �ndice para inserci�n en orden
				aux = (CvPoint *) cvGetSeqElem(vertical, j);
				j++;
			}while ((j < vertical->total) && (line[0].x > aux[0].x));
				

			 if ((aux != NULL) && (abs(aux[0].x - line[0].x) < RECT)) {				// Si tiene una l�nea paralela muy cerca
				ymax = MAX(MAX (aux[0].y, aux[1].y),  MAX (line[0].y, line[1].y));	// Combinar las dos l�neas (longitud m�xima)
				ymin = MIN(MIN (aux[0].y, aux[1].y),  MIN (line[0].y, line[1].y));	// Se modifica la línea insertada anteriormente

				aux[0].y = ymin;
				aux[1].y = ymax;
				aux[0].x = MAX(MAX (aux[0].x, aux[1].x),  MAX (line[0].x, line[1].x));	// Elegir el valor de X m�s alto --> peor caso (obst�culo m�s cerca)
				aux[1].x = aux[0].x;
				cvLine( color_dst, aux[0], aux[1], CV_RGB(255,255,0), 1, 8 );
			} else {																// Si va al final o no tiene paralelas cercanas
				cvSeqInsert(vertical, j - 1, line);
				cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 1, 8 );
			}
			
		}/*else if (abs((line[0].y - line[1].y) / (line[0].x - line[1].x)) > 5){				// L�neas semi-verticales		
			cvSeqPush(vertical, line);			
			cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 1, 8 );
			//printf ("Semivertical (%d, %d)(%d, %d) m=%f\n", line[1].x, line[1].y, line[0].x, line[0].y, (float) abs((line[0].y - line[1].y) / (line[0].x - line[1].x)));
		} */else if (line[0].y <= line[1].y){					// L�neas de pendiente negativa
			cvSeqPush(diagonal, line);			
		}
	
	}
	//cvSeqSort(vertical, cmp_vert, 0);					// Ordenar las l�neas verticales
	
	color_dst->origin = src->origin;
	cvShowImage ("Imagen disparidad", color_dst);

	cvReleaseImage (&color_dst);
	cvClearSeq(lines);
	cvReleaseMemStorage(&storage);

	cvResetImageROI(src);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void lineasV2(IplImage *src, Lineas *vertical, CvSeq *diagonal){         
	IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
	CvPoint *line;
    int i, j,
		nLines;


    cvCvtColor( src, color_dst, CV_GRAY2BGR );

	cvSetImageROI(src, cvRect(DISCARD, 0, src->width - DISCARD, src->height));	// Descartar la zona correspondiente a disparidades peque�as

	line = (CvPoint*) malloc (2 * sizeof (CvPoint));
	lines = cvHoughLines2( src, storage, CV_HOUGH_PROBABILISTIC, 2, 10*CV_PI/180, 20, 20, 30 ); //2 10*CV_PI/180 40 30 20


//printf ("Secuencia desordenada\n");
//for( i = 0; i < lines->total; i++ )
//{
//    CvPoint* pt = (CvPoint*)cvGetSeqElem( lines, i );
//    printf( "(%d,%d)\n", pt->x, pt->y );
//}

	cvSeqSort(lines, cmp_func, 0 /* userdata is not used here */ );   	

//printf ("Secuencia ordenada\n");     
//for( i = 0; i < lines->total; i++ )
//{
//    CvPoint* pt = (CvPoint*)cvGetSeqElem( lines, i );
//    printf( "(%d,%d)\n", pt->x + DISCARD, pt->y + DISCARD);
//}
//     
     
        	
	nLines = lines->total;
	for (i = 0; i < nLines; i++) {
	
		cvSeqPopFront(lines, line);						// Sacar el primer elemento de la lista
    
		line[0].x = line[0].x + DISCARD;						// Corregir para que quede en coordenadas de la imagen original
		line[1].x = line[1].x + DISCARD;
		
		j = 0;
  		if ((line[0].x == line[1].x)					// L�neas verticales
			|| (abs((line[0].y - line[1].y) / (line[0].x - line[1].x)) > RECT)){				// L�neas semi-verticales		
	
			//vertical->InsertGreedy(line, line[0].x, 5, true);
			vertical->InsertGreedy(line, MAX(line[0].x, line[1].x), WINDOW);
			
//			cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 1, 8 );
			
		} else if (line[0].y <= line[1].y){					// L�neas de pendiente negativa
			cvSeqPush(diagonal, line);			
		}
	}
	
	vertical->DrawLines(color_dst, CV_RGB(255,0,0));
	vertical->Sort();
	
	color_dst->origin = src->origin;
	cvShowImage ("Imagen disparidad", color_dst);

	cvReleaseImage (&color_dst);
	cvClearSeq(lines);
	cvReleaseMemStorage(&storage);

	cvResetImageROI(src);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
static int cmp_hor( const void* _a, const void* _b, void* userdata ) {
    CvPoint* a = (CvPoint*)_a;
    CvPoint* b = (CvPoint*)_b;
 
    int y_diff = a[1].y - b[1].y;
    return y_diff;
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void lineasH(IplImage *src, CvSeq *horizontal, CvSeq *pendpos, CvSeq *pendneg){         
	IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
	CvPoint *line, 
			*aux;
    int i, j,
		nLines,
		xmax, 
		xmin;

    cvCvtColor( src, color_dst, CV_GRAY2BGR );

	cvSetImageROI(src, cvRect(0, DISCARD, src->width, src->height - DISCARD));	// Descartar la zona correspondiente a disparidades peque�as

	line = (CvPoint*) malloc (2 * sizeof (CvPoint));
	lines = cvHoughLines2( src, storage, CV_HOUGH_PROBABILISTIC, 2, CV_PI/180, 20, 30, 1 ); // 2 10*CV_PI/180 40 30 1
        	
	nLines = lines->total;
	for (i = 0; i < nLines; i++) {
		cvSeqPopFront(lines, line);						// Sacar el primer elemento de la lista

		line[0].y = line[0].y + DISCARD;						// Corregir para que quede en coordenadas de la imagen original
		line[1].y = line[1].y + DISCARD;

		j = 0;
		if ((line[0].y == line[1].y)					// L�neas horizontales
			|| (abs((float)(line[0].y - line[1].y) / (float)(line[0].x - line[1].x)) < 1)){				// L�neas semi-horizontales	(ESTABLECER UN UMBRAL ADECUADO)
			do {										// Buscar el �ndice para inserci�n en orden
				aux = (CvPoint *) cvGetSeqElem(horizontal, j);
				j++;
			}while ((j < horizontal->total) && (line[0].y > aux[0].y));
				

			 if ((aux != NULL) && (abs(aux[0].y - line[0].y) < RECT)) {				// Si tiene una l�nea paralela muy cerca
				xmax = MAX(MAX (aux[0].x, aux[1].x),  MAX (line[0].x, line[1].x));	// Combinar las dos l�neas (longitud m�xima)
				xmin = MIN(MIN (aux[0].x, aux[1].x),  MIN (line[0].x, line[1].x));	// Se modifica la que ya está en la lista

//				aux[0].x = xmin;
//				aux[1].x = xmax;
//				aux[0].y = MAX(MAX (aux[0].y, aux[1].y),  MAX (line[0].y, line[1].y));	// Elegir el valor de X m�s alto --> peor caso (obst�culo m�s cerca)
//				aux[1].y = aux[0].y;
				cvLine( color_dst, aux[0], aux[1], CV_RGB(255,255,0), 1, 8 );
			} else {																// Si va al final o no tiene paralelas cercanas
				cvSeqInsert(horizontal, j - 1, line);
				cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 1, 8 );
			}
		}/*else if (abs((float)(line[0].y - line[1].y) / (float)(line[0].x - line[1].x)) < 1){				// L�neas semi-horizontales	(ESTABLECER UN UMBRAL ADECUADO)	
			cvSeqPush(horizontal, line);			
			cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 1, 8 );
			printf ("Semihorizontal (%d, %d)(%d, %d) m=%f\n", line[1].x, line[1].y, line[0].x, line[0].y, (float) abs((float)(line[0].y - line[1].y) / (float)(line[0].x - line[1].x)));
		} */else if (line[0].y < line[1].y){					// L�neas de pendiente negativa
			cvSeqPush(pendpos, line);			
		} else {					// L�neas de pendiente positiva
			cvSeqPush(pendneg, line);			
		}
	}

	cvSeqSort(horizontal, cmp_hor, 0);					// Ordenar las l�neas horizontales
	cvShowImage ("Imagen disparidad H", color_dst);

	cvReleaseImage (&color_dst);		
	cvClearSeq(lines);
	cvReleaseMemStorage(&storage);

	cvResetImageROI(src);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void lineasH2(IplImage *src, Lineas *horizontal, CvSeq *pendpos, CvSeq *pendneg){         
	IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
	CvPoint *line; 
	int i,
	    nLines;
			
    cvCvtColor( src, color_dst, CV_GRAY2BGR );

	cvSetImageROI(src, cvRect(0, DISCARD, src->width, src->height - DISCARD));	// Descartar la zona correspondiente a disparidades peque�as

	line = (CvPoint*) malloc (2 * sizeof (CvPoint));
	lines = cvHoughLines2( src, storage, CV_HOUGH_PROBABILISTIC, 2, CV_PI/180, 20, 30, 1 ); // 2 10*CV_PI/180 40 30 1
        	
	nLines = lines->total;
	for (i = 0; i < nLines; i++) {
		cvSeqPopFront(lines, line);						// Sacar el primer elemento de la lista

		line[0].y = line[0].y + DISCARD;						// Corregir para que quede en coordenadas de la imagen original
		line[1].y = line[1].y + DISCARD;
		
		if ((line[0].y == line[1].y)					// L�neas horizontales
			|| (abs((float)(line[0].y - line[1].y) / (float)(line[0].x - line[1].x)) < 1)){				// L�neas semi-horizontales	(ESTABLECER UN UMBRAL ADECUADO)
				
			horizontal->Insert(line, MAX(line[0].y, line[1].y), WINDOW);
//			cvLine( color_dst, line[0], line[1], CV_RGB(255,255,0), 1, 8 );
			
		} else if (line[0].y < line[1].y){					// L�neas de pendiente negativa
			cvSeqPush(pendpos, line);			
		} else {					// L�neas de pendiente positiva
			cvSeqPush(pendneg, line);			
		}
	}
	horizontal->DrawLines(color_dst, CV_RGB(255,255,0));
	horizontal->Sort();

	cvShowImage ("Imagen disparidad H", color_dst);

	cvReleaseImage (&color_dst);		
	cvClearSeq(lines);
	cvReleaseMemStorage(&storage);

	cvResetImageROI(src);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
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
	salida->origin = img->origin;

	cvThreshold(img, img, th, 255, CV_THRESH_BINARY);			// Umbralizar
	
	cvSetZero(salida);
	
	greatest = 0;
	for (i = 0, nMax = 0; i < MAXD; i++){
		cvSetImageROI(img, cvRect(i, 0, 1, img->height));					// Seleccionar columna
		counts[i] = cvCountNonZero(img);									// Contar puntos
		
		if ((i > MIND) & (i < MAXD - 1)){									// S�lo buscar m�ximos entre MIND y MAXD -> Descartar objetos lejanos
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
	
	cvShowImage("Lineas", salida);

	cvResetImageROI(img);
	cvReleaseImage(&salida);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
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
	   FUNCI�N:
	PAR�METROS:
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
	   FUNCI�N:
	PAR�METROS: IplImage *src --> Imagen a rotar
				IplImage *dst --> Imagen de destino (debe tener las medidas apropiadas)
				double degree --> �ngulo a rotar
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
		NOMBRE: 
	   FUNCI�N:
	PAR�METROS: 
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void marcObstacle (IplImage *sourceImage, CvSeq *vertical, CvSeq *horizontal ,CvSeq *diagonal ,CvSeq *pendpos, CvSeq *pendneg){
	IplImage* color_dst;
	int j, nVer;
	CvPoint *hor;
	CvPoint *ver;
	char auxText[255];

	CvFont font;
	double hScale,
		   vScale;
	int lineWidth; 
	CvScalar rectColor;


	/* Configurar la fuente para el texto en im�genes */
	hScale = 0.5;
	vScale = 0.5;
	lineWidth = 0;
	cvInitFont (&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, lineWidth);

	// Para mostrar el rect�ngulo
	color_dst = cvCloneImage (sourceImage);
	color_dst->origin = sourceImage->origin;

	for (int i= 0; i < horizontal->total; i ++){		// Dibujar rect�ngulos (detectar obst�culos)
		hor = (CvPoint *) cvGetSeqElem(horizontal, i);
	
		j = 0;
		nVer = vertical->total;
		while (/*(ver[0].x < hor[0].y) &&*/(j < nVer)) {
			ver = (CvPoint *) cvGetSeqElem(vertical, j);
	
			if ((abs(ver[0].x - hor[0].y) < 5) || (abs(ver[0].x - hor[1].y) < 5) ||
				(abs(ver[1].x - hor[0].y) < 5) || (abs(ver[1].x - hor[1].y) < 5)){   // Coincidencia
				if (ver[0].x > 46)
					rectColor = CV_RGB(255,0,0);
				else if (ver[0].x > 23)
					rectColor = CV_RGB(255,255,0);
				else
					rectColor = CV_RGB(0,255,0);

				cvRectangle(color_dst, cvPoint(MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y)), cvPoint(MAX(hor[0].x, hor[1].x), MAX(ver[0].y, ver[1].y)), rectColor);

				sprintf(auxText, "%.2f m",(float)(0.545*425/ver[0].x));
				cvPutText (color_dst, auxText, cvPoint(MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y) - 1), &font, rectColor);

				cvShowImage( "Obstaculos", color_dst );
				printf ("Posible obstaculo a %f m (disparidad %d)\n", (float)(0.545*425/ver[0].x), ver[0].x);
			
			} 

			j++;
		}
	}



	cvShowImage( "Obstaculos", color_dst );


	cvReleaseImage(&color_dst);
}
/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: 
	   FUNCI�N:
	PAR�METROS: 
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void marcObstacle2 (IplImage *sourceImage, Lineas *vertical, Lineas *horizontal){
	IplImage* color_dst;
	int nVer,
	    *index,
	    i, j;
	
	CvFont font;
	double hScale,
		   vScale;
	int lineWidth; 
	CvScalar rectColor;
	char auxText[255];

	CvSeq *hLines;
	CvPoint *hor, 
		    *ver;	
	
	printf ("Verticales:\n");
	vertical->Print();
	printf ("Horizontales:\n");
	horizontal->Print();
	
	/* Configurar la fuente para el texto en im�genes */
	hScale = 0.5;
	vScale = 0.5;
	lineWidth = 0;
	cvInitFont (&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, lineWidth);

	// Para mostrar el rect�ngulo
	color_dst = cvCloneImage (sourceImage);
	color_dst->origin = sourceImage->origin;

	nVer = vertical->GetN();
	index = vertical->GetIndex();
	
	for (i= 0; i < nVer; i ++){		// Dibujar rect�ngulos (detectar obst�culos)
		ver = (CvPoint *) cvGetSeqElem(vertical->GetLine(index[i]), 0);
		//printf ("Vertical: (%d %d) (%d %d)\n", ver[0].x, ver[0].y, ver[1].x, ver[1].y);
		
		hLines = horizontal->GetLine(index[i]);

		j = 0;
		while (j < hLines->total) {
			hor = (CvPoint *) cvGetSeqElem(hLines, j);
	
			//printf ("Horizontal: (%d %d) (%d %d)\n", hor[0].x, hor[0].y, hor[1].x, hor[1].y);

			if (ver[0].x > 46)
				rectColor = CV_RGB(255,0,0);
			else if (ver[0].x > 23)
				rectColor = CV_RGB(255,255,0);
			else
				rectColor = CV_RGB(0,255,0);

			cvRectangle(color_dst, cvPoint(MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y)), cvPoint(MAX(hor[0].x, hor[1].x), MAX(ver[0].y, ver[1].y)), rectColor);
//printf("Rectángulo: %d %d %d %d\n", MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y), MAX(hor[0].x, hor[1].x), MAX(ver[0].y, ver[1].y));


			sprintf(auxText, "%.2f m",(float)(0.545*425/ver[0].x));
			cvPutText (color_dst, auxText, cvPoint(MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y) - 1), &font, rectColor);

			cvShowImage( "Obstaculos", color_dst );
			printf ("Posible obstaculo a %f m (disparidad %d)\n", (float)(0.545*425/ver[0].x), ver[0].x);
			
			j++;
		}
		
			
	}

	cvShowImage( "Obstaculos", color_dst );

	cvReleaseImage(&color_dst);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: 
	   FUNCI�N:
	PAR�METROS: 
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void marcObstacle2 (IplImage *sourceImage, Lineas *vertical, Lineas *horizontal, int ventana){
	IplImage* color_dst;
	int nVer,
	    *index,
	    i, j, k;
	
	CvFont font;
	double hScale,
		   vScale;
	int lineWidth, lado; 
	CvScalar rectColor;
	char auxText[255];

	CvSeq *hLines;
	CvPoint *hor, 
		    *ver;	
	
	printf ("Verticales:\n");
	vertical->Print();
	printf ("Horizontales:\n");
	horizontal->Print();
	
	lado = round(ventana / 2);
		
	/* Configurar la fuente para el texto en im�genes */
	hScale = 0.5;
	vScale = 0.5;
	lineWidth = 0;
	cvInitFont (&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, lineWidth);

	// Para mostrar el rect�ngulo
	color_dst = cvCloneImage (sourceImage);
	color_dst->origin = sourceImage->origin;

	nVer = vertical->GetN();
	index = vertical->GetIndex();
	
	for (i= 0; i < nVer; i ++){		// Dibujar rect�ngulos (detectar obst�culos)
		ver = (CvPoint *) cvGetSeqElem(vertical->GetLine(index[i]), 0);
		//printf ("Vertical: (%d %d) (%d %d)\n", ver[0].x, ver[0].y, ver[1].x, ver[1].y);
		
		k = index[i] + lado;
		
		do {
			hLines = horizontal->GetLine(k);
			k--;
		} while ((hLines->total == 0) && (k >= 0) && (k >= index[i] - lado)); 

		j = 0;
		while (j < hLines->total) {
			hor = (CvPoint *) cvGetSeqElem(hLines, j);
	
			//printf ("Horizontal: (%d %d) (%d %d)\n", hor[0].x, hor[0].y, hor[1].x, hor[1].y);

			if (ver[0].x > 46)
				rectColor = CV_RGB(255,0,0);
			else if (ver[0].x > 23)
				rectColor = CV_RGB(255,255,0);
			else
				rectColor = CV_RGB(0,255,0);

			cvRectangle(color_dst, cvPoint(MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y)), cvPoint(MAX(hor[0].x, hor[1].x), MAX(ver[0].y, ver[1].y)), rectColor);
//printf("Rectángulo: %d %d %d %d\n", MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y), MAX(hor[0].x, hor[1].x), MAX(ver[0].y, ver[1].y));


			sprintf(auxText, "%.2f m",(float)(0.545*425/ver[0].x));
			cvPutText (color_dst, auxText, cvPoint(MIN(hor[0].x, hor[1].x), MIN(ver[0].y, ver[1].y) - 1), &font, rectColor);

			cvShowImage( "Obstaculos", color_dst );
			printf ("Posible obstaculo a %f m (disparidad %d)\n", (float)(0.545*425/ver[0].x), ver[0].x);
			
			j++;
		}
		
			
	}

	cvShowImage( "Obstaculos", color_dst );

	cvReleaseImage(&color_dst);
}


/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: 
	   FUNCI�N:
	PAR�METROS: 
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void crearRDM (IplImage *rdm, IplImage *dm, CvPoint *line){
		
		
		cvSetImageROI(dm, cvRect(line[0].x, MIN(line[0].y, line[1].y), dm->width - line[0].x, abs(line[0].y - line[1].y)));
		cvSetZero(rdm);
		cvSetImageROI(rdm, cvRect(line[0].x, MIN(line[0].y, line[1].y), dm->width - line[0].x, abs(line[0].y - line[1].y)));
		cvCmpS(dm, line[0].x, rdm, CV_CMP_EQ); 
		cvSet (rdm, cvScalar(line[0].x), rdm);
		cvResetImageROI (dm);
		cvResetImageROI (rdm);
		
		//cvWaitKey(1);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: 
	   FUNCI�N:
	PAR�METROS: 
	  DEVUELVE: void
-----------------------------------------------------------------------------------------------------------------*/
void crearRDM (IplImage *rdm, IplImage *dm, CvPoint *line, int ventana){
	IplImage *auxGE, 
			 *auxLE;
		
	auxGE = cvCreateImage(cvGetSize(dm), IPL_DEPTH_8U, 1);		
	auxLE = cvCreateImage(cvGetSize(dm), IPL_DEPTH_8U, 1);
		
	cvSetImageROI(dm, cvRect(line[0].x, MIN(line[0].y, line[1].y), dm->width - line[0].x, abs(line[0].y - line[1].y)));
	
	cvSetZero(rdm);
	cvSetZero(auxGE);
	cvSetZero(auxLE);
	
	cvSetImageROI(rdm, cvRect(line[0].x, MIN(line[0].y, line[1].y), dm->width - line[0].x, abs(line[0].y - line[1].y)));
	cvSetImageROI(auxGE, cvRect(line[0].x, MIN(line[0].y, line[1].y), dm->width - line[0].x, abs(line[0].y - line[1].y)));
	cvSetImageROI(auxLE, cvRect(line[0].x, MIN(line[0].y, line[1].y), dm->width - line[0].x, abs(line[0].y - line[1].y)));

		
	cvCmpS(auxLE, line[0].x - round(ventana/2), rdm, CV_CMP_LE); 
	cvCmpS(auxGE, line[0].x + round(ventana/2), rdm, CV_CMP_GE);
	cvAnd(rdm, auxGE, auxLE);
	
//	cvSet (rdm, cvScalar(line[0].x), rdm);
	cvResetImageROI (dm);
	cvResetImageROI (rdm);
	
	cvReleaseImage(&auxGE);
	cvReleaseImage(&auxLE);
}

/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE: disparity
	   FUNCI�N:
	PAR�METROS: IplImage *left    -> Imagen izquierda. Se asume una imagen RGB de 3 planos.
				IplImage *right   -> Imagen derecha. Se asume una imagen RGB de 3 planos.
				parameter adjusts -> Ajustes para los algoritmos que componen la disparidad.
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
void disparity (IplImage *left, IplImage* right, parameter adjusts){
	IplImage *izquierda,		// Imagen izquierda en escala de grises
			 *derecha,			// Imagen derecha en escala de grises
			 *mapaDisparidad,	// Mapa de disparidad
			 *imagenDisparidadH,
			 *imagenDisparidad,
			 *mapaDisparidadReducido;

	CvPoint *ver;

	CvSeq *diagonal,
		  *pendpos,
		  *pendneg; 

	Lineas *vLines,
		   *hLines;
	int nVer, 
		*index, 
		i;
	
	hLines = new Lineas(MAXD);
	vLines = new Lineas(MAXD);

	CvMemStorage *storageD,
				 *storageMP,
				 *storageMN;
	
	mapaDisparidad = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	imagenDisparidad = cvCreateImage(cvSize(MAXD,cvGetSize(left).height), IPL_DEPTH_8U, 1);
	imagenDisparidadH = cvCreateImage(cvSize(cvGetSize(left).width, MAXD), IPL_DEPTH_8U, 1);

	mapaDisparidadReducido = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);

	izquierda = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 1);
	derecha = cvCreateImage(cvGetSize(right), IPL_DEPTH_8U, 1);

	cvCvtColor(left, izquierda, CV_RGB2GRAY);					// Pasar a escala de grises
	cvCvtColor(right, derecha, CV_RGB2GRAY);					// Pasar a escala de grises

	mapaDisparidad->origin = left->origin;
	imagenDisparidad->origin = left->origin;
	izquierda->origin = left->origin;
	derecha->origin = right->origin;	

cvShowImage ("Izquierda", izquierda);	
cvShowImage ("Derecha", derecha);	

	
clock_t start = clock();

	iguala1D(derecha, izquierda);						// Igualar brillo y contraste de ambas im�genes

//Opci�n 1 de 
	ternarizacion (izquierda, adjusts.filtro, adjusts.sobel, adjusts.umbral);
	ternarizacion (derecha, adjusts.filtro, adjusts.sobel, adjusts.umbral);

// Opci�n 2 de preprocesado
//preprocesado (izquierda, derecha, adjusts.filtro);

	correlacion (izquierda, derecha, MAXD, mapaDisparidad);
	
	/* Disparidad V */
	crearImagen (mapaDisparidad, imagenDisparidad);
	cvThreshold(imagenDisparidad, imagenDisparidad, 10, 255, CV_THRESH_BINARY);			// Umbralizar

	/* Disparidad U */
	crearImagenH(mapaDisparidad, imagenDisparidadH);
	cvThreshold(imagenDisparidadH, imagenDisparidadH, 5, 255, CV_THRESH_BINARY);			// Umbralizar


	//	obstaculos(imagenDisparidad, adjusts.umbralObstaculos, adjusts.porcentaje * 10);
	storageD= cvCreateMemStorage(0);
	diagonal = cvCreateSeq( CV_32SC4, sizeof(CvSeq), 2 * sizeof(CvPoint), storageD);

	// Horizontales
	storageMP= cvCreateMemStorage(0);
	storageMN = cvCreateMemStorage(0);
	pendpos = cvCreateSeq( CV_32SC4, sizeof(CvSeq), 2 * sizeof(CvPoint), storageMP);
	pendneg = cvCreateSeq( CV_32SC4, sizeof(CvSeq), 2 * sizeof(CvPoint), storageMN);

	lineasV2(imagenDisparidad, vLines, diagonal);

/********************* Definiendo regi�n de inter�s en base a las l�neas verticales ****************/
	nVer = vLines->GetN();
	index = vLines->GetIndex();
	
	for (i= 0; i < nVer; i ++){		
		ver = (CvPoint *) cvGetSeqElem(vLines->GetLine(index[i]), 0);

		crearRDM (mapaDisparidadReducido, mapaDisparidad, ver);
		
		crearImagenH2(mapaDisparidadReducido, imagenDisparidadH, ver[0].x);
		lineasH2(imagenDisparidadH, hLines, pendpos, pendneg);

		// Rodear la regi�n de inter�s en el mapa de disparidad reducido
		cvRectangle(mapaDisparidadReducido, cvPoint(ver[0].x, MIN(ver[0].y, ver[1].y)), cvPoint(mapaDisparidadReducido->width, MAX(ver[0].y, ver[1].y)), CV_RGB(255,255,255));	
		cvShowImage ("Mapa disparidad reducido", mapaDisparidadReducido);
	}
/*******************************************************************************/

	marcObstacle2 (left, vLines, hLines, 5);

clock_t stop = clock();
printf("%.10lf\n", (double)(stop - start)/CLOCKS_PER_SEC);

	cvShowImage ("Mapa disparidad", mapaDisparidad);	


	delete vLines;
	delete hLines;
	
	cvClearSeq(diagonal);
	cvClearSeq(pendpos);
	cvClearSeq(pendneg);
	
	cvReleaseMemStorage (&storageD);
	cvReleaseMemStorage (&storageMP);
	cvReleaseMemStorage (&storageMN);

	cvReleaseImage(&mapaDisparidadReducido);

	cvReleaseImage(&mapaDisparidad);
	cvReleaseImage(&imagenDisparidad);
	cvReleaseImage(&imagenDisparidadH);
	cvReleaseImage(&izquierda);
	cvReleaseImage(&derecha);

	//cvWaitKey(1);
}



/*-----------------------------------------------------------------------------------------------------------------
		NOMBRE:
	   FUNCI�N:
	PAR�METROS:
	  DEVUELVE:
-----------------------------------------------------------------------------------------------------------------*/
int main (int argc, char* argv[]){
	IplImage *izquierda,		// Imagen izquierda
			 *derecha;			// Imagen derecha

//	LPWSTR *lista;
//	int totalDisp = 0;
//	CCapturaVLC captura;
	parameter ajustes;
	int frameNr;
	char filename[30];
	const char *prefix = "Series/puerta";
	
	bool trackbar;
		
	trackbar = false;

	CvCapture *videoIzq = 0;
	CvCapture *videoDer = 0;

	source = 4;					// Origen de las im�genes 0->Fichero imagen, 1 -> Tiempo real, 2-> Fichero v�deo, 3->T.Real Capturando frames, 4-> Frames capturados

	switch (source) {
		case 1:
		case 3:{										// Inicializar la captura desde c�maras
//			lista = captura.listaDispositivos(&totalDisp);
	
//			printf("TotalDisp = %d\n", totalDisp);

//			for (int i = 0; i < totalDisp; i++) {
//				printf("%d: %S\n", i + 1, lista[i]);
//			}
			system ("v4l2-ctl --dev /dev/video0 -i 0");
			system ("v4l2-ctl --dev /dev/video1 -i 1");

			videoIzq = cvCaptureFromCAM(0); // capture from video device #0
			videoDer = cvCaptureFromCAM(1); // capture from video device #1

			break;	   
		}

		case 2:{										// Inicializar los v�deos
			videoIzq = cvCaptureFromFile("Izquierda.avi");
			if (!videoIzq) {
				printf ("Error. Video izquierdo no encontrado\n.");
				exit (-1);			
			}

			videoDer = cvCaptureFromFile("Derecha.avi");
			if (!videoDer) {
				printf ("Error. Video derecho no encontrado\n.");
				exit (-1);			
			}

			break;
		}
	}


//	izquierda = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
//	derecha = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);

	cvNamedWindow("Izquierda", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Derecha", CV_WINDOW_AUTOSIZE);
//	cvNamedWindow("Lineas", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Obstaculos", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Mapa disparidad", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Imagen disparidad", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Imagen disparidad H", CV_WINDOW_AUTOSIZE);

	cvNamedWindow("Mapa disparidad reducido", CV_WINDOW_AUTOSIZE);

	/* Crear la ventana de controles */
	cvNamedWindow("Controles", CV_WINDOW_AUTOSIZE);
	ajustes.filtro = 9;
	ajustes.sobel = 3;
	ajustes.umbral = 4;
	ajustes.umbralObstaculos = 10;
	ajustes.porcentaje = 5;
	cvCreateTrackbar ("Filtro", "Controles", &ajustes.filtro, 21, checkFilter);
	cvCreateTrackbar ("Sobel", "Controles", &ajustes.sobel, 7, checkSobel);
	cvCreateTrackbar ("Umbral Ter", "Controles", &ajustes.umbral, 10, NULL);
	cvCreateTrackbar ("Umbral Obs", "Controles", &ajustes.umbralObstaculos, 15, NULL);
	cvCreateTrackbar ("Porcentaje", "Controles", &ajustes.porcentaje, 10, NULL);

	frameNr = 1;

	do {

		switch (source){
			case 0: {		// Im�genes est�ticas
				izquierda = cvLoadImage("Series/puerta_left_50.bmp");
				derecha = cvLoadImage("Series/puerta_right_50.bmp");

				if (!izquierda || !derecha){
					printf ("Error leyendo im�genes\n");
					exit (-1);
				}
				
				izquierda->origin = 0;
				derecha->origin = 0;

				break;
			}
			/*case 1: {		// Tiempo real
				if (!cvGrabFrame(videoIzq)){              // capture a frame 
  					printf("Could not grab a frame\n\7");
  					exit(0);
				}

				if (!cvGrabFrame(videoDer)){              // capture a frame 
  					printf("Could not grab a frame\n\7");
  					exit(0);
				}
			
				izquierda=cvRetrieveFrame(videoIzq);           // retrieve the captured frame
				derecha=cvRetrieveFrame(videoDer);           // retrieve the captured frame


				//izquierda = captura.captura(lista[0]);					// Capturar imagen izquierda
				//derecha = captura.captura(lista[1]);					// Capturar imagen derecha		

				izquierda->origin = 1;
				derecha->origin = 1;

				break;		
			}		*/	
			case 1:
			case 2: {		// V�deo
				if (!cvGrabFrame(videoIzq)){
					printf ("Error leyendo v�deo izquierdo\n");
					exit (-1);
				}
				if (!cvGrabFrame(videoDer)){
					printf ("Error leyendo v�deo derecho\n");
					exit (-1);
				}

				izquierda = cvRetrieveFrame(videoIzq);
				derecha = cvRetrieveFrame(videoDer);
				
				izquierda->origin = 0;
				derecha->origin = 0;

				break;
			}
			case 3: {		// Tiempo real con captura de secuencia de im�genes
				cvGrabFrame(videoIzq);
				cvGrabFrame(videoDer);

				izquierda = cvRetrieveFrame(videoIzq);
				derecha = cvRetrieveFrame(videoDer);

				sprintf(filename, "%s_left_%d.bmp", prefix, frameNr); 
				cvSaveImage (filename, izquierda);
				sprintf(filename, "%s_right_%d.bmp", prefix, frameNr); 
				cvSaveImage (filename, derecha);

				izquierda->origin = 0;
				derecha->origin = 0;

				break;		
			}

			case 4: {
				sprintf(filename, "%s_left_%d.bmp", prefix, frameNr); 
				izquierda = cvLoadImage(filename);

				sprintf(filename, "%s_right_%d.bmp", prefix, frameNr); 
				derecha = cvLoadImage(filename);

				if (!izquierda || !derecha) {		// Si se ha llegado al final de la secuencia -> reiniciar
					cvCreateTrackbar ("Frame", "Controles", &frameNr, frameNr-1, NULL);
					
					frameNr = 1;
					trackbar = true;
					
					sprintf(filename, "%s_left_%d.bmp", prefix, frameNr); 
					izquierda = cvLoadImage(filename);
	
					sprintf(filename, "%s_right_%d.bmp", prefix, frameNr); 
					derecha = cvLoadImage(filename);

				}
				
				if (trackbar) {
					cvSetTrackbarPos ("Frame", "Controles", frameNr);
				}

				izquierda->origin = 0;
				derecha->origin = 0;

				break;
			}

		}
		
		disparity (izquierda, derecha, ajustes);
//cvShowImage ("Izquierda", izquierda);	
//cvShowImage ("Derecha", derecha);	


		if ((source == 4) || (source == 0)){					// La propia funci�n de captura se encarga de la memoria
			cvReleaseImage(&izquierda);
			cvReleaseImage(&derecha);
		}

	
		frameNr++;
		printf ("Frame %d\n", frameNr);
	} while (cvWaitKey(10) == -1);
	
	cvReleaseCapture (&videoIzq);
	cvReleaseCapture (&videoDer);
	
	return (0);
}

