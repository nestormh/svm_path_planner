/* 
 * File:   CAntColony.cpp
 * Author: neztol
 * 
 * Created on 13 de mayo de 2010, 9:15
 */

#include "CAntColony.h"
#include "../ViewMorphing.h"

extern CvPoint p1;
extern CvPoint p2;
extern CvPoint p3;
extern CvPoint p4;

CAntColony::CAntColony(CvSize size) {
    bordeSup = 124;
    horizonSlider = 90;
    edgeSlider = 12; //75
    kBar = 80;
    kpBar = 70;
    kdBar = 30;
    izqBar = 160;
    dchaBar = 160;
    searchAreas = 50;
    consigna = 0;
    anguloCamara = 170;

    colony = (colonyStruct*) malloc(sizeof (colonyStruct));

    attractionX = 160;
    attractionY = 270;
    attractionXAnt = refLeftY = refRightY = aRef = bRef = cRef = dRef = -1;
    selectedImage = 1;
    corte = 0;

    storage = cvCreateMemStorage(0);

    cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE); //Creación y colocación de las ventanas
    cvNamedWindow("sliderWin", CV_WINDOW_AUTOSIZE);

    //Creación de las imágenes
    
    gray = cvCreateImage(size, 8, 1);
    gray->origin = 1;
    dst = cvCreateImage(size, 8, 1);
    dst->origin = 1;
    segmented = cvCreateImage(size, 8, 1);
    segmented->origin = 1;
    bordes = cvCreateImage(size, 8, 1);
    bordes->origin = 1;
    shadowMask = cvCreateImage(size, 8, 1);
    shadowMask->origin = 1;
    mask = cvCreateImage(size, 8, 3);
    mask->origin = 1;
    shadows = cvCreateImage(size, 8, 1);
    shadows->origin = 1;

    traces = cvCreateImage(size, 8, 3);
    traces->origin = 1;
    tempCapture = cvCreateImage(size, 8, 3);
    tempCapture->origin = 1;

    tracesaux = cvCreateImage(size, 8, 3);
    tracesaux->origin = 1;
    tempImage = cvCreateImage(size, 8, 1);
    tempImage->origin = 1;
    ed = cvCreateImage(size, 8, 1);
    ed->origin = 1;

    // Creación de las barras de desplazamiento

    //cvCreateTrackbar("Horiz.", "sliderWin", &horizonSlider, 200, on_trackbar);
    //cvCreateTrackbar("Bordes", "sliderWin", &edgeSlider, 255, NULL);
    //cvCreateTrackbar("Pos.Ini.", "sliderWin", &searchAreas, 170, NULL);
    cvShowImage("sliderWin", traces);

    cvSet(shadowMask, cvScalar(1));
    consigna = 0;
}

CAntColony::CAntColony(const CAntColony& orig) {
}

CAntColony::~CAntColony() {
    cvReleaseImage(&tempImage); //liberamos la memoria de las imágenes
    cvReleaseImage(&bordes);
    cvReleaseImage(&shadowMask);
    cvReleaseImage(&segmented);
    cvReleaseImage(&hough);
    cvReleaseImage(&mask);
    cvReleaseImage(&shadows);

    cvReleaseImage(&traces);
    cvReleaseImage(&tracesaux);
    cvReleaseImage(&tempCapture);
    cvReleaseImage(&gray);
    cvReleaseImage(&dst);
    cvReleaseImage(&ed);

    cvDestroyWindow("mainWin"); //Liberamos la memoria de las ventanas
    cvDestroyWindow("sliderWin");
}

CvPoint * CAntColony::iterate(IplImage * img) {
    img->origin = 1;
    cvConvertImage(img, img, 1); //flip vertical para mostrar la imágen correctamente.
    //cvCvtColor(img,hls,CV_BGR2HLS);						//en caso de que queramos trabajar con la luminancia
    //cvSplit(hls,h,l,s,0);



    cvCvtColor(img, gray, CV_RGB2GRAY); //pasamos la imagen de entrada a escala de grises    
    //cvCopyImage(img, gray);

    //cvSmooth(gray,gray,2,5);								//emborronamos para filtrar las altas frecuencias

    cvCanny(gray, segmented, edgeSlider, edgeSlider + 70); //aplicamos canny para detección de bordes

    cvDilate(segmented, segmented, 0, 3); //operación morfológica para dilatar los bordes y
    //crear dos zonas diferenciadas en la imágen: los márgenes
    //en blanco y la carretera en negro.

    cvCopy(segmented, dst); //se necesita una imagen temporal para calcular los contornos

    cvFindContours(dst, storage, &contour, sizeof (CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    cvZero(bordes);
    for (; contour != 0; contour = contour->h_next) {
        if (contour->total > 20) {
            cvDrawContours(bordes, contour, cvScalar(255), cvScalar(255), -1, 1, 8);
        }
    }

    attractionY = bordes->height - bordeSup + 20; //altura del punto de atracción

    //inicialización de la colonia de la izquierda
    initColony(bordes, traces, colony, bordes->width, bordes->height - bordeSup, attractionX, attractionY, searchAreas, &refLeftY);

    //Cálculo del camino mínimo por la izquierda
    acoMetaheuristic(bordes, traces, colony, &shortestLeftPath, 3);

    cvCopy(traces, tracesaux); //se salva el camino más corto de la izquierda

    //inicialización de la colonia de la derecha
    initColony(bordes, traces, colony, bordes->width, bordes->height - bordeSup, attractionX, attractionY, img->width - searchAreas, &refRightY);

    //cálculo del camino mínimo por la derecha
    acoMetaheuristic(bordes, traces, colony, &shortestRightPath, 3);

    cvAdd(tracesaux, traces, tracesaux); //juntamos los dos caminos-solución en una imágen

    cvAdd(img, tracesaux, tracesaux); //solapamos los caminos-solución con la imágen de entrada de fondo

    cvSetZero(mask); //inicialización del patrón de la carretera

    //actualización del punto de atracción
    setPointofAttraction(mask, shadowMask, &shortestLeftPath, &shortestRightPath, &attractionX, &aRef, &bRef, &cRef, &dRef, &corte, consigna);

    cvAdd(img, mask, img); //solapamos el patrón con la imágen de entrada de fondo

    if (attractionXAnt == -1) attractionXAnt = attractionX;

    /***************Depuración***********************
     * Dependiendo del número pulsado se mostrarán
     * distintas imágenes correspondientes a diferentes
     * etapas del procesado. Si se pulsa otra tecla,
     * se termina la ejecución.
     * **********************************************/
    switch (selectedImage) {
        case 1:
            cvShowImage("mainWin", img);
            break;
        case 2:
            cvShowImage("mainWin", tracesaux);
            break;
        case 3:
            cvShowImage("mainWin", bordes);
            break;
        case 4:
            cvShowImage("mainWin", segmented);
            break;
    }
    
    keyCode = cvWaitKey(2);
    if (keyCode == '1')
        selectedImage = 1;
    else if (keyCode == '2')
        selectedImage = 2;
    else if (keyCode == '3')
        selectedImage = 3;
    else if (keyCode == '4')
        selectedImage = 4;    
    /**********Fin de depuración***************/

    //cvSetTrackbarPos("Horiz.", "sliderWin", bordeSup);
    //cvSetTrackbarPos("Bordes", "sliderWin", edgeSlider);

    CvPoint * result = new CvPoint[4];
    result[0] = cvPoint(p1.x, img->height - p1.y - 1);
    result[1] = cvPoint(p2.x, img->height - p2.y - 1);
    result[2] = cvPoint(p3.x, img->height - p3.y - 1);
    result[3] = cvPoint(p4.x, img->height - p4.y - 1);

    return result;
}