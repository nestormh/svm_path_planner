// MathingPorFunciones.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ViewMorphing.h"
#include "ImageRegistration.h"
#include "CRutaDB.h"
#include "CRealMatches.h"
#include "CStatistics.h"
#include "CImageSearch.h"
#include "CImageNavigation.h"
#include "SurfGPU.h"
#include "CUDAlib.h"

#define EJECUCION 22

#define METHOD_CHEBYSHEV 0
#define METHOD_HARRIS 1

#define COCINA -13
#define CAMA -12
#define TELEVISION -11
#define FRUTAS -10
#define BOTELLAS -9
#define LIBROS -8
#define CIRCUITOS -7
#define DTO_PNG -6
#define DTO_JPG -5
#define TALADRADORA_PNG -4
#define TALADRADORA_JPG -3
#define EL_HIERRO_SABINA1 -2
#define EL_HIERRO_SABINA2 -1
#define MANZANILLA 0
#define YO_EN_CARRETERA 1			// OK
#define REJAS_Y_FURGON 2			// OK...
#define COCHE_DE_FRENTE 3			// OK si no hay obst�culos
#define FAROLA_Y_TODOTERRENO 4		// OK
#define PALMERAS 5					// ---
#define CONO1 6						// MUY DIFERENTES
#define CONO2 7						// OK!
#define CARRETERA_VACIA 8			// OK
#define COCHE_Y_REJAS_LEJANO 9		// OK, pero pocas correspondencias debido a los obst�culos
#define ART_EXP0_ILLUM1 10
#define ART_EXP1_ILLUM1 11
#define ART_EXP2_ILLUM1 12
#define ART_EXP0_ILLUM2 13
#define ART_EXP1_ILLUM2 14
#define ART_EXP2_ILLUM2 15
#define ART_EXP0_ILLUM3 16
#define ART_EXP1_ILLUM3 17
#define ART_EXP2_ILLUM3 18
#define COMPUTER_EXP0_ILLUM1 19
#define COMPUTER_EXP1_ILLUM1 20
#define COMPUTER_EXP2_ILLUM1 21
#define COMPUTER_EXP0_ILLUM2 22
#define COMPUTER_EXP1_ILLUM2 23
#define COMPUTER_EXP2_ILLUM2 24
#define COMPUTER_EXP0_ILLUM3 25
#define COMPUTER_EXP1_ILLUM3 26
#define COMPUTER_EXP2_ILLUM3 27
#define DRUM_EXP0_ILLUM1 28
#define DRUM_EXP1_ILLUM1 29
#define DRUM_EXP2_ILLUM1 30
#define DRUM_EXP0_ILLUM2 31
#define DRUM_EXP1_ILLUM2 32
#define DRUM_EXP2_ILLUM2 33
#define DRUM_EXP0_ILLUM3 34
#define DRUM_EXP1_ILLUM3 35
#define DRUM_EXP2_ILLUM3 36
#define Books_EXP0_ILLUM1 37
#define Books_EXP1_ILLUM1 38
#define Books_EXP2_ILLUM1 39
#define Books_EXP0_ILLUM2 40
#define Books_EXP1_ILLUM2 41
#define Books_EXP2_ILLUM2 42
#define Books_EXP0_ILLUM3 43
#define Books_EXP1_ILLUM3 44
#define Books_EXP2_ILLUM3 45
#define DOLLS_EXP0_ILLUM1 46
#define DOLLS_EXP1_ILLUM1 47
#define DOLLS_EXP2_ILLUM1 48
#define DOLLS_EXP0_ILLUM2 49
#define DOLLS_EXP1_ILLUM2 50
#define DOLLS_EXP2_ILLUM2 51
#define DOLLS_EXP0_ILLUM3 52
#define DOLLS_EXP1_ILLUM3 53
#define DOLLS_EXP2_ILLUM3 54
#define LAUNDRY_EXP0_ILLUM1 55
#define LAUNDRY_EXP1_ILLUM1 56
#define LAUNDRY_EXP2_ILLUM1 57
#define LAUNDRY_EXP0_ILLUM2 58
#define LAUNDRY_EXP1_ILLUM2 59
#define LAUNDRY_EXP2_ILLUM2 60
#define LAUNDRY_EXP0_ILLUM3 61
#define LAUNDRY_EXP1_ILLUM3 62
#define LAUNDRY_EXP2_ILLUM3 63
#define DWARVES_EXP0_ILLUM1 64
#define DWARVES_EXP1_ILLUM1 65
#define DWARVES_EXP2_ILLUM1 66
#define DWARVES_EXP0_ILLUM2 67
#define DWARVES_EXP1_ILLUM2 68
#define DWARVES_EXP2_ILLUM2 69
#define DWARVES_EXP0_ILLUM3 70
#define DWARVES_EXP1_ILLUM3 71
#define DWARVES_EXP2_ILLUM3 72
#define REINDEER_EXP0_ILLUM1 73
#define REINDEER_EXP1_ILLUM1 74
#define REINDEER_EXP2_ILLUM1 75
#define REINDEER_EXP0_ILLUM2 76
#define REINDEER_EXP1_ILLUM2 77
#define REINDEER_EXP2_ILLUM2 78
#define REINDEER_EXP0_ILLUM3 79
#define REINDEER_EXP1_ILLUM3 80
#define REINDEER_EXP2_ILLUM3 81
#define MOEBIUS_EXP0_ILLUM1 82
#define MOEBIUS_EXP1_ILLUM1 83
#define MOEBIUS_EXP2_ILLUM1 84
#define MOEBIUS_EXP0_ILLUM2 85
#define MOEBIUS_EXP1_ILLUM2 86
#define MOEBIUS_EXP2_ILLUM2 87
#define MOEBIUS_EXP0_ILLUM3 88
#define MOEBIUS_EXP1_ILLUM3 89
#define MOEBIUS_EXP2_ILLUM3 90

CStatistics stat;

void inicio1(IplImage * img1, IplImage * img2, int method);

void getImgFromDataSet(int val, IplImage* &img1, IplImage *&img2, IplImage *&img1C, IplImage *&img2C) {	
	int exp = -1;
	int illum = -1;
	char * cad; 
	if (val == ART_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Art"; }
	if (val == ART_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Art"; }
	if (val == ART_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Art"; }
	if (val == ART_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Art"; }
	if (val == ART_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Art"; }
	if (val == ART_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Art"; }
	if (val == ART_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Art"; }
	if (val == ART_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Art"; }
	if (val == ART_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Art"; }

	if (val == COMPUTER_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Computer"; }
	if (val == COMPUTER_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Computer"; }
	if (val == COMPUTER_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Computer"; }
	if (val == COMPUTER_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Computer"; }
	if (val == COMPUTER_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Computer"; }
	if (val == COMPUTER_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Computer"; }
	if (val == COMPUTER_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Computer"; }
	if (val == COMPUTER_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Computer"; }
	if (val == COMPUTER_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Computer"; }

	if (val == DRUM_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Drumsticks"; }
	if (val == DRUM_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Drumsticks"; }
	if (val == DRUM_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Drumsticks"; }
	if (val == DRUM_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Drumsticks"; }
	if (val == DRUM_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Drumsticks"; }
	if (val == DRUM_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Drumsticks"; }
	if (val == DRUM_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Drumsticks"; }
	if (val == DRUM_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Drumsticks"; }
	if (val == DRUM_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Drumsticks"; }

	if (val == Books_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Books"; }
	if (val == Books_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Books"; }
	if (val == Books_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Books"; }
	if (val == Books_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Books"; }
	if (val == Books_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Books"; }
	if (val == Books_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Books"; }
	if (val == Books_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Books"; }
	if (val == Books_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Books"; }
	if (val == Books_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Books"; }

	if (val == DOLLS_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Dolls"; }
	if (val == DOLLS_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Dolls"; }
	if (val == DOLLS_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Dolls"; }
	if (val == DOLLS_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Dolls"; }
	if (val == DOLLS_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Dolls"; }
	if (val == DOLLS_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Dolls"; }
	if (val == DOLLS_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Dolls"; }
	if (val == DOLLS_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Dolls"; }
	if (val == DOLLS_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Dolls"; }

	if (val == LAUNDRY_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Laundry"; }
	if (val == LAUNDRY_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Laundry"; }
	if (val == LAUNDRY_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Laundry"; }
	if (val == LAUNDRY_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Laundry"; }
	if (val == LAUNDRY_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Laundry"; }
	if (val == LAUNDRY_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Laundry"; }
	if (val == LAUNDRY_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Laundry"; }
	if (val == LAUNDRY_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Laundry"; }
	if (val == LAUNDRY_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Laundry"; }

	if (val == DWARVES_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Dwarves"; }
	if (val == DWARVES_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Dwarves"; }
	if (val == DWARVES_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Dwarves"; }
	if (val == DWARVES_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Dwarves"; }
	if (val == DWARVES_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Dwarves"; }
	if (val == DWARVES_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Dwarves"; }
	if (val == DWARVES_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Dwarves"; }
	if (val == DWARVES_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Dwarves"; }
	if (val == DWARVES_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Dwarves"; }

	if (val == REINDEER_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Reindeer"; }
	if (val == REINDEER_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Reindeer"; }
	if (val == REINDEER_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Reindeer"; }
	if (val == REINDEER_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Reindeer"; }
	if (val == REINDEER_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Reindeer"; }
	if (val == REINDEER_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Reindeer"; }
	if (val == REINDEER_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Reindeer"; }
	if (val == REINDEER_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Reindeer"; }
	if (val == REINDEER_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Reindeer"; }

	if (val == MOEBIUS_EXP0_ILLUM1) { exp = 0; illum = 1; cad = "Moebius"; }
	if (val == MOEBIUS_EXP1_ILLUM1) { exp = 1; illum = 1; cad = "Moebius"; }
	if (val == MOEBIUS_EXP2_ILLUM1) { exp = 2; illum = 1; cad = "Moebius"; }
	if (val == MOEBIUS_EXP0_ILLUM2) { exp = 0; illum = 2; cad = "Moebius"; }
	if (val == MOEBIUS_EXP1_ILLUM2) { exp = 1; illum = 2; cad = "Moebius"; }
	if (val == MOEBIUS_EXP2_ILLUM2) { exp = 2; illum = 2; cad = "Moebius"; }
	if (val == MOEBIUS_EXP0_ILLUM3) { exp = 0; illum = 3; cad = "Moebius"; }
	if (val == MOEBIUS_EXP1_ILLUM3) { exp = 1; illum = 3; cad = "Moebius"; }
	if (val == MOEBIUS_EXP2_ILLUM3) { exp = 2; illum = 3; cad = "Moebius"; }

	char cadena1[1024];
	char cadena2[1024];
	sprintf(cadena1, "/home/neztol/doctorado/Datos/imagenesSueltas/DataSets/%s/Illum%d/Exp%d/view1.png", cad, illum, exp);
	sprintf(cadena2, "/home/neztol/doctorado/Datos/imagenesSueltas/DataSets/%s/Illum%d/Exp%d/view5.png", cad, illum, exp);
		
	IplImage * imgA = cvLoadImage(cadena1, 0);
	IplImage * imgB = cvLoadImage(cadena2, 0);

	img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
	img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

	cvResize(imgA, img1);
	cvResize(imgB, img2);

	cvReleaseImage(&imgA);
	cvReleaseImage(&imgB);
		
	imgA = cvLoadImage(cadena1, 1);
	imgB = cvLoadImage(cadena2, 1);

	img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
	img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);	

	cvResize(imgA, img1C);
	cvResize(imgB, img2C);

	cvReleaseImage(&imgA);
	cvReleaseImage(&imgB);
}

int inicio0(int inicio, int optionImg) {
	//int optionImg = ART_EXP0_ILLUM1;
	int method = METHOD_HARRIS;
	//int inicio = 1;

	IplImage * img1, * img2, * img1C, * img2C, * mask, * prevImg = NULL;
        if (optionImg == EL_HIERRO_SABINA2) {
		IplImage * imgA = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3255.JPG", 0);
                IplImage * imgB = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3256.JPG", 0);

		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);

		imgA = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3255.JPG", 1);
                imgB = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3256.JPG", 1);

		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
        if (optionImg == EL_HIERRO_SABINA1) {
		IplImage * imgA = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3258.JPG", 0);
                IplImage * imgB = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3261.JPG", 0);

		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);

		imgA = cvLoadImage("/media/shared/fotos/DSC_0103.JPG", 1);
                imgB = cvLoadImage("/media/shared/fotos/DSC_0121.JPG", 1);

		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
        if (optionImg == MANZANILLA) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/Estadisticas/DSC_0103.JPG", 0);
                IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/Estadisticas/DSC_0121.JPG", 0);

		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);
                img1 = imgA;
                img2 = imgB;

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
                //img1 = imgA;
                //img2 = imgB;

		imgA = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3255.JPG", 1);
                imgB = cvLoadImage("/media/shared/Mis Documentos/MIs imagenes/El Hierro Septiembre 2009/El Hierro Bea/DSCN3256.JPG", 1);

		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
                //img1C = imgA;
                //img2C = imgB;
	}
        if (optionImg == TALADRADORA_PNG) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora1.png", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora2.png", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora1.png", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora2.png", 1);		
	}
        if (optionImg == TALADRADORA_JPG) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora1.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora2.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora1.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/taladradora2.jpg", 1);
	}
        if (optionImg == DTO_PNG) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto1.png", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto2.png", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto1.png", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto2.png", 1);
	}
        if (optionImg == DTO_JPG) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto1.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto2.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto1.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/dto2.jpg", 1);
	}
	if (optionImg == YO_EN_CARRETERA) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen3825a.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen391a.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen3825a.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen391a.jpg", 1);
		mask = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/mascarasPrueba/Imagen3825m.jpg", 0);
                prevImg = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen3824a.jpg", 0);
	}
	if (optionImg == REJAS_Y_FURGON) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen1401a.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen305a.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen1401a.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen305a.jpg", 1);
		mask = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/mascarasPrueba/Imagen1401m.jpg", 0);
                prevImg = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen1400a.jpg", 0);
	}
	if (optionImg == COCHE_DE_FRENTE) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen892a.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen95a.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen892a.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen95a.jpg", 1);
		mask = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/mascarasPrueba/Imagen892m.jpg", 0);
                prevImg = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen891a.jpg", 0);
	}

	if (optionImg == FAROLA_Y_TODOTERRENO) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_080.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_081.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);

		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_080.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_081.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == PALMERAS) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_082.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_083.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);

		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_082.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_083.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == CONO1) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_076.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_077.jpg", 0);

		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_076.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_077.jpg", 1);

		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == CONO2) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_078.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_079.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_078.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_079.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == CARRETERA_VACIA) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen2908a.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen3a.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen2908a.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen3a.jpg", 1);
		mask = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/mascarasPrueba/Imagen2908m.jpg", 0);
                prevImg = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen2907a.jpg", 0);
	}
	if (optionImg == COCHE_Y_REJAS_LEJANO) {
		img1 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen4133a.jpg", 0);
		img2 = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen715a.jpg", 0);
		img1C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen4133a.jpg", 1);
		img2C = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneConObs/Imagen715a.jpg", 1);
		mask = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/mascarasPrueba/Imagen4133m.jpg", 0);
                prevImg = cvLoadImage("/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/Imagen4132a.jpg", 0);
	}
	if (optionImg == CIRCUITOS) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/circuitos1.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/circuitos2.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/circuitos1.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/circuitos2.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == TELEVISION) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/tele2.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/tele1.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/tele2.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/tele1.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == FRUTAS) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/frutas2.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/frutas1.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/frutas2.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/frutas1.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == BOTELLAS) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/botellas2.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/botellas1.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/botellas2.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/botellas1.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == COCINA) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cocina2.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cocina1.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cocina2.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cocina1.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == CAMA) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cama1.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cama2.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cama1.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/cama2.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg == LIBROS) {
		IplImage * imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/libros1.jpg", 0);
		IplImage * imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/libros2.jpg", 0);
		
		img1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		img2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		cvResize(imgA, img1);
		cvResize(imgB, img2);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
		
		imgA = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/libros1.jpg", 1);
		imgB = cvLoadImage("/home/neztol/doctorado/Datos/imagenesSueltas/libros2.jpg", 1);
		
		img1C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img2C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);

		cvResize(imgA, img1C);
		cvResize(imgB, img2C);

		cvReleaseImage(&imgA);
		cvReleaseImage(&imgB);
	}
	if (optionImg >= ART_EXP0_ILLUM1 && optionImg <= 90) {		
		getImgFromDataSet(optionImg, img1, img2, img1C, img2C);
	}

	cvNamedWindow("Img1", 1);
	cvNamedWindow("Img2", 1);
	cvShowImage("Img1", img1);
	cvShowImage("Img2", img2);        
        if (prevImg != NULL) {
            cvNamedWindow("Prev", 1);
            cvShowImage("Prev", prevImg);            
        }

        if (inicio == 2) {
		//cvSet(mask, cvScalar(255));
		inicio2(img1, img2, img1C, img2C, NULL);
	} else if (inicio == 3) {
            CImageRegistration registration(cvGetSize(img1));
            registration.registration(img1, prevImg, img2);
        } else if (inicio == 4) {
            t_moment * moments1, * moments2;
            int nMoments1, nMoments2;
            mesrTest(img1, "mser1", moments1, nMoments1);
            mesrTest(img2, "mser2", moments2, nMoments2);
            vector<t_moment *> regionPairs;
            matchMserByMoments(img1, img2, moments1, moments2, nMoments1, nMoments2, "Match", regionPairs);
            cout << "Detectados " << regionPairs.size() << endl;
        } else if (inicio == 5) {
            cjtosImagenes();
        } else if (inicio == 6) {
            pruebaRutas();
        } else if (inicio == 7) {
            stat.statistics();
            exit(0);
        } else if (inicio == 8) {
            cvReleaseImage(&img1);
            cvReleaseImage(&img2);
            cvReleaseImage(&img1C);
            cvReleaseImage(&img2C);
            if (prevImg != NULL) {
                cvReleaseImage(&prevImg);
            }
            cvDestroyWindow("Prev");
            char filename[1024];
            for (int i = 0; i < 3; i++) {
                sprintf(filename, "/home/neztol/doctorado/Datos/Aerop/Masked/Imagen%da.jpg", i + 1);
                img1 = cvLoadImage(filename, 0);
                img1C = cvLoadImage(filename, 1);
                sprintf(filename, "/home/neztol/doctorado/Datos/Aerop/Masked/Imagen%db.jpg", i + 1);
		img2 = cvLoadImage(filename, 0);
		img2C = cvLoadImage(filename, 1);
                sprintf(filename, "/home/neztol/doctorado/Datos/Aerop/Masked/Imagen%dam.jpg", i + 1);
		IplImage * mask1 = cvLoadImage(filename, 0);
                sprintf(filename, "/home/neztol/doctorado/Datos/Aerop/Masked/Imagen%dbm.jpg", i + 1);
                IplImage * mask2 = cvLoadImage(filename, 0);

                cvDilate(mask1, mask1, 0, 2);
                cvDilate(mask2, mask2, 0, 2);
                IplImage * tmpImg = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
                cvZero(tmpImg);
                cvCopy(img1, tmpImg, mask1);
                cvReleaseImage(&img1);
                img1 = tmpImg;                
                tmpImg = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
                cvZero(tmpImg);
                cvCopy(img2, tmpImg, mask2);
                cvReleaseImage(&img2);
                img2 = tmpImg;//*/

                cvNamedWindow("Mask1", 1);
                cvNamedWindow("Mask2", 1);
        	cvShowImage("Img1", img1);
                cvShowImage("Img2", img2);
                cvShowImage("Mask1", mask1);
                cvShowImage("Mask2", mask2);

                CImageRegistration registration(cvGetSize(img1));
                registration.registration(img1, prevImg, img2);
                int key = cvWaitKey(0);
                if (key == 27) exit(0);
            }
            exit(0);
        }

        int key = cvWaitKey(0);        

	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&img1C);
	cvReleaseImage(&img2C); 
        if (prevImg != NULL) {
            cvReleaseImage(&prevImg);
        }

        if (key == 27)
            exit(0);

	return 0;
}

int main(int argc, _TCHAR argv[]) {    
	/*for (int i = DTO_PNG; i < ART_EXP0_ILLUM1; i++) {
		inicio0(EJECUCION, i);
	}//*/
        //inicio0(2, MANZANILLA);
	/*for (int i = YO_EN_CARRETERA; i < FAROLA_Y_TODOTERRENO; i++) {
		inicio0(EJECUCION, i);
	}
	//*/
	/*for (int i = ART_EXP0_ILLUM1; i < MOEBIUS_EXP2_ILLUM3 + 1; i++) {
		inicio0(EJECUCION, i);
	} //*/
    switch (EJECUCION) {    
        case 7: {
            /*for (int i = 0; i < argc; i++) {
                cout << i << ": " << argv[i] << endl;
            }//*/
            int testIdx = atoi(argv[1]);
            int index = atoi(argv[2]);
            int size = atoi(argv[3]);
            int zoom = atoi(argv[4]);
            int b1 = atoi(argv[5]);
            int b2 = atoi(argv[6]);

            cout << "TEST" << testIdx;
            cout << ": index = " << index << ", ";
            cout << "size = " << size << ", ";
            cout << "zoom = " << zoom << ", ";
            cout << "b1 = " << b1 << ", ";
            cout << "b2 = " << b2 << endl;

            try {
                stat.statistics(testIdx, index, zoom, size, b1, b2);
            } catch (exception &e) {
                cout << "[EXCEPTION]: ";
                cout << "TEST" << testIdx;
                cout << ": index = " << index << ", ";
                cout << "size = " << size << ", ";
                cout << "zoom = " << zoom << ", ";
                cout << "b1 = " << b1 << ", ";
                cout << "b2 = " << b2 << " ::: ";
                cout << e.what() << endl;
            }
            //stat.statistics();            
            break;
        }
        case 8: {
            stat.tests(0);
            break;
        }
        case 9: {
            CRealMatches rm(false, SIZE3);
            //rm.startTest(string("/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/"), string("datos.txt"), string("testOFlow1"));
            //rm.startTest(string("/home/neztol/doctorado/Datos/Estadisticas/"), string("datos.txt"), string("testOFlow1"));
            rm.startTest(string("/home/neztol/doctorado/Datos/EstadisticasITER/tripode1/"), string("datos.txt"), string("testOFlow1"));
            break;
        }
        case 10: {
            CRealMatches rm;
            rm.startTest2();
            break;
        }
        case 11: {
            CRealMatches rm(false);
            rm.startTest3();
            break;
        }
        case 12: {
            CRealMatches rm(false, SIZE2);
            rm.startTest4();
            break;
        }
        case 13: {
            stat.MRTP_test(SIZE2);
            break;
        }
        case 14: {
            CRealMatches rm(false);
            rm.startTest5();
            break;
        }
        case 15: {
            CRealMatches rm(false);
            rm.startTestRoadDetection();
            break;
        }
        case 16: {
            //CRealMatches rm(false);
            CRealMatches rm(false, SIZE2);
            rm.startTest6();
            break;
        }
        case 17: {
            CRealMatches rm(false, SIZE2);
            rm.startTest7();
            break;
        }
        case 18: {
            //CImageSearch is("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/urbRadazulDiciembre08Base", "Rutas/urbRadazulDiciembre08obs", "/home/neztol/doctorado/Datos/DB", false, cvRect(0, 0, 310, 228));
            CImageSearch is("/home/neztol/doctorado/Datos/DB/navEntorno3.sqlite", "Rutas/pruebaITERBase2", "Rutas/pruebaITERConObs2", "/home/neztol/doctorado/Datos/DB", false, cvRect(5, 0, 310, 240), cvSize(176, 144));
            is.startTest();
            break;
        }
        case 19: {
            //CImageNavigation in("/home/neztol/doctorado/Datos/DB/Rutas/urbRadazulDiciembre08Base/Camera0/");
            CImageNavigation in("/home/neztol/doctorado/Datos/MRPT_Data/malaga2009_campus_2L/Images_rect/", "jpg");
            //CImageNavigation in("/home/neztol/doctorado/Datos/EstadisticasITER/tripode3/");
            in.makePairsOFlow();
            break;
        }
        case 20: {
            SurfGPU surf;
            surf.testSurf(string("/home/neztol/doctorado/Datos/EstadisticasITER/tripode1/DSC_0555.JPG"), string("/home/neztol/doctorado/Datos/EstadisticasITER/tripode1/DSC_0559.JPG"));
            
            //surf.testSurf(string("/home/neztol/doctorado/Datos/imagenesSueltas/cocina1.jpg"), string("/home/neztol/doctorado/Datos/imagenesSueltas/cocina2.jpg"));
            break;
        }
        case 21: {
            enumerateDevices();
            sumaArrays();
            break;
        }
        case 22: {
            string testName = "testSinCUDA";
            CRealMatches rm1(false, cvSize(512, 512));
            rm1.startTestCMU(testName, true);
            rm1.~CRealMatches();
            CRealMatches rm2(false, cvSize(768, 768));
            rm2.startTestCMU(testName);
            rm2.~CRealMatches();
            CRealMatches rm3(false, cvSize(1024, 1024));
            rm3.startTestCMU(testName);
            rm3.~CRealMatches();

            break;
        }
        default: {
            inicio0(EJECUCION, -1);
        }
    }
}
