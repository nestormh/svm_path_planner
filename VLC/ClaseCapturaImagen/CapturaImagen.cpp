// CapturaImagen.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "CapturaVLC.h"


int _tmain(int argc, _TCHAR* argv[])
{
	CCapturaVLC captura;

	int totalDisp = 0;

	LPWSTR * lista = captura.listaDispositivos(&totalDisp);
	
	printf("TotalDisp = %d\n", totalDisp);

	for (int i = 0; i < totalDisp; i++) {
		printf("%d: %S\n", i + 1, lista[i]);
	}

	IplImage * img = captura.captura(lista[0]);

	while(true) {
		img = captura.captura(lista[0]);
		
		//cvAddS(img, cvScalar(0, 0, 128), img);

		cvNamedWindow("Test", 1);
		cvShowImage("Test", img);

		cvReleaseImage(&img);

		cvWaitKey(20);
	}

	getchar();

	return 0;
}

