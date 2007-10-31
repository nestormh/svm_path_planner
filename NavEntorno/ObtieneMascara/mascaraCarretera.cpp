// matchImagenes5.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "CRuta.h"
#include "tieneObstaculos.h"

int inicio = 2, fin = 2594, vel = 0;
int pos = 0, next = 0;

int bSize = 3, params = 6;
int puntos = 8;
int bufSize = 5;

int orAnd = 3;

int umbral = 128;

tieneObstaculos obs;

void updateTrackBSize(int val) {
	val = obs.updateBSize(val);
	cvSetTrackbarPos("BlockSize", "Control", val);	
	pos--;
}

void updateTrackParams(int val) {
	val = obs.updateParams(val);
	cvSetTrackbarPos("Params", "Control", val);
	pos--;
}

void updateTrackBuffer(int val) {
	val = obs.updateBuffer(val);
	cvSetTrackbarPos("Buffer", "Control", val);		
}

void updateTrackPuntos(int val) {
	val = obs.updatePuntos(val);
	cvSetTrackbarPos("Puntos", "Control", val);
	pos--;
}

void updateTrackOrAnd(int val) {
	val = obs.updateOrAnd(val);
	cvSetTrackbarPos("Or/And", "Control", val);	
	pos--;
}

void updateTrackUmbral(int val) {
	val = obs.updateUmbral(val);
	cvSetTrackbarPos("Umbral", "Control", val);	
	pos--;
}

void initControl() {
	cvNamedWindow("Control", 0);

	cvCreateTrackbar("Inicio", "Control", &inicio, 726, NULL);
	cvCreateTrackbar("Fin", "Control", &fin, 726, NULL);
	cvCreateTrackbar("Pos", "Control", &pos, 726, NULL);
	cvCreateTrackbar("Vel", "Control", &vel, 1000, NULL);
	cvCreateTrackbar("Next", "Control", &next, 100, NULL);	
	cvCreateTrackbar("BlockSize", "Control", &bSize, 50, updateTrackBSize);
	cvCreateTrackbar("Params", "Control", &params, 50, updateTrackParams);
	cvCreateTrackbar("Puntos", "Control", &puntos, 20, updateTrackPuntos);
	cvCreateTrackbar("Buffer", "Control", &bufSize, 100, updateTrackBuffer);
	cvCreateTrackbar("Or/And", "Control", &orAnd, 3, updateTrackOrAnd);
	cvCreateTrackbar("Umbral", "Control", &umbral, 256, updateTrackUmbral);
}

void initGUI() {
	cvNamedWindow("Imagen", 1);
	cvNamedWindow("Pyramid", 1);
	cvNamedWindow("Poligono", 1);

	/*cvNamedWindow("Mascara1", 1);
	cvNamedWindow("Mascara2", 1);
	cvNamedWindow("Mascara3", 1);
	cvNamedWindow("Mascara4", 1);*/

	cvNamedWindow("Umbral", 1);		

	cvNamedWindow("Debug", 1);		

	initControl();
}

int _tmain(int argc, _TCHAR* argv[]) {	
	cout << "NOTA: Va lento porque el número de hormigas es muy alto para mejorar precisión" << endl;

	char * dir = "C:\\Proyecto\\Datos";

	CRuta ruta(dir, "iter");	

	initGUI();	

	char * pref = "Imagen";
	char * post = "n.jpg";
	char * path = "C:\\Proyecto\\Datos\\iter";
	char * imagen = new char[strlen(path) + strlen(pref) + strlen(post) + 4];
	
	for (pos = inicio - 1; pos < fin; pos++) {
		cvSetTrackbarPos("Pos", "Control", pos);			
		//cout << pos << endl;

		IplImage * img = ruta.getImagenCercana(ruta.getPosicion(pos));

		IplImage * mascara = obs.getMask(img);

		sprintf(imagen, "%s\\%s%d%s", path, pref, pos, post);
		//cvSaveImage(imagen, mascara);

		cvWaitKey(vel);
		
		if (pos > fin)
			pos = inicio - 1;	
	}	

	return 0;
}

