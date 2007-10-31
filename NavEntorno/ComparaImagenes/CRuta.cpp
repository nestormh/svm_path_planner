#include "StdAfx.h"
#include "CRuta.h"

CRuta::CRuta(char * path_in, char * file) {
	path = new char[strlen(path_in) + strlen(file) + 1];
	sprintf(path, "%s\\%s", path_in, file);	

	loadRuta(path_in, file);

	masde50a = 0;
	masde100a = 0;
	masde150a = 0;
	masde200a = 0;
	masde250a = 0;
	masde50b = 0;
	masde100b = 0;
	masde150b = 0;
	masde200b = 0;
	masde250b = 0;
}

CRuta::~CRuta(void) {		
}

void CRuta::loadRuta(char * path_in, char * file) {
	char * fichero = new char[strlen(path_in) + strlen(file) + 5];

	sprintf(fichero, "%s\\%s.txt", path_in, file);	

	ifstream ifs(fichero, ifstream::in);

	if (ifs.is_open()) {			
		tCoord val;
		while (ifs.good()) {
			ifs >> val.x;			
			ifs >> val.y;			
			ifs >> val.z;	
			ifs >> val.ang;			
			ruta.push_back(val);
		}			
		ifs.close();
	} else {
		cout << "Error al abrir el fichero" << endl;		
	}	
}

tCoord CRuta::getPosicion(int i) {
	if (i > ruta.size())
		exit(0);
	return ruta[i];
}

double CRuta::getDifAngulos(double ang1, double ang2) {	
	int hemisferio1 = 0, hemisferio2 = 0;
	if (ang1 > CV_PI) hemisferio1 = 1;
	if (ang2 > CV_PI) hemisferio2 = 1;

	if (hemisferio1 != hemisferio2) {
		if (hemisferio1 == 0) ang1 += 2 * CV_PI;
		if (hemisferio2 == 0) ang2 += 2 * CV_PI;
	}

	return abs(ang1 - ang2);	
}

IplImage * CRuta::getImagenAt(int i) {
	if (i > ruta.size())
		exit(0);
	char * pref = "Imagen";
	char * post = "a.jpg";
	char * imagen = new char[strlen(path) + strlen(pref) + strlen(post) + 4];

	sprintf(imagen, "%s\\%s%d%s", path, pref, i, post);	

	IplImage * retorno = cvLoadImage(imagen);	

	return retorno;
}

IplImage * CRuta::getImagenCercana(tCoord pos) {
	minPos = 0;	
	double minDist = sqrt(pow(ruta[0].x - pos.x, 2) + pow(ruta[0].y - pos.y, 2) + pow(ruta[0].z - pos.z, 2));
	double difAng = getDifAngulos(pos.ang, ruta[0].ang);
	minDist += 2 * difAng;
	for (int i = 0; i < ruta.size(); i++) {
		double distancia = sqrt(pow(ruta[i].x - pos.x, 2) + pow(ruta[i].y - pos.y, 2) + pow(ruta[i].z - pos.z, 2));
		difAng = getDifAngulos(pos.ang, ruta[i].ang);
		distancia += 2 * difAng;

		if (distancia < minDist) {
			minDist = distancia;
			minPos = i;			
		}
	}
	cout << "Original: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << endl;
	cout << "Encontrada: (" << ruta[minPos].x << ", " << ruta[minPos].y << ", " << ruta[minPos].z << ")" << endl;

	cout << "minPos = " << minPos << endl;

	cout << "Distancia en X: " << abs(pos.x - ruta[minPos].x) << endl;
	cout << "Distancia en Y: " << abs(pos.y - ruta[minPos].y) << endl;
	cout << "Distancia en Z: " << abs(pos.z - ruta[minPos].z) << endl;	
	cout << "Distancia en ang: " << (getDifAngulos(pos.ang, ruta[minPos].ang) * 180 / CV_PI);
	cout << "(" << getDifAngulos(pos.ang, ruta[minPos].ang) << ")\n";
	cout << "Distancia absoluta: " << sqrt(pow(ruta[minPos].x - pos.x, 2) + pow(ruta[minPos].y - pos.y, 2) + pow(ruta[minPos].z - pos.z, 2)) << endl;
	cout << "Distancia ponderada: " << minDist << endl;

	double absol = sqrt(pow(ruta[minPos].x - pos.x, 2) + pow(ruta[minPos].y - pos.y, 2) + pow(ruta[minPos].z - pos.z, 2));
	
	if (absol > 0.50)
		masde50a++;
	if (absol > 1)
		masde100a++;
	if (absol > 1.50)
		masde150a++;
	if (absol > 2)
		masde200a++;
	if (absol > 2.50)
		masde250a++;
	if (minDist > 0.50)
		masde50b++;
	if (minDist > 1)
		masde100b++;
	if (minDist > 1.50)
		masde150b++;
	if (minDist > 2)
		masde200b++;
	if (absol > 2.50)
		masde250b++;



	char * pref = "Imagen";
	char * post = "a.jpg";
	char * imagen = new char[strlen(path) + strlen(pref) + strlen(post) + 4];

	sprintf(imagen, "%s\\%s%d%s", path, pref, minPos, post);	

	cout << imagen << endl;

	IplImage * retorno = cvLoadImage(imagen);	

	return retorno;
}

IplImage *  CRuta::getMask() {
	char * pref = "Imagen";
	char * post = "m.jpg";
	char * imagen = new char[strlen(path) + strlen(pref) + strlen(post) + 4];

	sprintf(imagen, "%s\\%s%d%s", path, pref, minPos, post);	

	cout << imagen << endl;

	IplImage * retorno = cvLoadImage(imagen, 0);	

	return retorno;
}

void CRuta::muestraDistancias() {
	cout << "Distancias = " << endl;
	cout << "50 cm. = " << masde50a << endl;
	cout << "100 cm. = " << masde100a << endl;
	cout << "150 cm. = " << masde150a << endl;
	cout << "200 cm. = " << masde200a << endl;
	cout << "250 cm. = " << masde250a << endl;
	cout << "Distancias Ponderadas = " << endl;
	cout << "50 cm. = " << masde50b << endl;
	cout << "100 cm. = " << masde100b << endl;
	cout << "150 cm. = " << masde150b  << endl;
	cout << "200 cm. = " << masde200b  << endl;
	cout << "250 cm. = " << masde250b  << endl;
}
	
	